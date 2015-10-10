/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#include "rasterize.h"

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include "rasterizeTools.h"
#include "../stream_compaction/efficient.h"

//TODO: Experiment with these values
#define VERTBLOCKSIZE 256
#define FRAGBLOCKSIZE 256

static int width = 0;
static int height = 0;
static Scene *scene = NULL;
static int *dev_bufIdx = NULL;
static VertexIn *dev_bufVertex = NULL; //TODO Shouldn't this really be changed to indicate that it is in?
static VertexOut *dev_bufVertexOut = NULL;
static Triangle *dev_primitives = NULL;
static int *dev_depth = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;
static int primitiveCount = 0;
static Triangle *dev_compactionOutput = NULL;

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

// Writes fragment colors to the framebuffer
__global__
void render(int w, int h, Fragment *depthbuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        framebuffer[index] = depthbuffer[index].color;
    }
}

/**
 * Clears the depth buffers and primitive buffer.
 */
void clearDepthBuffer() {
	cudaMemset(dev_depth, scene->farPlane * 10000, width * height * sizeof(int));
	cudaMemset(dev_depthbuffer, 0.0f, width * height * sizeof(Fragment));
}

/**
 * Apply vertex transformations and transfer to vertex out buffer
 */
__global__
void vertexShading(int w, int h, int nearPlane, int farPlane, int vertexCount, const VertexIn *vertexBufferIn, 
	VertexOut *vertexBufferOut, const glm::mat4 modelView) {
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * w);
	
	if (index < vertexCount) {
		glm::vec4 clipCoordinates = modelView * glm::vec4(vertexBufferIn[index].pos, 1.0f);
		glm::vec3 normDeviceCoordinates = glm::vec3(clipCoordinates.x, clipCoordinates.y, clipCoordinates.z) / clipCoordinates.w;

		vertexBufferOut[index].pos = glm::vec3(w * 0.5f * (normDeviceCoordinates.x + 1.0f),
			h * 0.5f * (normDeviceCoordinates.y + 1.0f), 0.5f * ((farPlane - nearPlane) 
			* normDeviceCoordinates.z + (farPlane + nearPlane)));
		vertexBufferOut[index].col = vertexBufferIn[index].col;
		vertexBufferOut[index].nor = vertexBufferIn[index].nor;
		vertexBufferOut[index].model_pos = vertexBufferIn[index].pos;
	}
}

/**
 * Assemble primitives from vertex out buffer data.
 */
__global__
void assemblePrimitives(int primitiveCount, const VertexOut *vertexBufferOut, Triangle *primitives, const int *bufIdx) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < primitiveCount) {
		for (int i = 0; i < 3; i++) {
			primitives[index].v[i] = vertexBufferOut[bufIdx[3 * index + i]];
		}

		primitives[index].boundingBox = getAABBForTriangle(primitives[index]);
		primitives[index].visible = true;
	}
}

__global__
void rasterization(int w, int h, int primitiveCount, Triangle *primitives, Fragment *depthbuffer, int *depth) {
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * w);

	if (index < primitiveCount) {
		// Only doing scanline triangle atm
		int minX = fmaxf(round(primitives[index].boundingBox.min.x), 0.0f), minY = fmaxf(round(primitives[index].boundingBox.min.y), 0.0f);
		int maxX = fminf(round(primitives[index].boundingBox.max.x), (float)w), maxY = fminf(round(primitives[index].boundingBox.max.y), (float)h);
		glm::vec3 baryCentricCoordiante;
		// Temp until barycentric coord calc handles triangles

		// Loop through each scanline, then each pixel on the line
		for (int y = maxY; y >= minY; y--) {
			for (int x = minX; x <= maxX; x++) {
				// TODO: Update to handle triangles coming in, not an array
				baryCentricCoordiante = calculateBarycentricCoordinate(primitives[index], glm::vec2(x, y));
				if (isBarycentricCoordInBounds(baryCentricCoordiante)) {
					// TODO: Update to handle triangle
					int z = getZAtCoordinate(baryCentricCoordiante, primitives[index]) * 10000.0f;
					int depthIndex = w - x + (h - y) * w;

					atomicMin(&depth[depthIndex], z);

					if (depth[depthIndex] == z) {
						depthbuffer[depthIndex].color = baryCentricCoordiante.x * primitives[index].v[0].col + baryCentricCoordiante.y
							* primitives[index].v[1].col + baryCentricCoordiante.z * primitives[index].v[2].col;
						depthbuffer[depthIndex].position = baryCentricCoordiante.x * primitives[index].v[0].pos + baryCentricCoordiante.y
							* primitives[index].v[1].pos + baryCentricCoordiante.z * primitives[index].v[2].pos;
						depthbuffer[depthIndex].normal = baryCentricCoordiante.x * primitives[index].v[0].nor + baryCentricCoordiante.y
							* primitives[index].v[1].nor + baryCentricCoordiante.z * primitives[index].v[2].nor;
					}
				}
			}
		}
	}
}

__global__
void pointRasterization(int w, int h, int primitiveCount, Triangle *primitives, Fragment *depthbuffer, int *depth) {
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * w);

	if (index < primitiveCount) {
		Triangle primitive = primitives[index];
		int x = round(primitive.v[1].pos.x), y = round(primitive.v[1].pos.y);
		int z = primitive.v[1].pos.z * 10000.0f;
		int depthIndex = w - x + (h - y) * w;

		atomicMin(&depth[depthIndex], z);

		if (depth[depthIndex] == z) {
			Fragment fragment;
			fragment.color = primitive.v[1].col;
			fragment.position = primitive.v[1].pos;
			fragment.normal = primitive.v[1].nor;
			depthbuffer[depthIndex] = fragment;
		}
	}
}

__global__
void lineRasterization(int w, int h, int primitiveCount, Triangle *primitives, Fragment *depthbuffer, int *depth) {
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * w);

	if (index < primitiveCount) {
		Triangle primitive = primitives[index];
		glm::vec3 minPosition = primitive.v[0].pos, maxPosition = primitive.v[1].pos;

		if (round(minPosition.x) == round(maxPosition.x)) {
			// Get straight vertical line
			int x = round(minPosition.x);
			if (minPosition.y > maxPosition.y) {
				// Flip
				minPosition = primitive.v[1].pos;
				maxPosition = primitive.v[0].pos;
			}

			for (int y = round(maxPosition.y); y >= round(minPosition.y); y--) {
				float minMaxRatio = __fdividef(y - minPosition.y, maxPosition.y - minPosition.y);
				int depthIndex = w - x + (h - y) * w;
				int z = -(minMaxRatio * round(minPosition.z) + (1.0f - minMaxRatio) * round(maxPosition.z));

				atomicMin(&depth[depthIndex], z);

				if (depth[depthIndex] == z) {
					Fragment fragment;
					fragment.color = primitive.v[1].col;
					fragment.position = glm::vec3(x, y, -z);
					fragment.normal = glm::normalize(primitive.v[0].nor + primitive.v[1].nor);
					depthbuffer[depthIndex] = fragment;
				}
			}
		}
		else {
			//Have to calculate a Bresenham line
			if (round(minPosition.x) > round(maxPosition.x)) {
				// Swap
				minPosition = primitive.v[1].pos;
				maxPosition = primitive.v[0].pos;
			}

			float slope = (maxPosition.y - minPosition.y) / (maxPosition.x - minPosition.x);

			for (int x = round(minPosition.x); x <= round(maxPosition.x); x++) {
				int y = slope * (x - round(minPosition.x)) + minPosition.y;
				float minMaxRatio = __fdividef(y - minPosition.y, maxPosition.y - minPosition.y);
				int depthIndex = w - x + (h - y) * w;
				int z = -(minMaxRatio * minPosition.z + (1.0f - minMaxRatio) * maxPosition.z);

				atomicMin(&depth[depthIndex], z);

				if (depth[depthIndex] == z) {
					Fragment fragment;
					fragment.color = primitive.v[1].col;
					fragment.position = glm::vec3(x, y, -z);
					fragment.normal = glm::normalize(primitive.v[0].nor + primitive.v[1].nor);
					depthbuffer[depthIndex] = fragment;
				}
			}
		}
	}
}

/**
* Fragment shader
*/
__global__
void fragmentShading(int w, int h, Fragment *depthBuffer, const Light light) {
	// TODO: Handle an array of lights
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < (w * h)) {
		Fragment fragment = depthBuffer[index];
		depthBuffer[index].color = (glm::dot(glm::normalize(light.position - fragment.position), fragment.normal)
			* fragment.color * light.color);
	}
}

__global__
void backFaceCulling(int w, int primitiveCount, Triangle *primitives, glm::vec3 cameraPosition) {
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * w);

	if (index < primitiveCount) {
		Triangle primitive = primitives[index];
		if (glm::dot(primitive.v[0].model_pos - cameraPosition, primitive.v[0].nor) >= 0.0f) {
			//TODO: ^^ Actually shouldn't I interpolate between the three vertices here?
			primitives[index].visible = false;
		}
	}
}

__global__
void scissorTest(int w, int primitiveCount, Triangle *primitives, const glm::vec2 scissorMax, const glm::vec2 scissorMin) {
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * w);

	if (index < primitiveCount) {
		Triangle primitive = primitives[index];
		if (primitive.boundingBox.min.y > scissorMax.y || primitive.boundingBox.max.y < scissorMin.y ||
			primitive.boundingBox.max.x > scissorMax.x || primitive.boundingBox.max.x < scissorMin.x) {
			primitives[index].visible = false;
		}
	}
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h, Scene *s) {
    width = w;
    height = h;
	scene = s;

	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));
	cudaMemset(dev_depth, scene->farPlane * 10000.0f, width * height * sizeof(int));
    cudaFree(dev_depthbuffer);
    cudaMalloc(&dev_depthbuffer,   width * height * sizeof(Fragment));
    cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    checkCUDAError("rasterizeInit");
}

/**
 * Set all of the buffers necessary for rasterization.
 */
void rasterizeSetBuffers(
        int _bufIdxSize, int *bufIdx,
        int _vertCount, float *bufPos, float *bufNor, float *bufCol) {
    bufIdxSize = _bufIdxSize;
    vertCount = _vertCount;
	primitiveCount = vertCount / 3;

    cudaFree(dev_bufIdx);
    cudaMalloc(&dev_bufIdx, bufIdxSize * sizeof(int));
    cudaMemcpy(dev_bufIdx, bufIdx, bufIdxSize * sizeof(int), cudaMemcpyHostToDevice);

    VertexIn *bufVertex = new VertexIn[_vertCount];
    for (int i = 0; i < vertCount; i++) {
        int j = i * 3;
        bufVertex[i].pos = glm::vec3(bufPos[j + 0], bufPos[j + 1], bufPos[j + 2]);
        bufVertex[i].nor = glm::vec3(bufNor[j + 0], bufNor[j + 1], bufNor[j + 2]);
        bufVertex[i].col = glm::vec3(bufCol[j + 0], bufCol[j + 1], bufCol[j + 2]);
    }
    cudaFree(dev_bufVertex);
    cudaMalloc(&dev_bufVertex, vertCount * sizeof(VertexIn));
    cudaMemcpy(dev_bufVertex, bufVertex, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);

	cudaFree(dev_bufVertexOut);
	cudaMalloc(&dev_bufVertexOut, vertCount * sizeof(VertexOut));
	cudaMemset(dev_bufVertexOut, 0, vertCount * sizeof(VertexIn));

    cudaFree(dev_primitives);
    cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
    cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

	cudaFree(dev_compactionOutput);
	cudaMalloc(&dev_compactionOutput, vertCount / 3 * sizeof(Triangle));
	cudaMemset(dev_compactionOutput, 0, vertCount / 3 * sizeof(Triangle));

    checkCUDAError("rasterizeSetBuffers");
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockCount2d((width + blockSize2d.x - 1) / blockSize2d.x,
		(height + blockSize2d.y - 1) / blockSize2d.y);
	int vertexBlockSize = VERTBLOCKSIZE, fragmentBlockSize = FRAGBLOCKSIZE;
	int vertexGridSize = (vertCount + VERTBLOCKSIZE - 1) / VERTBLOCKSIZE;
	int fragmentGridSize = (width * height + FRAGBLOCKSIZE - 1) / FRAGBLOCKSIZE;

	primitiveCount = vertCount / 3;
	
    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	// Clear depth buffer
	clearDepthBuffer();

	// Vertex shading
	vertexShading<<<vertexGridSize, vertexBlockSize>>>(width, height, scene->nearPlane, scene->farPlane, vertCount, dev_bufVertex, dev_bufVertexOut, scene->modelView);
	
	// Primitive Assembly
	assemblePrimitives<<<vertexGridSize, vertexBlockSize>>>(primitiveCount, dev_bufVertexOut, dev_primitives, dev_bufIdx);

	// Culling after Primitive assembly
	if (scene->culling) {
		backFaceCulling<<<blockCount2d, blockSize2d>>>(width, primitiveCount, dev_primitives, scene->camera.position);
		primitiveCount = StreamCompaction::Efficient::Compact(primitiveCount, dev_compactionOutput, dev_primitives);
		cudaMemcpy(dev_primitives, dev_compactionOutput, primitiveCount * sizeof(Triangle), cudaMemcpyDeviceToDevice);
	}

	// Scissor test
	if (scene->scissor) {
		scissorTest<<<blockCount2d, blockSize2d>>>(width, primitiveCount, dev_primitives, scene->scissorMax, scene->scissorMin);
		primitiveCount = StreamCompaction::Efficient::Compact(primitiveCount, dev_compactionOutput, dev_primitives);
		cudaMemcpy(dev_primitives, dev_compactionOutput, primitiveCount * sizeof(Triangle), cudaMemcpyDeviceToDevice);
	}

	// rasterization
	// Choose between primitive types based on scene file
	if (scene->pointRasterization) {
		pointRasterization<<<blockCount2d, blockSize2d>>>(width, height, primitiveCount, dev_primitives, dev_depthbuffer, dev_depth);
	}
	else if (scene->lineRasterization) {
		lineRasterization<<<blockCount2d, blockSize2d>>>(width, height, primitiveCount, dev_primitives, dev_depthbuffer, dev_depth);
	}
	else {
		// Standard triangle rasterization
		rasterization<<<blockCount2d, blockSize2d>>>(width, height, primitiveCount, dev_primitives, dev_depthbuffer, dev_depth);
	}

	// Fragment shading
	fragmentShading<<<fragmentGridSize, fragmentBlockSize>>>(width, height, dev_depthbuffer, scene->light);

    // Copy depthbuffer colors into framebuffer
    render<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer, dev_framebuffer);

    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("rasterize");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {
    cudaFree(dev_bufIdx);
    dev_bufIdx = NULL;

    cudaFree(dev_bufVertex);
    dev_bufVertex = NULL;

    cudaFree(dev_primitives);
    dev_primitives = NULL;

    cudaFree(dev_depthbuffer);
    dev_depthbuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	cudaFree(dev_compactionOutput);
	dev_compactionOutput = NULL;

    checkCUDAError("rasterizeFree");
}
