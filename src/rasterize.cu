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

//TODO: Experiment with these values
#define VERTBLOCKSIZE 128
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
	// TODO: Should I be clearing primitives ever? If I add mouse movement?
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
		VertexIn vertexIn = vertexBufferIn[index];
		VertexOut vertexOut;
		glm::vec4 clipCoordinates = modelView * glm::vec4(vertexIn.pos, 1.0f);
		glm::vec3 normDeviceCoordinates = glm::vec3(clipCoordinates.x, clipCoordinates.y, clipCoordinates.z) / clipCoordinates.w;

		vertexOut.pos = glm::vec3(w * 0.5f * (normDeviceCoordinates.x + 1.0f),
			h * 0.5f * (normDeviceCoordinates.y + 1.0f), 0.5f * ((farPlane - nearPlane) 
			* normDeviceCoordinates.z + (farPlane + nearPlane)));
		vertexOut.col = vertexIn.col;
		vertexOut.nor = vertexIn.nor;
		vertexOut.model_pos = vertexIn.pos;
		vertexBufferOut[index] = vertexOut;
	}
}

/**
 * Assemble primitives from vertex out buffer data.
 */
__global__
void assemblePrimitives(int primitiveCount, const VertexOut *vertexBufferOut, Triangle *primitives, const int *bufIdx) {
	// Currently only supports triangles
	// TODO: How will I differentiate between points and lines?
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < primitiveCount) {
		Triangle primitive;
		for (int i = 0; i < 3; i++) {
			primitive.v[i] = vertexBufferOut[bufIdx[3 * index + i]];
		}

		primitive.boundingBox = getAABBForTriangle(primitive);
		primitive.visible = true;
		primitives[index] = primitive;
	}
}

__global__
void raserization(int w, int h, int primitiveCount, Triangle *primitives, Fragment *depthbuffer, int *depth) {
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * w);

	if (index < primitiveCount) {
		// Only doing scanline triangle atm
		Triangle primitive = primitives[index];
		int minX = fmaxf(round(primitive.boundingBox.min.x), 0.0f), minY = fmaxf(round(primitive.boundingBox.min.y), 0.0f);
		int maxX = fminf(round(primitive.boundingBox.max.x), (float)w), maxY = fminf(round(primitive.boundingBox.max.y), (float)h);
		glm::vec3 baryCentricCoordiante;
		// Temp until barycentric coord calc handles triangles
		glm::vec3 coordinate[3] = {
			primitive.v[0].pos,
			primitive.v[1].pos,
			primitive.v[2].pos,
		};

		// Loop through each scanline, then each pixel on the line
		for (int y = maxY; y >= minY; y--) {
			for (int x = minX; x <= maxX; x++) {
				// TODO: Update to handle triangles coming in, not an array
				baryCentricCoordiante = calculateBarycentricCoordinate(coordinate, glm::vec2(x, y));
				if (isBarycentricCoordInBounds(baryCentricCoordiante)) {
					// TODO: Update to handle triangle
					int z = getZAtCoordinate(baryCentricCoordiante, coordinate) * 10000;
					int depthIndex = w - x + (h - y) * w;

					atomicMin(&depth[depthIndex], z);

					if (depth[depthIndex] == z) {
						Fragment fragment;
						fragment.color = baryCentricCoordiante.x * primitive.v[0].col + baryCentricCoordiante.y 
							* primitive.v[1].col + baryCentricCoordiante.z * primitive.v[2].col;
						fragment.position = baryCentricCoordiante.x * primitive.v[0].pos + baryCentricCoordiante.y
							* primitive.v[1].pos + baryCentricCoordiante.z * primitive.v[2].pos;
						fragment.normal = baryCentricCoordiante.x * primitive.v[0].nor + baryCentricCoordiante.y
							* primitive.v[1].nor + baryCentricCoordiante.z * primitive.v[2].nor;
						depthbuffer[depthIndex] = fragment;
					}
				}
			}
		}
	}
}

/**
* Fragment shader
*/
__global__
void fragmentShading(int w, int h, Fragment *depthBuffer, const Light light1, const Light light2) {
	// TODO: Handle an array of lights
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < (w * h)) {
		Fragment fragment = depthBuffer[index];
		depthBuffer[index].color = (glm::dot(glm::normalize(light1.position - fragment.position), fragment.normal) 
			* fragment.color * light1.color) + (glm::dot(glm::normalize(light2.position - fragment.position), 
			fragment.normal) * fragment.color * light2.color);
	}
}

__global__
void backFaceCulling(int w, int primitiveCount, Triangle *primitives, glm::vec3 cameraPosition) {
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + (((blockIdx.y * blockDim.y) + threadIdx.y) * w);

	if (index < primitiveCount) {
		Triangle primitive = primitives[index];
		if (glm::dot(primitive.v[0].model_pos - cameraPosition, primitive.v[0].nor) >= 0.0f) {
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
	cudaMemset(dev_depth, scene->farPlane * 10000, width * height * sizeof(int));
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
	int vertexGridSize = (width * height + VERTBLOCKSIZE - 1) / VERTBLOCKSIZE;
	int fragmentGridSize = (width * height + FRAGBLOCKSIZE - 1) / FRAGBLOCKSIZE;
	
    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	// Clear depth buffer
	clearDepthBuffer();

	// Vertex shading
	vertexShading<<<vertexGridSize, vertexBlockSize>>>(width, height, scene->nearPlane, scene->farPlane, vertCount, dev_bufVertex, dev_bufVertexOut, scene->modelView);
	
	// Primitive Assembly
	assemblePrimitives<<<vertexGridSize, vertexBlockSize>>>(primitiveCount, dev_bufVertexOut, dev_primitives, dev_bufIdx);

	// Culling after Primitive assembly

	// rasterization
	raserization<<<blockCount2d, blockSize2d>>>(width, height, primitiveCount, dev_primitives, dev_depthbuffer, dev_depth);

	// Fragment shading
	fragmentShading<<<fragmentGridSize, fragmentBlockSize>>>(width, height, dev_depthbuffer, scene->light1, scene->light2);

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

    checkCUDAError("rasterizeFree");
}
