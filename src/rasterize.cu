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
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include "rasterizeTools.h"

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};
struct VertexOut {
    // TODO
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;
};
struct Triangle {
    VertexOut v[3];
};
struct FragmentIn {
    glm::vec3 color;
	glm::vec3 normal;
	float depth;
};
struct FragmentOut {
	glm::vec3 color;
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static VertexIn *dev_bufVertex = NULL;
static VertexOut *dev_shadedVertices = NULL;
static Triangle *dev_primitives = NULL;
static FragmentIn *dev_fragsIn = NULL;
static FragmentOut *dev_fragsOut = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;

/**
 * Kernel that writes a background color into dev_fragsOut.
 */
__global__ void clearFragsOut(glm::vec3 bgColor, int w, int h, FragmentOut *dev_fragsOut) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (x < w && y < h) {
		dev_fragsOut[x + (y * w)].color = bgColor;
		//glm::vec3 peek = dev_fragsOut[x + (y * w)].color;
		//bgColor = bgColor;
	}
}

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
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
__global__ void render(int w, int h, FragmentOut *frags, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);
	// frameBuffer code assumes (0,0) is at the bottom left in pix coords,
	// but this code assumes it's at the top right.
	int frameBufferIndex = (w - 1 - x) + (h - 1 - y) * w;
    if (x < w && y < h) {
		framebuffer[index] = frags[frameBufferIndex].color;
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
    cudaFree(dev_fragsIn);
	cudaMalloc(&dev_fragsIn, width * height * sizeof(FragmentIn));
    cudaMemset(dev_fragsIn, 0, width * height * sizeof(FragmentIn));

    cudaFree(dev_fragsOut);
	cudaMalloc(&dev_fragsOut, width * height * sizeof(FragmentOut));
    cudaMemset(dev_fragsOut, 0, width * height * sizeof(FragmentOut));
       
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

    cudaFree(dev_bufIdx);
    cudaMalloc(&dev_bufIdx, bufIdxSize * sizeof(int));
    cudaMemcpy(dev_bufIdx, bufIdx, bufIdxSize * sizeof(int), cudaMemcpyHostToDevice);

    VertexIn *bufVertex = new VertexIn[_vertCount];
	VertexOut *bufVertexOut = new VertexOut[_vertCount];
    for (int i = 0; i < vertCount; i++) {
        int j = i * 3;
        bufVertex[i].pos = glm::vec3(bufPos[j + 0], bufPos[j + 1], bufPos[j + 2]);
        bufVertex[i].nor = glm::vec3(bufNor[j + 0], bufNor[j + 1], bufNor[j + 2]);
        bufVertex[i].col = glm::vec3(bufCol[j + 0], bufCol[j + 1], bufCol[j + 2]);
    }
    cudaFree(dev_bufVertex);
    cudaMalloc(&dev_bufVertex, vertCount * sizeof(VertexIn));
    cudaMemcpy(dev_bufVertex, bufVertex, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);

	cudaFree(dev_shadedVertices);
	cudaMalloc(&dev_shadedVertices, vertCount * sizeof(VertexOut));

    cudaFree(dev_primitives);
	cudaMalloc(&dev_primitives, bufIdxSize / 3 * sizeof(Triangle));
	cudaMemset(dev_primitives, 0, bufIdxSize / 3 * sizeof(Triangle));

	delete bufVertex;
	delete bufVertexOut;

    checkCUDAError("rasterizeSetBuffers");
}

// minimal vertex shader
__global__ void minVertexShader(int vertCount, glm::mat4 tf, VertexIn *dev_verticesIn, VertexOut *dev_verticesOut) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < vertCount) {
		dev_verticesOut[i].pos = tfPoint(tf, dev_verticesIn[i].pos);
		dev_verticesOut[i].nor = dev_verticesIn[i].nor;
		glm::vec3 pos = dev_verticesOut[i].pos;
		glm::vec3 untf = dev_verticesIn[i].pos;
		dev_verticesOut[i].col = dev_verticesIn[i].col;
	}
}

// primitive assembly. 1D linear blocks expected
__global__ void minPrimitiveAssembly(int numPrimitives, VertexOut *dev_vertices, Triangle *dev_primitives) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numPrimitives) {
		dev_primitives[i].v[0] = dev_vertices[i * 3];
		dev_primitives[i].v[1] = dev_vertices[i * 3 + 1];
		dev_primitives[i].v[2] = dev_vertices[i * 3 + 2];
	}
}

// scanline rasterization. 1D linear blocks expected
__global__ void minScanlineRasterization(int w, int h, int numPrimitives, Triangle *dev_primitives, FragmentIn *dev_frags) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numPrimitives) {
		// get the AABB of the triangle
		glm::vec3 v[3];
		v[0] = dev_primitives[i].v[0].pos;
		v[1] = dev_primitives[i].v[1].pos;
		v[2] = dev_primitives[i].v[2].pos;

		AABB triangleBB = getAABBForTriangle(v);

		// triangle should have been "smooshed" to screen coordinates already.
		// walk and fill frags.
		float pixWidth = 2.0f / (float) w; // NDC goes from -1 to 1 in x and y
		float pixHeight = 2.0f / (float) h;

		int BBYmin = triangleBB.min.y * (h / 2);
		int BBYmax = triangleBB.max.y * (h / 2);
		
		int BBXmin = triangleBB.min.x * (w / 2);
		int BBXmax = triangleBB.max.x * (w / 2);

		// scan over the AABB
		for (int y = BBYmin; y < BBYmax; y++) {
			for (int x = BBXmin; x < BBXmax; x++) {
				// compute x y coordinates of the center of "this fragment"
				//printf("%i %i\n", x, y);
				glm::vec2 fragCoord = glm::vec2(x * pixWidth + pixWidth * 0.5f,
					y * pixHeight + pixHeight * 0.5f);
				// check if it's in dev_primitives[i].v using bary
				glm::vec3 baryCoordinate = calculateBarycentricCoordinate(v, fragCoord);
				if (!isBarycentricCoordInBounds(baryCoordinate)) {
					continue;
				}
				// check depth using bary
				float zDepth = getZAtCoordinate(baryCoordinate, v);
				// we're pretending NDC is -1 to +1 along each axis
				// so a fragIndx(0,0) is at NDC -1 -1
				// btw, going from NDC back to pixel coordinates:
				// I've flipped the drawing system, so now it assumes 0,0 is in the bottom left.
				int fragIndex = (x + (w / 2)) + ((y + (h / 2)) * w);
				// if all things pass ok, then insert into fragment.
				if (zDepth <= dev_frags[fragIndex].depth) {
					dev_frags[fragIndex].depth = zDepth;
					// interpolate color
					glm::vec3 interpColor = dev_primitives[i].v[0].col * baryCoordinate[0];
					interpColor += dev_primitives[i].v[1].col * baryCoordinate[1];
					interpColor += dev_primitives[i].v[2].col * baryCoordinate[2];
					dev_frags[fragIndex].color = interpColor;

					// interpolate normal
					glm::vec3 interpNorm = dev_primitives[i].v[0].nor * baryCoordinate[0];
					interpNorm += dev_primitives[i].v[1].nor * baryCoordinate[1];
					interpNorm += dev_primitives[i].v[2].nor * baryCoordinate[2];
					dev_frags[fragIndex].normal = interpNorm;					
				}
			}
		}
	}
}

__global__ void minFragmentShading(int numFrags, FragmentIn *dev_fragsIn, FragmentOut *dev_fragsOut) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numFrags) {
		//dev_fragsOut[i].color = dev_fragsIn[i].color * abs(dev_fragsIn[i].depth);
		glm::vec3 norm = dev_fragsIn[i].normal;
		dev_fragsOut[i].color[0] = abs(dev_fragsIn[i].normal[0]);
		dev_fragsOut[i].color[1] = abs(dev_fragsIn[i].normal[1]);
		dev_fragsOut[i].color[2] = abs(dev_fragsIn[i].normal[2]);
	}

}

/**
 * Perform rasterization.
 */
void firstTryRasterize(uchar4 *pbo, glm::mat4 sceneGraphTransform, glm::mat4 cameraMatrix) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockSize1d(sideLength2d * sideLength2d);

    dim3 blockCount2d_display((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	// 1) clear fragment buffer with some default value.
	clearFragsOut << <blockCount2d_display, blockSize2d >> >(glm::vec3(0.5f, 0.5f, 0.5f), width, height, dev_fragsOut);

	// 2) vertex shade
	glm::mat4 tf = cameraMatrix * sceneGraphTransform;
	dim3 blockCount1d_vertices((vertCount - 1) / blockSize1d.x + 1);

	minVertexShader << <blockCount1d_vertices, blockSize1d >> >(vertCount, tf, dev_bufVertex, dev_shadedVertices);
	checkCUDAError("debug: vertex shading");

	// 3) primitive assembly
	int numPrimitives = bufIdxSize / 3;
	dim3 blockCount1d_primitives((numPrimitives - 1) / blockSize1d.x + 1);
	minPrimitiveAssembly<<<blockCount1d_primitives, blockSize1d>>>(numPrimitives, dev_shadedVertices, dev_primitives);
	checkCUDAError("debug: primitive assembly");

	// 4) rasterization
	minScanlineRasterization<<<blockCount1d_primitives, blockSize1d>>>(width, height, numPrimitives,
		dev_primitives, dev_fragsIn);
	checkCUDAError("debug: scanline rasterization");

	// 5) fragment shading
	dim3 blockCount1d_fragments(width * height);
	minFragmentShading<<<blockCount1d_fragments, blockSize1d>>>(width * height, dev_fragsIn, dev_fragsOut);
	checkCUDAError("debug: primitive fragment shading");

	// 6) fragments to depth buffer

	// 7) depth buffer for storing depth testing fragments

	// 8) frag to frame buffer
    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d_display, blockSize2d >> >(width, height, dev_fragsOut, dev_framebuffer);
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
	sendImageToPBO << <blockCount2d_display, blockSize2d >> >(pbo, width, height, dev_framebuffer);
    checkCUDAError("rasterize");
}

void rasterize(uchar4 *pbo, glm::mat4 sceneGraphTransform, glm::mat4 cameraMatrix) {
	 int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockSize1d(sideLength2d * sideLength2d);

    dim3 blockCount2d_display((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);
	// 1) vertex shade: apply vertex transformation
	//	- first compute the overall matrix from transform and view
	//	- AND THEN TRANSFORM IT!
	// 2) primitive assembly
	// 3) rasterize (use scanline, but don't do any depth stuff there
	// 4) fragment shade
	// 5) fragment to depth buffer
	// 6) depth buffer and test
	// 7) fragment to frame buffer write

}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {
    cudaFree(dev_bufIdx);
    dev_bufIdx = NULL;

    cudaFree(dev_bufVertex);
    dev_bufVertex = NULL;

	cudaFree(dev_shadedVertices);
	dev_shadedVertices = NULL;

    cudaFree(dev_primitives);
    dev_primitives = NULL;

    cudaFree(dev_fragsIn);
    dev_fragsIn = NULL;

    cudaFree(dev_fragsOut);
    dev_fragsOut = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

    checkCUDAError("rasterizeFree");
}
