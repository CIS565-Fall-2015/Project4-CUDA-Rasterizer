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

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};
struct VertexOut {
    // TODO: Mirrors VertexIn?
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;
};
struct Triangle {
    VertexOut v[3];
};
struct Fragment {
    glm::vec3 color;
	// will probably need to add more here? Will require updating my clear method
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static VertexIn *dev_bufVertex = NULL; //TODO Shouldn't this really be changed to indicate that it is in?
static VertexOut *dev_bufVertexOut = NULL;
static Triangle *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;

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
 * Clears the depth buffer with a black color.
 */
__global__
void clearDepthBuffer(int w, int h, Fragment *depthbuffer) {
	// TODO: Block stuff wrong?
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		depthbuffer[index].color = glm::vec3(0.0f);
	}
}

/**
 * Apply vertex transformations and transfer to vertex out buffer
 */
__global__
void vertexShading(int w, int h, VertexIn *vertexBufferIn, VertexOut *vertexBufferOut) {
	// TODO: Block stuff wrong?
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	// in this iteration just copy shit over. Later iterations will do transforms
	if (x < w && y < h) {
		vertexBufferOut[index].col = vertexBufferIn[index].col;
		vertexBufferOut[index].nor = vertexBufferIn[index].nor;
		vertexBufferOut[index].pos = vertexBufferIn[index].pos;
	}
}

/**
 * Assemble primitives from vertex out buffer data.
 */
__global__
void assemblePrimitives(int w, int h, VertexOut *vertexBufferOut, Triangle *primitives) {
	// TODO: Block stuff is WAY wrong here. 
	// Currently only supports triangles
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		Triangle primitive = primitives[index / 3];
		for (int i = 0; i < 3; i++) {
			primitive.v[i] = vertexBufferOut[3 * index + i]; // three times as many verts as triangles, gotta offset by index
			// You're getting the 3 verts that make up a triangle
		}
	}
}

__global__
void scanlineRaserization(int w, int h, Triangle *primitives, Fragment *depthbuffer) {
	// this is going to be the most incomplete one.
	// how large is hte depthbuffer vs the number of primitives? how do I insert them?
	// which buffer are they going to right now?
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
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
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)
	

	//so each stage of the pipeline is just anotehr kernel function?
	// Or should they be combined somewhat into one or two kerns?

	// first clear the depth buffer with some default value (0?)
	// TODO: Block stuff wrong?
	clearDepthBuffer<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer);

	// Vertex shading
	// TODO: Block stuff wrong?
	vertexShading<<<blockCount2d, blockSize2d>>>(width, height, dev_bufVertex, dev_bufVertexOut);

	// Primitive Assembly
	// TODO: Block stuff wrong?
	assemblePrimitives<<<blockCount2d, blockSize2d>>>(width, height, dev_bufVertexOut, dev_primitives);

	// rasterization

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
