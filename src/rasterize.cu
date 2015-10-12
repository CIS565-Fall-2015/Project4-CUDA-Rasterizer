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

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static int *d_mutex = NULL;
static VertexIn *dev_bufVertex = NULL;
static VertexOut *dev_vOut = NULL;
static Triangle *dev_primitives = NULL;
static float *dev_depthbuffer = NULL;
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

/*
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
*/

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
    cudaFree(dev_depthbuffer);
    cudaMalloc(&dev_depthbuffer,   width * height * sizeof(float));
	cudaMemset(dev_depthbuffer, -INFINITY, width * height * sizeof(float));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	cudaMalloc(&d_mutex, width * height * sizeof(int));
	cudaMemset(d_mutex, 0, width * height * sizeof(int));

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

	cudaFree(dev_vOut);
	cudaMalloc(&dev_vOut, vertCount * sizeof(VertexIn));

    cudaFree(dev_primitives);
    cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
    cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

    checkCUDAError("rasterizeSetBuffers");
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, glm::mat4 viewProjection) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

	clearBuffers << <blockCount2d, blockSize2d >> > 
		(dev_depthbuffer, dev_framebuffer, width, height);
	checkCUDAError("clearDepthBuffer");

    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	int bSize = 64;
	int numBlock = ceil(((float)vertCount) / bSize);
	vertexShader << <numBlock, bSize >> > (dev_bufVertex, dev_vOut, vertCount, viewProjection);
	checkCUDAError("vShader");

	numBlock = ceil(((float)bufIdxSize / 3) / bSize);
	primitiveAssembly <<<numBlock, bSize >>> (dev_vOut, dev_bufIdx, bufIdxSize / 3, dev_primitives);
	checkCUDAError("primitiveAssembly");

	rasterization << <numBlock, bSize >> > (dev_primitives, bufIdxSize / 3, 
		dev_depthbuffer, dev_framebuffer, width, height, d_mutex);
	checkCUDAError("rasterization");

	/*
	fragmentShader << < blockCount2d, blockSize2d >> >(dev_framebuffer, dev_depthbuffer, width, height);
	checkCUDAError("fragmentShader");
	*/

	// rClr << < blockCount2d, blockSize2d >> >(dev_primitives, bufIdxSize / 3,
	//	dev_framebuffer, width, height);

    // Copy depthbuffer colors into framebuffer
    //render<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer, dev_framebuffer);
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

	cudaFree(dev_vOut);
	dev_vOut = NULL;

    checkCUDAError("rasterizeFree");
}


__global__ void clearBuffers(float* dev_depthbuffer, glm::vec3* dev_framebuffer,
	int screenWidth, int screenHeight){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= screenWidth || j >= screenHeight) return;

	dev_depthbuffer[j * screenWidth + i] = -INFINITY;
	dev_framebuffer[j * screenWidth + i] = glm::vec3(0,0,0);
}
//per vertex
__global__ void vertexShader(VertexIn* d_vertsIn, VertexOut* d_vertsOut, int vertsNum, 
	glm::mat4 viewProjection){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= vertsNum) return;

	VertexIn in = d_vertsIn[i];
	VertexOut out;

	glm::vec4 tmp = viewProjection * glm::vec4(in.pos, 1);
	out.pos = glm::vec3(tmp / tmp.w);

	tmp = viewProjection * glm::vec4(in.nor, 0);
	out.nor = glm::vec3(tmp);

	out.col = in.col;

	d_vertsOut[i] = out;
}

//per triangle!
__global__ void primitiveAssembly(VertexOut* d_vertsOut, int* d_idx, int triangleNo, Triangle* d_tri){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= triangleNo) return;

	Triangle out;
	out.v[0] = d_vertsOut[d_idx[(3*i)]];
	out.v[1] = d_vertsOut[d_idx[(3*i)+1]];
	out.v[2] = d_vertsOut[d_idx[(3*i)+2]];

	d_tri[i] = out;
}

//perform rasterization per Triangle
__global__ void rasterization(Triangle* d_tri, int triNo,
	float* dev_depthbuffer, glm::vec3* dev_framebuffer,
	int screenWidth, int screenHeight, int* mutex)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= triNo) return;

	Triangle t = d_tri[i];
	glm::vec3 tri[3] = {t.v[0].pos, t.v[1].pos, t.v[2].pos};
	AABB bbox = getAABBForTriangle(tri);

	if (bbox.min.x > 1 || bbox.min.y > 1 || bbox.min.z > 1 ||
		bbox.max.x < -1 || bbox.max.y < -1 || bbox.max.z < 0)
		return;

	//start rasterizing from min to max
	//dont forget that the screen starts from -1 to 1
	int maxY = ceil((1 - bbox.min.y) * screenHeight / 2);
	if (maxY > screenHeight) maxY = screenHeight;
	int maxX = ceil((bbox.max.x + 1) * screenWidth / 2);
	if (maxX > screenWidth) maxX = screenWidth;
	int y = (1 - bbox.max.y) * screenHeight / 2;
	if (y < 0) y = 0;
	int minX = (bbox.min.x + 1) * screenWidth / 2;
	if (minX < 0) minX = 0;

	glm::vec2 p;
	for (; y < maxY; y++){
		p.y = 1 - ((y + 0.5f) / screenHeight * 2);

		for (int x = minX; x < maxX; x++){
			p.x = -1 + ((x + 0.5f) / screenWidth * 2);

			glm::vec3 bCoord = calculateBarycentricCoordinate(tri, p);

			if (isBarycentricCoordInBounds(bCoord)){
				glm::vec3 pos = (bCoord.x * tri[0]) + (bCoord.y * tri[1]) + (bCoord.z * tri[2]);
				int ptr = y * screenWidth + x;

				// mutex code from stackOverflow
				// Loop-wait until this thread is able to execute its critical section.
				bool isSet;
				do {
					isSet = (atomicCAS(&mutex[ptr], 0, 1) == 0);
					if (isSet) {
						// Critical section goes here.
						// The critical section MUST be inside the wait loop;
						// if it is afterward, a deadlock will occur.

						if (-pos.z > dev_depthbuffer[ptr]){
							
							glm::vec3 clr = (bCoord.x * t.v[0].col) + (bCoord.y * t.v[1].col) + (bCoord.z * t.v[2].col);
							glm::vec3 nor = (bCoord.x * t.v[0].nor) + (bCoord.y * t.v[1].nor) + (bCoord.z * t.v[2].nor);
		
							clr = glm::dot(glm::normalize(glm::vec3(0, 1, 0) - pos), nor) * clr;
							dev_framebuffer[ptr] = clr;
							dev_depthbuffer[ptr] = -pos.z;
						}
					}
					if (isSet) {
						mutex[ptr] = 0;
					}
				} while (!isSet);
			}
		}
	}
}