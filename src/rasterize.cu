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
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static glm::vec3 *d_lightSourcePos = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;
#define ANTIALIASING 1.0f
#define TWOAA 1.0f

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
	cudaMalloc(&dev_depthbuffer, TWOAA * width * height * sizeof(Fragment));
	cudaMemset(dev_depthbuffer, 0, TWOAA * width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	cudaMalloc(&d_mutex, TWOAA * width * height * sizeof(int));
	cudaMemset(d_mutex, 0, TWOAA * width * height * sizeof(int));

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
	cudaMalloc(&dev_primitives, bufIdxSize / 3 * sizeof(Triangle));

	cudaFree(d_lightSourcePos);
	cudaMalloc(&d_lightSourcePos, sizeof(glm::vec3));

    checkCUDAError("rasterizeSetBuffers");
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, glm::mat4 viewProjecition) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

	clearBuffers << <blockCount2d, blockSize2d >> > 
		(dev_depthbuffer, width, height);
	checkCUDAError("clearDepthBuffer");

    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	glm::vec3 lightSource(0, 2, 0);
	cudaMemcpy(d_lightSourcePos, &lightSource, sizeof(glm::vec3), cudaMemcpyHostToDevice);

	int bSize = 64;
	int numBlock = ceil(((float)vertCount) / bSize);
	vertexShader << <numBlock, bSize >> > (dev_bufVertex, dev_vOut, vertCount, viewProjecition);
	checkCUDAError("vShader");

	int numTri = bufIdxSize / 3;
	numBlock = ceil((float)numTri / bSize);
	primitiveAssembly << <numBlock, bSize >> > (dev_bufVertex, dev_vOut, dev_bufIdx, numTri, dev_primitives);
	checkCUDAError("primitiveAssembly");

	//backface culling
	Triangle* new_end = thrust::remove_if(thrust::device, dev_primitives, dev_primitives + numTri, facing_backward());
	numTri = new_end - dev_primitives;

	glm::ivec2 scissorMin(0, 0);
	glm::ivec2 scissorMax(800, 800);

	rasterization << <numBlock, bSize >> > (dev_primitives, numTri,
		dev_depthbuffer, width, height, d_mutex, d_lightSourcePos, scissorMin, scissorMax);
	checkCUDAError("rasterization");

	copyToFrameBuffer << < blockCount2d, blockSize2d >> >(dev_framebuffer,
		dev_depthbuffer, width, height);
	checkCUDAError("copyToFrameBuffer");

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


__global__ void clearBuffers(Fragment* dev_depthbuffer,
	int screenWidth, int screenHeight){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= screenWidth || j >= screenHeight) return;

	int ptr = (j * TWOAA * screenWidth) + (i * TWOAA);
	for (int offset = 0; offset < TWOAA; offset++){
		dev_depthbuffer[ptr + offset].depth = INFINITY;
		dev_depthbuffer[ptr + offset].col = glm::vec3(0, 0, 0);
	}
}
//per vertex
__global__ void vertexShader(VertexIn* d_vertsIn, VertexOut* d_vertsOut, int vertsNum, 
	glm::mat4 viewProjection){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= vertsNum) return;

	VertexIn in = d_vertsIn[i];
	VertexOut out;

	glm::vec4 v = viewProjection * glm::vec4(in.pos, 1);
	out.pos = glm::vec3(v / v.w);

	d_vertsOut[i] = out;
}

//per triangle!
__global__ void primitiveAssembly(VertexIn* d_vertsIn, VertexOut* d_vertsOut, int* d_idx, int triangleNo, Triangle* d_tri){
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= triangleNo) return;

	Triangle out;
	out.vOut[0] = d_vertsOut[d_idx[(3*i)]];
	out.vOut[1] = d_vertsOut[d_idx[(3*i)+1]];
	out.vOut[2] = d_vertsOut[d_idx[(3*i)+2]];

	out.vIn[0] = d_vertsIn[d_idx[(3*i)]];
	out.vIn[1] = d_vertsIn[d_idx[(3*i)+1]];
	out.vIn[2] = d_vertsIn[d_idx[(3*i)+2]];

	d_tri[i] = out;
}

//perform rasterization per Triangle
__global__ void rasterization(Triangle* d_tri, int triNo,
	Fragment* dev_depthbuffer, int screenWidth, int screenHeight, int* mutex,
	glm::vec3 *lightSourcePos,
	glm::ivec2 scissorMin = glm::ivec2(0, 0), 
	glm::ivec2 scissorMax = glm::ivec2(width, height))
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= triNo) return;

	Triangle t = d_tri[i];
	glm::vec3 tri[3] = { t.vOut[0].pos, t.vOut[1].pos, t.vOut[2].pos };
	float signedAreaTri = calculateSignedArea(tri);
	if (signedAreaTri <= 0) return;

	AABB bbox = getAABBForTriangle(tri);

	if (bbox.max.z > 0 && bbox.min.z < -1) return;

	//start rasterizing from min to max
	//dont forget that the screen starts from -1 to 1
	int maxY = ceil((1 - bbox.min.y) * screenHeight / 2);
	if (maxY > scissorMax.y) maxY = scissorMax.y;

	int maxX = ceil((bbox.max.x + 1) * screenWidth / 2);
	if (maxX > scissorMax.x) maxX = scissorMax.x;

	int y = (1 - bbox.max.y) * screenHeight / 2;
	if (y < scissorMin.y) y = scissorMin.y;

	int minX = (bbox.min.x + 1) * screenWidth / 2;
	if (minX < scissorMin.x) minX = scissorMin.x;


	glm::vec2 p;
	for (; y < maxY; y++){
		for (int x = minX; x < maxX; x++){
			for (int k = 0; k < ANTIALIASING; k++){
				for (int l = 0; l < ANTIALIASING; l++){
					float offsetY = (0.5f / ANTIALIASING) + (1.0f / ANTIALIASING)*k;
					float offsetX = (0.5f / ANTIALIASING) + (1.0f / ANTIALIASING)*l;

					p.x = -1 + ((x + offsetX) / screenWidth * 2);
					p.y = 1 - ((y + offsetY) / screenHeight * 2);
					glm::vec3 bCoord = calculateBarycentricCoordinate(tri, p, signedAreaTri);

					if (isBarycentricCoordInBounds(bCoord)){
						float depth = getZAtCoordinate(bCoord, tri);

						int ptr = (y * TWOAA * screenWidth) + (x * TWOAA) +
							(k * ANTIALIASING) + l;

						// mutex code from stackOverflow
						// Loop-wait until this thread is able to execute its critical section.
						bool isSet;
						do {
							isSet = (atomicCAS(&mutex[ptr], 0, 1) == 0);
							if (isSet) {
								// Critical section goes here.
								// The critical section MUST be inside the wait loop;
								// if it is afterward, a deadlock will occur.

								if (depth < dev_depthbuffer[ptr].depth){
									dev_depthbuffer[ptr].depth = depth;
								
									glm::vec3 pos = (bCoord.x * t.vIn[0].pos) + (bCoord.y * t.vIn[1].pos) + (bCoord.z * t.vIn[2].pos);
									glm::vec3 clr = (bCoord.x * t.vIn[0].col) + (bCoord.y * t.vIn[1].col) + (bCoord.z * t.vIn[2].col);
									glm::vec3 nor = (bCoord.x * t.vIn[0].nor) + (bCoord.y * t.vIn[1].nor) + (bCoord.z * t.vIn[2].nor);

									dev_depthbuffer[ptr].col = glm::dot(glm::normalize(*lightSourcePos - pos), nor) * clr;
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
	}
}

__global__ void copyToFrameBuffer(glm::vec3* dev_framebuffer, Fragment* dev_depthbuffer, int width, int height){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int i = x + (y * width);

	if (x >= width || y >= height) return;

	glm::vec3 clrOut(0, 0, 0);

	int ptr = (y * TWOAA * width) + (x * TWOAA);
	for (int offset = 0; offset < TWOAA; offset++){
		if (dev_depthbuffer[ptr + offset].depth != INFINITY){
			clrOut += dev_depthbuffer[ptr + offset].col;
		}
		else 
			clrOut += glm::vec3(0.3, 0.3, 0.3);
	}
	
	dev_framebuffer[i] = clrOut / TWOAA;
}