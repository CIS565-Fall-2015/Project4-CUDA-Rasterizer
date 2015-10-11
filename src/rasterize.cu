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

	cudaFree(dev_vOut);
	cudaMalloc(&dev_vOut, vertCount * sizeof(VertexIn));

    cudaFree(dev_primitives);
    cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
    cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

	cudaFree(d_mutex);
	cudaMalloc(&d_mutex, sizeof(int));
	cudaMemset(d_mutex, 0, sizeof(int));

    checkCUDAError("rasterizeSetBuffers");
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, glm::mat4 viewProjection) {
    int sideLength2d = 8;
	int bSize = sideLength2d * sideLength2d;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

	clearDepthBuffer << <blockCount2d, blockSize2d >> > (dev_depthbuffer, width, height);
	checkCUDAError("clearDepthBuffer");

    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	int numBlock = ceil(((float)vertCount) / bSize);
	vertexShader << <numBlock, bSize >> > (dev_bufVertex, dev_vOut, vertCount, viewProjection);
	checkCUDAError("vShader");

	numBlock = ceil(((float)bufIdxSize / 3) / bSize);
	primitiveAssembly <<<numBlock, bSize >>> (dev_vOut, dev_bufIdx, bufIdxSize / 3, dev_primitives);
	checkCUDAError("primitiveAssembly");

	rasterization << <numBlock, bSize >> > (dev_primitives, bufIdxSize / 3, dev_depthbuffer, width, height, d_mutex);
	checkCUDAError("rasterization");

	fragmentShader << < blockCount2d, blockSize2d >> >(dev_framebuffer, dev_depthbuffer, width, height);
	checkCUDAError("fragmentShader");
	
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


__global__ void clearDepthBuffer(Fragment* dev_depthbuffer, int screenWidth, int screenHeight){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= screenWidth || j >= screenHeight) return;

	dev_depthbuffer[j * screenWidth + i].depth = -INFINITY;
	dev_depthbuffer[j * screenWidth + i].t = NULL;
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

	tmp = viewProjection * glm::vec4(in.nor, 1);
	out.nor = glm::vec3(tmp / tmp.w);

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
	Fragment* d_fragment, int screenWidth, int screenHeight, int* mutex)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= triNo) return;

	glm::vec3 tri[3] = {d_tri[i].v[0].pos, d_tri[i].v[1].pos, d_tri[i].v[2].pos};
	AABB bbox = getAABBForTriangle(tri);

	//start rasterizing from min to max
	//dont forget that the screen starts from -1 to 1
	int hW = screenWidth / 2;
	int hH = screenHeight / 2;

	int maxY = (1 - bbox.min.y) * hH;
	int maxX = (bbox.max.x + 1) * hW;

	for (int y = (1 - bbox.max.y) * hH; y < maxY; y++){
		for (int x = (bbox.min.x + 1) * hW; x < maxX; x++){
			glm::vec2 p;
			p.x = -1 + ((x + 0.5f) / screenWidth) * 2;
			p.y = 1 - ((y + 0.5f) / screenHeight) * 2;

			glm::vec3 bCoord = calculateBarycentricCoordinate(tri, p);

			if (isBarycentricCoordInBounds(bCoord)){
				float newDepth = getZAtCoordinate(bCoord, tri);

				int ptr = y * screenWidth + x;
				
				// mutex code from stackOverflow
				// Loop-wait until this thread is able to execute its critical section.
				bool isSet;
				do {
					isSet = (atomicCAS(mutex, 0, 1) == 0);
					if (isSet) {
						// Critical section goes here.
						// The critical section MUST be inside the wait loop;
						// if it is afterward, a deadlock will occur.

						if (newDepth <= 0 && newDepth > d_fragment[ptr].depth){
							d_fragment[ptr].depth = newDepth;
							d_fragment[ptr].bCoord = bCoord;
							d_fragment[ptr].t = &(d_tri[i]);
						}
					}
					if (isSet) {
						*mutex = 0;
					}
				} while (!isSet);
			}

		}
	}
}

__global__ void fragmentShader(glm::vec3* framebuffer, Fragment* d_fragment, int screenWidth, int screenHeight){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= screenWidth || j >= screenHeight) return;

	glm::vec3 lightSource(0,1,0);

	Triangle* t = d_fragment[j * screenWidth + i].t;
	if (t != NULL){
		glm::vec3 bCoord  = d_fragment[j * screenWidth + i].bCoord;
		glm::vec3 clr = (bCoord.x * t->v[0].col) + (bCoord.y * t->v[1].col) + (bCoord.z * t->v[2].col);
		glm::vec3 pos = (bCoord.x * t->v[0].pos) + (bCoord.y * t->v[1].pos) + (bCoord.z * t->v[2].pos);
		glm::vec3 nor = (bCoord.x * t->v[0].nor) + (bCoord.y * t->v[1].nor) + (bCoord.z * t->v[2].nor);
		
		glm::vec3 lightSource = glm::normalize(lightSource - pos);
		framebuffer[j * screenWidth + i] = glm::dot(pos, nor) * clr;
	}
}

//perform rasterization per pixel.
__global__ void rClr(Triangle* d_tri, int triNo, 
								glm::vec3* d_fragment, int screenWidth, int screenHeight)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= screenWidth || j >= screenHeight) return;

	glm::vec2 p;
	p.x = -1 + ((i + 0.5f) / screenWidth) * 2;
	p.y = 1 - ((j + 0.5f) / screenHeight) * 2;

	glm::vec3 targetBCoord;
	int targetTriPtr = -1;
	float depth = -INFINITY;

	for (int x = 0; x < triNo; x++){
		glm::vec3 tri[3] = {d_tri[x].v[0].pos, d_tri[x].v[1].pos, d_tri[x].v[2].pos};
		glm::vec3 bCoord = calculateBarycentricCoordinate(tri, p);

		if (isBarycentricCoordInBounds(bCoord)){
			float newDepth = getZAtCoordinate(bCoord, tri);
			if (newDepth <= 0 && newDepth > depth){
				depth = newDepth;
				targetBCoord = bCoord;
				targetTriPtr = x;
			}
		}
	}

	//fragment shader!
	if (targetTriPtr != -1){
		d_fragment[j * screenWidth + i] = glm::vec3(1, 1, 1);
	}
}