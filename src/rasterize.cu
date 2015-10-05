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
#include <glm/gtc/matrix_transform.hpp>

#define MAX_THREADS 512
#define FIXED_SIZE 1000

struct Light {
	glm::vec3 pos;
	glm::vec3 color;
};

struct Cam {
	glm::vec3 pos;
	glm::vec3 focus;
	glm::vec3 up;
	int height;
	int width;
	float aspect;
	float fovy;
	float zNear;
	float zFar;
};

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};
struct VertexOut {
	glm::vec3 pos;
	glm::vec3 ndc_pos;
	glm::vec3 nor;
	glm::vec3 col;
};
struct Triangle {
    VertexOut v[3];
};
struct Fragment {
    glm::vec3 color;
	glm::vec3 norm;
	glm::vec3 pos;
	float depth;
	int fixed_depth;
	VertexOut v;
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static VertexIn *dev_bufVertex = NULL;
static VertexOut *dev_bufVertexOut = NULL;
static Triangle *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;

//TODO: Change these so we can move the camera around
static Cam cam; 
static glm::mat4 Mview;
static glm::mat4 Mmod;
static glm::mat4 Mproj;

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

__global__ void initDepths(int n, Fragment* depthbuffer){
	int index = threadIdx.x + (blockDim.x*blockIdx.x);

	if (index < n){
		depthbuffer[index].fixed_depth = 1 * FIXED_SIZE;
	}
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

	cam.width = width;
	cam.height = height;
	cam.pos = glm::vec3(0.0f, 0.5f, 10.0f);
	cam.focus = glm::vec3(0.0f, 0.0f, 0.0f);
	cam.up = glm::vec3(0.0f, 1.0f, 0.0f);
	cam.fovy = 30.0f;
	cam.zNear = 0.01f;
	cam.zFar = 100.0f;
	cam.aspect = 1.0f;

	Mmod = glm::mat4(1.0f)*1.0f;
	Mmod[3][3] = 1.0f;
	Mview = glm::lookAt(cam.pos, cam.focus, cam.up);
	Mproj = glm::perspective(cam.fovy, cam.aspect, cam.zNear, cam.zFar);
	//Mproj = glm::perspective(35.0f, 1.0f, 0.1f, 100.0f);
	//printf("%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n", Mmod[0][0], Mmod[0][1], Mmod[0][2], Mmod[0][3], 
	//												  Mmod[1][0], Mmod[1][1], Mmod[1][2], Mmod[1][3],
	//												  Mmod[2][0], Mmod[2][1], Mmod[2][2], Mmod[2][3], 
	//												  Mmod[3][0], Mmod[3][1], Mmod[3][2], Mmod[3][3]);
	//printf("");
}

/**
 * Set all of the buffers necessary for rasterization.
 */
void rasterizeSetBuffers(
        int _bufIdxSize, int *bufIdx,
        int _vertCount, float *bufPos, float *bufNor, float *bufCol) {
    bufIdxSize = _bufIdxSize;
    vertCount = _vertCount;

	int numBlocks = (width*height - 1) / MAX_THREADS + 1;
	initDepths<<<numBlocks, MAX_THREADS>>>(width*height, dev_depthbuffer);

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

    cudaFree(dev_primitives);
    cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
    cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

    checkCUDAError("rasterizeSetBuffers");
}

__global__ void kernShadeVertices(int n, VertexOut* vs_output, VertexIn* vs_input, glm::mat4 Mp, glm::mat4 Mv, glm::mat4 Mm){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		vs_output[index].pos = vs_input[index].pos;
		
		//vs_output[index].ndc_pos = Mview * glm::vec4(vs_input[index].pos, 1.0f);
		glm::vec4 new_pos = Mp * Mv * Mm * glm::vec4(vs_input[index].pos, 1.0f);

		vs_output[index].ndc_pos = glm::vec3(new_pos / new_pos.w);
		//vs_output[index].ndc_pos = glm::vec4(vs_input[index].pos, 1.0f);
		vs_output[index].nor = vs_input[index].nor;
		vs_output[index].col = vs_input[index].col;
	}
}

__global__ void kernAssemblePrimitives(int n, Triangle* primitives, VertexOut* vs_output, int* idx){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		int idx0 = idx[3 * index + 0];
		int idx1 = idx[3 * index + 1];
		int idx2 = idx[3 * index + 2];
		primitives[index].v[0] = vs_output[idx0];
		primitives[index].v[1] = vs_output[idx1];
		primitives[index].v[2] = vs_output[idx2];
	}
}

// Each thread is responsible for rasterizing a single triangle
__global__ void kernRasterize(int n, Cam cam, Fragment* fs_input, Triangle* primitives){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		Triangle prim = primitives[index];

		glm::vec3* tri = new glm::vec3[3];

		tri[0] = glm::vec3(prim.v[0].ndc_pos);
		tri[1] = glm::vec3(prim.v[1].ndc_pos);
		tri[2] = glm::vec3(prim.v[2].ndc_pos);

		AABB aabb = getAABBForTriangle(tri);
		glm::vec3 bary;
		glm::vec2 point;

		// TODO: Snap i,j to nearest fragment coordinate
		float dx = 2.0f / (float)cam.width;
		float dy = 2.0f / (float)cam.height;

		float x;
		float y;

		int mini = max((int)(aabb.min.x / dx) + cam.width / 2, 0);
		int minj = max((int)(aabb.min.y / dy) + cam.height / 2, 0);
		int maxi = min((int)(aabb.max.x / dx) + cam.width / 2, cam.width);
		int maxj = min((int)(aabb.max.y / dy) + cam.height / 2, cam.height);

		float depth;
		int fixed_depth;

		for (int j = minj; j < maxj; j++){
			for (int i = mini; i < maxi; i++){
				x = dx*i - dx*cam.width/2.0f;
				y = dy*j - dy*cam.height/2.0f;

				point[0] = x;
				point[1] = y;
				bary = calculateBarycentricCoordinate(tri, point);

				if (isBarycentricCoordInBounds(bary)){
					//printf("bary: %f %f %f\n", prim.v[0].col.r, prim.v[0].col.g, prim.v[0].col.b);
					depth = bary[0] * prim.v[0].ndc_pos[2] + bary[1] * prim.v[1].ndc_pos[2] + bary[2] * prim.v[2].ndc_pos[2];

					fixed_depth = (int)(depth * FIXED_SIZE);

					atomicMin(&fs_input[i + j*cam.width].fixed_depth, fixed_depth);

					if (fs_input[i + j*cam.width].fixed_depth == fixed_depth){
						fs_input[i + j*cam.width].depth = depth;
						fs_input[i + j*cam.width].color = bary[0] * prim.v[0].col + bary[1] * prim.v[1].col + bary[2] * prim.v[2].col; //glm::vec3(1.0, 0.0, 0.0);// prim.v[0].col;
						fs_input[i + j*cam.width].norm = bary[0] * prim.v[0].nor + bary[1] * prim.v[1].nor + bary[2] * prim.v[2].nor;
						fs_input[i + j*cam.width].pos = bary[0] * prim.v[0].pos + bary[1] * prim.v[1].pos + bary[2] * prim.v[2].pos;
						fs_input[i + j*cam.width].color = fs_input[i + j*cam.width].norm;
					}
				}
			}
		}
	}
}

__global__ void kernFragmentShading(int n, Fragment* fs_output, Fragment* fs_input){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		fs_output[index] = fs_input[index];
	}
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

	// TODO move this somewhere where it only happens once

	// Vertex shading
	int fragCount = cam.width * cam.height;
	int primCount = vertCount / 3;
	int numVertBlocks = (vertCount - 1) / MAX_THREADS + 1;
	int numPrimBlocks = (primCount - 1) / MAX_THREADS + 1;
	int numPixBlocks = (fragCount - 1) / MAX_THREADS + 1;

	kernShadeVertices<<<numVertBlocks, MAX_THREADS>>>(vertCount, dev_bufVertexOut, dev_bufVertex, Mproj, Mview, Mmod);
	
	/*
	VertexOut* hst_bufVertexOut = (VertexOut*)malloc(vertCount*sizeof(VertexOut));
	cudaMemcpy(hst_bufVertexOut,dev_bufVertexOut,vertCount*sizeof(VertexOut),cudaMemcpyDeviceToHost);
	printf("%d\n",vertCount);
	for (int i = 0; i < vertCount; i++){
		printf("%f %f %f\n", hst_bufVertexOut[i].ndc_pos[0], hst_bufVertexOut[i].ndc_pos[1], hst_bufVertexOut[i].ndc_pos[2]);
	}
	*/

	// Primitive Assembly
	kernAssemblePrimitives<<<numPrimBlocks, MAX_THREADS>>>(primCount, dev_primitives, dev_bufVertexOut, dev_bufIdx);

	// Rasterization
	kernRasterize<<<numPrimBlocks, MAX_THREADS>>>(primCount, cam, dev_depthbuffer, dev_primitives);

	// Fragment shading

	// Fragments to depth buffer

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

	cudaFree(dev_bufVertexOut);
	dev_bufVertexOut = NULL;

    cudaFree(dev_primitives);
    dev_primitives = NULL;

    cudaFree(dev_depthbuffer);
    dev_depthbuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

    checkCUDAError("rasterizeFree");
}
