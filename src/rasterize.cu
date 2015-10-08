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
#include <glm/gtc/constants.hpp>

#define MAX_THREADS 512

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
static Light light;

static int fragCount;
static int primCount;
static int numVertBlocks;
static int numPrimBlocks;
static int numFragBlocks;

//TODO: Change these so we can move the camera around
//static Cam cam; 
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
		depthbuffer[index].fixed_depth = 1 * INT_MAX;
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

	light.pos = glm::vec3(3.0, 3.0, 3.0);
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

	// Vertex shading
	fragCount = width * height;
	primCount = vertCount / 3;
	numVertBlocks = (vertCount - 1) / MAX_THREADS + 1;
	numPrimBlocks = (primCount - 1) / MAX_THREADS + 1;
	numFragBlocks = (fragCount - 1) / MAX_THREADS + 1;

	printf("fragment count: %d\n", fragCount);
	printf("vertex count: %d\n", vertCount);
	printf("primitive count: %d\n", primCount);

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
		
		glm::vec4 new_pos = Mp * Mv * Mm * glm::vec4(vs_input[index].pos, 1.0f);
		vs_output[index].ndc_pos = glm::vec3(new_pos / new_pos.w);
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
		primitives[index].v[0].col = glm::vec3(1.0, 0.0, 0.0);
		primitives[index].v[1].col = glm::vec3(1.0, 0.0, 0.0);
		primitives[index].v[2].col = glm::vec3(1.0, 0.0, 0.0);
	}
}

// Each thread is responsible for rasterizing a single triangle
__global__ void kernRasterize(int n, Cam cam, Fragment* fs_input, Triangle* primitives, glm::mat4 Mvm, glm::mat4 Mp){//, int* mutex){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		Triangle prim = primitives[index];
		glm::vec3* tri = new glm::vec3[3];

		tri[0] = glm::vec3(prim.v[0].ndc_pos);
		tri[1] = glm::vec3(prim.v[1].ndc_pos);
		tri[2] = glm::vec3(prim.v[2].ndc_pos);

		glm::vec4 viewpoint(0,0,cam.width,cam.height);

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
					fixed_depth = (int)(depth * INT_MAX);
					
					/*
					int ind = i + j * cam.width;

					bool isSet = false;
					do
					{
						if (isSet = atomicCAS(&mutex[ind], 0, 1) == 0)
						{
							if (fs_input[ind].depth > depth){
								fs_input[i + j*cam.width].depth = depth;
								fs_input[i + j*cam.width].color = bary.x * prim.v[0].col + bary.y * prim.v[1].col + bary.z * prim.v[2].col; //glm::vec3(1.0, 0.0, 0.0);// prim.v[0].col;
								fs_input[i + j*cam.width].norm = bary.x * prim.v[0].nor + bary.y * prim.v[1].nor + bary.z * prim.v[2].nor;
								fs_input[i + j*cam.width].pos = bary.x * prim.v[0].pos + bary.y * prim.v[1].pos + bary.z * prim.v[2].pos;
							}
							mutex[ind] = 0;
						}
						if (isSet)
						{
							mutex[ind] = 0;
						}
					} while (!isSet);
					*/

					int old = atomicMin(&fs_input[i + j*cam.width].fixed_depth, fixed_depth);

					if (fs_input[i + j*cam.width].fixed_depth == fixed_depth){
					//if (fs_input[i + j*cam.width].fixed_depth != old){
						fs_input[i + j*cam.width].depth = depth;
						fs_input[i + j*cam.width].color = bary.x * prim.v[0].col + bary.y * prim.v[1].col + bary.z * prim.v[2].col; //glm::vec3(1.0, 0.0, 0.0);// prim.v[0].col;
						fs_input[i + j*cam.width].norm = bary.x * prim.v[0].nor + bary.y * prim.v[1].nor + bary.z * prim.v[2].nor;
						fs_input[i + j*cam.width].pos = bary.x * prim.v[0].pos + bary.y * prim.v[1].pos + bary.z * prim.v[2].pos;
						
						//fs_input[i + j*cam.width].color = glm::vec3(1.0, 0.0, 0.0);
						//fs_input[i + j*cam.width].color = fs_input[i + j*cam.width].norm;
						//printf("%f  %f  %f\n", (depth + 1.0f) / 2.0f, depth, prim.v[0].ndc_pos[2]);
						//fs_input[i + j*cam.width].color = glm::vec3((depth + 1.0f)/2.0f);
					}
					
				}
			}
		}
	}
}

__global__ void kernShadeFragments(int n, Fragment* fs_input, Light light){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		if (fs_input[index].color != glm::vec3(0.0)){
			glm::vec3 light_ray = glm::normalize(fs_input[index].pos - light.pos);
			fs_input[index].color = fs_input[index].color * -(glm::dot(glm::normalize(fs_input[index].norm), light_ray));
		}
	}
}

void resetRasterize(){
	cudaFree(dev_bufVertexOut);
	cudaMalloc(&dev_bufVertexOut, vertCount * sizeof(VertexOut));

	cudaFree(dev_primitives);
	cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
	cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

	cudaFree(dev_depthbuffer);
	cudaMalloc(&dev_depthbuffer, fragCount * sizeof(Fragment));
	cudaMemset(dev_depthbuffer, 0, fragCount * sizeof(Fragment));
	initDepths << <numFragBlocks, MAX_THREADS >> >(width*height, dev_depthbuffer);

	cudaFree(dev_framebuffer);
	cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, Cam cam) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

	resetRasterize();

	Mmod = glm::mat4(3.0f)*1.0f;
	Mmod[3][3] = 1.0f;
	Mview = glm::lookAt(cam.pos, cam.focus, cam.up);
	Mproj = glm::perspective(cam.fovy, cam.aspect, cam.zNear, cam.zFar);

	//glm::vec4 viewport(0,0,800,800);
	//glm::vec3 h = glm::project(glm::vec3(0.99, 0.0, 0.0), Mview*Mmod, Mproj, viewport);
	//printf("%f %f %f\n",h.x,h.y,h.z);

	kernShadeVertices<<<numVertBlocks, MAX_THREADS>>>(vertCount, dev_bufVertexOut, dev_bufVertex, Mproj, Mview, Mmod);

	// Primitive Assembly
	kernAssemblePrimitives<<<numPrimBlocks, MAX_THREADS>>>(primCount, dev_primitives, dev_bufVertexOut, dev_bufIdx);

	// Rasterization
	//int* dev_mutex;
	//cudaMalloc((void**)&dev_mutex, fragCount*sizeof(int));
	//cudaMemset(dev_mutex, 0, fragCount*sizeof(int));
	//kernRasterize<<<numPrimBlocks, MAX_THREADS>>>(primCount, cam, dev_depthbuffer, dev_primitives, Mview*Mmod,Mproj, dev_mutex);
	kernRasterize<<<numPrimBlocks, MAX_THREADS>>>(primCount, cam, dev_depthbuffer, dev_primitives, Mview*Mmod, Mproj);

	// Fragment shading
	kernShadeFragments<<<numFragBlocks, MAX_THREADS>>>(fragCount, dev_depthbuffer, light);

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
