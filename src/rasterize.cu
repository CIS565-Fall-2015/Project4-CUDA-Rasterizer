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

#define MAX_THREADS 128

#define PARTICLE_MODE

static int iter;

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static int *dev_bufIdx2 = NULL;
static VertexIn *dev_bufVertex = NULL;
static VertexOut *dev_bufVertexOut = NULL;
static VertexOut *dev_bufVertexOut2 = NULL;
static Triangle *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int* dev_locks = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;
static Light light;

static int fragCount;
static int primCount;
static int numVertBlocks;
static int numPrimBlocks;
static int numFragBlocks;

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
		depthbuffer[index].fixed_depth = INT_MAX;
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
	iter = 0;
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

	//cudaFree(dev_bufVertexOut);
	//cudaMalloc(&dev_bufVertexOut, vertCount * sizeof(VertexOut));

    //cudaFree(dev_primitives);
    //cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
    //cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

    checkCUDAError("rasterizeSetBuffers");
}

__global__ void kernShadeVertices(int n, VertexOut* vs_output, VertexIn* vs_input, glm::mat4 Mpvm){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		vs_output[index].pos = vs_input[index].pos;
		
		glm::vec4 new_pos = Mpvm * glm::vec4(vs_input[index].pos, 1.0f);
		vs_output[index].ndc_pos = glm::vec3(new_pos / new_pos.w);
		vs_output[index].nor = vs_input[index].nor;
		vs_output[index].col = vs_input[index].col;
	}
}

__global__ void kernShadeGeometries(int n, VertexOut* out_vertices, int* idx, VertexOut* in_vertices){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		VertexOut vi = in_vertices[index];
		idx[index * 3] = 3*index;
		idx[index * 3 + 1] = 3*index + 1;
		idx[index * 3 + 2] = 3*index + 2;
		out_vertices[index * 3].ndc_pos = vi.ndc_pos;
		out_vertices[index * 3].col = vi.col;
		out_vertices[index * 3].pos = vi.pos;
		out_vertices[index * 3].nor = vi.nor;
		out_vertices[index * 3 + 1].ndc_pos = vi.ndc_pos + glm::vec3(0.01,0.0,0.0);
		out_vertices[index * 3 + 1].col = vi.col;
		out_vertices[index * 3 + 1].pos = vi.pos;
		out_vertices[index * 3 + 1].nor = vi.nor;
		out_vertices[index * 3 + 2].ndc_pos = vi.ndc_pos + glm::vec3(0.0, 0.01, 0.0);
		out_vertices[index * 3 + 2].col = vi.col;
		out_vertices[index * 3 + 2].pos = vi.pos;
		out_vertices[index * 3 + 2].nor = vi.nor;
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
		primitives[index].ndc_pos[0] = vs_output[idx0].ndc_pos;
		primitives[index].ndc_pos[1] = vs_output[idx1].ndc_pos;
		primitives[index].ndc_pos[2] = vs_output[idx2].ndc_pos;
		primitives[index].v[0].col = glm::vec3(1.0, 0.0, 0.0);
		primitives[index].v[1].col = glm::vec3(1.0, 0.0, 0.0);
		primitives[index].v[2].col = glm::vec3(1.0, 0.0, 0.0);
	}
}

// Each thread is responsible for rasterizing a single triangle
__global__ void kernRasterize(int n, Cam cam, Fragment* fs_input, Triangle* primitives, glm::mat4 Mvm, glm::mat4 Mp){
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (index < n){
		
		Triangle prim = primitives[index];
		glm::vec4 viewport(0,0,cam.width,cam.height);

		AABB aabb = getAABBForTriangle(primitives[index].ndc_pos);
		glm::vec3 bary;
		glm::vec2 point;
		glm::vec3 points;

		// Snap i,j to nearest fragment coordinate
		float dx = 2.0f / (float)cam.width;
		float dy = 2.0f / (float)cam.height;

		float x;
		float y;

		int mini = max((int)(aabb.min.x / dx) + cam.width / 2 - 2, 0);
		int minj = max((int)(aabb.min.y / dy) + cam.height / 2 - 2, 0);
		int maxi = min((int)(aabb.max.x / dx) + cam.width / 2 + 2, cam.width-1);
		int maxj = min((int)(aabb.max.y / dy) + cam.height / 2 + 2, cam.height-1);

		float depth;
		int fixed_depth;
		int ind;

		// Iterate through pixel coordinates
		for (int j = minj; j < maxj; j++){
			for (int i = mini; i < maxi; i++){

				ind = i + j * cam.width;
				
				// Get the NDC coordinate
				x = dx*i - dx*cam.width/2.0f;
				y = dy*j - dy*cam.height/2.0f;

				point[0] = x;
				point[1] = y;

				bary = calculateBarycentricCoordinate(primitives[index].ndc_pos, point);

				if (isBarycentricCoordInBounds(bary)){
					depth = getZAtCoordinate(bary, prim.ndc_pos);
					fixed_depth = (int)(depth * INT_MAX);

					int old = atomicMin(&fs_input[ind].fixed_depth, fixed_depth);

					if (fs_input[ind].fixed_depth == fixed_depth){
						fs_input[ind].depth = depth;
						fs_input[ind].color = bary.x * prim.v[0].col + bary.y * prim.v[1].col + bary.z * prim.v[2].col; //glm::vec3(1.0, 0.0, 0.0);// prim.v[0].col;
						fs_input[ind].norm = bary.x * prim.v[0].nor + bary.y * prim.v[1].nor + bary.z * prim.v[2].nor;
						fs_input[ind].pos = bary.x * prim.v[0].pos + bary.y * prim.v[1].pos + bary.z * prim.v[2].pos;
						fs_input[ind].ndc_pos = bary.x * prim.v[0].ndc_pos + bary.y * prim.v[1].ndc_pos + bary.z * prim.v[2].ndc_pos;
						fs_input[ind].prim_ind = index;
						//fs_input[ind].color = fs_input[ind].norm;
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
			fs_input[index].color = fs_input[index].color * abs((glm::dot(glm::normalize(fs_input[index].norm), light_ray)));
		}
	}
}

void resetRasterize(){
	cudaFree(dev_bufVertexOut);
	cudaMalloc(&dev_bufVertexOut, vertCount * sizeof(VertexOut));

	cudaFree(dev_bufVertexOut2);
	cudaMalloc((void**)&dev_bufVertexOut2, vertCount*3*sizeof(VertexOut));

	cudaFree(dev_bufIdx2);
	cudaMalloc((void**)&dev_bufIdx2, sizeof(int)*vertCount*3);

	cudaFree(dev_primitives);
	cudaMalloc(&dev_primitives, vertCount * sizeof(Triangle));
	cudaMemset(dev_primitives, 0, vertCount * sizeof(Triangle));

	cudaFree(dev_depthbuffer);
	cudaMalloc(&dev_depthbuffer, fragCount * sizeof(Fragment));
	cudaMemset(dev_depthbuffer, 0, fragCount * sizeof(Fragment));
	initDepths<<<numFragBlocks, MAX_THREADS >> >(width*height, dev_depthbuffer);

	cudaFree(dev_framebuffer);
	cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
	checkCUDAError("resetBuffers");
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

	glm::mat4 Mvm = Mview * Mmod;
	glm::mat4 Mpvm = Mproj * Mvm;

	kernShadeVertices<<<numVertBlocks, MAX_THREADS>>>(vertCount, dev_bufVertexOut, dev_bufVertex, Mpvm);
	checkCUDAError("shadeVertices");

	kernShadeGeometries<<<numVertBlocks, MAX_THREADS>>>(vertCount, dev_bufVertexOut2, dev_bufIdx2, dev_bufVertexOut);
	checkCUDAError("shadeGeometries");
	int numPrimBlocks3 = (primCount*3 - 1) / MAX_THREADS + 1;

	// Primitive Assembly
	//kernAssemblePrimitives<<<numPrimBlocks, MAX_THREADS>>>(primCount, dev_primitives, dev_bufVertexOut, dev_bufIdx);
	kernAssemblePrimitives<<<numPrimBlocks3, MAX_THREADS>>>(primCount*3, dev_primitives, dev_bufVertexOut2, dev_bufIdx2);
	checkCUDAError("assemblePrimitives");

	// Rasterization
	kernRasterize<<<numPrimBlocks3, MAX_THREADS>>>(primCount*3, cam, dev_depthbuffer, dev_primitives, Mvm, Mproj);
	checkCUDAError("rasterizePrimitives");

	// Fragment shading
	//kernShadeFragments<<<numFragBlocks, MAX_THREADS>>>(fragCount, dev_depthbuffer, light);
	//checkCUDAError("shadeFragments");

    // Copy depthbuffer colors into framebuffer
    render<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer, dev_framebuffer);
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("rasterize");

	iter += 1;
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
