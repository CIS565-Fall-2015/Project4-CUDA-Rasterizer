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
#include "sceneStructs.h"

#include <glm/gtx/transform.hpp>

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};
struct VertexOut {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;
    // TODO
};
struct Triangle {
    VertexOut v[3];
	AABB box;
};
struct Fragment {
	glm::vec3 pos;
	glm::vec3 nor;
    glm::vec3 col;
};

static int width = 0;
static int height = 0;
__constant__ static int *dev_bufIdx = NULL;
__constant__ static int *dev_depth = NULL;
__constant__ static VertexIn *dev_bufVertex = NULL;
__constant__ static VertexOut *dev_bufShadedVert = NULL;
__constant__ static Triangle *dev_primitives = NULL;
__constant__ static Fragment *dev_depthbuffer = NULL;
__constant__ static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;
static MVP *mvp = NULL;

/*
// Model translation
static glm::mat4 model = glm::mat4(1.0f);
// Camera matrix
glm::vec3 camPosition = glm::vec3(0,0,3);
glm::mat4 view = glm::lookAt(
	camPosition,
	glm::vec3(0,0,0),	// Direction; looking at here from position
	glm::vec3(0,1,0)	// Up
	);
// Perspective projection box
float nearPlane = 0.1f;
float farPlane = 100.0f;
static glm::mat4 projection = glm::perspective(45.0f, 1.0f / 1.0f, -nearPlane, -farPlane);

// Combined matrix
static glm::mat4 mvp = projection*view*model;
*/

static glm::vec3 light = glm::vec3(100.0f, 100.0f, 100.0f)*glm::vec3(80.0f, 80.0f, 80.0f);
static glm::vec3 lightCol = glm::vec3(0.8f);

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

__global__ void sendImageToPBO(uchar4 *pbo, int w, int h, Fragment *image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		glm::vec3 color;
		color.x = glm::clamp(image[index].col.x, 0.0f, 1.0f) * 255.0;
		color.y = glm::clamp(image[index].col.y, 0.0f, 1.0f) * 255.0;
		color.z = glm::clamp(image[index].col.z, 0.0f, 1.0f) * 255.0;
		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

// Writes fragment colors to the framebuffer
__global__ void render(int w, int h, Fragment *depthbuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        framebuffer[index] = depthbuffer[index].col;
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h, MVP *hst_mvp) {
    width = w;
	height = h;
	mvp = hst_mvp;
    //cudaFree(dev_depthbuffer);
	cudaMalloc(&dev_depthbuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
    //cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	cudaMalloc(&dev_depth, width * height * sizeof(int));
	cudaMemset(dev_depth, mvp->farPlane*10000, width * height * sizeof(int));
    checkCUDAError("rasterizeInit");
}

void flushDepthBuffer(){
	cudaMemset(dev_depth, mvp->farPlane * 10000, width * height * sizeof(int));
	checkCUDAError("rasterize flush");
}

/**
 * Set all of the buffers necessary for rasterization.
 */
void rasterizeSetBuffers(
        int _bufIdxSize, int *bufIdx,
        int _vertCount, float *bufPos, float *bufNor, float *bufCol) {
    bufIdxSize = _bufIdxSize;
    vertCount = _vertCount;

    //cudaFree(dev_bufIdx);
    cudaMalloc(&dev_bufIdx, bufIdxSize * sizeof(int));
    cudaMemcpy(dev_bufIdx, bufIdx, bufIdxSize * sizeof(int), cudaMemcpyHostToDevice);

    VertexIn *bufVertex = new VertexIn[_vertCount];
    for (int i = 0; i < vertCount; i++) {
        int j = i * 3;
        bufVertex[i].pos = glm::vec3(bufPos[j + 0], bufPos[j + 1], bufPos[j + 2]);
        bufVertex[i].nor = glm::vec3(bufNor[j + 0], bufNor[j + 1], bufNor[j + 2]);
        bufVertex[i].col = glm::vec3(bufCol[j + 0], bufCol[j + 1], bufCol[j + 2]);
    }
    //cudaFree(dev_bufVertex);
    cudaMalloc(&dev_bufVertex, vertCount * sizeof(VertexIn));
    cudaMemcpy(dev_bufVertex, bufVertex, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);

	//cudaFree(dev_bufShadedVert);
	cudaMalloc(&dev_bufShadedVert, vertCount * sizeof(VertexOut));

    //cudaFree(dev_primitives);
    cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
    cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

    checkCUDAError("rasterizeSetBuffers");
}

__global__ void shadeVertex(VertexOut *vOut, VertexIn *vIn, const int vertCount, const int width, const int height, const glm::mat4 mvp, const float near, const float far){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * width);

	if (index < vertCount) {
		// http://www.songho.ca/opengl/gl_transform.html
		VertexOut o;
		glm::vec4 clip = mvp*glm::vec4(vIn[index].pos, 1.0f);
		glm::vec3 ndc = glm::vec3(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
		o.pos = glm::vec3(width/2*(ndc.x+1), height/2*(ndc.y+1), (far-near)/2*ndc.z+(far+near)/2);
		o.nor = vIn[index].nor;
		o.col = vIn[index].col;
		vOut[index] = o;
	}
}

__device__ void findIntersect(glm::vec3 &i, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 p4){
	// http://paulbourke.net/geometry/pointlineplane/
	float d = (p4.y - p3.y)*(p2.x - p1.x) - (p4.x - p3.x)*(p2.y - p1.y);
	if (abs(d) < ZERO_ABSORPTION_EPSILON){
		// Parallel
		i = glm::vec3(0.0f);
	}
	else {
		float n = (p4.x - p3.x)*(p1.y - p3.y) - (p4.y - p3.y)*(p1.x - p3.x);
		float ua = n / d;
		i.x = p1.x + ua*(p2.x - p1.x);
		i.y = p1.y + ua*(p2.y - p1.y);
		i.z = 1;
	}
}

__global__ void assemblePrimitive(Triangle *pOut, VertexOut *vIn, int *triIdx, const int triCount, const int width){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * width);

	if (index < triCount) {
		Triangle t;
		t.v[0] = vIn[triIdx[3 * index + 0]];
		t.v[1] = vIn[triIdx[3 * index + 1]];
		t.v[2] = vIn[triIdx[3 * index + 2]];
		glm::vec3 coord[3] = { t.v[0].pos, t.v[1].pos, t.v[2].pos };
		t.box = getAABBForTriangle(coord);
		pOut[index] = t;
	}
}

__global__ void testCover(Fragment *dBuf, int *depth, Triangle *pIn, const int triCount, const int width, const int height, const glm::vec3 camPos){
	int xt = (blockIdx.x * blockDim.x) + threadIdx.x;
	int yt = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = xt + (yt * width);

	if (index < triCount) {
		Triangle t = pIn[index];
		Fragment f;
		float minX = t.box.min.x, maxX = t.box.max.x;
		for (int y = round(t.box.max.y); y >= round(t.box.min.y); y--){
			int dp;
			glm::vec3 coord[3] = { t.v[0].pos, t.v[1].pos, t.v[2].pos };
			for (int x = round(minX); x <= round(maxX); x++){
				glm::vec3 bcc = calculateBarycentricCoordinate(coord, glm::vec2(x, y));
				if (isBarycentricCoordInBounds(bcc)){
					dp = getZAtCoordinate(bcc, coord)* 10000;

					int flatIdx = width-x + (height-y)*width;

					if (flatIdx >= 0 && flatIdx < width*height && width-x >= 0 && width-x <= width){
						atomicMin(&depth[flatIdx], dp);

						if (depth[flatIdx] == dp) {
							// Shallowest
							glm::vec3 bcc = calculateBarycentricCoordinate(coord, glm::vec2(x, y));
							f.pos = bcc.x * t.v[0].pos + bcc.y*t.v[1].pos + bcc.z*t.v[2].pos;
							f.nor = bcc.x * t.v[0].nor + bcc.y*t.v[1].nor + bcc.z*t.v[2].nor;
							f.col = bcc.x * t.v[0].col + bcc.y*t.v[1].col + bcc.z*t.v[2].col;
							dBuf[flatIdx] = f;
						}
					}
				}
			}
		}
	}
}

__global__ void shadeFragment(Fragment *fBuf, const int pxCount, const int width, const glm::vec3 light, const glm::vec3 lightCol){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * width);

	if (index < pxCount) {
		glm::vec3 L = glm::normalize(light - fBuf[index].pos);
		fBuf[index].col = glm::dot(L, fBuf[index].nor)*fBuf[index].col*lightCol;
		//fBuf[index].col = normalize(fBuf[index].pos);
		//fBuf[index].col = fBuf[index].nor;
	}
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
	
	dim3 blockCount2d((width + blockSize2d.x - 1) / blockSize2d.x,
						(height + blockSize2d.y - 1) / blockSize2d.y);

	// Vertex shading
	shadeVertex << <blockCount2d, blockSize2d >> >(dev_bufShadedVert, dev_bufVertex, vertCount, width, height, mvp->mvp, mvp->nearPlane, mvp->farPlane);
	checkCUDAError("Vert shader");

	// Primitive assembly
	assemblePrimitive << <blockCount2d, blockSize2d >> >(dev_primitives, dev_bufShadedVert, dev_bufIdx, vertCount/3, width);
	checkCUDAError("Prim assembly");
	
	// Rasterization
	testCover << <blockCount2d, blockSize2d >> >(dev_depthbuffer, dev_depth, dev_primitives, vertCount / 3, width, height, mvp->camPosition);
	checkCUDAError("Rasterization");

	// Fragment shading
	shadeFragment << <blockCount2d, blockSize2d >> >(dev_depthbuffer, height*width, width, light, lightCol);
	checkCUDAError("Frag shader");

    // Copy depthbuffer colors into framebuffer
    render<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer, dev_framebuffer);
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
	//sendImageToPBO << <blockCount2d, blockSize2d >> >(pbo, width, height, dev_framebuffer);
	sendImageToPBO << <blockCount2d, blockSize2d >> >(pbo, width, height, dev_depthbuffer);
    checkCUDAError("rasterize");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {
    cudaFree(dev_bufIdx);
	cudaFree(dev_bufVertex);
	cudaFree(dev_bufShadedVert);
    cudaFree(dev_primitives);
    cudaFree(dev_depthbuffer);
    cudaFree(dev_framebuffer);
	cudaFree(dev_depth);

    checkCUDAError("rasterizeFree");
}
