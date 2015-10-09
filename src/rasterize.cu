/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#include "rasterize.h"

#include <cmath>
#include <vector>
#include <cstdio>
#include <cuda.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include "rasterizeTools.h"
#include <glm/gtc/matrix_transform.hpp>
#define DEG2RAD  PI/180.f
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
	//	glm::vec3 col;
};
struct Triangle {
	VertexOut v[3];
};
struct Fragment {
	glm::vec3 color;
};
int mat = 0;
int dev = 0;
static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static VertexIn *dev_bufVertex = NULL;
static VertexOut *dev_vsOutput = NULL;
static Triangle *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static Fragment *dev_fmInput = NULL;
static Fragment *dev_fmOutput = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;

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
__global__ void cleanDepth(Fragment* dev_depthbuffer, int w,int h)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);
	if (index < w*h)
	{
		dev_depthbuffer[index].color = glm::vec3(0, 0, 0);
	}
}
// Writes fragment colors to the framebuffer
__global__ void render(int w, int h, Fragment *depthbuffer, glm::vec3 *framebuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		//*******************************//
		//depthbuffer[index].color = glm::vec3(1, 0, 0);
		//*********************************//
		framebuffer[index] = depthbuffer[index].color;
	}
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
	width = w;
	height = h;
	cudaFree(dev_depthbuffer);
	cudaMalloc(&dev_depthbuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
	cudaFree(dev_framebuffer);
	cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	cudaFree(dev_fmInput);
	cudaMalloc(&dev_fmInput, width * height * sizeof(Fragment));
	cudaMemset(dev_fmInput, 0, width * height * sizeof(Fragment));

	cudaFree(dev_fmOutput);
	cudaMalloc(&dev_fmOutput, width * height * sizeof(Fragment));
	cudaMemset(dev_fmOutput, 0, width * height * sizeof(Fragment));
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

	cudaFree(dev_vsOutput);
	cudaMalloc(&dev_vsOutput, vertCount * sizeof(VertexOut));

	cudaFree(dev_primitives);
	cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
	cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

	checkCUDAError("rasterizeSetBuffers");
}

__global__ void vertexShader(VertexIn *dev_bufVertex, VertexOut *dev_vsOutput, int vertexCount, glm::mat4 ViewProj){

	int id = (blockIdx.x * blockDim.x) + threadIdx.x;


	if (id < vertexCount){
		//simple orthordox projection 
		//dev_vsOutput[id].pos = dev_bufVertex[id].pos;
		//dev_vsOutput[id].nor = dev_bufVertex[id].nor;
		glm::vec4 temp_p = glm::vec4(dev_bufVertex[id].pos, 1)*ViewProj;
		glm::vec4 temp_n = glm::vec4(dev_bufVertex[id].nor, 1)*ViewProj;
		dev_vsOutput[id].pos = glm::vec3(temp_p[0], temp_p[1], temp_p[2]);
		dev_vsOutput[id].nor = glm::vec3(temp_n[0], temp_n[1], temp_n[2]);
		//dev_vsOutput[id].col = dev_bufVertex[id].col;
	}

}
__global__ void PrimitiveAssembly(VertexOut *dev_vsOutput, Triangle * dev_primitives, int verCount)
{
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < verCount / 3){
		dev_primitives[id].v[0] = dev_vsOutput[3 * id];
		dev_primitives[id].v[1] = dev_vsOutput[3 * id + 1];
		dev_primitives[id].v[2] = dev_vsOutput[3 * id + 2];
	}
}


__host__ __device__  bool fequal(float a, float b){
	if (a > b - 0.000001&&a < b + 0.000001){ return true; }
	else return false;
}

//scan_line??
__global__ void rasterization(Triangle * dev_primitives, Fragment *dev_fmInput, int vertexcount, int w, int h)
{
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (id < vertexcount / 3.f)
	{
		//potimized boundingbox;
		glm::vec3 tri[3];
		for (int i = 0; i < 3; i++){
			tri[i] = dev_primitives[id].v[i].pos;
			tri[i].x += 1;
			tri[i].y += 1;
			tri[i].x *= w / 2.f;
			tri[i].y *= h / 2.f;
		}
		AABB aabb;
		aabb = getAABBForTriangle(tri);

		for (int i = aabb.min.x - 1; i < aabb.max.x + 1; i++){
			for (int j = aabb.min.y - 1; j < aabb.max.y + 1; j++){

				glm::vec2 point(i, j);
				glm::vec3 baryc = calculateBarycentricCoordinate(tri, point);
				if (isBarycentricCoordInBounds(baryc)){
					dev_fmInput[i*w + j].color = dev_primitives[id].v[0].nor;
				}
			}
		}
	}
}

/* scan_line:brute force
glm::vec3 tri[3];
for (int i = 0; i < 3; i++){
tri[i] = dev_primitives[id].v[i].pos;
tri[i].x += 1;
tri[i].y += 1;
tri[i].x *= w / 2.f;
tri[i].y *= h / 2.f;
}
for (int i = 0; i < w; i++){
for (int j = 0; j < h; j++){
glm::vec2 point(i, j);
glm::vec3 baryc = calculateBarycentricCoordinate(tri, point);
if (isBarycentricCoordInBounds(baryc)){
dev_fmInput[i*w + j].color = glm::vec3(1, 0, 0);
}
}*/


__global__ void	fragmentShading(Fragment *dev_fmInput, Fragment *dev_fmOutput, int totalpix)
{
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < totalpix){
		dev_fmOutput[id].color = dev_fmInput[id].color;
	}
}

__global__ void SetDepth(Fragment*dev_fmOutput, Fragment * dev_depthbuffer, int totalpix)
{
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < totalpix){
		dev_depthbuffer[id].color = dev_fmOutput[id].color;
	}
}
/*
 * Perform rasterization.
 */
void RotateAboutRight(float deg,glm::vec3 &ref,const glm::vec3 right,const glm::vec3 eye)
{
	deg *= DEG2RAD;
	glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), deg, right);
	ref = ref - eye;
	ref = glm::vec3(rotation * glm::vec4(ref, 1));
	ref = ref + eye;

}
void TranslateAlongRight(float amt,glm::vec3 &ref,const glm::vec3 right,glm::vec3 &eye)
{
	glm::vec3 translation = right * amt;
	eye += translation;
	ref += translation;
}
glm::mat4 camera(int all_mat,int all_dev)
{
	glm::vec3 eye = glm::vec3(0, 0, 15);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 ref = glm::vec3(0, 0, 0);
	float near_clip = 1.0f;
	float far_clip = 1000.f;
	float width = 800; 
	float height = 800;
	float aspect = (float)width / (float)height;
	float fovy = 45;
	glm::vec3 world_up = glm::vec3(0, 1, 0);

	glm::vec3 look = glm::normalize(ref - eye);
	glm::vec3 right = glm::normalize(glm::cross(look, world_up));
	RotateAboutRight(all_dev, ref,right,eye);
	TranslateAlongRight(all_mat, ref, right, eye);
	glm::mat4 viewMatrix = glm::lookAt(eye, ref, up);
	glm::mat4 projectionMatrix = glm::perspective(fovy, aspect, near_clip, far_clip);//fovy,aspect, zNear, zFar;
	glm::mat4 getViewProj = projectionMatrix*viewMatrix;
	return getViewProj;
}

void rasterize(uchar4 *pbo,int all_mat,int all_dev) {
	int sideLength2d = 8;
	dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);
	
	std::cout << "ss " << all_mat << "and"<<all_dev<<std::endl;
	std::cout << "dd" << mat << "and" << dev << std::endl;
	// TODO: Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)
	//step1.vertex shading
	int blockSize1d = 256;
	int blockCount1d = (vertCount + blockSize1d - 1) / blockSize1d;

	//clean depth buffer
	cleanDepth << <blockCount2d, blockSize2d >> >(dev_depthbuffer, width,height);


	glm::mat4 getViewProj= camera(all_mat, all_dev);
	vertexShader << <blockCount1d, blockSize1d >> >(dev_bufVertex, dev_vsOutput, vertCount, getViewProj);
	checkCUDAError("vertexShader");
	//step2.primitive assembly
	int blockCount1d_tri = blockCount1d / 3 + 1;
	PrimitiveAssembly << < blockCount1d_tri, blockSize1d >> >(dev_vsOutput, dev_primitives, vertCount);
	checkCUDAError("PrimitiveAssembly");
	//step3.rasterization
	rasterization << < blockCount1d_tri, blockSize1d >> >(dev_primitives, dev_depthbuffer, vertCount, width, height);
	checkCUDAError("rasterization");
	//step4.fragment shading
	//fragmentShading << <blockCount1d, blockSize1d >> >(dev_fmInput, dev_fmOutput, width*height);
	//step5.fragment to depth buffer
	//SetDepth << <blockCount1d, blockSize1d >> >(dev_fmOutput, dev_depthbuffer, width*height);
	checkCUDAError("setDepth");
	//step6.a depth buffer for storing and depth testing fragment

	//step7.fragment buffer writing.
	// Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_depthbuffer, dev_framebuffer);
	// Copy framebuffer into OpenGL buffer for OpenGL previewing
	sendImageToPBO << <blockCount2d, blockSize2d >> >(pbo, width, height, dev_framebuffer);
	checkCUDAError("sendToPBO");
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
