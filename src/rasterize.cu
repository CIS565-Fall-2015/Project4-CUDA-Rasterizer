/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#include "rasterize.h"
//<seqan / parallel.h>

#include <cmath>
#include <vector>
#include <cstdio>
#include <cuda.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include "rasterizeTools.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define DEG2RAD  PI/180.f
#define Tess 1
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
	int dis;
	glm::vec3 color;
	glm::vec3 normal;
	glm::vec3 pos;
};
int N = 0;
int M = 0;
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
__global__ void cleanDepth(Fragment* dev_depthbuffer, int w, int h)
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
	int _vertCount, float *bufPos, float *bufNor, float *bufCol, bool resselation) {
	//********************
	resselation = Tess;
	//********************
	bufIdxSize = _bufIdxSize;
	vertCount = _vertCount;

	cudaFree(dev_bufIdx);
	cudaMalloc(&dev_bufIdx, bufIdxSize * sizeof(int));
	cudaMemcpy(dev_bufIdx, bufIdx, bufIdxSize * sizeof(int), cudaMemcpyHostToDevice);


	VertexIn *bufVertex = new VertexIn[_vertCount];
	float maxv = -1.f;

	for (int i = 0; i < vertCount; i++) {
		int j = i * 3;
		bufVertex[i].pos = glm::vec3(bufPos[j + 0], bufPos[j + 1], bufPos[j + 2]);
		bufVertex[i].nor = glm::vec3(bufNor[j + 0], bufNor[j + 1], bufNor[j + 2]);
		bufVertex[i].col = glm::vec3(bufCol[j + 0], bufCol[j + 1], bufCol[j + 2]);
		//***********check here....*******//
		float temp = std::max(bufVertex[i].pos.x, std::max(bufVertex[i].pos.y, bufVertex[i].pos.y));
		if (temp>maxv){ maxv = temp; }
	}
	N = (int)maxv + 1;
	cudaFree(dev_bufVertex);
	cudaMalloc(&dev_bufVertex, vertCount * sizeof(VertexIn));
	cudaMemcpy(dev_bufVertex, bufVertex, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);

	cudaFree(dev_vsOutput);
	cudaMalloc(&dev_vsOutput, vertCount * sizeof(VertexOut));

	if (!resselation)
	{
		cudaFree(dev_primitives);
		cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
		cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));
		checkCUDAError("rasterizeSetBuffers");
	}
	else
	{
		cudaFree(dev_primitives);
		cudaMalloc(&dev_primitives, vertCount / 3*4 * sizeof(Triangle));
		cudaMemset(dev_primitives, 0, vertCount / 3*4 * sizeof(Triangle));
		checkCUDAError("rasterizeSetBuffers");
	}

}


__global__ void vertexShader(VertexIn *dev_bufVertex, VertexOut *dev_vsOutput, int vertexCount, glm::mat4 ViewProj){

	int id = (blockIdx.x * blockDim.x) + threadIdx.x;


	if (id < vertexCount){
		//simple orthordox projection 
		//dev_vsOutput[id].pos = dev_bufVertex[id].pos;
		//dev_vsOutput[id].nor = dev_bufVertex[id].nor;

		dev_vsOutput[id].pos = multiplyMV(ViewProj, glm::vec4(dev_bufVertex[id].pos, 1));
		dev_vsOutput[id].nor = multiplyMV(ViewProj, glm::vec4(dev_bufVertex[id].nor, 0));
		dev_vsOutput[id].nor = glm::normalize(dev_vsOutput[id].nor);
		//dev_vsOutput[id].col = dev_bufVertex[id].col;
		//interpolate the normal:smooth normal color??


	}

}
__global__ void PrimitiveAssembly(VertexOut *dev_vsOutput, Triangle * dev_primitives, int verCount)
{
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < verCount / 3){
		dev_primitives[id].v[0].pos = dev_vsOutput[3 * id].pos;//012,345,678
		dev_primitives[id].v[1].pos = dev_vsOutput[3 * id + 1].pos;
		dev_primitives[id].v[2].pos = dev_vsOutput[3 * id + 2].pos;

		dev_primitives[id].v[0].nor = dev_vsOutput[3 * id].nor;//012,345,678
		dev_primitives[id].v[1].nor = dev_vsOutput[3 * id + 1].nor;
		dev_primitives[id].v[2].nor = dev_vsOutput[3 * id + 2].nor;
	}
}

__host__ __device__  bool fequal(float a, float b){
	if (a > b - 0.000001&&a < b + 0.000001){ return true; }
	else return false;
}

__device__ int _atomicMin(int *addr, int val)
{
	int old = *addr, assumed;
	if (old <= val) return old;
	do{
		assumed = old;
		old = atomicCAS(addr, assumed, val);
	} while (old != assumed);
	return old;
}

__global__ void rasterization(Triangle * dev_primitives, Fragment *dev_fmInput, int vertexcount, int w, int h, int N)
{
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (id < vertexcount / 3.f)
	{
		//potimized boundingbox;
		glm::vec3 tri[3];
		for (int i = 0; i < 3; i++){//(-1,1)+1*w/2
			//(-10,10)+10*w/20
			tri[i] = dev_primitives[id].v[i].pos;
			tri[i].x += N;
			tri[i].y += N;
			tri[i].z += N;
			tri[i].x *= w / (float)(2.f*N);
			tri[i].y *= h / (float)(2.f*N);
			tri[i].z *= w / (float)(2.f*N);
			//because the image is cube anyway...I think multiply should have better result than devide...
		}
		AABB aabb;
		aabb = getAABBForTriangle(tri);

		for (int i = aabb.min.x - 1; i < aabb.max.x + 1; i++){
			for (int j = aabb.min.y - 1; j < aabb.max.y + 1; j++){

				glm::vec2 point(i, j);
				glm::vec3 baryc = calculateBarycentricCoordinate(tri, point);
				//simple clip..
				if (tri[0].x > w || tri[0].x < 0 || tri[0].y>h || tri[0].x < 0)continue;
				if (isBarycentricCoordInBounds(baryc)){
					//these three normal should be the same since they are on the same face (checked)
					int intdepth = getZAtCoordinate(baryc, tri);
					_atomicMin(&dev_fmInput[i*w + j].dis, intdepth);
					if (intdepth == dev_fmInput[i*w + j].dis){
						dev_fmInput[i*w + j].color = dev_primitives[id].v[0].nor;
						dev_fmInput[i*w + j].normal = dev_primitives[id].v[0].nor;
						dev_fmInput[i*w + j].pos = (dev_primitives[id].v[0].pos + dev_primitives[id].v[1].pos + dev_primitives[id].v[2].pos) / 3.f;
					}
				}
			}
		}
	}
}
//#version 450 core
//layout(vertices = 3) out;
//layout(triangles, equal_spacing, cw) in;
__global__ void Tesselation(bool active, VertexOut *dev_vertin, Triangle *dev_triout, int vercount)
{

	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (active&&id < vercount / 3.f)
	{
		int tessel_number = 3;
		glm::vec3 tri[3];
		tri[0] = dev_vertin[3 * id].pos;
		tri[1] = dev_vertin[3 * id + 1].pos;
		tri[2] = dev_vertin[3 * id + 2].pos;
		//default tesselation,generate 4 triangles automativaly
		glm::vec3 vnew[3];
		vnew[0] = (tri[0] + tri[1]) / 2.f;
		vnew[1] = (tri[0] + tri[2]) / 2.f;
		vnew[2] = (tri[2] + tri[1]) / 2.f;

		dev_triout[4 * id].v[0].pos = tri[0];
		dev_triout[4 * id].v[1].pos = vnew[0];
		dev_triout[4 * id].v[2].pos = vnew[1];

		dev_triout[4 * id + 1].v[0].pos = vnew[0];
		dev_triout[4 * id + 1].v[1].pos = tri[1];
		dev_triout[4 * id + 1].v[2].pos = vnew[2];

		dev_triout[4 * id + 2].v[0].pos = vnew[0];
		dev_triout[4 * id + 2].v[1].pos = vnew[2];
		dev_triout[4 * id + 2].v[2].pos = vnew[1];

		dev_triout[4 * id + 3].v[0].pos = vnew[1];
		dev_triout[4 * id + 3].v[1].pos = vnew[2];
		dev_triout[4 * id + 3].v[2].pos = tri[2];
		/*for (int i = 0; i < 4; i++){
			for (int j = 0; j < 3; j++)
			{
				dev_triout[4 * id + i].v[j].nor = dev_vertin[3 * id].nor;
			}
		}*/
		//in order to check :change the normal a little
		for (int i = 0; i < 3; i++){
			{
				dev_triout[4 * id ].v[i].nor = glm::normalize(dev_vertin[3 * id].nor+glm::vec3(0.3,0,0));
				dev_triout[4 * id + 1].v[i].nor = glm::normalize(dev_vertin[3 * id].nor + glm::vec3(0, 0.3, 0));
				dev_triout[4 * id + 2].v[i].nor = glm::normalize(dev_vertin[3 * id].nor + glm::vec3(0, 0, 0));
				dev_triout[4 * id + 3].v[i].nor = glm::normalize(dev_vertin[3 * id].nor + glm::vec3(0, 0, 0.3));
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

glm::vec3 SetLight()
{
	glm::vec3 light_pos = glm::vec3(2, 1, 2);

	return light_pos;
}
//blin phong
__global__ void	fragmentShading(Fragment *dev_fmInput, Fragment *dev_fmOutput, int w, int h, glm::vec3 light_pos, glm::vec3 camera_pos)
{
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < w*h){

		float specular_power = 10;
		glm::vec3 specular_color = dev_fmInput[id].color;
		glm::vec3 lightray = light_pos - dev_fmInput[id].pos;
		glm::vec3 inray = camera_pos - dev_fmInput[id].pos;
		glm::vec3 H = glm::normalize(inray) + glm::normalize(lightray);
		H = glm::vec3(H.x / 2.0, H.y / 2.0, H.z / 2.0);
		float hdot = glm::dot(H, dev_fmInput[id].normal);
		float x = pow(hdot, specular_power);
		if (x < 0)x = 0.f;
		glm::vec3 spec = x*specular_color;

		glm::vec3 Lambert = glm::vec3(1, 1, 1);
		glm::vec3 Ambient = glm::vec3(1, 1, 1);
		float diffuse = glm::clamp(glm::dot(dev_fmInput[id].normal, glm::normalize(lightray)), 0.0f, 1.0f);
		Lambert *= diffuse;

		glm::vec3 phong_color = 0.5f*spec + 0.4f*Lambert + 0.1f*Ambient;//where is ambient light?
		phong_color = glm::clamp(phong_color, 0.f, 1.f);

		dev_fmOutput[id].color = phong_color;
	}
}

__global__ void SetDepth(Fragment*dev_fmOutput, Fragment * dev_depthbuffer, int totalpix)
{

	int id = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (id < totalpix){
		int intdepth;
		dev_depthbuffer[id].color = dev_fmOutput[id].color;
	}
}
/*
 * Perform rasterization.
 */
void RotateAboutRight(float deg, glm::vec3 &ref, const glm::vec3 right, const glm::vec3 eye)
{
	deg *= DEG2RAD;
	glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), deg, right);
	ref = ref - eye;
	ref = glm::vec3(rotation * glm::vec4(ref, 1));
	ref = ref + eye;

}
void TranslateAlongRight(float amt, glm::vec3 &ref, const glm::vec3 right, glm::vec3 &eye)
{
	glm::vec3 translation = right * amt;
	eye += translation;
	ref += translation;
}
glm::mat4 camera(int all_mat, int all_dev, glm::vec3 &camerapos)
{
	glm::vec3 eye = glm::vec3(0, 0, 13);
	glm::vec3 up = glm::vec3(0, 1, 0);
	glm::vec3 ref = glm::vec3(0, 0, 0);
	camerapos = eye;

	float near_clip = 1.0f;
	float far_clip = 1000.f;
	float width = 800;
	float height = 800;
	float aspect = (float)width / (float)height;
	float fovy = 45.f;
	glm::vec3 world_up = glm::vec3(0, 1, 0);
	glm::vec3 look = glm::normalize(ref - eye);
	glm::vec3 right = glm::normalize(glm::cross(look, world_up));
	RotateAboutRight(all_dev, ref, right, eye);
	TranslateAlongRight(all_mat, ref, right, eye);
	glm::mat4 viewMatrix = glm::lookAt(eye, ref, up);
	glm::mat4 projectionMatrix = glm::perspective(fovy, aspect, near_clip, far_clip);//fovy,aspect, zNear, zFar;

	glm::mat4 getViewProj = projectionMatrix*viewMatrix;
	return getViewProj;
}

void rasterize(uchar4 *pbo, int all_mat, int all_dev) {
	int sideLength2d = 8;
	dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockCount2d((width - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);
	//key_test:
	//std::cout << "ss " << all_mat << "and"<<all_dev<<std::endl;
	//std::cout << "dd" << mat << "and" << dev << std::endl;

	//step1.vertex shading
	int blockSize1d = 256;
	int blockCount1d = (vertCount + blockSize1d - 1) / blockSize1d;

	//clean depth buffer
	cleanDepth << <blockCount2d, blockSize2d >> >(dev_depthbuffer, width, height);

	int image_blockSize1d = 256;
	int image_blockCount1d = (width*height + blockSize1d - 1) / image_blockSize1d;
	glm::vec3 camera_pos = glm::vec3(0);
	glm::vec3 light_pos = SetLight();
	glm::mat4 getViewProj = camera(all_mat, all_dev, camera_pos);
	getViewProj = glm::mat4(1);

	vertexShader << <blockCount1d, blockSize1d >> >(dev_bufVertex, dev_vsOutput, vertCount, getViewProj);
	checkCUDAError("vertexShader");
	//step2.primitive assembly
	int blockCount1d_tri;
	bool tesselation = Tess;
	if (!tesselation)
	{
		//vertexnumber: vertcount,triangle number:vertcount/3.0
		blockCount1d_tri = blockCount1d / 3 + 1;

		PrimitiveAssembly << < blockCount1d_tri, blockSize1d >> >(dev_vsOutput, dev_primitives, vertCount);
		checkCUDAError("PrimitiveAssembly");
		rasterization << < blockCount1d_tri, blockSize1d >> >(dev_primitives, dev_fmInput, vertCount, width, height, N);
		checkCUDAError("rasterization");
	}
	else
	{
		blockCount1d_tri = blockCount1d / 3 * 4 + 1;
		//vertexnumber: vertcount*12,triangle number:vertcount*12/3.0
		Tesselation << <blockCount1d_tri, blockSize1d >> >(1, dev_vsOutput, dev_primitives, vertCount);
		checkCUDAError("Tesselation");
		rasterization << < blockCount1d_tri, blockSize1d >> >(dev_primitives, dev_fmInput, vertCount*4, width, height, N);
		checkCUDAError("rasterization");
	}
		
		fragmentShading << <image_blockCount1d, image_blockSize1d >> >(dev_fmInput, dev_depthbuffer, width, height, light_pos, camera_pos);
		checkCUDAError("setDepth");
		render << <blockCount2d, blockSize2d >> >(width, height, dev_depthbuffer, dev_framebuffer);
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
