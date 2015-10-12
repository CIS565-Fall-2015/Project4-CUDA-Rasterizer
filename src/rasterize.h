/**
 * @file      rasterize.h
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void rasterizeInit(int width, int height);
void rasterizeSetBuffers(
        int bufIdxSize, int *bufIdx,
        int vertCount, float *bufPos, float *bufNor, float *bufCol);
void rasterize(uchar4 *pbo, glm::mat4 viewProjection);
void rasterizeFree();

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
};
struct Triangle {
	VertexOut v[3];
};

__global__ void clearBuffers(float* dev_depthbuffer, glm::vec3* dev_framebuffer,
	int screenWidth, int screenHeight);

__global__ void vertexShader(VertexIn* d_vertsIn, VertexOut* d_vertsOut, int vertsNum, glm::mat4 viewProjection);
__global__ void primitiveAssembly(VertexOut* d_vertsOut, int* d_idx, int idxNo, Triangle* d_tri);
__global__ void rasterization(Triangle* d_tri, int triNo, 
		float* dev_depthbuffer, glm::vec3* dev_framebuffer,
		int screenWidth, int screenHeight, int* mutex);