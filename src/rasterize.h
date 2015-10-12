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
#include <thrust/remove.h>
#include <thrust/execution_policy.h>

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
};

struct Triangle {
	VertexOut vOut[3]; 
	VertexIn vIn[3];
};

struct Fragment{
	Triangle *t;
	float depth;
	glm::vec3 bCoord;
};

struct facing_backward
{
	__host__ __device__
	bool operator()(const Triangle t)
	{
		glm::vec3 v1 = glm::normalize(t.vOut[1].pos - t.vOut[0].pos);
		glm::vec3 v2 = glm::normalize(t.vOut[2].pos - t.vOut[0].pos);
		v1 = glm::cross(v1, v2);
		return glm::dot(v1, glm::vec3(0,0,1)) < 0;
	}
};

__global__ void clearBuffers(Fragment* dev_depthbuffer, int screenWidth, int screenHeight);

__global__ void vertexShader(VertexIn* d_vertsIn, VertexOut* d_vertsOut, int vertsNum, glm::mat4 viewProjection);
__global__ void primitiveAssembly(VertexIn* d_vertsIn, VertexOut* d_vertsOut, int* d_idx, int idxNo, Triangle* d_tri);
__global__ void rasterization(Triangle* d_tri, int triNo,
	Fragment* dev_depthbuffer, int screenWidth, int screenHeight, int* mutex);

__global__ void fragmentShader(glm::vec3* dev_framebuffer, Fragment* dev_depthbuffer,
	int width, int height, glm::vec3 *lightSourcePos);