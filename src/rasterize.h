/**
 * @file      rasterize.h
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#pragma once

#include <glm/glm.hpp>
#include <vector_types.h>

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
	glm::vec3 ndc_pos[3];
};
struct Fragment {
	glm::vec3 color;
	glm::vec3 norm;
	glm::vec3 pos;
	glm::vec3 ndc_pos;
	float depth;
	int fixed_depth;
	VertexOut v;
};

void rasterizeInit(int width, int height);
void rasterizeSetBuffers(
        int bufIdxSize, int *bufIdx,
        int vertCount, float *bufPos, float *bufNor, float *bufCol);
void rasterize(uchar4 *pbo, Cam cam);
void rasterizeFree();
