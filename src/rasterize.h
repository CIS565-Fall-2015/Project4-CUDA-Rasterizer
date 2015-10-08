/**
 * @file      rasterize.h
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#pragma once

#include <glm/glm.hpp>

#define NUM_LIGHTS (2)

enum ShaderMode
{
	SHADER_NORMAL = 0,
	SHADER_WHITE_MATERIAL,
	SHADER_TEXTURE
};





enum LightType
{
	POINT_LIGHT = 0,
	DIRECTION_LIGHT
};


struct Light
{
	LightType type;

	//glm::vec3 ambient;
	//glm::vec3 diffuse;
	//glm::vec3 specular;
	glm::vec3 intensity;

	//Point light
	glm::vec3 vec;

	bool enabled;
};





void rasterizeInit(int width, int height);
void rasterizeSetBuffers(
        int bufIdxSize, int *bufIdx,
        int vertCount, float *bufPos, float *bufNor, float *bufCol);

void vertexShader(const glm::mat4 & M, const glm::mat4 & M_model_view, const glm::mat4 & inv_trans_M);
void rasterize(uchar4 * pbo);
void rasterizeFree();


void changeShaderMode();