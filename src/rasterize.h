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

enum GeomMode
{
	GEOM_COMPLETE = 0,
	GEOM_WIREFRAME,
	GEOM_VERTEX
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
		int vertCount, int _faceCount, float *bufPos, float *bufNor, float *bufCol, bool useTexture, float *bufUV);

//texture init
void initTextureData(bool has_diffuse_tex, int d_w, int d_h, glm::vec3 * diffuse_tex,
	bool has_specular_tex, int s_w, int s_h, glm::vec3 * specular_tex,
	const glm::vec3 & ambient, const glm::vec3 & diffuse, const glm::vec3 & specular, float Ns);


void vertexShader(const glm::mat4 & M, const glm::mat4 & M_model_view, const glm::mat4 & inv_trans_M);
void rasterize(uchar4 * pbo);
void rasterizeFree();


void changeShaderMode();
void changeGeomMode();