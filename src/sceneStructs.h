#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <glm/gtx/transform.hpp>

#pragma once

struct Scissor {
	glm::vec2 min;
	glm::vec2 max;
};

struct MVP {
	// Perspective projection box
	const float nearPlane = 0.1f;
	const float farPlane = 100.0f;
	//
	float fov;
	glm::mat4 projection;
	// Camera matrix
	glm::vec3 camPosition;
	glm::vec3 camLookAt;
	glm::vec3 camRight;
	glm::vec3 camUp;
	glm::vec3 camPosOld;
	glm::vec3 camLookAtOld;
	glm::vec3 camRightOld;
	//
	glm::mat4 view;
	// Model translation
	const glm::mat4 model = glm::mat4(1.0f);

	// Combined matrix
	glm::mat4 mvp;

	// Mouse input control
	bool mouseLeftDown;
	bool mouseRightDown;
	double mouseDownX, mouseDownY;
	bool initPos;

	// Scissor test
	bool doScissor;
	Scissor scissor;

	// Shade mode
	int shadeMode;

	// Geometry shader
	bool geomShading;

	// Point shader
	bool pointShading;

	// Backface culling
	bool culling;
};

struct AABB {
	glm::vec3 min;
	glm::vec3 max;
};

struct VertexIn {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;
};
struct VertexOut {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;
	glm::vec3 mpos;
};
struct Triangle {
	VertexOut v[3];
	AABB box;
	bool isPoint;
	bool isLine;
	bool isValidGeom;
};
struct Fragment {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;
};

struct is_invalid{
	__host__ __device__
		bool operator()(const Triangle &t)
	{
		return !t.isValidGeom;
	}
};