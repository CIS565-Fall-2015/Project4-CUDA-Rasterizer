#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <glm/gtx/transform.hpp>

#pragma once

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
};