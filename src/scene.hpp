#pragma once

#include "glm/glm.hpp"

using namespace std;

struct Camera {
	float fieldOfView;
	glm::vec3 position;
	glm::vec3 lookAt;
	glm::vec3 right;
	glm::vec3 up;
};

class Scene {
private:
	Camera camera;

	float nearPlane;
	float farPlane;

	glm::mat4 view;
	glm::mat4 projection;
	const glm::mat4 model = glm::mat4(1.0f);

public:
	glm::mat4 modelView;

	Scene();
	Scene(float fieldOfView, int nearPlane, int farPlane, glm::vec3 cameraPosition, glm::vec3 cameraLookAt, glm::vec3 cameraRight);
	~Scene();

	void updateModelView();
};