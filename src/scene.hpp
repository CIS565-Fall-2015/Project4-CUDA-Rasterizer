#pragma once

#include "glm/glm.hpp"

using namespace std;

struct MouseState {
	bool initialPositionsSet;
	int x, y;
	bool leftPressed;
	bool middlePressed;
};

struct Camera {
	float fieldOfView;
	glm::vec3 position;
	glm::vec3 lookAt;
	glm::vec3 right;
	glm::vec3 up;
};

struct Light {
	glm::vec3 position;
	glm::vec3 color;
};

class Scene {
private:
	glm::mat4 view;
	glm::mat4 projection;
	const glm::mat4 model = glm::mat4(1.0f);

public:
	Camera camera;

	MouseState mouseState;

	float nearPlane;
	float farPlane;
	glm::mat4 modelView;

	//TODO: This should be an array of lights?
	Light light;

	bool culling;
	bool scissor;
	bool pointRasterization;
	bool lineRasterization;

	glm::vec2 scissorMin;
	glm::vec2 scissorMax;

	Scene();
	Scene(float fieldOfView, int nearPlane, int farPlane, glm::vec3 cameraPosition, 
		glm::vec3 cameraLookAt, glm::vec3 cameraRight, glm::vec3 lightPosition, glm::vec3 lightColor, bool culling);
	~Scene();

	void updateModelView();
};