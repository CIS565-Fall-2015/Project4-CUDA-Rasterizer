#include "scene.hpp"
#include <glm/gtc/matrix_transform.hpp>

Scene::Scene() {
	// Create with some default values
	this->nearPlane = 0.1f;
	this->farPlane = 100.0f;
	camera.fieldOfView = 45.0f;
	camera.position = glm::vec3(0.0f, 0.0f, 3.0f);
	camera.lookAt = glm::vec3(0.0f);
	camera.right = glm::vec3(1.0f, 0.0f, 0.0f);

	light1.position = glm::vec3(1000.0f);
	light1.color = glm::vec3(1.0f);
	light2.position = glm::vec3(-1000.0f, 1000.0f, -1000.0f);
	light2.color = glm::vec3(1.0f);

	culling = true;
	pointRasterization = false;
	lineRasterization = false;

	mouseState.initialPositionsSet = false;

	// Then have to calculate hte model view matrix
	updateModelView();
}

Scene::Scene(float fieldOfView, int nearPlane, int farPlane, glm::vec3 cameraPosition, glm::vec3 cameraLookAt, 
	glm::vec3 cameraRight, glm::vec3 light1Position, glm::vec3 light1Color, glm::vec3 light2Position, glm::vec3 light2Color, bool culling) {
	this->nearPlane = nearPlane;
	this->farPlane = farPlane;
	camera.fieldOfView = fieldOfView;
	camera.position = cameraPosition;
	camera.lookAt = cameraLookAt;
	camera.right = cameraRight;

	light1.position = light1Position;
	light1.color = light1Color;
	light2.position = light2Position;
	light2.color = light2Color;

	this->culling = culling;
	this->pointRasterization = false;
	this->lineRasterization = false;
	mouseState.initialPositionsSet = false;
	mouseState.leftPressed = false;
	mouseState.middlePressed = false;

	// Then have to calculate hte model view matrix
	updateModelView();
}

Scene::~Scene() {

}

// Used for initial calculation and any updates we might want to make if we add mouse/keyboard control
void Scene::updateModelView() {
	glm::vec3 cameraDirection;

	cameraDirection = glm::normalize(camera.lookAt - camera.position);
	camera.up = glm::cross(camera.right, cameraDirection);
	view = glm::lookAt(camera.position, camera.lookAt, camera.up);
	projection = glm::perspective(camera.fieldOfView, 1.0f, -nearPlane, -farPlane);
	modelView = projection * view * model;
}