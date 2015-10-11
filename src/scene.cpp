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

	light.position = 1000.0f * camera.position;
	light.color = glm::vec3(1.0f);

	culling = false;
	scissor = false;
	pointRasterization = false;
	lineRasterization = false;

	mouseState.initialPositionsSet = false;

	// Then have to calculate the model view matrix
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

	light.position = 1000.0f * camera.position;
}