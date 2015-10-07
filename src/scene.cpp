#include "scene.hpp"
#include <glm/gtc/matrix_transform.hpp>

Scene::Scene() {
	// Create with some default values
	Scene(45.0f, 0.1f, 100.0f, glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
}

Scene::Scene(float fieldOfView, int nearPlane, int farPlane, glm::vec3 cameraPosition, glm::vec3 cameraLookAt, glm::vec3 cameraRight) {
	this->nearPlane = nearPlane;
	this->farPlane = farPlane;
	camera.fieldOfView = fieldOfView;
	camera.position = cameraPosition;
	camera.lookAt = cameraLookAt;
	camera.right = cameraRight;

	// Then have to calculate hte model view matrix
	updateModelView();
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