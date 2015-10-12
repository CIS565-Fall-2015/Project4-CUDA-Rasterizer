//camera
#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"
#include "glm\gtx\rotate_vector.hpp"

#include <iostream>

class Camera{
private:
	glm::vec3 pos;
	glm::vec3 direction;
	glm::vec3 up;

	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 viewProjection;
	bool updateView;
	bool updateProjection;
	float fovy_rad;
	float aspect;

	void calculateMatrices();

public:
	Camera(int width, int height);
	glm::mat4 getViewProjection();
	void rotateBy(float x, float y, float z);
	void translateBy(float x, float y, float z);
};

Camera::Camera(int width, int height){
	pos = glm::vec3(0, 0, 4);
	direction = glm::vec3(0, 0, -1);
	up = glm::vec3(0, 1, 0);

	fovy_rad = glm::radians(45.0f);
	aspect = (float)width / height;

	updateView = true;
	updateProjection = true;
}

void Camera::calculateMatrices(){
	if (updateView){
		view = glm::lookAt(pos, pos + direction, up);
	}

	if (updateProjection){
		projection = glm::perspective(fovy_rad, aspect, 0.1f, 1000.0f);
	}

	if (updateView || updateProjection){
		viewProjection = projection * view;
	}

	updateView = false;
	updateProjection = false;
}


void Camera::rotateBy(float x, float y, float z){
	direction = glm::vec3(glm::rotateX(glm::vec4(direction, 0), x));
	direction = glm::vec3(glm::rotateY(glm::vec4(direction, 0), y));
	direction = glm::vec3(glm::rotateZ(glm::vec4(direction, 0), z));

	//std::cout << angle_x << " " << angle_y << " " << angle_z << std::endl;

	updateView = true;
}

void Camera::translateBy(float x, float y, float z){
	glm::vec3 side = glm::cross(direction, up);
	pos += side * x + up * y + direction * z;
	updateView = true;
}

glm::mat4 Camera::getViewProjection(){
	if (updateView || updateProjection)
		calculateMatrices();
	return viewProjection;
}