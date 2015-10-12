//camera
#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"
#include "glm\gtx\rotate_vector.hpp"

#include <iostream>

class Camera{
private:
	glm::vec3 pos;
	float angle_x;
	float angle_y;
	float angle_z;
	glm::mat4 rot;
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
	angle_x = 0;
	angle_y = 0;
	angle_z = 0;
	fovy_rad = glm::radians(45.0f);
	aspect = (float)width / height;

	updateView = true;
	updateProjection = true;
}

void Camera::calculateMatrices(){
	if (updateView){
		view = glm::translate(glm::mat4(1.0), pos) * rot;
		view = glm::inverse(view);
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
	angle_x += x;
	angle_y += y;
	angle_z += z;

	float two_pi = glm::two_pi<float>();
	if (angle_x >= two_pi) angle_x -= two_pi;
	if (angle_x < 0) angle_x += two_pi;
	if (angle_y >= two_pi) angle_y -= two_pi;
	if (angle_y < 0) angle_y += two_pi;
	if (angle_z >= two_pi) angle_z -= two_pi;
	if (angle_z < 0) angle_z += two_pi;

	rot = glm::rotate(glm::mat4(), angle_z, glm::vec3(0, 0, 1));
	rot = glm::rotate(rot, angle_y, glm::vec3(0, 1, 0));
	rot = glm::rotate(rot, angle_x, glm::vec3(1, 0, 0));

	//std::cout << angle_x << " " << angle_y << " " << angle_z << std::endl;

	updateView = true;
}

void Camera::translateBy(float x, float y, float z){
	glm::vec3 dir_x(rot[0][0], rot[0][1], rot[0][2]);
	dir_x *= x;
	glm::vec3 dir_y(rot[1][0], rot[1][1], rot[1][2]);
	dir_y *= y;
	glm::vec3 dir_z(rot[2][0], rot[2][1], rot[2][2]);
	dir_z *= z;

	pos += dir_x + dir_y + dir_z;

	updateView = true;

	//std::cout << dir_z.x << " " << dir_z.y << " " << dir_z.z << std::endl;
}

glm::mat4 Camera::getViewProjection(){
	if (updateView || updateProjection)
		calculateMatrices();
	return viewProjection;
}