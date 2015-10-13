//camera
#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"
#include "glm\gtx\rotate_vector.hpp"

#include <iostream>

class Camera{
private:
	glm::vec3 zoomRadius;
	glm::vec3 target;
	glm::vec3 up;

	float angle_x;
	float angle_y;

	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 viewProjection;
	glm::mat4 invTViewProjection;
	bool updateView;
	bool updateProjection;
	float fovy_rad;
	float aspect;

public:
	Camera(int width, int height);
	glm::mat4 Camera::getViewProjection();
	void zoomInBy(float z);
	void rotateBy(float x, float y);
	void translateBy(float x, float y, float z);
};

Camera::Camera(int width, int height){
	zoomRadius = glm::vec3(0, 0, 5);
	target = glm::vec3(0, 0, 0);
	up = glm::vec3(0, 1, 0);

	fovy_rad = glm::radians(90.0f);
	aspect = (float)width / height;

	updateView = true;
	updateProjection = true;

	angle_x = 0;
	angle_y = 0;
}

void Camera::rotateBy(float x, float y){
	angle_x += x;
	angle_y += y;

	float twoPi = glm::two_pi<float>();
	if (angle_x < 0) angle_x += twoPi;
	if (angle_y < 0) angle_y += twoPi;
	if (angle_x > twoPi) angle_x -= twoPi;
	if (angle_y > twoPi) angle_y -= twoPi;

	updateView = true;
}

void Camera::zoomInBy(float z){
	zoomRadius.z += z;
	updateView = true;
}

void Camera::translateBy(float x, float y, float z){
	target.x += x;
	target.y += y;
	target.z += z;
	updateView = true;
}

glm::mat4 Camera::getViewProjection(){
	if (updateView){
		view = glm::translate(target) * 
			glm::rotate(angle_y, glm::vec3(0, 1, 0)) *
			glm::rotate(angle_x, glm::vec3(1, 0, 0)) *
			glm::translate(zoomRadius);

		view = glm::inverse(view);
	}
	if (updateProjection){
		float near = 0.1f;
		float far = 50.0f;

		float top = near * tan(fovy_rad / 2);
		float bottom = -top;
		float right = top * aspect;
		float left = -right;

		projection = glm::mat4(1.0);
		projection[0][0] = 2 * near / (right - left);
		projection[0][2] = -(right + left) / (right - left);

		projection[1][1] = 2 * near / (top - bottom);
		projection[1][2] = -(top + bottom) / (top - bottom);

		projection[2][2] = far / (far - near);
		projection[2][3] = -(far * near) / (far - near);

		projection[3][2] = 1;
		projection[3][3] = 0;
	}
	if (updateView || updateProjection){
		viewProjection = projection * view;
	}
	return viewProjection;
}