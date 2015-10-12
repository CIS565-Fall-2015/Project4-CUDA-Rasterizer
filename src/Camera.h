//camera
#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"
#include "glm\gtx\rotate_vector.hpp"

class Camera{
private:
	glm::vec3 pos;
	float angle_x;
	float angle_y;
	float angle_z;
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
		view = glm::mat4(1.0);
		view = glm::translate(view, pos);
		view = glm::rotate(view, angle_z, glm::vec3(0, 0, 1));
		view = glm::rotate(view, angle_y, glm::vec3(0, 1, 0));
		view = glm::rotate(view, angle_x, glm::vec3(1, 0, 0));

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
	updateView = true;
}

void Camera::translateBy(float x, float y, float z){
	glm::vec3 dir(x, y, z);
	dir = glm::rotateX(dir, angle_x);
	dir = glm::rotateY(dir, angle_y);
	dir = glm::rotateZ(dir, angle_z);

	pos += dir;
	updateView = true;
}

glm::mat4 Camera::getViewProjection(){
	if (updateView || updateProjection)
		calculateMatrices();
	return viewProjection;
}