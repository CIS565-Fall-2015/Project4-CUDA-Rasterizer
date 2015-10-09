/*
 * Scene.cpp
 *
 *  Created on: Oct 8, 2015
 *      Author: sanchitgarg
 */

#include "Scene.h"
#include <glm/gtc/matrix_transform.hpp>

Scene::Scene(int w, int h)
{
	run = true;
	width = w;
	height = h;
    setDefaultCamera();
    setLights();
}

void Scene::setWidthHeight(int w, int h)
{
	width = w;
	height = h;
}

void Scene::configureCameraMatrix()
{
	cam.view = glm::lookAt(cam.pos, cam.lookat, cam.up);
	////	cam.projection = glm::frustum<float>(-1, 1, -1, 1, -1, 1);
	cam.projection = glm::perspective<float>(45.0f, float(width)/ float(height), -100.0f, 100.0f);
	cam.model = glm::mat4();
	cam.cameraMatrix = cam.projection * cam.view * cam.model;
	cam.dir = glm::normalize(cam.lookat - cam.pos);
}

void Scene::setLights()
{
	light.pos = glm::vec3(1,1,1);
	light.col = glm::vec3(1,1,1);
}

void Scene::setDefaultCamera()
{
	cam.pos = glm::vec3(0,0,5);
	cam.lookat = glm::vec3(0,0,0);
	cam.up = glm::vec3(0,-1,0);

	configureCameraMatrix();
}

void Scene::updateCameraPos(glm::vec3 p)
{
	cam.pos += p;
	run = true;
	configureCameraMatrix();
}

void Scene::updateCameraLookAt(glm::vec3 p)
{
	cam.lookat += p;
	run = true;
	configureCameraMatrix();
}
