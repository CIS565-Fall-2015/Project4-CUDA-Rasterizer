/*
 * Scene.cpp
 *
 *  Created on: Oct 8, 2015
 *      Author: sanchitgarg
 */

#include "Scene.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <util/utilityCore.hpp>
#include <iostream>

Scene::Scene(int w, int h)
{
	renderMode = TRIANGLES;
	backFaceCulling = true;
	run = true;
	width = w;
	height = h;
	imageColor = new glm::vec3[width*height];
    setDefaultCamera();
    setLights();
}

//Scene::~Scene()
//{
//	delete(imageColor);
//}


void Scene::setWidthHeight(int w, int h)
{
	width = w;
	height = h;
}

void Scene::configureCameraMatrix()
{
	cam.cameraMatrix = cam.projection * cam.view * cam.model;
	cam.inverseModel = glm::inverse(cam.model);
	cam.inverseTransposeModel = glm::inverseTranspose(cam.model);
	cam.dir = glm::normalize(cam.lookat - cam.pos);
}

void Scene::setLights()
{
	light1.pos = glm::vec3(100,100,100);
	light1.col = glm::vec3(1,1,1);

	light2.pos = glm::vec3(-100,100,-100);
	light2.col = glm::vec3(1,1,1);
}

void Scene::setDefaultCamera()
{
	cam.pos = glm::vec3(0,0,5);
	cam.lookat = glm::vec3(0,0,0);
	cam.up = glm::vec3(0,-1,0);
	cam.view = glm::lookAt(cam.pos, cam.lookat, cam.up);
	////	cam.projection = glm::frustum<float>(-1, 1, -1, 1, -1, 1);
	cam.projection = glm::perspective<float>(45.0f, float(width)/ float(height), -100.0f, 100.0f);
	cam.model = glm::mat4();

	run = true;
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

void Scene::moveModel(glm::vec3 m)
{
	cam.model = utilityCore::buildTransformationMatrix(m, glm::vec3(0,0,0), glm::vec3(1,1,1)) * cam.model;
	run = true;
	configureCameraMatrix();
}

void Scene::rotateModel(glm::vec3 r)
{
	cam.model = utilityCore::buildTransformationMatrix(glm::vec3(0,0,0), r, glm::vec3(1,1,1)) * cam.model;
	run = true;
	configureCameraMatrix();
}

void Scene::updateRenderMode()
{
	renderMode = (renderMode + 1) % 2;

//	std::cout<<"Here now";
	run = true;
	if(renderMode == TRIANGLES)
	{
		backFaceCulling = true;
	}
	else
	{
		backFaceCulling = false;
	}
}
