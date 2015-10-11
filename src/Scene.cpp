/*
 * Scene.cpp
 *
 *  Created on: Oct 8, 2015
 *      Author: sanchitgarg
 */

#include "Scene.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <util/utilityCore.hpp>
#include <iostream>

Scene::Scene(int w, int h)
{
	renderMode = TRIANGLES;
	backFaceCulling = true;
	antiAliasing = false;
	run = true;

	mouse.dragging = false;
	mouse.clicked = false;
	mouse.left = false;
	mouse.right = false;
	mouse.middle = false;

	width = w;
	height = h;
	imageColor = new glm::vec3[width*height];
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
	cam.cameraMatrix = cam.projection * cam.view * cam.model;
	cam.inverseModel = glm::inverse(cam.model);
	cam.inverseTransposeModel = glm::inverseTranspose(cam.model);
	cam.dir = glm::normalize(cam.lookat - cam.pos);
}

void Scene::configureMVPMatrices()
{
	cam.view = glm::lookAt(cam.pos, cam.lookat, cam.up);
	cam.projection = glm::perspective<float>(45.0f, float(width)/ float(height), -100.0f, 100.0f);
	configureCameraMatrix();
}

void Scene::setLights()
{
	light1.pos = glm::vec3(100,100,100);
	light1.col = glm::vec3(1,1,1);

	light2.pos = glm::vec3(-100,100,-100);
	light2.col = glm::vec3(1,1,1);
}

void Scene::moveLights()
{
	glm::vec4 l1(light1.pos, 1.0f);
	l1 = cam.model * l1;

	glm::vec4 l2(light2.pos, 1.0f);
	l2 = cam.model * l2;

	light1.pos = glm::vec3(l1);
	light2.pos = glm::vec3(l2);
}

void Scene::setDefaultCamera()
{
	cam.pos = glm::vec3(0,0,5);
	cam.lookat = glm::vec3(0,0,0);
	cam.up = glm::vec3(0,-1,0);
	cam.view = glm::lookAt(cam.pos, cam.lookat, cam.up);
	cam.projection = glm::perspective<float>(45.0f, float(width)/ float(height), -100.0f, 100.0f);
	cam.model = glm::mat4();

	model.translate = glm::vec3(0.0f);
	model.rotate = glm::vec3(0.0f);
	model.scale = glm::vec3(1.0f);

	run = true;
	configureCameraMatrix();
}

void Scene::updateCameraPos(glm::vec3 p)
{
	cam.pos += p;
	run = true;
	configureMVPMatrices();
}

void Scene::updateCameraLookAt(glm::vec3 p)
{
	cam.lookat += p;
	run = true;
	configureMVPMatrices();
}

void Scene::moveModel(glm::vec3 m)
{
	model.translate += m;
	cam.model = utilityCore::buildTransformationMatrix(model.translate, model.rotate, model.scale);
//	moveLights();
	run = true;
	configureCameraMatrix();
}

void Scene::rotateModel(glm::vec3 r)
{
	model.rotate += r;
	cam.model = utilityCore::buildTransformationMatrix(model.translate, model.rotate, model.scale);
//	moveLights();
	run = true;
	configureCameraMatrix();
}

void Scene::updateRenderMode()
{
	renderMode = (renderMode + 1) % 3;

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

void Scene::toggleAntiAliasing()
{
	antiAliasing = !antiAliasing;
	run = true;
}

void Scene::toggleBackFaceCulling()
{
	backFaceCulling = !backFaceCulling;
	run = true;
}
