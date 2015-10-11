/*
 * Scene.h
 *
 *  Created on: Oct 8, 2015
 *      Author: sanchitgarg
 */

#ifndef SCENE_H_
#define SCENE_H_

#include "sceneStructs.h"

enum RenderMode{
	TRIANGLES,
	POINTS,
	LINES
};


class Scene {

public:

	Scene(){}
	Scene(int w, int h);
//	~Scene();

	int renderMode;
	bool backFaceCulling;
	bool run;
	Camera cam;
	Light light1, light2;
	int width, height;
	glm::vec3 *imageColor;

	void setWidthHeight(int w, int h);
	void configureCameraMatrix();
	void setLights();
	void setDefaultCamera();
	void updateCameraPos(glm::vec3);
	void updateCameraLookAt(glm::vec3);
	void moveModel(glm::vec3);
	void rotateModel(glm::vec3);
	void updateRenderMode();
	void configureMVPMatrices();
	void moveLights();
};


#endif /* SCENE_H_ */
