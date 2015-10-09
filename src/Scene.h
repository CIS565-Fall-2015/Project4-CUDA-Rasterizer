/*
 * Scene.h
 *
 *  Created on: Oct 8, 2015
 *      Author: sanchitgarg
 */

#ifndef SCENE_H_
#define SCENE_H_

#include "sceneStructs.h"

class Scene {

public:

	Scene(){}
	Scene(int w, int h);
//	~Scene();

	bool run;
	Camera cam;
	Light light;
	int width, height;
	glm::vec3 *imageColor;

	void setWidthHeight(int w, int h);
	void configureCameraMatrix();
	void setLights();
	void setDefaultCamera();
	void updateCameraPos(glm::vec3);
	void updateCameraLookAt(glm::vec3);
};


#endif /* SCENE_H_ */
