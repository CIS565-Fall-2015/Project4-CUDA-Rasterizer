/*
 * sceneStructs.h
 *
 *  Created on: Oct 8, 2015
 *      Author: sanchitgarg
 */

#ifndef SCENESTRUCTS_H_
#define SCENESTRUCTS_H_

#include <glm/glm.hpp>

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
};

//Vertex in screen coordinates
struct VertexOut {
	glm::vec3 pos;
};

//Triangle, saves vertex in screen space and world space
struct Triangle {
    VertexOut vOut[3];
    VertexIn vIn[3];
    bool keep;
};

//One fragment, saves color and depth
struct Fragment {
    glm::vec3 color;
    int depth;
};

//Light struct
struct Light {
	glm::vec3 pos;
	glm::vec3 col;
};

//Camera struct
//TODO: model matrix should be moved out of this
struct Camera {
	glm::vec3 pos;
	glm::vec3 lookat;
	glm::vec3 up;
	glm::vec3 dir;

	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 cameraMatrix;
};

#endif /* SCENESTRUCTS_H_ */
