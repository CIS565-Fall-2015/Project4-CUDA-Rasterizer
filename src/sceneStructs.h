/*
 * sceneStructs.h
 *
 *  Created on: Oct 8, 2015
 *      Author: sanchitgarg
 */

#ifndef SCENESTRUCTS_H_
#define SCENESTRUCTS_H_

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};

struct VertexOut {
	glm::vec3 pos;
};

struct Triangle {
    VertexOut vOut[3];
    VertexIn vIn[3];
    bool keep;
};

struct Fragment {
    glm::vec3 color;
    float depth;
};

struct Light {
	glm::vec3 pos;
	glm::vec3 col;
};

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
