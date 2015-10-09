/**
 * @file      rasterize.h
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include "EasyBMP.h"

void rasterizeInit(int width, int height, std::string texture);
void initTexture(std::string fileName);
void rasterizeSetBuffers(
        int bufIdxSize, int *bufIdx,
        int vertCount, float *bufPos, float *bufNor, float *bufCol, float *bufTex, int *texIdx);
void rasterize(uchar4 *pbo,glm::vec3 lightPos,glm::vec3 cameraUp,glm::vec3 cameraFront,float fovy,float cameraDis,float rotation,bool outputImage,bool fog,int anti,int frame,float *time);
void rasterizeFree();
//void imageOutput(glm::vec3 *frameBuffer);
