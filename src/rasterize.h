/**
 * @file      rasterize.h
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#pragma once

#include <glm/glm.hpp>
#include "util\obj.hpp"

void rasterizeInit(int width, int height);
void rasterizeSetBuffers(obj * mesh,int tessLevel = 0);
void rasterize(uchar4 *pbo,glm::mat4 viewMat,glm::mat4 projMat,glm::vec3 eye);
void rasterizeFree();
