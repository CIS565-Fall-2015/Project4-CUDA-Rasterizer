/**
 * @file      rasterize_min.h
 * @brief     minimal, barely working kind of CUDA-accelerated skeleton rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 */

#pragma once
#include <glm/glm.hpp>

void minRasterizeInit(int width, int height);
void minRasterizeSetBuffers(
        int bufIdxSize, int *bufIdx,
        int vertCount, float *bufPos, float *bufNor, float *bufCol);
void minRasterizeFirstTry(uchar4 *pbo, glm::mat4 sceneGraphTransform, glm::mat4 cameraMatrix);
void minRasterizeFree();
