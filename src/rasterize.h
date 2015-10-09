/**
 * @file      rasterize.h
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#pragma once

#include <glm/glm.hpp>
#include <util/utilityCore.hpp>

void rasterizeTileInit();
void rasterizeInit(int width, int height, MVP *mvp);
void rasterizeSetBuffers(
        int bufIdxSize, int *bufIdx,
        int vertCount, float *bufPos, float *bufNor, float *bufCol);
void rasterize(uchar4 *pbo);
void rasterizeTile(uchar4 *pbo);
void rasterizeFree();
void flushDepthBuffer();
