/**
 * @file      rasterizeTools.h
 * @brief     Tools/utility functions for rasterization.
 * @authors   Yining Karl Li
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <util/utilityCore.hpp>
#include "sceneStructs.h"

/**
 * Multiplies a glm::mat4 matrix and a vec4.
 */
__host__ __device__ static
glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Finds the axis aligned bounding box for a given triangle.
 */
/*
__host__ __device__ static
AABB getAABBForTriangle(const glm::vec3 tri[3]) {
    AABB aabb;
    aabb.min = glm::vec3(
            glm::min(glm::min(tri[0].x, tri[1].x), tri[2].x),
			glm::min(glm::min(tri[0].y, tri[1].y), tri[2].y),
			glm::min(glm::min(tri[0].z, tri[1].z), tri[2].z));
    aabb.max = glm::vec3(
		glm::max(glm::max(tri[0].x, tri[1].x), tri[2].x),
		glm::max(glm::max(tri[0].y, tri[1].y), tri[2].y),
		glm::max(glm::max(tri[0].z, tri[1].z), tri[2].z));
    return aabb;
}
*/

__host__ __device__ static
AABB getAABBForTriangle(const Triangle tri) {
	AABB aabb;
	aabb.min = glm::vec3(
		glm::min(glm::min(tri.v[0].pos.x, tri.v[1].pos.x), tri.v[2].pos.x),
		glm::min(glm::min(tri.v[0].pos.y, tri.v[1].pos.y), tri.v[2].pos.y),
		glm::min(glm::min(tri.v[0].pos.z, tri.v[1].pos.z), tri.v[2].pos.z));
	aabb.max = glm::vec3(
		glm::max(glm::max(tri.v[0].pos.x, tri.v[1].pos.x), tri.v[2].pos.x),
		glm::max(glm::max(tri.v[0].pos.y, tri.v[1].pos.y), tri.v[2].pos.y),
		glm::max(glm::max(tri.v[0].pos.z, tri.v[1].pos.z), tri.v[2].pos.z));
	return aabb;
}

__host__ __device__ static
AABB getAABB2D(const Triangle tri) {
	AABB aabb;
	aabb.min = glm::vec3(
		glm::min(tri.v[0].pos.x, tri.v[1].pos.x),
		glm::min(tri.v[0].pos.y, tri.v[1].pos.y),
		glm::min(tri.v[0].pos.z, tri.v[1].pos.z));
	aabb.max = glm::vec3(
		glm::max(tri.v[0].pos.x, tri.v[1].pos.x),
		glm::max(tri.v[0].pos.y, tri.v[1].pos.y),
		glm::max(tri.v[0].pos.z, tri.v[1].pos.z));
	return aabb;
}

__host__ __device__ static
AABB getAABB1D(const Triangle tri) {
	AABB aabb;
	aabb.min = tri.v[0].pos;
	aabb.max = tri.v[0].pos;
	return aabb;
}

__host__ __device__ static
AABB getAABB1D(const glm::vec3 point) {
	AABB aabb;
	aabb.min = point;
	aabb.max = point;
	return aabb;
}

// CHECKITOUT
/**
 * Calculate the signed area of a given triangle.
 */
__host__ __device__ static
float calculateSignedArea(const glm::vec3 tri[3]) {
    return 0.5 * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

__host__ __device__ static
float calculateSignedArea(const VertexOut tri[3]) {
	return 0.5 * ((tri[2].pos.x - tri[0].pos.x) * (tri[1].pos.y - tri[0].pos.y) - (tri[1].pos.x - tri[0].pos.x) * (tri[2].pos.y - tri[0].pos.y));
}


// CHECKITOUT
/**
 * Helper function for calculating barycentric coordinates.
 */
__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, const glm::vec3 tri[3]) {
    glm::vec3 baryTri[3];
    baryTri[0] = glm::vec3(a, 0);
    baryTri[1] = glm::vec3(b, 0);
    baryTri[2] = glm::vec3(c, 0);
    return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

/**
* Helper function for calculating barycentric coordinates.
*/
__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, const float triArea) {
	glm::vec3 baryTri[3];
	baryTri[0] = glm::vec3(a, 0);
	baryTri[1] = glm::vec3(b, 0);
	baryTri[2] = glm::vec3(c, 0);
	return calculateSignedArea(baryTri) / triArea;
}

// CHECKITOUT
/**
 * Calculate barycentric coordinates.
 */
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const glm::vec3 tri[3], glm::vec2 point) {
    float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
    float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
    float alpha = 1.0 - beta - gamma;
    return glm::vec3(alpha, beta, gamma);
}

__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const Triangle tri, glm::vec2 point) {
	float beta = calculateBarycentricCoordinateValue(glm::vec2(tri.v[0].pos.x, tri.v[0].pos.y), point, glm::vec2(tri.v[2].pos.x, tri.v[2].pos.y), tri.signedArea);
	float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri.v[0].pos.x, tri.v[0].pos.y), glm::vec2(tri.v[1].pos.x, tri.v[1].pos.y), point, tri.signedArea);
	float alpha = 1.0 - beta - gamma;
	return glm::vec3(alpha, beta, gamma);
}

// CHECKITOUT
/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

// CHECKITOUT
/**
 * For a given barycentric coordinate, compute the corresponding z position
 * (i.e. depth) on the triangle.
 */
__host__ __device__ static
float getZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3]) {
    return -(barycentricCoord.x * tri[0].z
           + barycentricCoord.y * tri[1].z
           + barycentricCoord.z * tri[2].z);
}

__host__ __device__ static
float getZAtCoordinate(const glm::vec3 barycentricCoord, const Triangle tri) {
	return -(barycentricCoord.x * tri.v[0].pos.z
		+ barycentricCoord.y * tri.v[1].pos.z
		+ barycentricCoord.z * tri.v[2].pos.z);
}

__device__ bool boxOverlapTest(AABB a, AABB b){
	bool result;
	if (a.max.x < b.min.x) {
		result = false;
	}
	else if (a.min.x > b.max.x){
		result = false;
	}
	else if (a.max.y < b.min.y){
		result = false;
	}
	else if (a.min.y > b.max.y) {
		result = false;
	}
	else {
		result = true;
	}
	return result;
}