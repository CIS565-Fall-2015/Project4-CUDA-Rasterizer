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

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

struct VertexIn {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;
};

struct VertexOut {
	glm::vec3 pos;
	glm::vec3 model_pos; // Used for culling
	glm::vec3 nor;
	glm::vec3 col;
};

struct Triangle {
	VertexOut v[3];
	AABB boundingBox;
	bool visible;
};

struct Fragment {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
};

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

// CHECKITOUT
/**
 * Calculate the signed area of a given triangle.
 */
__host__ __device__ static
float calculateSignedArea(const Triangle tri) {
	return 0.5 * ((tri.v[2].pos.x - tri.v[0].pos.x) * (tri.v[1].pos.y - tri.v[0].pos.y) - (tri.v[1].pos.x - tri.v[0].pos.x) * (tri.v[2].pos.y - tri.v[0].pos.y));
}

// CHECKITOUT
/**
 * Helper function for calculating barycentric coordinates.
 */
__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, const Triangle tri) {
	Triangle baryTri;
	baryTri.v[0].pos = glm::vec3(a, 0);
	baryTri.v[1].pos = glm::vec3(b, 0);
	baryTri.v[2].pos = glm::vec3(c, 0);
	return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

// CHECKITOUT
/**
 * Calculate barycentric coordinates.
 * TODO: Update to handle triangles coming in and not the array
 */
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const Triangle tri, glm::vec2 point) {
	float beta = calculateBarycentricCoordinateValue(glm::vec2(tri.v[0].pos.x, tri.v[0].pos.y), point, glm::vec2(tri.v[2].pos.x, tri.v[2].pos.y), tri);
	float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri.v[0].pos.x, tri.v[0].pos.y), glm::vec2(tri.v[1].pos.x, tri.v[1].pos.y), point, tri);
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
float getZAtCoordinate(const glm::vec3 barycentricCoord, const Triangle tri) {
	return -(barycentricCoord.x * tri.v[0].pos.z
		+ barycentricCoord.y * tri.v[1].pos.z
		+ barycentricCoord.z * tri.v[2].pos.z);
}
