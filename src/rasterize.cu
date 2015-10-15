/**
* @file      rasterize.cu
* @brief     CUDA-accelerated rasterization pipeline.
* @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
* @date      2012-2015
* @copyright University of Pennsylvania & STUDENT
*/

#include "rasterize.h"

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include "rasterizeTools.h"
#include <stdint.h>

struct VertexIn {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;
};
struct VertexOut {
	glm::vec3 screenPos;
	glm::vec3 worldPos; // needed for computing lighting
	glm::vec3 worldNor; // needed for computing lighting
	glm::vec3 col;
};
struct Triangle {
	VertexOut v[3];
};

struct Fragment {
	glm::vec3 color;
	glm::vec3 worldNorm; // needed for computing lighting
	glm::vec3 worldPos; // needed for computing lighting
};

struct FragmentAA { // antialiased fragment
	/***************
	*         1
	*  4 
	*       0
	*            2
	*     3
	****************/
	Fragment subFrags[5];
	int primitiveID[5];
};

struct Light {
	glm::vec3 position;
	glm::vec3 ambient;
	glm::vec3 diffuse;
	glm::vec3 specular;
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static unsigned int *dev_intDepths = NULL;
static VertexIn *dev_bufVertex = NULL; // the raw vertices
static VertexOut *dev_tfVertex = NULL; // transformed (tesselated?) vertices
static Triangle *dev_primitives = NULL; // primitives of transformed verts
static Fragment *dev_depthbuffer = NULL; // stores visible fragments
static glm::vec3 *dev_framebuffer = NULL; // framebuffer of colors
static int bufIdxSize = 0;
static int vertCount = 0;

static Light *dev_lights = NULL; // buffer of lights
static int numLights = 0;
static int numInstances = 1;
static glm::mat4 *dev_modelTransforms = NULL;
static glm::mat4 *dev_vertexTransforms = NULL;

// antialiasing stuff
static bool antialiasing = false;
static FragmentAA *dev_depthbufferAA = NULL; // stores MSAA fragments
static unsigned int *dev_intDepthsAA = NULL; // stores depths of MSAA subfragments

/**
* Add Lights
*/
void addLights(std::vector<glm::vec3> &positions, std::vector<glm::vec3> &ambient, 
	std::vector<glm::vec3> &diffuse, std::vector<glm::vec3> &specular) {
	numLights = positions.size();
	cudaFree(dev_lights);
	cudaMalloc(&dev_lights, numLights * sizeof(Light));
	Light *hst_lights = new Light[numLights];
	for (int i = 0; i < numLights; i++) {
		hst_lights[i].position = positions.at(i);
		hst_lights[i].ambient = ambient.at(i);
		hst_lights[i].diffuse = diffuse.at(i);
		hst_lights[i].specular = specular.at(i);
	}
	cudaMemcpy(dev_lights, hst_lights, numLights * sizeof(Light), cudaMemcpyHostToDevice);

	delete hst_lights;
}

/**
* Kernel that writes the image to the OpenGL PBO directly.
*/
__global__
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		glm::vec3 color;
		color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
		color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
		color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

// Writes fragment colors to the framebuffer
__global__
void render(int w, int h, Fragment *depthbuffer, glm::vec3 *framebuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	// frameBuffer code assumes (0,0) is at the bottom left in pix coords,
	// but this code assumes it's at the top right.
	int frameBufferIndex = (w - 1 - x) + (h - 1 - y) * w;

	if (x < w && y < h) {
		framebuffer[index] = depthbuffer[frameBufferIndex].color;
	}
}

// Writes AAfragment colors to the framebuffer
__global__
void renderAA(int w, int h, FragmentAA *depthbufferAA, glm::vec3 *framebuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	// frameBuffer code assumes (0,0) is at the bottom left in pix coords,
	// but this code assumes it's at the top right.
	int frameBufferIndex = (w - 1 - x) + (h - 1 - y) * w;

	if (x < w && y < h) {
		framebuffer[index] = depthbufferAA[frameBufferIndex].subFrags[0].color;
		framebuffer[index] += depthbufferAA[frameBufferIndex].subFrags[1].color;
		framebuffer[index] += depthbufferAA[frameBufferIndex].subFrags[2].color;
		framebuffer[index] += depthbufferAA[frameBufferIndex].subFrags[3].color;
		framebuffer[index] += depthbufferAA[frameBufferIndex].subFrags[4].color;
		framebuffer[index] = framebuffer[index] / 5.0f;
	}
}

/**
* Called once at the beginning of the program to allocate memory.
*/
void rasterizeInit(int w, int h) {
	width = w;
	height = h;
	cudaFree(dev_depthbuffer);
	cudaMalloc(&dev_depthbuffer, width * height * sizeof(Fragment));

	cudaFree(dev_intDepths);
	cudaMalloc(&dev_intDepths, width * height * sizeof(unsigned int));
	cudaMemset(dev_intDepths, 0, width * height * sizeof(unsigned int));

	cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
	cudaFree(dev_framebuffer);
	cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	checkCUDAError("rasterizeInit");
}

/**
* Enable antialiasing. Call after init.
*/
void enableAA() {
	cudaFree(dev_depthbufferAA);
	cudaMalloc(&dev_depthbufferAA, width * height * sizeof(FragmentAA));
	
	cudaFree(dev_intDepthsAA);
	cudaMalloc(&dev_intDepthsAA, width * height * sizeof(unsigned int) * 5);
	cudaMemset(dev_intDepthsAA, 0, width * height * sizeof(unsigned int) * 5);
	
	antialiasing = true;
	checkCUDAError("antialiasing ON");
}

void disableAA() {
	antialiasing = false;
}

/**
* Set all of the buffers necessary for rasterization.
*/
void rasterizeSetBuffers(
	int _bufIdxSize, int *bufIdx,
	int _vertCount, float *bufPos, float *bufNor, float *bufCol) {
	bufIdxSize = _bufIdxSize;
	vertCount = _vertCount;

	cudaFree(dev_bufIdx);
	cudaMalloc(&dev_bufIdx, bufIdxSize * sizeof(int));
	cudaMemcpy(dev_bufIdx, bufIdx, bufIdxSize * sizeof(int), cudaMemcpyHostToDevice);

	VertexIn *bufVertex = new VertexIn[_vertCount];
	for (int i = 0; i < vertCount; i++) {
		int j = i * 3;
		bufVertex[i].pos = glm::vec3(bufPos[j + 0], bufPos[j + 1], bufPos[j + 2]);
		bufVertex[i].nor = glm::vec3(bufNor[j + 0], bufNor[j + 1], bufNor[j + 2]);
		bufVertex[i].col = glm::vec3(bufCol[j + 0], bufCol[j + 1], bufCol[j + 2]);
	}
	cudaFree(dev_bufVertex);
	cudaMalloc(&dev_bufVertex, vertCount * sizeof(VertexIn));
	cudaMemcpy(dev_bufVertex, bufVertex, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);

	cudaFree(dev_primitives);
	cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
	cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

	checkCUDAError("rasterizeSetBuffers");
}

/**
* Set all the buffers that may be influenced by tesselation. Call manually
* after setBuffers (and tesselation, if tesselating)
*/
void rasterizeSetVariableBuffers() {
	cudaFree(dev_tfVertex); // transformed (tesselated?) vertices
	cudaMalloc(&dev_tfVertex, vertCount * sizeof(VertexOut));

	cudaFree(dev_primitives); // primitives of transformed verts
	cudaMalloc(&dev_primitives, bufIdxSize / 3 * sizeof(Triangle));
}

/**
* Clears the depth buffer between draws. Expects single dimensional blocks
*/
__global__ 
void clearDepthBuffer(int bufSize, glm::vec3 bgColor, glm::vec3 defaultNorm,
	Fragment *dev_depthbuffer) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < bufSize) {
		dev_depthbuffer[i].color = bgColor;
		dev_depthbuffer[i].worldNorm = defaultNorm;
	}
}

/**
* Clears the depth buffer between draws. Expects single dimensional blocks
*/
__global__
void clearDepthBufferAA(int bufSize, glm::vec3 bgColor, glm::vec3 defaultNorm,
FragmentAA *dev_depthbufferAA) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < bufSize) {
		dev_depthbufferAA[i].subFrags[0].color = bgColor;
		dev_depthbufferAA[i].subFrags[0].worldNorm = defaultNorm;
		dev_depthbufferAA[i].subFrags[1].color = bgColor;
		dev_depthbufferAA[i].subFrags[1].worldNorm = defaultNorm;
		dev_depthbufferAA[i].subFrags[2].color = bgColor;
		dev_depthbufferAA[i].subFrags[2].worldNorm = defaultNorm;
		dev_depthbufferAA[i].subFrags[3].color = bgColor;
		dev_depthbufferAA[i].subFrags[3].worldNorm = defaultNorm;
		dev_depthbufferAA[i].subFrags[4].color = bgColor;
		dev_depthbufferAA[i].subFrags[4].worldNorm = defaultNorm;
		dev_depthbufferAA[i].primitiveID[0] = -1;
		dev_depthbufferAA[i].primitiveID[1] = -1;
		dev_depthbufferAA[i].primitiveID[2] = -1;
		dev_depthbufferAA[i].primitiveID[3] = -1;
		dev_depthbufferAA[i].primitiveID[4] = -1;
	}
}

/**
* Compute vertex transforms. expects single dimensional blocks
*/
__global__ void computeVertexTFs(int tfCount, glm::mat4 *dev_vertexTfs,
	glm::mat4 *dev_modelTfs, glm::mat4 cam_tf) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < tfCount) {
		dev_vertexTfs[i] = cam_tf * dev_modelTfs[i];
	}
}

/**
* Vertex shader
*/
__global__ void vertexShader(int vertCount, int numInstances, glm::mat4 *dev_vertTfs,
	VertexIn *dev_bufVertices, VertexOut *dev_tfVertices) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < vertCount * numInstances) {
		// figure out which matrix to use
		int instanceNumber = i / vertCount;
		// figure out what index to use for dev_vertTfs
		int vertexIndex = i % vertCount;

		dev_tfVertices[i].screenPos = tfPoint(dev_vertTfs[instanceNumber],
			dev_bufVertices[vertexIndex].pos);
		//glm::vec3 screenPos = tfPoint(dev_vertTfs[instanceNumber], dev_bufVertices[vertexIndex].pos); // debug
		dev_tfVertices[i].worldNor = dev_bufVertices[vertexIndex].nor;
		dev_tfVertices[i].col = dev_bufVertices[vertexIndex].col;
	}
}

// primitive assembly. 1D linear blocks expected
__global__ void primitiveAssembly(int numPrimitives, int numVertices, int numInstances,
	int *dev_idx, VertexOut *dev_vertices, Triangle *dev_primitives) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numPrimitives * numInstances) {
		// compute the index to get indices from
		int indicesIndex = i % numPrimitives;
		int offset = (i / numPrimitives) * numVertices;

		dev_primitives[i].v[0] = dev_vertices[dev_idx[indicesIndex * 3] + offset];
		dev_primitives[i].v[1] = dev_vertices[dev_idx[indicesIndex * 3 + 1] + offset];
		dev_primitives[i].v[2] = dev_vertices[dev_idx[indicesIndex * 3 + 2] + offset];
	}
}

__device__ void scanlineSingleFragment(Triangle primitive, glm::vec3 baryCoordinate,
	Fragment &frag, unsigned int &frag_depth, unsigned int zDepth, int *ID_slot,
	int primitive_ID) {
	// check if it's in dev_primitives[i].v using bary
	if (!isBarycentricCoordInBounds(baryCoordinate)) {
		return;
	}

	// do int depthTest using atomicMin. we'll use the whole range of uint to do this.
	// check depth using bary. the version in utils returns a negative z for some reason
	atomicMin(&frag_depth, zDepth);

	if (zDepth == frag_depth) {
		// interpolate color from bary
		glm::vec3 interpColor = primitive.v[0].col * baryCoordinate[0];
		interpColor += primitive.v[1].col * baryCoordinate[1];
		interpColor += primitive.v[2].col * baryCoordinate[2];
		frag.color = interpColor;

		// interpolate normal from bary
		glm::vec3 interpNorm = primitive.v[0].worldNor * baryCoordinate[0];
		interpNorm += primitive.v[1].worldNor * baryCoordinate[1];
		interpNorm += primitive.v[2].worldNor * baryCoordinate[2];
		frag.worldNorm = interpNorm;

		// interpolate world position from bary
		glm::vec3 interpWorld = primitive.v[0].worldPos * baryCoordinate[0];
		interpWorld += primitive.v[1].worldPos * baryCoordinate[1];
		interpWorld += primitive.v[2].worldPos * baryCoordinate[2];
		frag.worldPos = interpWorld;

		if (ID_slot != NULL) {
			*ID_slot = primitive_ID;
		}
	}
}

/**
* Perform scanline rasterization. 1D linear blocks expected.
*/
__global__ void scanlineRasterization(int w, int h, int numPrimitives,
	Triangle *dev_primitives, Fragment *dev_fragsDepths, unsigned int *dev_intDepths) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numPrimitives) {
		// get the AABB of the triangle
		glm::vec3 v[3];
		v[0] = dev_primitives[i].v[0].screenPos;
		v[1] = dev_primitives[i].v[1].screenPos;
		v[2] = dev_primitives[i].v[2].screenPos;

		AABB triangleBB = getAABBForTriangle(v);

		float pixWidth = 2.0f / (float)w; // NDC goes from -1 to 1 in x and y
		float pixHeight = 2.0f / (float)h;

		int BBYmin = triangleBB.min.y * (h / 2) - 1;
		int BBYmax = triangleBB.max.y * (h / 2) + 1;

		int BBXmin = triangleBB.min.x * (w / 2) - 1;
		int BBXmax = triangleBB.max.x * (w / 2) + 1;

		// clip
		if (BBYmin < -h / 2) BBYmin = -h / 2;
		if (BBXmin < -w / 2) BBXmin = -w / 2;
		if (BBYmax > h / 2) BBYmax = h / 2;
		if (BBXmax > w / 2) BBXmax = w / 2;

		// scan over the AABB
		for (int y = BBYmin; y < BBYmax; y++) {
			for (int x = BBXmin; x < BBXmax; x++) {
				glm::vec2 fragCoord = glm::vec2(x * pixWidth + pixWidth * 0.5f,
					y * pixHeight + pixHeight * 0.5f);

				// we're pretending NDC is -1 to +1 along each axis
				// so a fragIndx(0,0) is at NDC -1 -1
				// btw, going from NDC back to pixel coordinates:
				// I've flipped the drawing system, so now it assumes 0,0 is in the bottom left.
				int fragIndex = (x + (w / 2) - 1) + ((y + (h / 2) - 1) * w);
				glm::vec3 baryCoordinate = calculateBarycentricCoordinate(v, fragCoord);
				int zDepth = UINT16_MAX  * -getZAtCoordinate(baryCoordinate, v);

				scanlineSingleFragment(dev_primitives[i], baryCoordinate,
					dev_fragsDepths[fragIndex], dev_intDepths[fragIndex], zDepth, NULL, 0);
			}
		}
	}
}

/**
* Perform scanline rasterization. 1D linear blocks expected.
*/

__global__ void scanlineRasterizationAA(int w, int h, int numPrimitives,
	Triangle *dev_primitives, FragmentAA *dev_fragsDepthsAA, unsigned int *dev_intDepthsAA) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numPrimitives) {
		// get the AABB of the triangle
		glm::vec3 v[3];
		v[0] = dev_primitives[i].v[0].screenPos;
		v[1] = dev_primitives[i].v[1].screenPos;
		v[2] = dev_primitives[i].v[2].screenPos;

		AABB triangleBB = getAABBForTriangle(v);

		float pixWidth = 2.0f / (float)w; // NDC goes from -1 to 1 in x and y
		float pixHeight = 2.0f / (float)h;

		int BBYmin = triangleBB.min.y * (h / 2) - 1;
		int BBYmax = triangleBB.max.y * (h / 2) + 1;

		int BBXmin = triangleBB.min.x * (w / 2) - 1;
		int BBXmax = triangleBB.max.x * (w / 2) + 1;

		// clip
		if (BBYmin < -h / 2) BBYmin = -h / 2;
		if (BBXmin < -w / 2) BBXmin = -w / 2;
		if (BBYmax > h / 2) BBYmax = h / 2;
		if (BBXmax > w / 2) BBXmax = w / 2;

		// scan over the AABB
		for (int y = BBYmin; y < BBYmax; y++) {
			for (int x = BBXmin; x < BBXmax; x++) {

				// we're pretending NDC is -1 to +1 along each axis
				// so a fragIndx(0,0) is at NDC -1 -1
				// btw, going from NDC back to pixel coordinates:
				// I've flipped the drawing system, so now it assumes 0,0 is in the bottom left.
				int fragIndex = (x + (w / 2) - 1) + ((y + (h / 2) - 1) * w);

				/***************
				*         1
				*  4
				*       0
				*            2
				*     3
				****************/

				// do the middle fragment [0]
				glm::vec2 fragCoord = glm::vec2(x * pixWidth + pixWidth * 0.5f,
					y * pixHeight + pixHeight * 0.5f);
				glm::vec3 baryCoordinate = calculateBarycentricCoordinate(v, fragCoord);
				int zDepth = UINT16_MAX  * -getZAtCoordinate(baryCoordinate, v);


				scanlineSingleFragment(dev_primitives[i], baryCoordinate,
					dev_fragsDepthsAA[fragIndex].subFrags[0],
					dev_intDepthsAA[fragIndex * 5], zDepth,
					&dev_fragsDepthsAA[fragIndex].primitiveID[0], i);

				// do fragment 1: 1/6 over, 1/3 up
				fragCoord = glm::vec2(x * pixWidth + pixWidth * 0.5f + pixWidth / 6.0f,
					y * pixHeight + pixHeight * 0.5f + pixHeight / 3.0f);
				baryCoordinate = calculateBarycentricCoordinate(v, fragCoord);
				zDepth = UINT16_MAX  * -getZAtCoordinate(baryCoordinate, v);

				scanlineSingleFragment(dev_primitives[i], baryCoordinate,
					dev_fragsDepthsAA[fragIndex].subFrags[1],
					dev_intDepthsAA[fragIndex * 5 + 1], zDepth,
					&dev_fragsDepthsAA[fragIndex].primitiveID[1], i);

				// do fragment 2: 1/3 over, 1/6 down
				fragCoord = glm::vec2(x * pixWidth + pixWidth * 0.5f + pixWidth / 3.0f,
					y * pixHeight + pixHeight * 0.5f - pixHeight / 6.0f);
				baryCoordinate = calculateBarycentricCoordinate(v, fragCoord);
				zDepth = UINT16_MAX  * -getZAtCoordinate(baryCoordinate, v);

				scanlineSingleFragment(dev_primitives[i], baryCoordinate,
					dev_fragsDepthsAA[fragIndex].subFrags[2],
					dev_intDepthsAA[fragIndex * 5 + 2], zDepth,
					&dev_fragsDepthsAA[fragIndex].primitiveID[2], i);

				// do fragment 3: 1/6 over left, 1/3 down
				fragCoord = glm::vec2(x * pixWidth + pixWidth * 0.5f - pixWidth / 6.0f,
					y * pixHeight + pixHeight * 0.5f - pixHeight / 3.0f);
				baryCoordinate = calculateBarycentricCoordinate(v, fragCoord);
				zDepth = UINT16_MAX  * -getZAtCoordinate(baryCoordinate, v);

				scanlineSingleFragment(dev_primitives[i], baryCoordinate,
					dev_fragsDepthsAA[fragIndex].subFrags[3],
					dev_intDepthsAA[fragIndex * 5 + 3], zDepth,
					&dev_fragsDepthsAA[fragIndex].primitiveID[3], i);

				// do fragment 4: 1/3 over left, 1/6 up
				fragCoord = glm::vec2(x * pixWidth + pixWidth * 0.5f - pixWidth / 3.0f,
					y * pixHeight + pixHeight * 0.5f + pixHeight / 6.0f);
				baryCoordinate = calculateBarycentricCoordinate(v, fragCoord);
				zDepth = UINT16_MAX  * -getZAtCoordinate(baryCoordinate, v);

				scanlineSingleFragment(dev_primitives[i], baryCoordinate,
					dev_fragsDepthsAA[fragIndex].subFrags[4],
					dev_intDepthsAA[fragIndex * 5 + 4], zDepth,
					&dev_fragsDepthsAA[fragIndex].primitiveID[4], i);
			}
		}
	}
}

__device__ void shadeSingleFragment(Fragment &frag, int numLights, Light *dev_lights) {
	// https://www.opengl.org/sdk/docs/tutorials/ClockworkCoders/lighting.php
	glm::vec3 N = frag.worldNorm;
	if (glm::abs(N.x) < 0.1f && glm::abs(N.y) < 0.1f && glm::abs(N.z) < 0.1f) {
		return;
	}

	glm::vec3 V = frag.worldPos;
	glm::vec3 finalColor = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 tmp;
	for (int j = 0; j < numLights; j++) {
		glm::vec3 light = normalize(dev_lights[j].position - V);
		glm::vec3 eye = normalize(-V);
		glm::vec3 refl = normalize(-glm::reflect(light, N));

		//glm::vec3 amb = dev_lights[j].ambient; // debug
		//glm::vec3 diff = dev_lights[j].diffuse; // debug
		//glm::vec3 spec = dev_lights[j].specular; // debug

		// calculate ambient term
		finalColor += dev_lights[j].ambient;

		// calculate diffuse term
		tmp = dev_lights[j].diffuse * glm::max(dot(N, light), 0.0f);
		finalColor[0] += glm::clamp(tmp[0], 0.0f, 1.0f);
		finalColor[1] += glm::clamp(tmp[1], 0.0f, 1.0f);
		finalColor[2] += glm::clamp(tmp[2], 0.0f, 1.0f);

		// calculate specular term
		tmp = dev_lights[j].specular * powf(glm::max(glm::dot(refl, eye), 0.0f), 0.3);
		finalColor[0] += glm::clamp(tmp[0], 0.0f, 1.0f);
		finalColor[1] += glm::clamp(tmp[1], 0.0f, 1.0f);
		finalColor[2] += glm::clamp(tmp[2], 0.0f, 1.0f);
	}
	frag.color *= finalColor;
	//frag.color[0] = dev_fragsDepths[i].worldNorm[0]; // debug normals
	//frag.color[1] = dev_fragsDepths[i].worldNorm[1]; // debug normals
	//frag.color[2] = dev_fragsDepths[i].worldNorm[2]; // debug normals
}

__global__ void fragmentShader(int numFrags, Fragment *dev_fragsDepths, int numLights, Light *dev_lights) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numFrags) {
		shadeSingleFragment(dev_fragsDepths[i], numLights, dev_lights);
	}
}

__global__ void fragmentShaderMSAA(int numFrags, FragmentAA *dev_fragsDepthsAA,
	int numLights, Light *dev_lights) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numFrags) {
		for (int j = 0; j < 5; j++) {
			if (dev_fragsDepthsAA[i].primitiveID[j] > 0) {
				shadeSingleFragment(dev_fragsDepthsAA[i].subFrags[j], numLights, dev_lights);
				// propagate the computation
				for (int k = 0; k < 5; k++) {
					if (j == k) continue;
					if (dev_fragsDepthsAA[i].primitiveID[k] == dev_fragsDepthsAA[i].primitiveID[j]) {
						dev_fragsDepthsAA[i].primitiveID[j] = -1;
						dev_fragsDepthsAA[i].subFrags[k].color = dev_fragsDepthsAA[i].subFrags[j].color;
					}
				}
			}
		}
	}
}

__global__ void fragmentShaderFSAA(int numFrags, FragmentAA *dev_fragsDepthsAA,
	int numLights, Light *dev_lights) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numFrags) {
		shadeSingleFragment(dev_fragsDepthsAA[i].subFrags[0], numLights, dev_lights);
		shadeSingleFragment(dev_fragsDepthsAA[i].subFrags[1], numLights, dev_lights);
		shadeSingleFragment(dev_fragsDepthsAA[i].subFrags[2], numLights, dev_lights);
		shadeSingleFragment(dev_fragsDepthsAA[i].subFrags[3], numLights, dev_lights);
		shadeSingleFragment(dev_fragsDepthsAA[i].subFrags[4], numLights, dev_lights);
	}
}

/**
* Perform rasterization.
*/
void rasterize(uchar4 *pbo, glm::mat4 cameraMatrix) {
	int sideLength2d = 8;
	dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockCount2d_pix((width + blockSize2d.x - 1) / blockSize2d.x,
		(height + blockSize2d.y - 1) / blockSize2d.y);

	int sideLength1d = 16;
	dim3 blockSize1d(sideLength1d);
	dim3 blockCount1d_pix((width * height + sideLength1d - 1) / sideLength1d);

	dim3 blockCount1d_vertices((vertCount * numInstances + sideLength1d - 1) / sideLength1d);
	dim3 blockCount1d_primitives(((bufIdxSize / 3) * numInstances + sideLength1d - 1) / sideLength1d);
	dim3 blockCount1d_transformations((numInstances + sideLength1d - 1) / sideLength1d);

	// 1) clear depth buffer - should be able to pass in color, clear depth, etc.
	glm::vec3 bgColor = glm::vec3(0.1f, 0.1f, 0.1f);
	glm::vec3 defaultNorm = glm::vec3(0.0f, 0.0f, 0.0f);
	
	if (antialiasing) {
		clearDepthBufferAA << <blockCount1d_pix, blockSize1d >> >(width * height,
			bgColor, defaultNorm, dev_depthbufferAA);
	}
	else {
		clearDepthBuffer << <blockCount1d_pix, blockSize1d >> >(width * height,
			bgColor, defaultNorm, dev_depthbuffer);
	}

	int depth = UINT16_MAX; // really should get this from cam params somehow
	cudaMemset(dev_intDepths, depth, width * height * sizeof(int)); // clear the depths grid
	if (antialiasing) { // clear the depths grid for antialiasing
		cudaMemset(dev_intDepthsAA, depth, width * height * sizeof(int) * 5);
	}

	// 2) transform all the vertex tfs with the camera matrix
	computeVertexTFs <<<blockCount1d_transformations, blockSize1d>>>(numInstances,
		dev_vertexTransforms, dev_modelTransforms, cameraMatrix);

	// 3) vertex shading -> generates numInstances * vertCout screen space vertices
	vertexShader <<<blockCount1d_vertices, blockSize1d>>>(vertCount, numInstances,
		dev_vertexTransforms, dev_bufVertex, dev_tfVertex);

	int numPrimitivesTotal = (bufIdxSize / 3) * numInstances;

	// 4) primitive assembly -> generates numInstances * numPrimitives screen space triangles
	primitiveAssembly << <blockCount1d_primitives, blockSize1d >> >(bufIdxSize / 3, 
		vertCount, numInstances, dev_bufIdx, dev_tfVertex, dev_primitives);

	if (antialiasing) {
		// 5) rasterize and depth test
		scanlineRasterizationAA << <blockCount1d_primitives, blockSize1d >> >(
			width, height, numPrimitivesTotal, dev_primitives, dev_depthbufferAA,
			dev_intDepthsAA);

		// 6) fragment shade
		fragmentShaderMSAA << <blockCount1d_pix, blockSize1d >> >(width * height,
			dev_depthbufferAA, numLights, dev_lights);

		// 7) Copy depthbuffer colors into framebuffer
		renderAA << <blockCount2d_pix, blockSize2d >> >(width, height,
			dev_depthbufferAA, dev_framebuffer);
	}
	else {
		// 5) rasterize and depth test
		scanlineRasterization << <blockCount1d_primitives, blockSize1d >> >(width, height,
			numPrimitivesTotal, dev_primitives, dev_depthbuffer, dev_intDepths);

		// 6) fragment shade
		fragmentShader << <blockCount1d_pix, blockSize1d >> >(width * height, dev_depthbuffer, numLights, dev_lights);

		// 7) Copy depthbuffer colors into framebuffer
		render << <blockCount2d_pix, blockSize2d >> >(width, height, dev_depthbuffer, dev_framebuffer);
	}
	// Copy framebuffer into OpenGL buffer for OpenGL previewing
	sendImageToPBO << <blockCount2d_pix, blockSize2d >> >(pbo, width, height, dev_framebuffer);
	checkCUDAError("rasterize");
}

/**
* Called once per change of instancing... situation.
* - allocate space for and upload the matrices
* - allocate space for the new tf vertices on the device
* - set the number of instances
*/
void setupInstances(std::vector<glm::mat4> &modelTransform) {
	numInstances = modelTransform.size();

	cudaFree(dev_modelTransforms);
	cudaMalloc(&dev_modelTransforms, numInstances * sizeof(glm::mat4));
	cudaMemcpy(dev_modelTransforms, modelTransform.data(),
		numInstances * sizeof(glm::mat4), cudaMemcpyHostToDevice);

	cudaFree(dev_vertexTransforms);
	cudaMalloc(&dev_vertexTransforms, numInstances * sizeof(glm::mat4));

	cudaFree(dev_tfVertex);
	cudaMalloc(&dev_tfVertex, numInstances * vertCount * sizeof(VertexOut));

	cudaFree(dev_primitives); // primitives of transformed verts
	cudaMalloc(&dev_primitives, numInstances * bufIdxSize / 3 * sizeof(Triangle));
}

/**
* Called once at the end of the program to free CUDA memory.
*/
void rasterizeFree() {
	cudaFree(dev_bufIdx);
	dev_bufIdx = NULL;

	cudaFree(dev_bufVertex);
	dev_bufVertex = NULL;

	cudaFree(dev_primitives);
	dev_primitives = NULL;

	cudaFree(dev_depthbuffer);
	dev_depthbuffer = NULL;

	cudaFree(dev_intDepths);
	dev_intDepths = NULL;

	cudaFree(dev_framebuffer);
	dev_framebuffer = NULL;

	cudaFree(dev_tfVertex); // transformed (tesselated?) vertices
	dev_tfVertex = NULL;

	cudaFree(dev_primitives); // primitives of transformed verts
	dev_primitives = NULL;

	cudaFree(dev_lights);
	dev_lights = NULL;
	numLights = 0;

	cudaFree(dev_modelTransforms);
	dev_modelTransforms = NULL;

	cudaFree(dev_vertexTransforms);
	dev_vertexTransforms = NULL;

	cudaFree(dev_depthbufferAA);
	dev_depthbufferAA = NULL;

	numInstances = 1;

	checkCUDAError("rasterizeFree");
}