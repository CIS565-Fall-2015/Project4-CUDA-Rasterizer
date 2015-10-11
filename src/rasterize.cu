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


struct VertexIn {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;
};
struct VertexOut {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;
};
struct Triangle {
	VertexOut v[3];
};

struct Fragment {
	glm::vec3 color;
	glm::vec3 norm;
	float depth;
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static VertexIn *dev_bufVertex = NULL; // the raw vertices
//static int *dev_tesselatedIdx = NULL; // tesselated indices
//static VertexIn *dev_tesselatedVertex = NULL; // tesselated vertices
static VertexOut *dev_tfVertex = NULL; // transformed (tesselated?) vertices
static Triangle *dev_primitives = NULL; // primitives of transformed verts
static Fragment *dev_depthbuffer = NULL; // stores visible fragments
static glm::vec3 *dev_framebuffer = NULL; // framebuffer of colors
static int bufIdxSize = 0;
static int vertCount = 0;

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

/**
* Called once at the beginning of the program to allocate memory.
*/
void rasterizeInit(int w, int h) {
	width = w;
	height = h;
	cudaFree(dev_depthbuffer);
	cudaMalloc(&dev_depthbuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
	cudaFree(dev_framebuffer);
	cudaMalloc(&dev_framebuffer, width * height * sizeof(glm::vec3));
	cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));

	checkCUDAError("rasterizeInit");
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
	float defaultDepth, Fragment *dev_depthbuffer) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < bufSize) {
		dev_depthbuffer[i].color = bgColor;
		dev_depthbuffer[i].norm = defaultNorm;
		dev_depthbuffer[i].depth = defaultDepth;
	}
}


/**
* Vertex shader
*/
__global__ void vertexShader(int vertCount, glm::mat4 tf, VertexIn *dev_bufVertices,
	VertexOut *dev_tfVertices) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < vertCount) {
		dev_tfVertices[i].pos = tfPoint(tf, dev_bufVertices[i].pos);
		dev_tfVertices[i].nor = dev_bufVertices[i].nor;
		dev_tfVertices[i].col = dev_bufVertices[i].col;
	}
}

// primitive assembly. 1D linear blocks expected
__global__ void primitiveAssembly(int numPrimitives, int *dev_idx,
	VertexOut *dev_vertices, Triangle *dev_primitives) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numPrimitives) {
		dev_primitives[i].v[0] = dev_vertices[dev_idx[i * 3]];
		dev_primitives[i].v[1] = dev_vertices[dev_idx[i * 3 + 1]];
		dev_primitives[i].v[2] = dev_vertices[dev_idx[i * 3 + 2]];
	}
}

/**
* Perform scanline rasterization. 1D linear blocks expected.
*/
__global__ void scanlineRasterization(int w, int h, int numPrimitives,
	Triangle *dev_primitives, Fragment *dev_fragsDepths) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numPrimitives) {
		// get the AABB of the triangle
		glm::vec3 v[3];
		v[0] = dev_primitives[i].v[0].pos;
		v[1] = dev_primitives[i].v[1].pos;
		v[2] = dev_primitives[i].v[2].pos;

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
				// check if it's in dev_primitives[i].v using bary
				glm::vec3 baryCoordinate = calculateBarycentricCoordinate(v, fragCoord);
				if (!isBarycentricCoordInBounds(baryCoordinate)) {
					continue;
				}
				// check depth using bary. the version in utils returns a negative z for some reason,
				// so depths are assumed to be negative, with "more negative" indicating further away.
				float zDepth = getZAtCoordinate(baryCoordinate, v);

				// we're pretending NDC is -1 to +1 along each axis
				// so a fragIndx(0,0) is at NDC -1 -1
				// btw, going from NDC back to pixel coordinates:
				// I've flipped the drawing system, so now it assumes 0,0 is in the bottom left.
				int fragIndex = (x + (w / 2) - 1) + ((y + (h / 2) - 1) * w);
				if (zDepth > dev_fragsDepths[fragIndex].depth) { // remember, depths are negative
					dev_fragsDepths[fragIndex].depth = zDepth;
					// interpolate color
					glm::vec3 interpColor = dev_primitives[i].v[0].col * baryCoordinate[0];
					interpColor += dev_primitives[i].v[1].col * baryCoordinate[1];
					interpColor += dev_primitives[i].v[2].col * baryCoordinate[2];
					dev_fragsDepths[fragIndex].color = interpColor;

					// interpolate normal
					glm::vec3 interpNorm = dev_primitives[i].v[0].nor * baryCoordinate[0];
					interpNorm += dev_primitives[i].v[1].nor * baryCoordinate[1];
					interpNorm += dev_primitives[i].v[2].nor * baryCoordinate[2];
					dev_fragsDepths[fragIndex].norm = interpNorm;
				}
			}
		}
	}
}

__global__ void fragmentShader(int numFrags, Fragment *dev_fragsDepths) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < numFrags) {
		dev_fragsDepths[i].color[0] = dev_fragsDepths[i].norm[0];
		dev_fragsDepths[i].color[1] = dev_fragsDepths[i].norm[1];
		dev_fragsDepths[i].color[2] = dev_fragsDepths[i].norm[2];
	}
}

/**
* Perform rasterization.
*/
void rasterize(uchar4 *pbo, glm::mat4 sceneGraphTransform, glm::mat4 cameraMatrix) {
	int sideLength2d = 8;
	dim3 blockSize2d(sideLength2d, sideLength2d);
	dim3 blockCount2d_pix((width + blockSize2d.x - 1) / blockSize2d.x,
		(height + blockSize2d.y - 1) / blockSize2d.y);

	int sideLength1d = 16;
	dim3 blockSize1d(sideLength1d);
	dim3 blockCount1d_pix((width * height + sideLength1d - 1) / sideLength1d);

	dim3 blockCount1d_vertices((vertCount + sideLength1d - 1) / sideLength1d);
	dim3 blockCount1d_primitives(((bufIdxSize / 3) + sideLength1d - 1) / sideLength1d);

	// 1) clear depth buffer - should be able to pass in color, clear depth, etc.
	glm::vec3 bgColor = glm::vec3(0.1f, 0.1f, 0.1f);
	float depth = -HUGE_VAL; // -infinity. really should be extracting this from the camera or passing in.
	clearDepthBuffer << <blockCount1d_pix, blockSize1d >> >(width * height, bgColor,
		bgColor, depth, dev_depthbuffer);

	// 2) vertex shading - pass in vertex tf
	glm::mat4 tf = cameraMatrix * sceneGraphTransform;
	vertexShader <<<blockCount1d_vertices, blockSize1d>>>(vertCount, tf, dev_bufVertex, dev_tfVertex);

	// 3) primitive assembly
	primitiveAssembly << <blockCount1d_primitives, blockSize1d >> >(bufIdxSize / 3, dev_bufIdx,
		dev_tfVertex, dev_primitives);

	// 4) rasterize and depth test
	scanlineRasterization <<<blockCount1d_primitives, blockSize1d >> >(width, height, bufIdxSize / 3,
		dev_primitives, dev_depthbuffer);

	// 5) fragment shade
	fragmentShader <<<blockCount1d_pix, blockSize1d >>>(width * height, dev_depthbuffer);

	// 6) Copy depthbuffer colors into framebuffer
	render << <blockCount2d_pix, blockSize2d >> >(width, height, dev_depthbuffer, dev_framebuffer);
	// Copy framebuffer into OpenGL buffer for OpenGL previewing
	sendImageToPBO << <blockCount2d_pix, blockSize2d >> >(pbo, width, height, dev_framebuffer);
	checkCUDAError("rasterize");
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

	cudaFree(dev_framebuffer);
	dev_framebuffer = NULL;

	cudaFree(dev_tfVertex); // transformed (tesselated?) vertices
	dev_tfVertex = NULL;

	cudaFree(dev_primitives); // primitives of transformed verts
	dev_primitives = NULL;

	checkCUDAError("rasterizeFree");
}