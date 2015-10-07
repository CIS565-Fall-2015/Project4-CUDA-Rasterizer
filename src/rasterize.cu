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
#include <glm/gtc/matrix_transform.hpp>
#include "rasterizeTools.h"

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};
struct VertexOut {
	glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
    // TODO
};
struct Triangle {
    VertexOut v[3];
};
struct Fragment {
    glm::vec3 color;
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static VertexIn *dev_bufVertex = NULL;
static VertexOut *dev_bufTransformedVertex = NULL;
static Triangle *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
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

    if (x < w && y < h) {
        framebuffer[index] = depthbuffer[index].color;
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	printf("Width: %i, Height: %i", width, height);
    cudaFree(dev_depthbuffer);
    cudaMalloc(&dev_depthbuffer,   width * height * sizeof(Fragment));
    cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
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
	cudaMalloc(&dev_bufTransformedVertex, vertCount * sizeof(VertexOut));
    cudaMemcpy(dev_bufVertex, bufVertex, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);

    cudaFree(dev_primitives);
    cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
    cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

    checkCUDAError("rasterizeSetBuffers");
}

__global__ void vertexShader(VertexIn* inVerts, VertexOut* outVerts, int vertCount, glm::mat4 matrix) {
	int thrId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thrId < vertCount) {
		//printf("%i oldVert: (%i, %i, %i) \n", thrId, inVerts[thrId].pos[0], inVerts[thrId].pos[1], inVerts[thrId].pos[2]);
		glm::vec4 newVert = matrix * glm::vec4(inVerts[thrId].pos, 1.0);
		if (newVert[0] != 0) {
			outVerts[thrId].pos = glm::vec3(newVert[0] / newVert[3], newVert[1] / newVert[3], newVert[2] / newVert[3]);
			//printf("%i newVert: (%i, %i, %i) \n", thrId, outVerts[thrId].pos[0], outVerts[thrId].pos[1], outVerts[thrId].pos[2]);
		}
		else {
			outVerts[thrId].pos = glm::vec3(newVert[0], newVert[1], newVert[2]);
			//printf("%i newVert: (%i, %i, %i) \n", thrId, outVerts[thrId].pos[0], outVerts[thrId].pos[1], outVerts[thrId].pos[2]);
		}
	}

}

__global__ void primitiveAssemble(VertexOut* verts, Triangle* tris, int vertCount) {
	int thrId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thrId < vertCount) {
		if (thrId % 3 == 0) {
			tris[thrId/3].v[0] = verts[thrId];
			//printf("%i first: (%i, %i, %i) \n", thrId, verts[thrId].pos[0], verts[thrId].pos[1], verts[thrId].pos[2]);
		} 
		else if (thrId % 3 == 1) {
			tris[(thrId - 1)/3].v[1] = verts[thrId];
			//printf("%i second: (%i, %i, %i) \n", thrId, verts[thrId].pos[0], verts[thrId].pos[1], verts[thrId].pos[2]);
		}
		else {
			tris[(thrId - 2)/3].v[2] = verts[thrId];
			//printf("%i third: (%i, %i, %i) \n", thrId, verts[thrId].pos[0], verts[thrId].pos[1], verts[thrId].pos[2]);
		}
	}
}

__global__ void kernRasterize(Triangle* tris, Fragment* buf, int width, int height, int triCount) {
	int thrId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thrId < triCount) {
		//printf("here \n");
		glm::vec3 triangle[3];
		triangle[0] = tris[thrId].v[0].pos;
		triangle[1] = tris[thrId].v[1].pos;
		triangle[2] = tris[thrId].v[2].pos;
		AABB bbox = getAABBForTriangle(triangle);
		printf("min x: %i, max x: %i, min y: %i, max y: %i \n", bbox.min.x, bbox.max.x, bbox.min.y, bbox.max.y);
		for (int x = 0; x < width; x++) {
			for (int y = 0; y < height; y++) {
				
				
				float tempX = ((x / (float)width) * 2.0f) - 1.0f;
				float tempY = ((y / (float)height) * 2.0f) - 1.0f;
				
				glm::vec3 baryCoord = calculateBarycentricCoordinate(triangle, glm::vec2(tempX, tempY));
				if(isBarycentricCoordInBounds(baryCoord)) {
					printf("(%i, %i) \n", x, y);
					buf[x + y*width].color = glm::vec3(1.0f, 1.0f, 1.0f);
				}
				
			}
		}
	}
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);
    glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
    glm::mat4 projection = glm::perspective<float>(50.0, (float)width / (float)height, 0.0f, 1000.0f);
    glm::mat4 model = glm::mat4();
	glm::mat4 matrix = projection * view * model;
    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	//Set buffer to default value
	cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
	
	//Transfer from VertexIn to VertexOut (vertex shading)
	vertexShader<<<1, vertCount>>>(dev_bufVertex, dev_bufTransformedVertex, vertCount, matrix);
	checkCUDAError("rasterize");
	//Transfer from VertexOut to Triangles (primitive assembly)
	primitiveAssemble<<<1, vertCount>>>(dev_bufTransformedVertex, dev_primitives, vertCount);
	checkCUDAError("rasterize");
	//Scanline each triangle to get fragment color (rasterize)
	int triCount = vertCount / 3;
	printf("Tri count: %i \n", triCount);
	kernRasterize<<<1, triCount>>>(dev_primitives, dev_depthbuffer, width, height, triCount);
	checkCUDAError("rasterize");
    // Copy depthbuffer colors into framebuffer
    render<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer, dev_framebuffer);
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
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

    checkCUDAError("rasterizeFree");
}
