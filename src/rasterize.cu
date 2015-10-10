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
#include <glm/gtc/matrix_inverse.hpp>
#include <thrust/remove.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include "rasterizeTools.h"
extern glm::vec3 camCoords;
glm::vec3 cam(0.0f, 3.0f, 3.0f);
#define MAX_DEPTH INT_MAX
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
	bool erase;
};
struct Fragment {
    glm::vec3 color;
	float depth;
	int idepth;
	glm::vec3 nor;

};

struct removeTriangles {
	__host__ __device__ bool operator() (const Triangle tri) {
		return tri.erase;
	}
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static VertexIn *dev_bufVertex = NULL;
static VertexOut *dev_bufTransformedVertex = NULL;
static Triangle *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static Fragment *dev_fragbuffer = NULL;
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
	//printf("Width: %i, Height: %i", width, height);
    cudaFree(dev_depthbuffer);
    cudaMalloc(&dev_depthbuffer,   4 * width * height * sizeof(Fragment));
    cudaMemset(dev_depthbuffer, 0, 4 * width * height * sizeof(Fragment));
	cudaFree(dev_fragbuffer);
	cudaMalloc(&dev_fragbuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragbuffer, 0, width * height * sizeof(Fragment));
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

__global__ void setDepth(Fragment* depth, int width) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * width);
	depth[index].depth = 1.0f;
	depth[index].idepth = MAX_DEPTH;
	depth[index].color = glm::vec3(0.0f, 0.0f, 0.0f);
}

 __device__ __host__ glm::vec4 mul(glm::mat4 m, glm::vec4 v) {
    return glm::vec4(m[0].x*v.x + m[1].x*v.y + m[2].x*v.z + m[3].x*v.w,
                 m[0].y*v.x + m[1].y*v.y + m[2].y*v.z + m[3].y*v.w,
                 m[0].z*v.x + m[1].z*v.y + m[2].z*v.z + m[3].z*v.w,
                 m[0].w*v.x + m[1].w*v.y + m[2].w*v.z + m[3].w*v.w);
 }

__global__ void vertexShader(VertexIn* inVerts, VertexOut* outVerts, int vertCount, glm::mat4 matrix, glm::mat4 model) {
	int thrId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thrId < vertCount) {
		//printf("%i oldVert: (%f, %f, %f) \n", thrId, inVerts[thrId].pos[0], inVerts[thrId].pos[1], inVerts[thrId].pos[2]);
		//printf("matrix: (%f, %f, %f, %f) \n", matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]);
		//printf("matrix: (%f, %f, %f, %f) \n", matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]);
		
		//glm::vec4 newVert = matrix * glm::vec4(inVerts[thrId].pos[0], inVerts[thrId].pos[1], inVerts[thrId].pos[2], 1.0f);
		//newVert[2] = (matrix[2][2] * inVerts[thrId].pos[2]) + matrix[2][3];
		//newVert[3] = (matrix[3][2] * inVerts[thrId].pos[2]) + matrix[3][3];
		glm::vec4 oldVert(inVerts[thrId].pos, 1.0f);
		glm::vec4 newVert = mul(matrix, oldVert);

		glm::vec4 newNorm = mul(glm::inverseTranspose(model), glm::vec4(inVerts[thrId].nor, 0.0f));
		//printf("newVert 2: %f, new vert 3: %f \n", newVert[2], newVert[3]);
		if (newVert[3] != 0) {
			outVerts[thrId].pos = glm::vec3(newVert[0] / newVert[3], newVert[1] / newVert[3], newVert[2] / newVert[3]);
			outVerts[thrId].nor = glm::vec3(newNorm); //inVerts[thrId].nor; //
			outVerts[thrId].col = inVerts[thrId].col;
			//printf("%i newVert: (%f, %f, %f) \n", thrId, outVerts[thrId].pos[0], outVerts[thrId].pos[1], outVerts[thrId].pos[2]);
		}
		else {
			outVerts[thrId].pos = glm::vec3(newVert[0], newVert[1], newVert[2]);
			outVerts[thrId].nor = glm::vec3(newNorm); //inVerts[thrId].nor;
			outVerts[thrId].col = inVerts[thrId].col;
			//printf("%i newVert: (%f, %f, %f) \n", thrId, outVerts[thrId].pos[0], outVerts[thrId].pos[1], outVerts[thrId].pos[2]);
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

__global__ void backWard(Triangle* tris, int triCount) {
	int thrId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thrId < triCount) {
		glm::vec3 camera(0.0f, 3.0f, 3.0f);
		glm::vec3 normal = tris[thrId].v[0].nor; //glm::cross((tris[thrId].v[1].pos - tris[thrId].v[0].pos), (tris[thrId].v[2].pos - tris[thrId].v[0].pos));
		glm::vec3 newVec = tris[thrId].v[0].pos; //(tris[thrId].v[0].pos + tris[thrId].v[1].pos + tris[thrId].v[2].pos) / 3.0f;
		newVec = (newVec - camera);
		//printf("normal: (%f, %f, %f) \n", normal[0], normal[1], normal[2]);
		float NdotC = glm::dot(normal, newVec); 
		//printf("NdotC: %f \n", NdotC);
		if ( NdotC >= 0.0f) {
			tris[thrId].erase = true;
			//printf("thrID: %i, true \n", thrId);
		}
		else {
			tris[thrId].erase = false;
			//printf("thrID: %i, false \n", thrId);
		}
	}
}

__global__ void lineRasterize(Triangle* tris, Fragment* buf, int width, int height, int triCount, glm::mat4 matrix) {
	int thrId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thrId < triCount) {
		for (int m = 0; m < 3; m++) {
			glm::mat4 screenCoords(0.0f);
			screenCoords[0][0] = (float)width;
			screenCoords[1][1] = (float)height;
			screenCoords[2][2] = .5f;
			screenCoords[0][3] = (float)width;
			screenCoords[1][3] = (float)height;
			screenCoords[2][3] = .5f;
			screenCoords[3][3] = 1.0f;
			//printf("p0: (%f, %f, %f) \n", tris[thrId].v[2].pos[0], tris[thrId].v[2].pos[1], tris[thrId].v[2].pos[2]);
			//printf("p1: (%f, %f, %f) \n", tris[thrId].v[1].pos[0], tris[thrId].v[1].pos[1], tris[thrId].v[1].pos[2]);
			//glm::vec4 p0 = mul(screenCoords, glm::vec4(tris[thrId].v[0].pos, 1.0f));
			//p0 = mul(screenCoords, p0);
			//glm::vec4 p1 = mul(screenCoords, glm::vec4(tris[thrId].v[1].pos, 1.0f));
			//p1 = mul(screenCoords, p1);
			float p0x, p0y, p1x, p1y;
			if (m != 2) {
				p0x = (tris[thrId].v[m].pos[0] + 1) * (width / 2.0f);
				p0y = (tris[thrId].v[m].pos[1] + 1) * (height / 2.0f);
				p1x = (tris[thrId].v[m + 1].pos[0] + 1) * (width / 2.0f);
				p1y = (tris[thrId].v[m + 1].pos[1] + 1) * (height / 2.0f);
			}
			else {
				p0x = (tris[thrId].v[m].pos[0] + 1) * (width / 2.0f);
				p0y = (tris[thrId].v[m].pos[1] + 1) * (height / 2.0f);
				p1x = (tris[thrId].v[0].pos[0] + 1) * (width / 2.0f);
				p1y = (tris[thrId].v[0].pos[1] + 1) * (height / 2.0f);
			}
			glm::vec2 p0(p0x, p0y);
			glm::vec2 p1(p1x, p1y);
			//printf("point1: (%f, %f) \n point 2: (%f, %f) \n", p0.x, p0.y, p1.x, p1.y);

			//glm::vec2 p0(30, 0);
			//glm::vec2 p1(30, 400);
			float slope;
			if (p1.x - p0.x > 0.0001f || p1.x - p0.x < -.00001f) {
				slope = (p1.y - p0.y) / (p1.x - p0.x);
				//printf("slode: %f \n", slope);
				if (slope >= -1.0f && slope <= 1.0f) {
					if (p1.x > p0.x) {
						for (int x = ceil(p0.x); x < floor(p1.x); x++) {
							float y = p0.y + (((float)x - p0.x) / (p1.x - p0.x))*(p1.y - p0.y);
							int newY = glm::round(y);
							//printf("points: (%i, %i) \n", x, newY);
							if (x < width && x >= 0 && newY < height && newY >= 0) {
						
								buf[4*x + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 1 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 2 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 3 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								//buf[4*x + newY*4*width].idepth = -1;
							}
						}
					}
					else {
						for (int x = floor(p0.x); x > ceil(p1.x); x--) {
							float y = p0.y + (((float)x - p0.x) / (p1.x - p0.x))*(p1.y - p0.y);
							int newY = glm::round(y);
							//printf("points: (%i, %i) \n", x, newY);
							if (x < width && x >= 0 && newY < height && newY >= 0) {
								//printf("pixel: (%i, %i) \n", x, newY);
								buf[4*x + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 1 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 2 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 3 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								//buf[4*x + newY*4*width].idepth = -1;
							}
						}
					}
				}
				else {
					if (p1.y > p0.y) {
						for (int y = ceil(p0.y); y < floor(p1.y); y++) {
							float x = p0.x + (((float)y - p0.y)*(1.0f / slope));
							int newX = glm::round(x);
							//printf("points: (%i, %i) \n", newX, y);
							if (y < height && y >= 0 && newX < width && newX >= 0) {
								buf[4*newX + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 1 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 2 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 3 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								//buf[4*newX + y*4*width].idepth = -1;
							}
						}
					}
					else {
						for (int y = floor(p0.y); y > ceil(p1.y); y--) {
							float x = p0.x + (((float)y - p0.y)*(1.0f / slope));
							int newX = glm::round(x);
							//printf("points: (%i, %i) \n", newX, y);
							if (y < height && y >= 0 && newX < width && newX >= 0) {
								buf[4*newX + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 1 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 2 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 3 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								//buf[4*newX + y*4*width].idepth = -1;
							}
						}
					}
			
				}
			}
			else {
				if (glm::abs(p1.x - p0.x) < .00001f) {
					if (p1.y > p0.y) {
						for (int y = ceil(p0.y); y < floor(p1.y); y++) {
							int newX = p0.x;
							if (y < height && y >= 0 && newX < width && newX >= 0) {
								buf[4*newX + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 1 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 2 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 3 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								//buf[4*newX + y*4*width].idepth = -1;
							}
						}
					}
					else {
						for (int y = floor(p0.y); y > ceil(p1.y); y--) {
							int newX = p0.x;
							//printf("points: (%i, %i) \n", newX, y);
							if (y < height && y >= 0 && newX < width && newX >= 0) {
								buf[4*newX + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 1 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 2 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*newX + 3 + y*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								//buf[4*newX + y*4*width].idepth = -1;
							}
						}
					}
				}
				else {
					if (p1.x > p0.x) {
						for (int x = ceil(p0.x); x < floor(p1.x); x++) {
							int newY = p0.y;
							//printf("points: (%i, %i) \n", x, newY);
							if (x < width && x >= 0 && newY < height && newY >= 0) {
						
								buf[4*x + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 1 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 2 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 3 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								//buf[4*x + newY*4*width].idepth = -1;
							}
						}
					}
					else {
						for (int x = floor(p0.x); x > ceil(p1.x); x--) {
							int newY = p0.y;
							//printf("points: (%i, %i) \n", x, newY);
							if (x < width && x >= 0 && newY < height && newY >= 0) {
								//printf("pixel: (%i, %i) \n", x, newY);
								buf[4*x + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 1 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 2 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								buf[4*x + 3 + newY*4*width].color = glm::vec3(1.0f, 1.0f, 0.0f);
								//buf[4*x + newY*4*width].idepth = -1;
							}
						}
					}
				}
			}
		
		}
	}
}
__global__ void kernRasterize(Triangle* tris, Fragment* buf, int width, int height, int triCount) {
	int thrId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thrId < triCount) {
		//printf("heyo \n");
		glm::vec3 triangle[3];
		triangle[0] = tris[thrId].v[0].pos;
		triangle[1] = tris[thrId].v[1].pos;
		triangle[2] = tris[thrId].v[2].pos;
		//printf("1 (%f, %f, %f) \n", tris[thrId].v[0].pos[2], tris[thrId].v[1].pos[2], tris[thrId].v[2].pos[2]);
		/*printf("thrId: %i \n", thrId);
		printf("1 (%f, %f, %f) \n", tris[thrId].v[0].nor[0], tris[thrId].v[0].nor[1], tris[thrId].v[0].nor[2]);
		printf("2 (%f, %f, %f) \n", tris[thrId].v[1].nor[0], tris[thrId].v[1].nor[1], tris[thrId].v[1].nor[2]);
		printf("3 (%f, %f, %f) \n", tris[thrId].v[2].nor[0], tris[thrId].v[2].nor[1], tris[thrId].v[2].nor[2]);
		//printf("heyo2 \n");
		printf("normal (%f, %f, %f) \n", normal[0], normal[1], normal[2]);*/
		AABB bbox = getAABBForTriangle(triangle);
		//printf("min x: %f, max x: %f, min y: %f, max y: %f \n", bbox.min.x, bbox.max.x, bbox.min.y, bbox.max.y);
		float minX = (bbox.min.x + 1) * (width / 2.0f);
		float maxX = (bbox.max.x + 1) * (width / 2.0f);
		float minY = (bbox.min.y + 1) * (height / 2.0f);
		float maxY = (bbox.max.y + 1) * (height / 2.0f);
		//printf("heyo3 \n");
		for (int x = glm::max(0.0f, floor(minX)); x < glm::min(width - 1.0f, ceil(maxX)); x++) {
			for (int y = glm::max(0.0f, floor(minY)); y < glm::min(height - 1.0f, ceil(maxY)); y++) {
				
				//LOOP THROUGH 4 TIMES, jitter fragment placement, buf must be 4 times as big.  When in fragment shader average 4 depths together to get new color
				for (int m = 0; m < 4; m++) {
					float rand = .25f;
					//printf("heyo4 \n");
					float newX;
					float newY;
					if (m == 0) {
						newX = x + rand;
						newY = y + rand;
					}
					else if (m == 1) {
						newX = x + rand + .5f;
						newY = y + rand;
					}
					else if (m == 2) {
						newX = x + rand;
						newY = y + rand + .5f;
					}
					else {
						newX = x + rand + .5f;
						newY = y + rand + .5f;
					}
					float tempX = ((newX / (float)width) * 2.0f) - 1.0f;
					float tempY = ((newY / (float)height) * 2.0f) - 1.0f;
					//float minZ = (bbox.min.z + 1) * (width / 2.0f);
					//printf("z: %f, %f \n", minZ, bbox.min.z);
				
					//printf("depth int: %i \n", myDepth);
					if (tempX >= -1.0f && tempX <= 1.0f && tempY >= -1.0f && tempY <= 1.0f) {
						glm::vec3 baryCoord = calculateBarycentricCoordinate(triangle, glm::vec2(tempX, tempY));
						int myDepth = getZAtCoordinate(baryCoord, triangle) * INT_MAX;
						//printf("heyo5 \n");
						//__syncthreads();
						
						if(isBarycentricCoordInBounds(baryCoord)) {
							//printf("(%i, %i) \n", x, y);
							atomicMin(&buf[4*x + m + y*4*width].idepth, myDepth);
							if(myDepth == buf[4*x + m + y*4*width].idepth) {
								//printf("heyo6 \n");
								
						
								glm::vec3 normal = baryCoord[0] * tris[thrId].v[0].nor + baryCoord[1] * tris[thrId].v[1].nor + baryCoord[2] * tris[thrId].v[2].nor;
								normal = glm::normalize(normal);
								//normal = glm::abs(normal);
								buf[4*x + m + y*4*width].color = normal;
								buf[4*x + m + y*4*width].nor = normal;
								buf[4*x + m + y*4*width].depth = bbox.min.z;
					
							}
						}
					}
				}
			}
		}
	}
}

__global__ void fragmentShader(Fragment* depth, Fragment* frag, int width, int height) {
	glm::vec3 light(0.0f, 3.0f, 6.0f);
	glm::vec3 lightColor(1.0f, 1.0f, 1.0f);

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * width);
	int depthIndex = 4*x + y*4*width;
	if (index < width * height) {
		int minDepth = glm::min(depth[depthIndex + 3].idepth, glm::min(depth[depthIndex + 2].idepth, glm::min(depth[depthIndex].idepth, depth[depthIndex + 1].idepth)));
		//printf("%i minDepth: %i < %i \n", index, minDepth, MAX_DEPTH);
		if (minDepth < MAX_DEPTH && minDepth != 0) {
			glm::vec3 finalColor = (depth[depthIndex + 3].color + depth[depthIndex + 2].color + depth[depthIndex + 1].color + depth[depthIndex].color) / 4.0f;
			//printf("%i final color (%f, %f, %f) %i \n", index, finalColor.x, finalColor.y, finalColor.z, minDepth);
			float ambientTerm = 0.2f;
			float diffuseTerm = glm::dot(glm::normalize(depth[depthIndex].nor), glm::normalize(-light));
			if (diffuseTerm < 0.0f) diffuseTerm = 0.0f;
			if (diffuseTerm > 1.0f) diffuseTerm = 1.0f;
			//printf("diff: %f norm: (%f, %f, %f) \n", diffuseTerm, depth[depthIndex].nor[0], depth[depthIndex].nor[1], depth[depthIndex].nor[2]);
			//frag[index].color = diffuseTerm * finalColor;//depth[depthIndex].nor; //glm::dot(light, depth[depthIndex].nor)*depth[depthIndex].color;
			//printf("(%f, %f, %f) \n", frag[index].color[0], frag[index].color[1], frag[index].color[2]);
			
			float lightIntensity = diffuseTerm + ambientTerm;
			glm::vec3 Ia = finalColor;
			glm::vec3 Ii = lightColor*finalColor;
			glm::vec3 R = glm::reflect(glm::normalize(-light), glm::normalize(depth[depthIndex].nor));
			R = glm::normalize(R);
			glm::vec3 V = normalize(glm::vec3(0.0f, 3.0f, 3.0f));
			float spec = glm::dot(R, V);
			if (spec < 0.0f) spec = 0.0f;
			if (spec < 1.0f) spec = 1.0f;
	
			frag[index].color = finalColor; //ambientTerm*Ia + Ii*diffuseTerm;


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
	dim3 blockSize1d(64);
	dim3 blockCount1d((vertCount + 64 - 1) / 64);
	dim3 blockCountAnti((4*width - 1) / blockSize2d.x + 1, (height - 1) / blockSize2d.y + 1);
	//cam.x += camCoords.x;
	//cam.y += camCoords.y;
	//camCoords.x = 0.0;
	//camCoords.y = 0.0;
	glm::vec3 camera(0.0f, 3.0f, 3.0f);
	glm::mat4 model = utilityCore::buildTransformationMatrix(glm::vec3(0.0f), glm::vec3(-camCoords.y, -camCoords.x, 0.0f), glm::vec3(camCoords.z));
	glm::mat4 view = glm::lookAt(glm::vec3(0.0, 3, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
	glm::mat4 projection = glm::perspective<float>(50.0, (float)width / (float)height, 0.5f, 1000.0f);
	//glm::mat4 model = glm::mat4();
	glm::mat4 matrix = projection * view * model;

    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	//Set buffer to default value
	cudaMemset(dev_depthbuffer, 0, 4 * width * height * sizeof(Fragment));
	setDepth<<<blockCountAnti, blockSize2d>>>(dev_depthbuffer, width);
	setDepth<<<blockCount2d, blockSize2d>>>(dev_fragbuffer, width);
	//Transfer from VertexIn to VertexOut (vertex shading)
	vertexShader<<<blockCount1d, blockSize1d>>>(dev_bufVertex, dev_bufTransformedVertex, vertCount, matrix, model);
	checkCUDAError("rasterize");
	int k;
	//std::cin >> k;
	//Transfer from VertexOut to Triangles (primitive assembly)
	primitiveAssemble<<<blockCount1d, blockSize1d>>>(dev_bufTransformedVertex, dev_primitives, vertCount);
	checkCUDAError("rasterize");
	

	//Scanline each triangle to get fragment color (rasterize)
	int triCount = vertCount / 3;
	blockCount1d = ((triCount + 64 - 1) / 64);
	//printf("old triangle count: %i \n", triCount);
	//checkCUDAError("rasterize");
	//THRUST REMOVE IF
	
	backWard<<<blockCount1d, blockSize1d>>>(dev_primitives, triCount);
	Triangle* triEnd = thrust::remove_if(thrust::device, dev_primitives, dev_primitives + triCount, removeTriangles());
	//printf("triEnd: %i , %i, %i\n", sizeof(triEnd), sizeof(Triangle), sizeof(triEnd)/sizeof(Triangle));
	triCount = triEnd - dev_primitives;
	blockCount1d = ((triCount + 64 - 1) / 64);
	//printf("new triangle count: %i \n", triCount);
	//printf("tri count: %i, block count: %i \n", triCount, (triCount + 64 - 1) / 64);
	kernRasterize<<<blockCount1d, blockSize1d>>>(dev_primitives, dev_depthbuffer, width, height, triCount);

	lineRasterize<<<blockCount1d, blockSize1d>>>(dev_primitives, dev_depthbuffer, width, height, triCount, matrix);
	checkCUDAError("rasterize");
    
    //Fragment shader
	fragmentShader<<<blockCount2d, blockSize2d>>>(dev_depthbuffer, dev_fragbuffer, width, height);
	checkCUDAError("Fragment Shader");

    // Copy depthbuffer colors into framebuffer
    render<<<blockCount2d, blockSize2d>>>(width, height, dev_fragbuffer, dev_framebuffer);
    
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
