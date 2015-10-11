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
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <util/checkCUDAError.h>
#include "rasterizeTools.h"

//#include "sceneStructs.h"
#include "Scene.h"

extern Scene *scene;

struct keep
{
	__host__ __device__ bool operator()(const Triangle t)
	{
		return (!t.keep);
	}
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static VertexIn *dev_bufVertex = NULL;
static Triangle *dev_primitives = NULL;
static Edge *dev_edges = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;
static glm::mat4 matrix;
static glm::vec3 camDir;
static Light light;
static Camera cam;

//Things added
static VertexOut *dev_outVertex = NULL;

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

//Kernel function to figure out vertex in transformes space and NDC
__global__
void kernVertexShader(int numVertices, int w, int h, VertexIn * inVertex, VertexOut *outVertex, Camera cam)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numVertices)
	{
		glm::vec4 outPoint = glm::vec4(inVertex[index].pos.x, inVertex[index].pos.y, inVertex[index].pos.z, 1.0f);

		outVertex[index].transformedPos = multiplyMV(cam.model, outPoint);
		outPoint = cam.cameraMatrix * outPoint;

		if(outPoint.w != 0)
			outPoint /= outPoint.w;

		//In NDC
//		outVertex[index].pos = glm::vec3(outPoint);

		//In Device Coordinates
		outVertex[index].pos.x = outPoint.x * w;
		outVertex[index].pos.y = outPoint.y * h;
		outVertex[index].pos.z = outPoint.z;

		outVertex[index].nor = multiplyMV(cam.inverseTransposeModel, glm::vec4(inVertex[index].nor, 0.0f));

		//		outVertex[index].col = glm::vec3(0,0,1);
//		outVertex[index].nor = inVertex[index].nor;

//		printf ("InVertex : %f %f \nOutVertex : %f %f \n\n", inVertex[index].pos.x, inVertex[index].pos.y, outVertex[index].pos.x, outVertex[index].pos.y);
	}
}

//Kernel function to assemble triangles
__global__
void kernPrimitiveAssembly(int numTriangles, VertexOut *outVertex, VertexIn *inVertex, Triangle *triangles, int* indices, glm::vec3 camDir)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numTriangles)
	{
		int k_3 = 3 * index;

		Triangle &t = triangles[index];

		//Find the triangle normal
		glm::vec3 triNor = (outVertex[k_3].nor + outVertex[k_3+1].nor + outVertex[k_3+2].nor);

//		printf ("Tri Normal : %f %f %f\n", triNor.x, triNor.y, triNor.z);
//		printf ("Cam Dir : %f %f %f\n", camDir.x, camDir.y, camDir.z);

		if(glm::dot(triNor, camDir) > 0.0f)
		{
			//Triangle facing away from the camera
			//	Mark for deletion
			t.keep = false;
		}

		else
		{
			//Else save it
			t.keep = true;

			t.vOut[0] = outVertex[indices[k_3]];
			t.vOut[1] = outVertex[indices[k_3+1]];
			t.vOut[2] = outVertex[indices[k_3+2]];

			t.vIn[0] = inVertex[indices[k_3]];
			t.vIn[1] = inVertex[indices[k_3+1]];
			t.vIn[2] = inVertex[indices[k_3+2]];
		}
	}
}

//Kernel function to assemble edges
__global__
void kernEdgeAssembly(int numTriangles, VertexOut *outVertex, Edge *edge, int* indices)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numTriangles)
	{
		int k_3 = 3 * index;

		edge[indices[k_3]].v1 = outVertex[k_3].pos;
		edge[indices[k_3]].v2 = outVertex[k_3+1].pos;

		edge[indices[k_3+1]].v1 = outVertex[k_3+1].pos;
		edge[indices[k_3+1]].v2 = outVertex[k_3+2].pos;

		edge[indices[k_3+2]].v1 = outVertex[k_3+2].pos;
		edge[indices[k_3+2]].v2 = outVertex[k_3].pos;
	}
}

//Kernel function to draw axis
__global__
void kernDrawAxis(int w, int h, Fragment *fragments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
    {
		int index = x + (y * w);
		if((x - w*0.5f) == 0)
	    {
			fragments[index].color = glm::vec3(0, 1, 0);
	    }
	    else if((y - h*0.5f) == 0)
		{
			fragments[index].color = glm::vec3(1, 0, 0);
		}
	    else if(x == 0 || x == w-1)
		{
			fragments[index].color = glm::vec3(1);
		}
		else if(y == 0 || y == h)
		{
			fragments[index].color = glm::vec3(1);
		}
    }
}

//Kernel function to clear the depth and color buffer
__global__
void kernClearFragmentBuffer(int w, int h, Fragment *fragments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
    {
		int index = x + (y * w);

		fragments[index].color = glm::vec3(0, 0, 0);
		fragments[index].depth = INT_MAX;
    }
}

//Kernel function to rasterize the triangle
__global__
void kernRasterizeTraingles(int w, int h, Fragment *fragments, Triangle *triangles, int numTriangles, Camera cam, Light light1, Light light2)
{
	//Rasterization per triangle
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numTriangles)
	{
		Triangle &t = triangles[index];

		glm::vec3 tri[3];
		tri[0] = t.vOut[0].pos;
		tri[1] = t.vOut[1].pos;
		tri[2] = t.vOut[2].pos;

		AABB aabb = getAABBForTriangle(tri);
		glm::ivec3 min, max;

		//Attempted clipping
		min.x = glm::clamp(aabb.min.x, -(float)w*0.5f+1, (float)w*0.5f-1);
		min.y = glm::clamp(aabb.min.y, -(float)h*0.5f+1, (float)h*0.5f-1);
		max.x = glm::clamp(aabb.max.x, -(float)w*0.5f+1, (float)w*0.5f-1);
		max.y = glm::clamp(aabb.max.y, -(float)h*0.5f+1, (float)h*0.5f-1);

		for(int i=min.x-1; i<=max.x+1; ++i)
		{
			for(int j=min.y-1; j<=max.y+1; ++j)
			{
				glm::ivec2 point(i,j);
				glm::vec3 barycentric = calculateBarycentricCoordinate(tri, point);

				if(isBarycentricCoordInBounds(barycentric))
				{
					glm::vec3 triIn[3];
					VertexIn tvIn[3] = {t.vIn[0], t.vIn[1], t.vIn[2]};

					triIn[0] = t.vOut[0].transformedPos;
					triIn[1] = t.vOut[1].transformedPos;
					triIn[2] = t.vOut[2].transformedPos;

					glm::vec3 norm = barycentric.x * t.vOut[0].nor +
										barycentric.y * t.vOut[1].nor +
						                barycentric.z * t.vOut[2].nor;

					glm::vec3 pos = barycentric.x * t.vOut[0].transformedPos +
										barycentric.y * t.vOut[1].transformedPos +
										barycentric.z * t.vOut[2].transformedPos;

					glm::vec3 col = barycentric.x * tvIn[0].col +
										barycentric.y * tvIn[1].col +
										barycentric.z * tvIn[2].col;

					glm::vec3 lightVector1 = glm::normalize(light1.pos - pos);
					glm::vec3 lightVector2 = glm::normalize(light2.pos - pos);
					//glm::vec3 camVector = glm::normalize(cam.pos - pos);

					float diffusedTerm1 = glm::dot(lightVector1, norm);
					float diffusedTerm2 = glm::dot(lightVector2, norm);

//					if(diffusedTerm1 > 0.0f || diffusedTerm2 > 0.0f)
					{
						int fragIndex = int((i+w*0.5) + (j + h*0.5)*w);
						int depth = getZAtCoordinate(barycentric, triIn) * 10000;

						//TODO : Use cuda atomics to avoid race condition here
						if(depth < fragments[fragIndex].depth)
						{
							atomicMin(&fragments[fragIndex].depth, depth);
							if(diffusedTerm1 > 0.0f && diffusedTerm2 > 0.0f)
							{
								fragments[fragIndex].color = diffusedTerm1 * col * light1.col +
															 diffusedTerm2 * col * light2.col;
							}

							else if(diffusedTerm1 > 0.0f)
							{
								fragments[fragIndex].color = diffusedTerm1 * col * light1.col;
							}
							else if(diffusedTerm2 > 0.0f)
							{
								fragments[fragIndex].color = diffusedTerm2 * col * light2.col;
							}
							else
							{
								fragments[fragIndex].color = glm::vec3(0.0f);
							}
						}
					}
				}
			}
		}
	}
}

//Kernel function to rasterize points
__global__
void kernRasterizePoints(int numVertices, int w, int h, Fragment *fragments, VertexOut * vertices, Camera cam, Light light1, Light light2)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numVertices)
	{
		glm::ivec2 point(vertices[index].pos.x, vertices[index].pos.y);

		if(point.x > -w*0.5
				&& point.x < w*0.5f
				&& point.y > -h*0.5f
				&& point.y < h*0.5f )
		{

//			int fragIndex = ;
			fragments[int((point.x + w*0.5f) + (point.y + h*0.5f)*w)].color = glm::vec3(1.0f);
		}
	}
}

//Kernel function to rasterize lines
__global__
void kernRasterizeLines(int numVertices, int w, int h, Fragment *fragments, Edge *edge, Camera cam, Light light1, Light light2)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numVertices)
	{
		Edge &e = edge[index];

		glm::vec2 v1(e.v1.x, e.v1.y);
		glm::vec2 v2(e.v2.x, e.v2.y);

		v1.x = glm::clamp(v1.x, -(float)w*0.5f, (float)w*0.5f);
		v1.y = glm::clamp(v1.y, -(float)h*0.5f, (float)h*0.5f);
		v2.x = glm::clamp(v2.x, -(float)w*0.5f, (float)w*0.5f);
		v2.y = glm::clamp(v2.y, -(float)h*0.5f, (float)h*0.5f);

		float m = (v2.y - v1.y) / (v2.x - v1.x);

		int inc;

		if(m > 1)
		{
			if(v1.y > v2.y)
			{
				inc = -1;
			}
			else
			{
				inc = 1;
			}

			int i, j;

			for(j=v1.y; j!=(int)v2.y; j += inc)
			{
				i = ((float)j - v1.y) / m + v1.x;
				//			printf("i, j : %d %d\n", i, j);
				//			int fragIndex = int((i + w*0.5f) + (j + h*0.5f)*w);
				fragments[int((i + w*0.5f) + (j + h*0.5f)*w)].color = glm::vec3(1.0f);
			}
		}

		else
		{
			if(v1.x > v2.x)
			{
				inc = -1;
			}

			else
			{
				inc = 1;
			}

			int i, j;

			for(i=v1.x; i!=(int)v2.x; i += inc)
			{
				j = m * ((float)i - v1.x) + v1.y;
				//			printf("i, j : %d %d\n", i, j);
				//			int fragIndex = int((i + w*0.5f) + (j + h*0.5f)*w);
				fragments[int((i + w*0.5f) + (j + h*0.5f)*w)].color = glm::vec3(1.0f);
			}
		}
	}
}


/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;

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
    cudaMemcpy(dev_bufVertex, bufVertex, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);

    cudaFree(dev_primitives);
    cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
    cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

    cudaFree(dev_outVertex);
    cudaMalloc((void**)&dev_outVertex, vertCount * sizeof(VertexOut));

    cudaFree(dev_edges);
    cudaMalloc((void**)&dev_edges, vertCount * sizeof(Edge));

    checkCUDAError("rasterizeSetBuffers");
}

/**
 * Perform rasterization.
 */

void rasterize(uchar4 *pbo) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

    if(scene->run)
    {
    	int numThreads = 128;
		int numBlocks;
		int numTriangles = vertCount/3;
		scene->run = false;

		Camera &cam = scene->cam;
		Light &light1 = scene->light1;
		Light &light2 = scene->light2;

		//Clear the color and depth buffers
		kernClearFragmentBuffer<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer);

		//Drawing axis
		kernDrawAxis<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer);

		switch (scene->renderMode)
    	{
			case TRIANGLES:
			{
				Triangle *dev_primitivesEnd;

				//Do vertex shading
				numBlocks = (vertCount + numThreads -1)/numThreads;
				kernVertexShader<<<numBlocks, numThreads>>>(vertCount, width, height, dev_bufVertex, dev_outVertex, cam);

				//Do primitive assembly
				numBlocks = (numTriangles + numThreads -1)/numThreads;
				kernPrimitiveAssembly<<<numBlocks, numThreads>>>(numTriangles, dev_outVertex, dev_bufVertex, dev_primitives, dev_bufIdx, cam.dir);

				//Back face culling
				dev_primitivesEnd = dev_primitives + numTriangles;
				dev_primitivesEnd = thrust::remove_if(thrust::device, dev_primitives, dev_primitivesEnd, keep());
				numTriangles = dev_primitivesEnd - dev_primitives;
		////    	std::cout<<numTriangles;

				//Rasterization per triangle
				numBlocks = (numTriangles + numThreads -1)/numThreads;
				kernRasterizeTraingles<<<numBlocks, numThreads>>>(width, height, dev_depthbuffer, dev_primitives, numTriangles, cam, light1, light2);

				break;
			}

			case POINTS:
			{
//				std::cout<<"HERE";
				//Do vertex shading
				numBlocks = (vertCount + numThreads -1)/numThreads;
				kernVertexShader<<<numBlocks, numThreads>>>(vertCount, width, height, dev_bufVertex, dev_outVertex, cam);

				kernRasterizePoints<<<numBlocks, numThreads>>>(vertCount, width, height, dev_depthbuffer, dev_outVertex, cam, light1, light2);

				break;
			}

			case LINES:
			{
//				std::cout<<"HERE";
				//Do vertex shading
				numBlocks = (vertCount + numThreads -1)/numThreads;
				kernVertexShader<<<numBlocks, numThreads>>>(vertCount, width, height, dev_bufVertex, dev_outVertex, cam);

				//Do primitive assembly
				numBlocks = (numTriangles + numThreads -1)/numThreads;
				kernEdgeAssembly<<<numBlocks, numThreads>>>(numTriangles, dev_outVertex, dev_edges, dev_bufIdx);

				numBlocks = (vertCount + numThreads -1)/numThreads;
				kernRasterizeLines<<<numBlocks, numThreads>>>(vertCount, width, height, dev_depthbuffer, dev_edges, cam, light1, light2);

				break;
			}
    	}
    }

    // Copy depthbuffer colors into framebuffer
    render<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer, dev_framebuffer);
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);

    //Save image data to write to file
    cudaMemcpy(scene->imageColor, dev_framebuffer, width*height*sizeof(glm::vec3), cudaMemcpyDeviceToHost);

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

    cudaFree(dev_outVertex);
    dev_outVertex = NULL;

    cudaFree(dev_edges);
    dev_edges = NULL;

    checkCUDAError("rasterizeFree");
}
