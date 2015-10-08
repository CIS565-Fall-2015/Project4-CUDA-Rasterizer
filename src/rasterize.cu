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
#include <glm/gtc/matrix_transform.hpp>

extern glm::vec3 *imageColor;

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
};
struct Triangle {
    VertexOut vOut[3];
    VertexIn vIn[3];
    bool keep;
    glm::vec3 triNor;	//Used for back face culling
};
struct Fragment {
    glm::vec3 color;
    float depth;
};
struct Light {
	glm::vec3 pos;
	glm::vec3 col;
};

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
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;
static glm::mat4 matrix;
static glm::vec3 camDir;
static Light light;

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

__global__
void kernVertexShader(int numVertices, int w, int h, VertexIn * inVertex, VertexOut *outVertex, glm::mat4 matrix)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numVertices)
	{
		glm::vec4 outPoint = glm::vec4(inVertex[index].pos.x, inVertex[index].pos.y, inVertex[index].pos.z, 1.0f);

		outPoint = matrix * outPoint;

		if(outPoint.w != 0)
			outPoint /= outPoint.w;

		//In NDC
//		outVertex[index].pos = glm::vec3(outPoint);

		//In Device Coordinates
		outVertex[index].pos.x = outPoint.x * w;
		outVertex[index].pos.y = outPoint.y * h;
		outVertex[index].pos.z = outPoint.z;

		outVertex[index].col = glm::vec3(0,0,1);
		outVertex[index].nor = inVertex[index].nor;

//		printf ("InVertex : %f %f \nOutVertex : %f %f \n\n", inVertex[index].pos.x, inVertex[index].pos.y, outVertex[index].pos.x, outVertex[index].pos.y);
	}
}

__global__
void kernPrimitiveAssembly(int numTriangles, VertexOut *outVertex, VertexIn *inVertex, Triangle *triangles, int* indices, glm::vec3 camDir)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numTriangles)
	{
		int k_3 = 3 * index;

		Triangle &t = triangles[index];
		glm::vec3 triNor = glm::normalize(inVertex[k_3].nor + inVertex[k_3+1].nor + inVertex[k_3+2].nor);

//		printf ("Tri Normal : %f %f %f\n", triNor.x, triNor.y, triNor.z);
//		printf ("Cam Dir : %f %f %f\n", camDir.x, camDir.y, camDir.z);

		if(glm::dot(triNor, camDir) > -0.0001f)
		{
			t.keep = false;
		}

		else
		{
			t.keep = true;

			t.vOut[0].pos = outVertex[indices[k_3]].pos;
			t.vOut[1].pos = outVertex[indices[k_3+1]].pos;
			t.vOut[2].pos = outVertex[indices[k_3+2]].pos;

			//TODO:  figure out the normals
			t.vOut[0].nor = outVertex[indices[k_3]].nor;
			t.vOut[1].nor = outVertex[indices[k_3+1]].nor;
			t.vOut[2].nor = outVertex[indices[k_3+2]].nor;

			t.triNor = triNor;
		}
	}
}

__global__
void kernDrawAxis(int w, int h, Fragment *fragments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h)
    {
		if((x - w*0.5f) == 0)
	    {
			fragments[index].color = glm::vec3(0, 1, 0);
	    }
		else if((y - h*0.5f) == 0)
		{
			fragments[index].color = glm::vec3(1, 0, 0);
		}
    }
}

__global__
void kernRasterizePerFragment()
{
	//Rasterization per Fragment
//	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
//	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
//    int index = x + (y * w);
//
//    if (x < w && y < h)
//    {
//    	if((x - w*0.5f) == 0 || (y - h*0.5f) == 0)
//    	{
//    		fragments[index].color = glm::vec3(1);
//    	}
//    	else
//    	{
//			glm::vec2 point((x - w*0.5f), (y - h*0.5f));
//			for(int i=0; i<numTriangles; ++i)
//			{
//				glm::vec3 tri[3];
//				tri[0] = triangles[i].v[0].pos;
//				tri[1] = triangles[i].v[1].pos;
//				tri[2] = triangles[i].v[2].pos;
//
//	//    		AABB aabb = getAABBForTriangle(tri);
//				float signedArea = calculateSignedArea(tri);
//				glm::vec3 barycentric = calculateBarycentricCoordinate(tri, point);
//				if(isBarycentricCoordInBounds(barycentric))
//				{
//					fragments[index].color = glm::vec3(1);
//					fragments[index].depth = getZAtCoordinate(barycentric, tri);
//				}
//			}
//    	}
//    }
}

__global__
void kernRasterize(int w, int h, Fragment *fragments, Triangle *triangles, int numTriangles, glm::vec3 camDir, Light light)
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
	//				printf("\nMax : %f %f %f\nMin : %f %f %f\n", aabb.max.x, aabb.max.y, aabb.max.z, aabb.min.x, aabb.min.y, aabb.min.z);
				glm::ivec2 point(i,j);
				glm::vec3 barycentric = calculateBarycentricCoordinate(tri, point);

				if(isBarycentricCoordInBounds(barycentric))
				{
					//Then fragment in the triangle.
					//Implement lambert shading

					//Interpolate normal
					glm::vec3 norm = barycentric.x * t.vOut[0].nor +
										barycentric.y * t.vOut[1].nor +
						                barycentric.z * t.vOut[2].nor;

//					glm::vec3 lightVector =
					fragments[int((i+w*0.5) + (j + h*0.5)*w)].color = glm::normalize(norm);//triangles[index].v[0].col;
//						fragments[int((i+w*0.5) + (j + h*0.5)*w)].color = t.triNor;
//						fragments[(x+1) * y].depth = getZAtCoordinate(barycentric, tri);
				}
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

    imageColor = new glm::vec3[width*height];

    checkCUDAError("rasterizeSetBuffers");
}

/**
 * Perform rasterization.
 */
bool run = true;

void createCameraAndLight()
{
	//Camera stuff
	glm::vec3 camEye, camCenter, camUp;
	camEye = glm::vec3(0,0,5);
	camCenter = glm::vec3(0,0,0);
	camUp = glm::vec3(0,-1,0);

	glm::mat4 view = glm::lookAt(camEye, camCenter, camUp);
//	glm::mat4 projection = glm::frustum<float>(-1, 1, -1, 1, -1, 1);
	glm::mat4 projection = glm::perspective<float>(45.0f, float(width)/ float(height), -100.0f, 100.0f);
	glm::mat4 model = glm::mat4();
	glm::mat4 temp;

//	std::cout<<"View : "<<std::endl;
//	utilityCore::printMat4(view);
//	std::cout<<std::endl<<"Projection : "<<std::endl;
//	utilityCore::printMat4(projection);
//	std::cout<<std::endl<<"Model : "<<std::endl;
//	utilityCore::printMat4(model);
//	std::cout<<std::endl;

	matrix = projection * view * model;
	camDir = glm::normalize(camCenter - camEye);
	light.pos = glm::vec3(1,1,1);
	light.col = glm::vec3(1,1,1);
}

void rasterize(uchar4 *pbo) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);


    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

    int numThreads = 256;
    int numBlocks;
    int numTriangles = vertCount/3;

    Triangle *dev_primitivesEnd;

    if(run)
    {
    	createCameraAndLight();

    	numBlocks = (vertCount + numThreads -1)/numThreads;
    	kernVertexShader<<<numBlocks, numThreads>>>(vertCount, width, height, dev_bufVertex, dev_outVertex, matrix);

    	numBlocks = (numTriangles + numThreads -1)/numThreads;
    	kernPrimitiveAssembly<<<numBlocks, numThreads>>>(numTriangles, dev_outVertex, dev_bufVertex, dev_primitives, dev_bufIdx, camDir);

    	std::cout<<"Num Triangles before : "<<numTriangles<<std::endl;
    	dev_primitivesEnd = dev_primitives + numTriangles;
    	dev_primitivesEnd = thrust::remove_if(thrust::device, dev_primitives, dev_primitivesEnd, keep());
    	numTriangles = dev_primitivesEnd - dev_primitives;
    	std::cout<<"Num Triangles after : "<<numTriangles<<std::endl;

    	kernDrawAxis<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer);

    	numBlocks = (numTriangles + numThreads -1)/numThreads;
    	kernRasterize<<<numBlocks, numThreads>>>(width, height, dev_depthbuffer, dev_primitives, numTriangles, camDir, light);

//    	numBlocks = (width*height + numThreads -1)/numThreads;
//    	kernRasterize<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer, dev_primitives, numTriangles);

    	run = false;
    }

    // Copy depthbuffer colors into framebuffer
    render<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer, dev_framebuffer);
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);

    cudaMemcpy(imageColor, dev_framebuffer, width*height*sizeof(glm::vec3), cudaMemcpyDeviceToHost);
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

    checkCUDAError("rasterizeFree");
}
