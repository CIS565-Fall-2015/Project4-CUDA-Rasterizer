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
#include <glm/gtc/matrix_transform.hpp>


struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};
struct VertexOut {
	glm::vec3 pos;
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
static Triangle *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;
static glm::mat4 matrix;

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

//		printf("OutPoint : %f %f %f %f\n", outPoint.x, outPoint.y, outPoint.z, outPoint.w);

		if(outPoint.w != 0)
			outVertex[index].pos = glm::vec3(outPoint / outPoint.w);
//		printf ("InVertex : %f %f \nOutVertex : %f %f \n\n", inVertex[index].pos.x, inVertex[index].pos.y, outVertex[index].pos.x, outVertex[index].pos.y);
	}
}

__global__
void kernPrimitiveAssembly(int numTriangles, VertexOut *outVertex, Triangle *triangles, int* indices)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index < numTriangles)
	{
		int k_3 = 3 * index;
		triangles[index].v[0] = outVertex[indices[k_3]];
		triangles[index].v[1] = outVertex[indices[k_3+1]];
		triangles[index].v[2] = outVertex[indices[k_3+2]];

//		printf ("Triangle : %d\n", index);
//		printf ("Vertex 1 : %f %f\n", triangles[index].v[0].pos.x, triangles[index].v[0].pos.y);
//		printf ("Vertex 2 : %f %f\n", triangles[index].v[1].pos.x, triangles[index].v[1].pos.y);
//		printf ("Vertex 3 : %f %f\n", triangles[index].v[2].pos.x, triangles[index].v[2].pos.y);
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

    checkCUDAError("rasterizeSetBuffers");
}

/**
 * Perform rasterization.
 */
bool run = true;

void createCamera()
{
	//Camera stuff
	glm::vec3 camEye, camCenter, camUp;
	camEye = glm::vec3(0,0,-2);
	camCenter = glm::vec3(0,0,0);
	camUp = glm::vec3(0,1,0);

	glm::mat4 view = glm::lookAt(camEye, camCenter, camUp);
//	glm::mat4 projection = glm::frustum<float>(-1, 1, -1, 1, -1, 1);
	glm::mat4 projection = glm::perspective<float>(45.0f, float(width)/ float(height), 0.1f, 100.0f);
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
}

void rasterize(uchar4 *pbo) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);


    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

    int numTriangles = vertCount/3;

    if(run)
    {
    	createCamera();
//    	run = false;
    }

    //Todo change the number of threads based on input size.
    if(run)
    {
    	kernVertexShader<<<1, vertCount>>>(vertCount, width, height, dev_bufVertex, dev_outVertex, matrix);
//    	run = false;
    }

    if(run)
    {
    	kernPrimitiveAssembly<<<1, numTriangles>>>(numTriangles, dev_outVertex, dev_primitives, dev_bufIdx);
    	run = false;
    }

    if(run)
    {
    	//kernRasterize
    }

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

    cudaFree(dev_outVertex);
    dev_outVertex = NULL;
//
//    cudaFree(dev_view);
//    dev_view = NULL;
//
//    cudaFree(dev_model);
//    dev_model = NULL;
//
//    cudaFree(dev_projection);
//    dev_projection = NULL;

    checkCUDAError("rasterizeFree");
}
