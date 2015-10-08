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
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};
struct VertexOut {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;

	glm::vec3 ndc;
	glm::vec3 winPos;
    // TODO
};
struct Triangle {
    VertexOut v[3];
};
struct Fragment {
	int depth;
    glm::vec3 color;
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static VertexIn *dev_bufVertex = NULL;
VertexOut *dev_bufVtxOut = NULL;
static Triangle *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;
glm::mat4 M_win;
glm::mat4 M_view;

__global__
void kernBufInit(int w, int h, Fragment * depthbuffer, glm::vec3 *framebuffer)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) 
	{
		depthbuffer[index].depth = 1000000; //INFINITY;//!!!
		depthbuffer[index].color = glm::vec3(0.2, 0, 0);
	}
}

__global__			//(vertCount,     glm::mat4() ,       M_view,          projMat,            dev_bufVertex, dev_bufVtxOut, M_win);
void kernVertexShader(int vtxCount,glm::mat4 M_model, glm::mat4 M_view, glm::mat4 M_Projection, VertexIn *vtxI, VertexOut *vtxO, glm::mat4 M_win)
{
	//demo:http://www.realtimerendering.com/udacity/transforms.html
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < vtxCount)
	{
		glm::vec4 P_clip = M_Projection * M_view * M_model * glm::vec4(vtxI[index].pos, 1);	//clip coords
		glm::vec4 P_NDC = P_clip*(1 / P_clip.w);//!!!w-divide for NDC	: P_clip/w
		//!!!window coords		: M_win*P_NDC
		

		vtxO[index].ndc = glm::vec3(P_NDC);
		//vtxO[index].ndc = vtxI[index].pos;
		vtxO[index].nor = vtxI[index].nor;
		vtxO[index].col = vtxI[index].col;
		P_NDC = M_win*P_NDC;
		vtxO[index].winPos = glm::vec3(P_NDC);
	}
}

__global__
void kernPrimitiveAssembly(Triangle* primitives,int* bufIdx,int bufIdxSize, VertexOut * bufVtxOut)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index<bufIdxSize/3)
	{
		int i = 3 * index;
		primitives[index].v[0] = bufVtxOut[bufIdx[i]];
		primitives[index].v[1] = bufVtxOut[bufIdx[i+1]];
		primitives[index].v[2] = bufVtxOut[bufIdx[i+2]];
	}
}

__global__
void kernRasterizer(int w,int h,Fragment * depthbuffer, Triangle*primitives, int bufIdxSize )
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < bufIdxSize / 3)
	{
		//Scanline
		glm::vec3 tri[3];
		tri[0] = primitives[index].v[0].winPos;
		tri[1] = primitives[index].v[1].winPos;
		tri[2] = primitives[index].v[2].winPos;

		//!!! later interpolation
		glm::vec3 normal = glm::normalize(primitives[index].v[0].nor + primitives[index].v[1].nor + primitives[index].v[2].nor);
		/*
		//http://keisan.casio.com/exec/system/1223596129
		glm::vec3 A = primitives[index].v[0].ndc;
		glm::vec3 B = primitives[index].v[1].ndc;
		glm::vec3 C = primitives[index].v[2].ndc;

		float a = (B.y - A.y)*(C.z - A.z) - (C.y - A.y)*(B.z - A.z);
		float b = (B.z - A.z)*(C.x - A.x) - (C.z - A.z)*(B.x - A.x);
		float c = (B.x - A.x)*(C.y - A.y) - (C.x - A.x)*(B.y - A.y);
		float d = -(a*A.x + b*A.y + c*A.x);
		//ax+by+cz+d = 0;
		*/
		AABB triBox = getAABBForTriangle(tri);
		for (int x = triBox.min.x; x <= triBox.max.x; x++)
		{
			for (int y = triBox.min.y; y <= triBox.max.y; y++)
			{
				glm::vec3 bPoint = calculateBarycentricCoordinate(tri, glm::vec2(x, y));
				//!!! later line segment
				if (isBarycentricCoordInBounds(bPoint)) // Inside triangle
				//if (true)
				{
					//glm::vec4 crntNDC = glm::inverse(M_win)*glm::vec4(x, y, 1,1);
					//crntNDC.z = (a*crntNDC.x + b*crntNDC.y + d) / (-c);
					//crntNDC = M_win*crntNDC;
					//int crntDepth = (int)(tri[0].z * 1000);
					//int crntDepth = (int)(crntNDC.z * 1000);
					//!!! later clipping
					if (x<0 || x>w || y<0 || y>h)
						continue;
					int crntDepth = (int)(getZAtCoordinate(bPoint, tri)*1000000.f);
					int orig = atomicMin(&(depthbuffer[x+y*w].depth), crntDepth);
					//if (orig >= crntDepth)
					if (depthbuffer[x + y*w].depth==crntDepth)
					{
						depthbuffer[x + y*w].color = normal;// (normal + glm::vec3(1, 1, 1))*0.5f; //glm::vec3(0, 1, 0);
					}
					//else depthbuffer[x + y*w].color = glm::vec3(0, 0, 0);
					//printf("point(%d,%d):orig=%d,new=%d\n", x, y, orig, depthbuffer[x, y].depth);
				}
			}
		}

	}
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

	int hWidth = width / 2;
	int hHeight = height / 2;
	M_win = glm::mat4(\
		hWidth, 0, 0, 0, \
		0, hHeight, 0, 0, \
		0, 0, 0.5, 0,
		hWidth, hHeight, 0.5, 1
		);

    cudaFree(dev_depthbuffer);
    cudaMalloc(&dev_depthbuffer,   width * height * sizeof(Fragment));
    cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
	//cudaMemset(dev_depthbuffer, INFINITY, width * height * sizeof(Fragment));

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

	//!!!
	cudaFree(dev_bufVtxOut);
	cudaMalloc(&dev_bufVtxOut, vertCount * sizeof(VertexOut));

    cudaFree(dev_primitives);
    cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
    cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

    checkCUDAError("rasterizeSetBuffers");
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo,glm::mat4 viewMat,glm::mat4 projMat) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

	M_view = viewMat;//glm::lookAt(eye, center, up);
    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	int bSize_vtx = 128;
	int bSize_pri = 128;
	dim3 gSize_vtx((vertCount + bSize_vtx - 1) / bSize_vtx);
	dim3 gSize_pri((bufIdxSize/3 + bSize_pri - 1) / bSize_pri);
	//****** 1. Clear depth buffer
	kernBufInit << <blockCount2d, blockSize2d >> >(width, height, dev_depthbuffer, dev_framebuffer);
	//****** 2. Vertex Shading
	//	VertexIn[n] vs_input -> VertexOut[n] vs_output

	kernVertexShader << <gSize_vtx, bSize_vtx >> >(vertCount, glm::mat4(), M_view, projMat, dev_bufVertex, dev_bufVtxOut, M_win);

	VertexOut * textVtxOut = new VertexOut[vertCount];
	cudaMemcpy(textVtxOut, dev_bufVtxOut, vertCount*sizeof(VertexOut), cudaMemcpyDeviceToHost);
	/*for (int i = 0; i < vertCount; i++)
	{
		glm::vec3 temp = textVtxOut[i].ndc;
		printf("tri[%d] after VtxShader:%2f,%2f,%2f\n",i,temp.x,temp.y,temp.z);
	}*/

	//****** 3. Primitive Assembly
	//  VertexOut[n] vs_output -> Triangle[n/3] primitives
	kernPrimitiveAssembly<<<gSize_pri, bSize_pri >>>(dev_primitives, dev_bufIdx, bufIdxSize, dev_bufVtxOut);

	//****** 4. Rasterization
	//  Triangle[n/3] primitives -> FragmentIn[m] fs_input
	kernRasterizer <<<gSize_pri, bSize_pri >>>(width,height,dev_depthbuffer, dev_primitives, bufIdxSize);
	//****** 5. Fragment shading
	//****** 6. Fragments to depth buffer
	//****** 7. Depth buffer for storing & testing fragments
	//****** 8. Fragment to framebuffer writing

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
