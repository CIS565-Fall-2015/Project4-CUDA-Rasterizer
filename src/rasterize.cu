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
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include "rasterizeTools.h"

#include "glm/gtc/matrix_transform.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))


struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};

 
struct VertexOut {
    // TODO
	glm::vec4 pos;	//in NDS
	glm::vec3 color;

	glm::vec3 noraml_eye_space;
	
	float divide_w_clip;

	glm::vec2 uv;
};

struct Edge
{
	VertexOut v[2];

	float x, z;
	float dx, dz;


	//
	//VertexOut cur_v;	//used for interpolate between a scan line
	float gap_y;
};




//struct FragmentIn
//{
//	bool shade;
//
//	glm::vec3 color;
//	glm::vec3 normal_eye_space;
//	glm::vec2 uv;
//
//	float depth;
//
//	
//	//__host__ __device__ FragmentIn(){ shade = false; depth = FLT_MAX; }
//};


enum LightType
{
	POINT_LIGHT = 0,
	DIRECTION_LIGHT
};

struct Light
{
	LightType type;

	glm::vec3 ambient;
	glm::vec3 diffuse;
	glm::vec3 specular;

	//Point light
	glm::vec3 vec;

	bool enabled;
};


struct Triangle {
    VertexOut v[3];
};
struct Fragment {
	bool shade;

	glm::vec3 color;
	glm::vec3 normal_eye_space;
	glm::vec2 uv;

	float depth;

    //glm::vec3 color;
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

static int triCount = 0;

//static FragmentIn * dev_fragments = NULL;
static int * dev_fragmentLocks = NULL;




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



/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
    cudaFree(dev_depthbuffer);
    cudaMalloc(&dev_depthbuffer,   width * height * sizeof(Fragment));
    cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
    
	//cudaFree(dev_fragments);
	//cudaMalloc(&dev_fragments, width * height *sizeof(FragmentIn));
	//cudaMemset(dev_fragments, 0, width * height * sizeof(FragmentIn));
	
	cudaFree(dev_fragmentLocks);
	cudaMalloc(&dev_fragmentLocks, width * height *sizeof(int));
	cudaMemset(dev_fragmentLocks, 0, width * height * sizeof(int));
	
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

	//MY
	triCount = vertCount / 3;
	/////////

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






//-------------------------------------------------------------------------------
// Vertex Shader
//-------------------------------------------------------------------------------

/**
* each thread copy info for one vertex
*/
__global__ 
void kernVertexShader(int N,glm::mat4 M,glm::mat4 M_normal_view, VertexIn * dev_vertex, Triangle * dev_triangles)
{
	int vertexId = blockDim.x * blockIdx.x + threadIdx.x;
	

	if (vertexId < N)
	{
		int triangleId = vertexId / 3;
		int i = vertexId - triangleId * 3;
		VertexIn & vi = dev_vertex[vertexId];
		VertexOut & vo = dev_triangles[triangleId].v[i];

		vo.pos = M * glm::vec4(vi.pos, 1);
		vo.noraml_eye_space = glm::vec3(M_normal_view * glm::vec4(vi.nor, 0));
		
		vo.color = vi.col;

		//TODO: UV etc...
		
	}
}

/**
* MY:
* 
* VertexIn dev_bufVertex => Triangle VertexOut
* M model-view
*/
void vertexShader(const glm::mat4 & M, const glm::mat4 & inv_trans_M)
{
	const int blockSize = 192;
	dim3 blockCount( (vertCount + blockSize - 1 )/blockSize );

	// get M, M_normal_view

	kernVertexShader << <blockCount, blockSize >> >(vertCount, M, inv_trans_M, dev_bufVertex, dev_primitives);
}

//------------------------------------------------------------------------------



//MY
//atomic write FragmentIn

//__device__ 
//void atomicWriteFragmentIn(int * fragmentInLocks, FragmentIn * address, const FragmentIn & value)
//{
//	//atomicCAS
//
//}













//MY
//-------------------------------------------------------------------------------
// Rasterization
//-------------------------------------------------------------------------------

__host__ __device__
VertexOut interpolateVertexOut(const VertexOut & a, const VertexOut & b,float u)
{
	VertexOut c;

	if (u < 0.0f){ u = 0.0f; }
	else if (u > 1.0f){ u = 1.0f; }

	c.pos = (1 - u) * a.pos + u * b.pos;
	c.color = (1 - u) * a.color + u * b.color;
	c.uv = (1 - u) * a.uv + u * b.uv;

	c.divide_w_clip = (1 - u) * a.divide_w_clip + u * b.divide_w_clip;

	
	c.noraml_eye_space = glm::normalize( (1 - u) * a.noraml_eye_space + u * b.noraml_eye_space );

	return c;
}


//e.v[0] is the one with smaller y value
//scan from v[0] to v[1]
__device__ 
void constructEdge(Edge & e, const VertexOut & v0, const VertexOut & v1)
{
	if (v0.pos.y <= v1.pos.y)
	{
		e.v[0] = v0;
		e.v[1] = v1;
	}
	else
	{
		e.v[0] = v1;
		e.v[1] = v0;
	}

	//TODO: other members
	//e.cur_v = e.v[0];
	e.gap_y = 0.0f;

}


__device__
void initEdge(Edge & e, float y)
{
	e.gap_y = e.v[1].pos.y - e.v[0].pos.y;
	
	e.dx = (e.v[1].pos.x - e.v[0].pos.x) / (e.v[1].pos.y - e.v[0].pos.y);
	e.dz = (e.v[1].pos.z - e.v[0].pos.z) / (e.v[1].pos.y - e.v[0].pos.y);
	e.x = e.v[0].pos.x + (y - e.v[0].pos.y) * e.dx;
	e.z = e.v[0].pos.z + (y - e.v[0].pos.y) * e.dz;
}

__device__
void updateEdge(Edge & e)
{
	e.x += e.dx;
	e.z += e.dz;
}



__device__
void drawOneScanLine(int width, const Edge & e1, const Edge & e2, int y, Fragment * fragments, int * fragmentLocks)
{
	// Find the starting and ending x coordinates and
	// clamp them to be within the visible region
	int x_left = (int)(ceilf(e1.x) + EPSILON);
	int x_right = (int)(ceilf(e2.x) + EPSILON);

	if (x_left < 0)
	{
		x_left = 0;
	}
	
	if (x_right > width)
	{
		x_right = width;
	}

	// Discard scanline with no actual rasterization and also
	// ensure that the length is larger than zero
	if (x_left >= x_right) return;


	//TODO: get two interpolated segment end points
	VertexOut cur_v_e1 = interpolateVertexOut(e1.v[0], e1.v[1], (float)y / e1.gap_y);
	VertexOut cur_v_e2 = interpolateVertexOut(e2.v[0], e2.v[1], (float)y / e2.gap_y);


	//Initialize attributes
	float dz = (e2.z - e1.z) / (e2.x - e1.x);
	float z = e1.z + (x_left - e1.x) * dz;


	//Interpolate
	//printf("%d,%d\n", x_left, x_right);
	float gap_x = x_right - x_left;
	for (int x = x_left; x < x_right; ++x)
	{
		int idx = x + y * width;



		// Z-buffer comparision
		VertexOut p = interpolateVertexOut(cur_v_e1, cur_v_e2, (float)x / gap_x);
		



		//atomic 

		int assumed;
		int* address = &fragmentLocks[idx];
		int old = *address;

		//lock
		do{
			assumed = old;
			old = atomicCAS(address, assumed, 1);
		} while (assumed != old);

		//if (*address == 0)
		//{
		//	printf(" -%d- ", *address);
		//}
		

		if (fragments[idx].shade == false)
		{
			fragments[idx].shade = true;
			fragments[idx].depth = FLT_MAX;
		}

		if (z < fragments[idx].depth)
		{
			fragments[idx].depth = z;
			fragments[idx].color = p.color;
			fragments[idx].normal_eye_space = p.noraml_eye_space;
			fragments[idx].uv = p.uv;
		}
		

		//unlock
		old = *address;
		do{
			assumed = old;
			old = atomicCAS(address, assumed, 0);
		} while (assumed != old);
		//if (*address == 1)
		//{
		//	printf("%d,%d\t", *address,old);
		//}
		

		z += dz;
	}
}








/**
* Rasterize the area between two edges as the left and right limit.
* e1 - longest y span
*/
__device__
void drawAllScanLines(int width, int height, Edge & e1, Edge & e2, Fragment * fragments, int * fragmentLocks)
{
	// Discard horizontal edge as there is nothing to rasterize
	if (e2.v[1].pos.y - e2.v[0].pos.y == 0.0f) return;

	// Find the starting and ending y positions and
	// clamp them to be within the visible region
	int y_bot = (int)(ceilf(e2.v[0].pos.y) + EPSILON);
	int y_top = (int)(ceilf(e2.v[1].pos.y) + EPSILON);

	if (y_bot < 0)
	{
		y_bot = 0;
	}

	if (y_top > height)
	{
		y_top = height;
	}


	//Initialize edge's structure
	initEdge(e1, (float)y_bot);
	initEdge(e2, (float)y_bot);


	for (int y = y_bot; y < y_top; ++y)
	{
		if (e1.x <= e2.x)
		{
			drawOneScanLine(width, e1, e2, y, fragments, fragmentLocks);
		}
		else
		{
			drawOneScanLine(width, e2, e1, y, fragments, fragmentLocks);
		}

		//update edge
		updateEdge(e1);
		updateEdge(e2);
	}
}





/**
* Each thread handles one triangle
* rasterization
*/
__global__
void kernScanLineForOneTriangle(int width,int height
, Triangle * triangles, Fragment * depth_fragment, int * fragmentLocks)
{
	int triangleId = blockDim.x * blockIdx.x + threadIdx.x;

	Triangle tri = triangles[triangleId];	//copy

	//currently tri.v are in clipped coordinates
	//need to transform to viewport coordinate
	for (int i = 0; i < 3; i++)
	{
		tri.v[i].divide_w_clip = 1.0f / tri.v[i].pos.w;
		//view port
		tri.v[i].pos.x = 0.5f * width * (tri.v[i].pos.x * tri.v[i].divide_w_clip + 1.0f);
		tri.v[i].pos.y = 0.5f * height * (tri.v[i].pos.y * tri.v[i].divide_w_clip + 1.0f);
		tri.v[i].pos.z = 0.5f * (tri.v[i].pos.z * tri.v[i].divide_w_clip + 1.0f);
		tri.v[i].pos.w = 1.0f;

		//perspective correct interpolation
		tri.v[i].color *= tri.v[i].divide_w_clip;
		tri.v[i].noraml_eye_space *= tri.v[i].divide_w_clip;
		tri.v[i].uv *= tri.v[i].divide_w_clip;
		


		////////

	}


	//build edge
	// for line scan
	Edge edges[3];

	constructEdge(edges[0], tri.v[0], tri.v[1]);
	constructEdge(edges[1], tri.v[1], tri.v[2]);
	constructEdge(edges[2], tri.v[2], tri.v[0]);

	//Find the edge with longest y span
	float maxLength = 0.0f;
	int longEdge = -1;
	for (int i = 0; i < 3; ++i)
	{
		float length = edges[i].v[1].pos.y - edges[i].v[0].pos.y;
		if (length > maxLength)
		{
			maxLength = length;
			longEdge = i;
		}
	}


	// get indices for other two shorter edges
	int shortEdge0 = (longEdge + 1) % 3;
	int shortEdge1 = (longEdge + 2) % 3;

	// Rasterize two parts separately
	drawAllScanLines(width, height, edges[longEdge], edges[shortEdge0], depth_fragment, fragmentLocks);
	drawAllScanLines(width, height, edges[longEdge], edges[shortEdge1], depth_fragment, fragmentLocks);

	

}

//---------------------------------------------------------------------------



//-------------------------------------------------------------------------------
// Fragment Shader
//-------------------------------------------------------------------------------

//__global__ 
//void fragmentShader(int width, int height, Fragment* depthBuffer, FragmentIn* fragments   )
//{
//	//currently
//	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
//	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
//
//	if (x < width && y < height)
//	{
//		int index = x + y*width;
//
//		if (fragments[index].shade)
//		{
//			//depthBuffer[index].color = glm::vec3(1.0f);
//
//			//test: normal
//			depthBuffer[index].color = fragments[index].normal_eye_space;
//		}
//		else
//		{
//			depthBuffer[index].color = BACKGROUND_COLOR;
//		}
//	}
//}


// Writes fragment colors to the framebuffer
__global__
void render(int w, int h, Fragment *depthbuffer, glm::vec3 *framebuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * w);

	if (x < w && y < h) {
		//framebuffer[index] = depthbuffer[index].color;

		if (x < w && y < h)
		{

			if (depthbuffer[index].shade)
			{
				//depthBuffer[index].color = glm::vec3(1.0f);

				//test: normal
				framebuffer[index] = depthbuffer[index].normal_eye_space;
			}
			else
			{
				framebuffer[index] = BACKGROUND_COLOR;
			}
		}
	}
}

//--------------------------------------------------------------------------------





/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	//cudaMemset(dev_fragments, 0, width * height * sizeof(FragmentIn));
	cudaMemset(dev_fragmentLocks, 0, width * height * sizeof(int));
	cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));


	//rasterization
	dim3 blockSize_Rasterize(64);
	dim3 blockCount_tri((triCount + blockSize_Rasterize.x - 1) / blockSize_Rasterize.x);

	cudaDeviceSynchronize();
	kernScanLineForOneTriangle << <blockCount_tri, blockSize_Rasterize >> >(width, height, dev_primitives, dev_depthbuffer, dev_fragmentLocks);


	//fragment shader
	//fragmentShader << <blockCount2d, blockSize2d >> >(width, height, dev_depthbuffer, dev_fragments);


    // Copy depthbuffer colors into framebuffer
	cudaDeviceSynchronize();
    render<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer, dev_framebuffer);



    // Copy framebuffer into OpenGL buffer for OpenGL previewing
	cudaDeviceSynchronize();
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
	cudaDeviceSynchronize();
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

	//cudaFree(dev_fragments);
	//dev_fragments = NULL;

	cudaFree(dev_fragmentLocks);
	dev_fragmentLocks = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

    checkCUDAError("rasterizeFree");
}
