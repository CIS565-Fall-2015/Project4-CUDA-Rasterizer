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

//#define ENABLE_ANTI_ALIASING
//#define ENABLE_BLENDING

#define Alpha 0.5f

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};
struct VertexOut {
    // TODO
	glm::vec3 pos;
	glm::vec3 nor;
    glm::vec3 col;

};
struct Triangle {
    VertexOut v[3];
};
struct Fragment {
    glm::vec3 color;
	glm::vec3 nor;
	float z;	
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static VertexIn *dev_bufVertex = NULL;
static VertexOut *dev_bufVertex_out = NULL;
static Triangle *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;
static bool* dev_is_writable = NULL;  // mutex for race condition

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

//vertex shader function
__global__
void kern_vertex_shader(VertexIn *dev_bufVertex_in, VertexOut *dev_bufVertex_out, int vertCount,glm::mat4 MVP,glm::mat4 M_inv_T) //trans = proj*view*model
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//simple version for doing nothing

	
	if( index < vertCount)
	{
		VertexIn cur_v_in = dev_bufVertex_in[index];
		//calculate pos 
		dev_bufVertex_out[index].pos = multiplyMV(MVP,glm::vec4(cur_v_in.pos,1.f));
		
		//calculate normal
		dev_bufVertex_out[index].nor = multiplyMV(M_inv_T, glm::vec4(cur_v_in.nor,1.f));
		//calculate color
		dev_bufVertex_out[index].col = cur_v_in.col;

		
	}

}


//primitives assembly
__global__ 
void kern_premitive_assemble(VertexOut* dev_bufVertex_out,int* dev_bufIdx,Triangle* dev_primitives,int num_of_primitives)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(index<num_of_primitives)
	{
		int v1_index = dev_bufIdx[3*index+0];
		int v2_index = dev_bufIdx[3*index+1];
		int v3_index = dev_bufIdx[3*index+2];

		dev_primitives[index].v[0] = dev_bufVertex_out[v1_index];
		dev_primitives[index].v[1] = dev_bufVertex_out[v2_index];
		dev_primitives[index].v[2] = dev_bufVertex_out[v3_index];


	}
}

//Rasterization
__global__ 
void kern_rasterization(Triangle* dev_primitives,Fragment *dev_depthbuffer, int num_of_primitives, int width, int height, bool* dev_is_writable)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if( index < num_of_primitives)
	{
		Triangle cur_triangle = dev_primitives[index];
		glm::vec3 m_tri[3] = {cur_triangle.v[0].pos,cur_triangle.v[1].pos,cur_triangle.v[2].pos};
		glm::vec3 m_normals[3] = {cur_triangle.v[0].nor,cur_triangle.v[1].nor,cur_triangle.v[2].nor};
		glm::vec3 m_colors[3] = {cur_triangle.v[0].col,cur_triangle.v[1].col,cur_triangle.v[2].col};
		AABB cur_AABB = getAABBForTriangle(m_tri);

		float min_x = max(cur_AABB.min.x,-1.f) ;
		float min_y = max(cur_AABB.min.y,-1.f);
		float max_x = min(cur_AABB.max.x,1.f);
		float max_y = min(cur_AABB.max.y,1.f);

		if(min_x > 1 || max_x < -1 || min_y >1 || max_y<-1)
		{
			return;
		}
		
		
		float dx = 2.f/width;
		float dy = 2.f/height;

		int min_x_idx = max((int)((min_x+1)/dx),(int)0);
		int min_y_idx = max((int)((min_y+1)/dy),(int)0);
		int max_x_idx = min((int)((max_x+1)/dx),(int)width-1);
		int max_y_idx = min((int)((max_y+1)/dy),(int)height-1);


		
		//first try the center sampling method
		
		for(int i = min_y_idx;i<=max_y_idx;i++)
		{
			for(int j = min_x_idx ; j<=max_x_idx ;j++)
			{
				int buffer_index = (height-i)*width + j;

				//center point sample
				float cur_y = -1+ ((float)i*2+1.f)/(float)height;
				float cur_x = -1+ ((float)j*2+1.f)/(float)width;

				glm::vec2 cur_vec2 (cur_x,cur_y);

				glm::vec3 b_c = calculateBarycentricCoordinate(m_tri,cur_vec2);
				bool is_inside = isBarycentricCoordInBounds(b_c);

				//subpixel center 1
				float sub_y1 = cur_y + dy/4.f;
				float sub_x1 = cur_x - dx/4.f;
				glm::vec3 b_c1 = calculateBarycentricCoordinate(m_tri,glm::vec2(sub_x1,sub_y1));
				bool is_inside1 = isBarycentricCoordInBounds(b_c1);

				//subpixel center 2
				float sub_y2 = cur_y + dy/4.f;
				float sub_x2 = cur_x + dx/4.f;
				glm::vec3 b_c2 = calculateBarycentricCoordinate(m_tri,glm::vec2(sub_x2,sub_y2));
				bool is_inside2 = isBarycentricCoordInBounds(b_c2);
				
				//subpixel center 3
				float sub_y3 = cur_y - dy/4.f;
				float sub_x3 = cur_x - dx/4.f;
				glm::vec3 b_c3 = calculateBarycentricCoordinate(m_tri,glm::vec2(sub_x3,sub_y3));
				bool is_inside3 = isBarycentricCoordInBounds(b_c3);
				
				//subpixel center 4
				float sub_y4 = cur_y - dy/4.f;
				float sub_x4 = cur_x + dx/4.f;
				glm::vec3 b_c4 = calculateBarycentricCoordinate(m_tri,glm::vec2(sub_x4,sub_y4));
				bool is_inside4 = isBarycentricCoordInBounds(b_c4);
				
				int sample_res = (int)is_inside + (int)is_inside1 + (int)is_inside2 + (int)is_inside3 + (int)is_inside4;
				
				if(sample_res)
				{
					float cur_z = getZAtCoordinate(b_c,m_tri);
					if(cur_z<=1 && cur_z>= -1) //within the range 
					{
						if(dev_depthbuffer[buffer_index].z<cur_z)
						{
							// attention for the race condition
							//wait until it is writable
							while(!dev_is_writable[buffer_index])
							{}

							//enter critical area
							dev_is_writable[buffer_index] =false;


							dev_depthbuffer[buffer_index].z = cur_z;
							
							//interpolate the color
							
#ifdef ENABLE_ANTI_ALIASING
#ifdef ENABLE_BLENDING
							dev_depthbuffer[buffer_index].color =Alpha*(m_colors[0]*b_c.x +m_colors[1]*b_c.y+m_colors[2]*b_c.z)*((float)sample_res / 5.f) + (1.f- Alpha)*dev_depthbuffer[buffer_index].color;

#else
		dev_depthbuffer[buffer_index].color =(m_colors[0]*b_c.x +m_colors[1]*b_c.y+m_colors[2]*b_c.z)*((float)sample_res / 5.f);

#endif

							

#else
							dev_depthbuffer[buffer_index].color =(m_colors[0]*b_c.x +m_colors[1]*b_c.y+m_colors[2]*b_c.z)*((float)is_inside);
#endif
							//interpolate the normal
							dev_depthbuffer[buffer_index].nor = m_normals[0]*b_c.x +m_normals[1]*b_c.y+m_normals[2]*b_c.z;
						
							//leave critical area
							dev_is_writable[buffer_index] = true;
						
						}
#ifdef ENABLE_BLENDING
						if(dev_depthbuffer[buffer_index].z>cur_z)
						{
							while(!dev_is_writable[buffer_index])
							{}

							//enter critical area
							dev_is_writable[buffer_index] =false;

							dev_depthbuffer[buffer_index].color = Alpha*dev_depthbuffer[buffer_index].color + (1.f - Alpha)*(m_colors[0]*b_c.x +m_colors[1]*b_c.y+m_colors[2]*b_c.z)*((float)sample_res / 5.f);

							dev_is_writable[buffer_index] = true;
						}
#endif

					}
				}

			}
		}


	}
	
}

//fragment shader
__global__
void kern_fragment_shader(Fragment *dev_depthbuffer, int num_of_fragment)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(index < num_of_fragment)
	{
		// for now just doing nothing to test
		if(abs(dev_depthbuffer[index].z + M_INFINITE)>1e-6)
		{
			//dev_depthbuffer[index].color = glm::normalize(glm::vec3(abs(dev_depthbuffer[index].nor.x),abs(dev_depthbuffer[index].nor.y),abs(dev_depthbuffer[index].nor.z)));
			
			//light direction glm::vec3(0,0,-1)
			float dot_prod = glm::dot(glm::normalize(glm::vec3(1.0,0.0,0.0)),glm::normalize(dev_depthbuffer[index].nor));

			if(dot_prod>0)
			{
				dev_depthbuffer[index].color *= (dot_prod + 0.1);
			}
			else
			{
				dev_depthbuffer[index].color *= 0.1;
			}
		}
		

	}

}

//fragment init
__global__
void kern_fragment_init(Fragment *dev_depthbuffer, bool * dev_is_writable, int num_of_fragment)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(index < num_of_fragment)
	{
		// for now just doing nothing to test
		dev_depthbuffer[index].z = -M_INFINITE;
		dev_depthbuffer[index].color = glm::vec3(0.0);
		dev_depthbuffer[index].nor = glm::vec3(0.0);

		dev_is_writable[index] = true;

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


	cudaFree(dev_is_writable);
    cudaMalloc(&dev_is_writable,   width * height * sizeof(bool));
    cudaMemset(dev_is_writable, true, width * height * sizeof(bool));

	
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

	cudaFree(dev_bufVertex_out);
    cudaMalloc(&dev_bufVertex_out, vertCount * sizeof(VertexOut));

    cudaFree(dev_primitives);
    cudaMalloc(&dev_primitives, bufIdxSize / 3 * sizeof(Triangle));
    cudaMemset(dev_primitives, 0, bufIdxSize / 3 * sizeof(Triangle));

    checkCUDAError("rasterizeSetBuffers");
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, Camera *m_camera) 
{
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);
	
    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	//vertex shader 
	dim3 blockSize1d (THREADS_PER_BLOCK);
	dim3 blockCount1d (vertCount/THREADS_PER_BLOCK+1);

	glm::mat4 m_view =  glm::transpose(m_camera->GetViewMatrix());
	glm::mat4 m_proj =  m_camera->GetProjectionMatrix();
	glm::mat4 m_model = m_camera->GetModelMatrix();
	//glm::mat4 MVP = glm::mat4(1.0);
	//glm::mat4 MVP_inv_T = glm::mat4(1.0);

	glm::mat4 MVP = m_proj * m_view * m_model;
	glm::mat4 M_inv_T = glm::transpose(glm::inverse(m_model));


	kern_vertex_shader<<<blockCount1d,blockSize1d>>>(dev_bufVertex, dev_bufVertex_out, vertCount, MVP, M_inv_T);

	//primitive assembler
	int num_of_primitives = bufIdxSize/3;
	blockCount1d.x = num_of_primitives/THREADS_PER_BLOCK+1;

	
	kern_premitive_assemble<<<blockCount1d,blockSize1d>>>(dev_bufVertex_out,dev_bufIdx,dev_primitives, num_of_primitives);

	
	//fragment init
	int num_of_fragment = width * height;
	blockCount1d.x = num_of_fragment/THREADS_PER_BLOCK+1;

	kern_fragment_init<<<blockCount1d,blockSize1d>>>(dev_depthbuffer,dev_is_writable,num_of_fragment);
	
	
	//rasterization
	kern_rasterization<<<blockCount1d,blockSize1d>>>(dev_primitives, dev_depthbuffer, num_of_primitives, width, height, dev_is_writable);

	//fragment shader
	blockCount1d.x = num_of_fragment/THREADS_PER_BLOCK+1;

	kern_fragment_shader<<<blockCount1d,blockSize1d>>>(dev_depthbuffer,num_of_fragment);


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

	cudaFree(dev_bufVertex_out);
    dev_bufVertex_out = NULL;

    cudaFree(dev_primitives);
    dev_primitives = NULL;

    cudaFree(dev_depthbuffer);
    dev_depthbuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_is_writable);
	dev_is_writable = NULL;

    checkCUDAError("rasterizeFree");
}
