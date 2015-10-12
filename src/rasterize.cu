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
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
	glm::vec2 uv;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};
struct VertexOut {
    // TODO
	glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
	glm::vec2 uv;

};
struct Triangle {
    VertexOut v[3];
};
struct Fragment {
    glm::vec3 color;
	glm::vec3 position;
	glm::vec3 normal;
	int depth;
};

__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

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

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
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
__global__ void render(int w, int h, Fragment *depthbuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        framebuffer[index] = depthbuffer[index].color;
    }
}

__global__ void depthBufferClearing(int w, int h, Fragment *fragments) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

	if ( x < w && y < h) {
		fragments[index].depth = INT_MAX;
		fragments[index].color = glm::vec3(0.0f);
	}
}

__global__ void vertexShading(int n, glm::mat4 view_projection,
	VertexIn *vs_input, VertexOut *vs_output) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (index < n) {

		VertexOut vert_out;
		glm::vec4 input_pos = glm::vec4(
			vs_input[index].pos.x, 
			vs_input[index].pos.y, 
			vs_input[index].pos.z, 
			1.0f);

		glm::vec3 transformedPoint = multiplyMV(view_projection, input_pos);
		vert_out.pos = transformedPoint;

		glm::vec4 input_normal = glm::vec4(
			vs_input[index].nor.x, 
			vs_input[index].nor.y, 
			vs_input[index].nor.z, 
			1.0f);

		glm::vec3 output_normal = multiplyMV(view_projection,input_normal);
		vert_out.nor = output_normal;

		vert_out.col = vs_input[index].col;

		vs_output[index] = vert_out;
		
	}
}

__global__ void primitiveAssembling(int n, VertexOut *vs_output,
	Triangle *primitives) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (index < n) {
		primitives[index].v[0] = vs_output[3*index];
		primitives[index].v[1] = vs_output[3*index+1];
		primitives[index].v[2] = vs_output[3*index+2];
	}

}

__global__ void rasterizing(int n, int w, int h,
	Triangle *primitives, Fragment *fs_input) {

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < n) {
		Triangle tri = primitives[index];
		glm::vec3 tri_verts[3] = {tri.v[0].pos, tri.v[1].pos, tri.v[2].pos};
		AABB aabb = getAABBForTriangle(tri_verts);

		glm::vec2 pixel_min;
		pixel_min.x = (aabb.min.x + 1) * w / 2.0f;
		pixel_min.y = (aabb.min.y + 1) * h / 2.0f;

		glm::vec2 pixel_max;
		pixel_max.x = (aabb.max.x + 1) * w / 2.0f;
		pixel_max.y = (aabb.max.y + 1) * h / 2.0f;
		
		for (int i = glm::max(0.0f, pixel_min.x); i <= pixel_max.x; i++) {
			for (int j = glm::max(0.0f, pixel_min.y); j <= pixel_max.y; j++) {
				
				thrust::default_random_engine rng = makeSeededRandomEngine(0, index, 0);
				thrust::uniform_real_distribution<float> u01(0, 1);

				float x = ((i + u01(rng))/float(w)) * 2.0f - 1;
				float y = ((j + u01(rng))/float(h)) * 2.0f - 1;

				glm::vec3 barycentric = calculateBarycentricCoordinate(tri_verts,
					glm::vec2(x,y));
				if (isBarycentricCoordInBounds(barycentric)) {

					int frag_index = j*w + i;
					int depth = getZAtCoordinate(barycentric, tri_verts) * INT_MAX;
					atomicMin(&fs_input[frag_index].depth, depth);
					
					if(fs_input[frag_index].depth == depth) {

						Fragment frag;
						frag.color = (primitives[index].v[0].col
							+ primitives[index].v[1].col
							+ primitives[index].v[2].col) / 3.0f;

						frag.normal = (primitives[index].v[0].nor
							+ primitives[index].v[1].nor
							+ primitives[index].v[2].nor) / 3.0f;

						frag.position = barycentric;
						frag.depth = depth;

						fs_input[frag_index] = frag;
					}
				}
			}
		}

		
	}
}

__global__ void fragmentShading(int w, int h, Fragment *fs, glm::vec3 light_pos) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

	if ( x < w && y < h) {
		float diffuseTerm = 0.7f;
		glm::vec3 light_color = glm::vec3(1.0f);
		fs[index].color *= diffuseTerm * glm::max(0.0f, 
			glm::dot(glm::normalize(fs[index].normal),
			glm::normalize(light_pos - fs[index].position)));
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

	cudaFree(dev_bufVertex_out);
    cudaMalloc(&dev_bufVertex_out, vertCount * sizeof(VertexOut));
   
    cudaFree(dev_primitives);
    cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
    cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

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

	dim3 blockSize1d(128);
	dim3 blockCount1d((vertCount + 128 - 1) / 128);

    //-----RATERIZATION PIPELINE----------

	//---Clear Depth Buffer
	depthBufferClearing<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer);
	checkCUDAError("depth buffer clearing");

	//---Vertex Shader
	//view matrix 
	glm::mat4 view = glm::lookAt(
			glm::vec3(0.0f, 1.5f, 5.0f), 
			glm::vec3(0.0f, 0.0f, -1.0f), 
			glm::vec3(0.0f, -1.0f, 0.0f));
	
	//projection matrix
	glm::mat4 projection = glm::perspective(
		45.0f, float(width)/float(height), 1.0f, 100.0f);

	glm::mat4 view_projection = projection * view;

	vertexShading<<<blockCount1d, blockSize1d>>>(vertCount, view_projection,
		dev_bufVertex, dev_bufVertex_out);
	checkCUDAError("vertex shader");

	//---Primitive Assembly
	primitiveAssembling<<<blockCount1d, blockSize1d>>>(vertCount/3, 
		dev_bufVertex_out, dev_primitives);
	checkCUDAError("primitive assembling");

	//---Rasterization
	rasterizing<<<blockCount1d, blockSize1d>>>(vertCount/3, width, height,
		dev_primitives, dev_depthbuffer);
	checkCUDAError("triangle rasterizing");

	//--Fragment Shader
	glm::vec3 light_pos = glm::vec3(-3.0f, 5.0f, 10.0f);
	fragmentShading<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer,
		light_pos);
	checkCUDAError("fragment shading");
   
	
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

    checkCUDAError("rasterizeFree");
}
