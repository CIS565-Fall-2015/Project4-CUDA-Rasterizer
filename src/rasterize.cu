/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#include "rasterize.h"

#include <iostream>
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
};

static int width = 0;
static int height = 0;
static int *dev_bufIdx = NULL;
static int *dev_depth=NULL;
static VertexIn *dev_bufVertex = NULL;
static Triangle *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static VertexOut *dev_vertexOut=NULL;
static glm::vec3 *dev_framebuffer = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;

__global__ void vertexShadingTest(VertexIn *vs_input,VertexOut *vs_output,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		vs_output[index].col=vs_input[index].col;
		vs_output[index].nor=vs_input[index].nor;
		vs_output[index].pos=vs_input[index].pos;
	}
}

__global__ void vertexShading(VertexIn *vs_input,VertexOut *vs_output,
							  glm::vec3 cameraUp,glm::vec3 cameraFront,float fovy,float cameraDis,float rotation,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		vs_output[index].col=vs_input[index].col;
		vs_output[index].nor=vs_input[index].nor;
		glm::mat4 m;
		//m=glm::translate(m,glm::vec3(0,0,1));
        //m=m*glm::rotate(m, glm::radians(rotation), glm::vec3(0.0f, 1.0f, 0.0f));
		m=m*glm::perspective(glm::radians(fovy),1.0f,0.1f,100.0f);
		m=m*glm::lookAt(-cameraFront,glm::vec3(0,0,0),cameraUp);
		//m=m*glm::rotate(m, glm::radians(rotation), glm::vec3(0.0f, 1.0f, 0.0f));
		vs_output[index].pos=multiplyMV(m,glm::vec4(vs_input[index].pos,1));
		vs_output[index].pos/=cameraDis;
		if(vs_output[index].pos.x<-1) vs_output[index].pos.x=-1;
		if(vs_output[index].pos.x>1) vs_output[index].pos.x=1;
		if(vs_output[index].pos.y<-1) vs_output[index].pos.y=-1;
		if(vs_output[index].pos.y>1) vs_output[index].pos.y=1;
		if(vs_output[index].pos.z<-1) vs_output[index].pos.z=-1;
		if(vs_output[index].pos.z>1) vs_output[index].pos.z=1;
	}
}

__global__ void primitiveAssemblyTest(VertexOut *vs_output,int *indices,Triangle *primitives,int N){
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index<N){
		primitives[index].v[0]=vs_output[indices[3*index]];
		primitives[index].v[1]=vs_output[indices[3*index+1]];
		primitives[index].v[2]=vs_output[indices[3*index+2]];
	}
}

__global__ void setColorToBlack(Fragment *fg_out,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		fg_out[index].color=glm::vec3(0,0,0);
	}
}

__global__ void setFlagZero(int *flag,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		flag[index]=0;
	}
}

__global__ void setDepthMax(int *depth,int N){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		depth[index]=1e12;
	}
}

__global__ void rasterizationTest(Triangle *primitives,Fragment *fg_out,int N,int Len){//no race condition considered, since only one triangle here.
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		float _len=2.0/Len;
		glm::vec3 tri[3];
		tri[0]=primitives[index].v[0].pos;
		tri[1]=primitives[index].v[1].pos;
		tri[2]=primitives[index].v[2].pos;
		AABB boundary=getAABBForTriangle(tri);
		for(int i=(1-boundary.max.y)/_len;i<(1-boundary.min.y)/_len;++i){
			for(int j=(boundary.min.x+1)/_len;j<(boundary.max.x+1)/_len;++j){
				glm::vec2 p(j*_len-1,1-i*_len);
				glm::vec3 tmp=calculateBarycentricCoordinate(tri,p);
				if(isBarycentricCoordInBounds(tmp)){
					fg_out[i*Len+j].color=glm::vec3(1,1,1);
				}
			}
		}
	}
}

__global__ void rasterization(Triangle *primitives,Fragment *fg_out,int *depth,glm::vec3 lightPos,glm::vec3 eyePos,int N,int Len){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<N){
		float _len=2.0/Len;
		glm::vec3 tri[3];
		tri[0]=primitives[index].v[0].pos;
		tri[1]=primitives[index].v[1].pos;
		tri[2]=primitives[index].v[2].pos;
		AABB boundary=getAABBForTriangle(tri);
		for(int i=(1-boundary.max.y)/_len;i<(1-boundary.min.y)/_len;++i){
			for(int j=(boundary.min.x+1)/_len;j<(boundary.max.x+1)/_len;++j){
				glm::vec2 p(j*_len-1,1-i*_len);
				glm::vec3 tmp=calculateBarycentricCoordinate(tri,p);
				if(isBarycentricCoordInBounds(tmp)){
					int currentDepth=(int)(1e6*getZAtCoordinate(tmp,tri));
					atomicMin(&depth[i*Len+j],currentDepth);
					if(currentDepth==depth[i*Len+j]){
						glm::vec3 n1=primitives[index].v[0].nor;
						glm::vec3 n2=primitives[index].v[1].nor;
						glm::vec3 n3=primitives[index].v[2].nor;
						glm::vec3 normal=n1*tmp.x+n2*tmp.y+n3*tmp.z;

						glm::vec3 pos=tri[0]*tmp.x+tri[1]*tmp.y+tri[2]*tmp.z;
						glm::vec3 dir=glm::normalize(lightPos-pos);
						glm::vec3 diffuse=glm::vec3(1,1,1)*(glm::dot(dir,normal));
						//diffuse

						glm::vec3 ref=dir-2.0f*normal*glm::dot(dir,normal);
						glm::vec3 eyeDir=glm::normalize(-eyePos-pos);
						glm::vec3 specular=glm::vec3(1,1,1)*max(0.0f,(float)pow(glm::dot(ref,eyeDir),20.0f));

						fg_out[i*Len+j].color=diffuse*0.7f+specular*0.3f;
						//fg_out[i*Len+j].color=normal;
						//fg_out[i*Len+j].color=glm::vec3(1,1,1);
					}
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
    cudaFree(dev_depthbuffer);
    cudaMalloc(&dev_depthbuffer,   width * height * sizeof(Fragment));
    cudaMemset(dev_depthbuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
	cudaMalloc(&dev_depth, width*height*sizeof(int));
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
	//std::cout<<vertCount<<std::endl;
    VertexIn *bufVertex = new VertexIn[_vertCount];
    for (int i = 0; i < vertCount; i++) {
        int j = i * 3;
        bufVertex[i].pos = glm::vec3(bufPos[j + 0], bufPos[j + 1], bufPos[j + 2]);
        bufVertex[i].nor = glm::vec3(bufNor[j + 0], bufNor[j + 1], bufNor[j + 2]);
        bufVertex[i].col = glm::vec3(bufCol[j + 0], bufCol[j + 1], bufCol[j + 2]);
		//std::cout<<bufVertex[i].nor.x<<","<<bufVertex[i].nor.y<<","<<bufVertex[i].nor.z<<std::endl;
    }
    cudaFree(dev_bufVertex);
    cudaMalloc(&dev_bufVertex, vertCount * sizeof(VertexIn));
    cudaMemcpy(dev_bufVertex, bufVertex, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);

	cudaFree(dev_vertexOut);
    cudaMalloc(&dev_vertexOut, vertCount * sizeof(VertexOut));

    cudaFree(dev_primitives);
    cudaMalloc(&dev_primitives, vertCount / 3 * sizeof(Triangle));
    cudaMemset(dev_primitives, 0, vertCount / 3 * sizeof(Triangle));

    checkCUDAError("rasterizeSetBuffers");
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo,glm::vec3 lightPos,glm::vec3 cameraUp,glm::vec3 cameraFront,float fovy,float cameraDis,float rotation) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);
	
    // TODO: Execute your rasterization pipeline here
    // (See README for rasterization pipeline outline.)

	vertexShading<<<(vertCount+127)/128,128>>>(dev_bufVertex,dev_vertexOut,cameraUp,cameraFront,fovy,cameraDis,rotation,vertCount);
	primitiveAssemblyTest<<<(bufIdxSize/3+127)/128,128>>>(dev_vertexOut,dev_bufIdx,dev_primitives,bufIdxSize/3);
	setColorToBlack<<<(width*height+127)/128,128>>>(dev_depthbuffer,width*height);
	
	setDepthMax<<<(width*height+127)/128,128>>>(dev_depth,width*height);
	rasterization<<<(bufIdxSize/3+127)/128,128>>>(dev_primitives,dev_depthbuffer,dev_depth,lightPos,cameraFront,bufIdxSize/3,width);
	
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

	cudaFree(dev_vertexOut);
	dev_vertexOut=NULL;

	cudaFree(dev_depth);
	dev_depth=NULL;

    checkCUDAError("rasterizeFree");
}
