/*
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania & STUDENT
 */

#include "rasterize.h"

#include <cmath>
#include <cstdio>
#include <climits>
#include <cuda.h>

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/transform.hpp>

#include <util/utilityCore.hpp>
#include <util/checkCUDAError.h>
#include "rasterizeTools.h"
#include "sceneStructs.h"

/************************* Struct Definitions *********************************/

struct VertexIn {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;
};
struct VertexOut {
    glm::vec3 pos;
    glm::vec3 nor;
    glm::vec3 col;

    glm::vec3 worldPos;
};
struct Triangle {
    glm::vec3 pos[3];
    glm::vec3 nor[3];
    glm::vec3 col[3];

    glm::vec3 worldPos[3];
    bool valid;
    bool middle;
};
struct Fragment {
    glm::vec3 color;
    Triangle tri;
    glm::vec3 baryCoords;
    int z;
    bool valid;
};

static float t = 0;

static int width = 0;
static int height = 0;
static int bufIdxSize = 0;
static int vertCount = 0;
static int primMultFactor = 4;

static int       *dev_bufIdx         = NULL;
static VertexIn  *dev_bufVertexIn    = NULL;
static VertexOut *dev_bufVertexOut   = NULL;
static Triangle  *dev_origPrimitives = NULL;
static Triangle  *dev_genPrimitives  = NULL;
static Fragment  *dev_depthbuffer    = NULL;
static glm::vec3 *dev_framebuffer    = NULL;

__device__ VertexOut transformVertex(glm::vec3 pos, glm::vec3 nor,
        glm::mat4 model, glm::mat4 invModel, glm::mat4 mvp) {
    VertexOut vo;
    vo.pos = pos;
    vo.worldPos = multiplyMV(model, glm::vec4(pos, 1));
    vo.pos = multiplyMV(mvp, glm::vec4(pos, 1));
    vo.nor = glm::vec3(invModel * glm::vec4(nor, 0));
    vo.col = glm::vec3(0.f);
    return vo;
}

__device__ Triangle buildTriangle(VertexOut v0, VertexOut v1, VertexOut v2) {
    Triangle tri;
    tri.pos[0] = v0.pos;
    tri.pos[1] = v1.pos;
    tri.pos[2] = v2.pos;

    tri.nor[0] = v0.nor;
    tri.nor[1] = v1.nor;
    tri.nor[2] = v2.nor;

    tri.col[0] = v0.col;
    tri.col[1] = v1.col;
    tri.col[2] = v2.col;

    tri.worldPos[0] = v0.worldPos;
    tri.worldPos[1] = v1.worldPos;
    tri.worldPos[2] = v2.worldPos;
    tri.valid = true;
    tri.middle = false;
    return tri;
}

__device__ void printVec3(glm::vec3 v) {
    printf("(%f, %f, %f)\n", v.x, v.y, v.z);
}

__device__ void printMat4(const glm::mat4 &m) {
    printf("%f, %f, %f, %f\n", m[0][0], m[1][0], m[2][0], m[3][0]);
    printf("%f, %f, %f, %f\n", m[0][1], m[1][1], m[2][1], m[3][1]);
    printf("%f, %f, %f, %f\n", m[0][2], m[1][2], m[2][2], m[3][2]);
    printf("%f, %f, %f, %f\n", m[0][3], m[1][3], m[2][3], m[3][3]);
}

/************************* Output to Screen ***********************************/

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

/**
 * Writes fragment colors to the framebuffer
 */
__global__ void render(int w, int h, Fragment *depthbuffer,
        glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < w && y < h) {
        int frameidx = x + (y * w);
        int depthidx = 4*frameidx;
        glm::vec3 color = depthbuffer[depthidx].color +
            depthbuffer[depthidx+1].color +
            depthbuffer[depthidx+2].color +
            depthbuffer[depthidx+3].color;
        framebuffer[frameidx] = color / 4.f;
    }
}

/************************* Initialization *************************************/

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;

    cudaFree(dev_depthbuffer);
    cudaMalloc(&dev_depthbuffer,    4*width * height * sizeof(Fragment));
    cudaMemset( dev_depthbuffer, 0, 4*width * height * sizeof(Fragment));

    cudaFree(dev_bufVertexOut);
    cudaMalloc(&dev_bufVertexOut,    width * height * sizeof(VertexOut));
    cudaMemset( dev_bufVertexOut, 0, width * height * sizeof(VertexOut));

    cudaFree(dev_genPrimitives);
    cudaMalloc(&dev_genPrimitives,    primMultFactor * width * height * sizeof(Triangle));
    cudaMemset( dev_genPrimitives, 0, primMultFactor * width * height * sizeof(Triangle));

    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,    width * height * sizeof(glm::vec3));
    cudaMemset( dev_framebuffer, 0, width * height * sizeof(glm::vec3));
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

    VertexIn *bufVertexIn = new VertexIn[_vertCount];
    for (int i = 0; i < vertCount; i++) {
        int j = i * 3;
        bufVertexIn[i].pos = glm::vec3(bufPos[j + 0], bufPos[j + 1], bufPos[j + 2]);
        bufVertexIn[i].nor = glm::vec3(bufNor[j + 0], bufNor[j + 1], bufNor[j + 2]);
        bufVertexIn[i].col = glm::vec3(bufCol[j + 0], bufCol[j + 1], bufCol[j + 2]);
    }
    cudaFree(dev_bufVertexIn);
    cudaMalloc(&dev_bufVertexIn, vertCount * sizeof(VertexIn));
    cudaMemcpy( dev_bufVertexIn, bufVertexIn, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);

    cudaFree(dev_origPrimitives);
    cudaMalloc(&dev_origPrimitives, vertCount / 3 * sizeof(Triangle));
    cudaMemset(dev_origPrimitives, 0, vertCount / 3 * sizeof(Triangle));

    checkCUDAError("rasterizeSetBuffers");
}

/************************* Rasterization Pipeline *****************************/

__device__ void clearFragment(int idx, Fragment *depthbuffer) {
    depthbuffer[idx].valid = false;
    depthbuffer[idx].z = INT_MAX;
    depthbuffer[idx].color = glm::vec3(.15, .15, .15);
}

__global__ void clearDepthBuffer(int width, int height, Fragment *depthbuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < width && y < height) {
        int index = x + (y * width);
        clearFragment(4*index  , depthbuffer);
        clearFragment(4*index+1, depthbuffer);
        clearFragment(4*index+2, depthbuffer);
        clearFragment(4*index+3, depthbuffer);
    }
}

// Applies vertex transformations (from given model-view-projection matrix)
__global__ void vertexShader(int vertcount,
        VertexIn *verticesIn, VertexOut *verticesOut,
        glm::mat4 model, glm::mat4 invModel, glm::mat4 mvp) {
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (k < vertcount) {
        VertexIn vin = verticesIn[k];

        //VertexOut vo = transformVertex(vin.pos, vin.nor, model, invModel, mvp);
        VertexOut vo;
        vo.pos = vin.pos;
        vo.nor = vin.nor;
        verticesOut[k] = vo;
    }
}

// Assembles sets of 3 vertices into Triangles.
__global__ void assemblePrimitives(int primitivecount, VertexOut *vertices,
        int *indices, Triangle *primitives) {
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (k < primitivecount) {
        VertexOut v0 = vertices[indices[k*3  ]];
        VertexOut v1 = vertices[indices[k*3+1]];
        VertexOut v2 = vertices[indices[k*3+2]];
        primitives[k] = buildTriangle(v0, v1, v2);
    }
}

__device__ void subdivide(int idx, Triangle tri, Triangle *genprimitives,
        glm::mat4 model, glm::mat4 invModel, glm::mat4 mvp) {
    // New vertices
    glm::vec3 v0pos = tri.pos[0];
    glm::vec3 v1pos = tri.pos[1];
    glm::vec3 v2pos = tri.pos[2];
    glm::vec3 v3pos = midpoint(v0pos, v1pos);
    glm::vec3 v4pos = midpoint(v0pos, v2pos);
    glm::vec3 v5pos = midpoint(v1pos, v2pos);

    // New normals
    glm::vec3 v0nor = tri.nor[0];
    glm::vec3 v1nor = tri.nor[1];
    glm::vec3 v2nor = tri.nor[2];
    glm::vec3 v3nor = midpoint(v0nor, v1nor);
    glm::vec3 v4nor = midpoint(v0nor, v2nor);
    glm::vec3 v5nor = midpoint(v1nor, v2nor);

    VertexOut v0 = transformVertex(v0pos, v0nor, model, invModel, mvp);
    VertexOut v1 = transformVertex(v1pos, v1nor, model, invModel, mvp);
    VertexOut v2 = transformVertex(v2pos, v2nor, model, invModel, mvp);
    VertexOut v3 = transformVertex(v3pos, v3nor, model, invModel, mvp);
    VertexOut v4 = transformVertex(v4pos, v4nor, model, invModel, mvp);
    VertexOut v5 = transformVertex(v5pos, v5nor, model, invModel, mvp);

    genprimitives[idx  ] = buildTriangle(v0, v3, v4);
    genprimitives[idx+1] = buildTriangle(v3, v1, v5);
    genprimitives[idx+2] = buildTriangle(v4, v5, v2);
    genprimitives[idx+3] = buildTriangle(v3, v5, v4); // middle triangle
    genprimitives[idx+3].middle = true;
}

__global__ void geometryShader(int primitivecount, int multFactor, Triangle *primitives,
        Triangle *genprimitives, glm::vec3 eye,
        glm::mat4 model, glm::mat4 invModel, glm::mat4 mvp) {
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (k < primitivecount) {
        int genidx = k * multFactor;

        // Back-face culling
        Triangle tri = primitives[k];
        glm::vec3 n = glm::cross(tri.pos[1] - tri.pos[0], tri.pos[2] - tri.pos[0]);
        float dir = glm::dot(eye - tri.pos[0], n);
        subdivide(genidx, tri, genprimitives, model, invModel, mvp);
        if (dir < 0.f) {
            // Triangle tessellation
            //subdivide(genidx, tri, genprimitives);
        } else {
            //subdivide(genidx, tri, genprimitives);
            //genprimitives[genidx  ].valid = false;
            //genprimitives[genidx+1].valid = false;
            //genprimitives[genidx+2].valid = false;
            //genprimitives[genidx+3].valid = false;
        }
    }
}

__device__ void storeFragment(float x, float y, float width, float height,
        int fragmentidx, Triangle tri, Fragment *fragments) {

    glm::vec3 bary = calculateBarycentricCoordinate(tri.pos, glm::vec2(x, y));

    if (isBarycentricCoordInBounds(bary)) {
        Fragment prev = fragments[fragmentidx];

        float z = getZAtCoordinate(tri.worldPos, bary);
        int depth = z * INT_MAX;
        atomicMin(&fragments[fragmentidx].z, depth);

        if (fragments[fragmentidx].z == depth) {
            fragments[fragmentidx] = (Fragment) { glm::vec3(0, 0, 0), tri, bary, depth, true};
        }
    } else {
    }
}

// Scans across triangles to generate primitives (pixels).
__global__ void scanline(int w, int h, int tricount,
        Triangle *primitives, Fragment *fragments) {
    int k = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (k < tricount) {
        Triangle tri = primitives[k];
        if (tri.valid == false) {
            return;
        }

        float ystep = 2.f / h;
        float xstep = 2.f / w;

        float yjit = ystep / 4;
        float xjit = xstep / 4;

        AABB bb = getAABBForTriangle(tri.pos);

        float ymin = glm::max(-1.f, (int) (bb.min.y / ystep) * ystep);
        float xmin = glm::max(-1.f, (int) (bb.min.x / xstep) * xstep);
        float ymax = glm::min(1.f, bb.max.y);
        float xmax = glm::min(1.f, bb.max.x);
        for (float y = ymin; y < ymax; y += ystep) {
            for (float x = xmin; x < xmax; x += xstep) {
                glm::vec2 pos = fromNDC(x, y, w, h);
                int fragmentidx = 4*(pos.x + (pos.y * w));

                storeFragment(x,      y,      w, h, fragmentidx,   tri, fragments);
                storeFragment(x+xjit, y,      w, h, fragmentidx+1, tri, fragments);
                storeFragment(x,      y+yjit, w, h, fragmentidx+2, tri, fragments);
                storeFragment(x+xjit, y+yjit, w, h, fragmentidx+3, tri, fragments);
            }
        }
    }
}

__device__ void colorFragment(Fragment &frag, glm::vec3 light) {
    if (frag.valid) {
        glm::vec3 norm = barycentricInterpolate(frag.tri.nor, frag.baryCoords);
        glm::vec3 pos = barycentricInterpolate(frag.tri.worldPos, frag.baryCoords);
        glm::vec3 lightdir = glm::normalize(light - pos);
        frag.color = glm::dot(lightdir, norm) * glm::vec3(1, 0, 0);
        frag.color = glm::abs(glm::normalize(norm));
        if (frag.tri.middle == true) {
            frag.color *= .75;
        }
    }
}

__global__ void fragmentShader(int width, int height,
        Fragment *fragments, glm::vec3 light) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < width && y < height) {
        int index = 4*(x + (y * width));
        colorFragment(fragments[index]  , light);
        colorFragment(fragments[index+1], light);
        colorFragment(fragments[index+2], light);
        colorFragment(fragments[index+3], light);
    }
}

struct terminator {
    __device__ bool operator()(const Triangle tri) {
        return tri.valid == false;
    }
};

int compactPrimitives(int primitivecount, Triangle *primitives) {
    Triangle *new_end = thrust::remove_if(thrust::device,
            primitives, primitives+primitivecount, terminator());
    return (new_end - primitives);
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo) {
    //t += 0.01f;

    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

    int tricount = bufIdxSize / 3;

    int sideLength1d = 16;
    dim3 blockSize1d(sideLength1d);
    dim3 vertBlockCount((vertCount + sideLength1d - 1) / sideLength1d);
    dim3 triBlockCount((tricount + sideLength1d - 1) / sideLength1d);

    Camera c;
    // Suzanne
//    c.position = glm::vec3(1, 1, 4);
//    c.view = glm::vec3(0, 1, 0);
//    c.up = glm::vec3(0, -1, 0);
//    c.light = glm::vec3(2, 5, -1);
//    c.fovy = glm::radians(45.f);

    // Cow
    c.position = glm::vec3(0, .2, -0.5);
    c.view = glm::vec3(0, .2, 0);
    c.up = glm::vec3(0, 1, 0);
    c.light = glm::vec3(0, 4, 5);
    c.fovy = 17.f;

    // Cube
//    c.position = glm::vec3(0, 1, 1);
//    c.view = glm::vec3(0, 0, 0);
//    c.up = glm::vec3(0, 1, 0);
//    c.light = glm::vec3(0, 4, 5);
//    c.fovy = glm::radians(40.f);

    glm::mat4 model = glm::rotate(t, glm::vec3(0.f, 1.f, 0.f));
    glm::mat4 invModel = glm::inverseTranspose(model);
    glm::mat4 view = glm::lookAt(c.position, c.view, c.up);
    glm::mat4 persp = glm::perspective(c.fovy, 1.f, 1.f, 10.f);
    glm::mat4 mvp = persp * view * model;

    // Set CudaEvents
    float vShadeTime, assPrimitivesTime, scanlineTime, fShadeTime;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    // Clear Depth Buffer
    clearDepthBuffer<<<blockCount2d, blockSize2d>>>(width, height, dev_depthbuffer);

    // VertexIn -> VertexOut
        cudaEventRecord(begin);
    vertexShader<<<vertBlockCount, blockSize1d>>>(vertCount, dev_bufVertexIn,
            dev_bufVertexOut, model, invModel, mvp);
        checkCUDAError("");

        cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&vShadeTime, begin, end);

    // VertexOut -> Triangle
        cudaEventRecord(begin);
    assemblePrimitives<<<triBlockCount, blockSize1d>>>(tricount,
            dev_bufVertexOut, dev_bufIdx, dev_origPrimitives);
        checkCUDAError("");

        cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&assPrimitivesTime, begin, end);

    // Triangle -> Triangle
        cudaEventRecord(begin);
    geometryShader<<<triBlockCount, blockSize1d>>>(tricount, primMultFactor,
            dev_origPrimitives, dev_genPrimitives,
            c.position, model, invModel, mvp);
        checkCUDAError("");

        cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&assPrimitivesTime, begin, end);

    int genPrimitiveCount = compactPrimitives(tricount * primMultFactor, dev_genPrimitives);
    dim3 genPrimCount((genPrimitiveCount + sideLength1d - 1) / sideLength1d);

    // Triangle -> Fragment
        cudaEventRecord(begin);
    scanline<<<genPrimitiveCount, blockSize1d>>>(width, height, tricount *
            primMultFactor, dev_genPrimitives, dev_depthbuffer);
        checkCUDAError("");

        cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&scanlineTime, begin, end);

    // Fragment -> Fragment
        cudaEventRecord(begin);
    fragmentShader<<<blockCount2d, blockSize2d>>>(width, height,
            dev_depthbuffer, c.light);
        checkCUDAError("");

        cudaEventRecord(end); cudaEventSynchronize(end); cudaEventElapsedTime(&fShadeTime, begin, end);

    // Clear CudaEvents
    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    //fprintf(stderr, "%f %f %f %f\n", vShadeTime, assPrimitivesTime, scanlineTime, fShadeTime);

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

    cudaFree(dev_bufVertexOut);
    dev_bufVertexOut = NULL;

    cudaFree(dev_bufVertexIn);
    dev_bufVertexIn = NULL;

    cudaFree(dev_origPrimitives);
    dev_origPrimitives = NULL;

    cudaFree(dev_genPrimitives);
    dev_genPrimitives = NULL;

    cudaFree(dev_depthbuffer);
    dev_depthbuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

    checkCUDAError("rasterizeFree");
}
