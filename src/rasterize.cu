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

	glm::vec3 tex;
    // TODO (optional) add other vertex attributes (e.g. texture coordinates)
};
struct VertexOut {
	glm::vec3 pos;
	glm::vec3 nor;
	glm::vec3 col;

	glm::vec3 ndc;
	glm::vec3 winPos;

	glm::vec3 tex;
	bool DispAdded = false;
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
VertexOut *dev_bufVtxOut_tess = NULL;
VertexOut *dev_bufVtxOut = NULL;
static Triangle *dev_primitives = NULL;
static Fragment *dev_depthbuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
glm::vec3 **dev_textures = NULL;
glm::vec2 * dev_texInfo = NULL;
static int bufIdxSize = 0;
static int vertCount = 0;
static int bufTexSize = 0;
static 	int tessIncre = 1;
static int tessLevel = 0;
static int lastLevel = 0;
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
		depthbuffer[index].depth =2* MAX_DEPTH; //INFINITY;//!!!
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
		
		vtxO[index].pos = vtxI[index].pos;
		vtxO[index].ndc = glm::vec3(P_NDC);
		//vtxO[index].ndc = vtxI[index].pos;
		vtxO[index].nor = vtxI[index].nor;
		vtxO[index].col = vtxI[index].col;
		P_NDC = M_win*P_NDC;
		vtxO[index].winPos = glm::vec3(P_NDC);
		vtxO[index].tex = vtxI[index].tex;
	}
}

__device__ VertexOut EdgeTessellator_MipP(VertexOut vtxA, VertexOut vtxB, glm::mat4 Mats, glm::mat4 M_win)
{
	//now, all linear. (mid point)
	//!!!later..catmull??
	VertexOut vtxC;
	vtxC.pos = (vtxA.pos + vtxB.pos)*0.5f;
	vtxC.nor = glm::normalize(vtxA.nor + vtxB.nor);

	//vtxC.pos += vtxC.nor*0.1f;

	glm::vec4 clip = Mats*glm::vec4(vtxC.pos,1);
	vtxC.ndc = glm::vec3(clip*(1 / clip.w));
	vtxC.winPos = glm::vec3(M_win*glm::vec4(vtxC.ndc,1));// (vtxA.winPos + vtxB.winPos)*0.5f;
	vtxC.tex = (vtxA.tex + vtxB.tex)*0.5f;
	vtxC.col = (vtxA.col + vtxB.col)*0.5f;
	vtxC.DispAdded = false;
	return vtxC;
}

__global__ void kernPrimitiveAssembly(Triangle* primitives,int* bufIdx,int bufIdxSize, VertexOut * bufVtxOut,int tessLevel,int tessIncre)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//bufIdxSize *= tessIncre;
	if (index<bufIdxSize/3)
	{
		int i = 3 * index;
		//index*tessIncre~index*tessIncre+3
		index *= tessIncre;
		/*if (tessLevel>0)
		{
			VertexOut v0 = bufVtxOut[bufIdx[i + 0]];
			VertexOut v1 = bufVtxOut[bufIdx[i + 1]];
			VertexOut v2 = bufVtxOut[bufIdx[i + 2]];

			VertexOut m0 = EdgeTessellator(v0, v1, Mats, M_win);
			VertexOut m1 = EdgeTessellator(v0, v2, Mats, M_win);
			VertexOut m2 = EdgeTessellator(v1, v2, Mats, M_win);

			primitives[index + 0].v[0] = v0;
			primitives[index + 0].v[1] = m0;
			primitives[index + 0].v[2] = m1;

			primitives[index + 1].v[0] = m0;
			primitives[index + 1].v[1] = m2;
			primitives[index + 1].v[2] = m1;

			primitives[index + 2].v[0] = v1;
			primitives[index + 2].v[1] = m0;
			primitives[index + 2].v[2] = m2;

			primitives[index + 3].v[0] = m1;
			primitives[index + 3].v[1] = m2;
			primitives[index + 3].v[2] = v2;
		}
		else*/
		//{
			primitives[index].v[0] = bufVtxOut[bufIdx[i + 0]];		//p0
			primitives[index].v[1] = bufVtxOut[bufIdx[i + 1]];	//p1
			primitives[index].v[2] = bufVtxOut[bufIdx[i + 2]];	//p2
		//}

	}
}

__device__ VertexOut VtxUpdate(VertexOut v,glm::mat4 M_model,glm::mat4 M_view,glm::mat4 M_Proj,glm::mat4 M_win)
{
	VertexOut outV = v;
	glm::vec4 P_clip = M_Proj * M_view * M_model * glm::vec4(v.pos, 1);	//clip coords
	glm::vec4 P_NDC = P_clip*(1 / P_clip.w);//!!!w-divide for NDC	: P_clip/w

	outV.ndc = glm::vec3(P_NDC);
	P_NDC = M_win*P_NDC;
	outV.winPos = glm::vec3(P_NDC);

	return outV;
}

__global__ void kernPrimUpdate(Triangle* primitives, int primSize, glm::mat4 M_model, glm::mat4 M_view, glm::mat4 M_Proj, glm::mat4 M_win)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//bufIdxSize *= tessIncre;
	if (primSize)
	{
		primitives[index].v[0] = VtxUpdate(primitives[index].v[0], M_model, M_view, M_Proj, M_win);
		primitives[index].v[1] = VtxUpdate(primitives[index].v[1], M_model, M_view, M_Proj, M_win);
		primitives[index].v[2] = VtxUpdate(primitives[index].v[2], M_model, M_view, M_Proj, M_win);
	}
}

__global__ void kernTessellation_aftPri(Triangle* primitives,int crntSize,  int crntInce, glm::mat4 Mats, glm::mat4 M_win)
{
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < crntSize)
	{
		index *= crntInce;
		crntInce = crntInce/4;
		VertexOut v0 = primitives[index ].v[0];
		VertexOut v1 = primitives[index ].v[1];
		VertexOut v2 = primitives[index ].v[2];

		VertexOut m0 = EdgeTessellator_MipP(v0, v1, Mats, M_win);
		VertexOut m1 = EdgeTessellator_MipP(v0, v2, Mats, M_win);
		VertexOut m2 = EdgeTessellator_MipP(v1, v2, Mats, M_win);

		primitives[index + 0 * crntInce].v[0] = v0;
		primitives[index + 0 * crntInce].v[1] = m0;
		primitives[index + 0 * crntInce].v[2] = m1;

		primitives[index + 1 * crntInce].v[0] = m0;
		primitives[index + 1 * crntInce].v[1] = m2;
		primitives[index + 1 * crntInce].v[2] = m1;

		primitives[index + 2 * crntInce].v[0] = v1;
		primitives[index + 2 * crntInce].v[1] = m0;
		primitives[index + 2 * crntInce].v[2] = m2;

		primitives[index + 3 * crntInce].v[0] = m1;
		primitives[index + 3 * crntInce].v[1] = m2;
		primitives[index + 3 * crntInce].v[2] = v2;

	}
	
}

__host__ __device__ glm::vec3 ColorInTex(int texId, glm::vec3**texs, glm::vec2*info, glm::vec2 uv)
{
	int xSize = info[texId].x;
	int ySize = info[texId].y;
	if (uv.x < 0 || uv.y < 0 || uv.x >1 || uv.y >1) return glm::vec3(0, 0, 0);
	float u = (float)(uv.x*(float)xSize);
	float v = (float)(uv.y*(float)ySize);
	int k = u;
	int j = v;
	//if (k == 0 || k == xSize - 1 || j == 0 || j == ySize - 1)//!!!border
	if (true)
	{
		return texs[texId][(j * xSize) + k];
	}
	//else return glm::vec3(0, 0, 0);
	//bilinear filtering - within
	//https://en.wikipedia.org/wiki/Bilinear_interpolation


}

__host__ __device__ glm::vec3 ColorInTexBilinear(int texId, glm::vec3**texs, glm::vec2*info, glm::vec2 uv,float repeat)
{
	//https://en.wikipedia.org/wiki/Bilinear_filtering

	uv *= repeat;

	int xSize = info[texId].x;
	int ySize = info[texId].y;
	//if (uv.x < 0 || uv.y < 0 || uv.x >1 || uv.y >1) return glm::vec3(0, 0, 0);
	uv.x = uv.x < 0 ? (uv.x - int(uv.x) + 1) : (uv.x>1 ? uv.x - (int)uv.x : uv.x);
	uv.y = uv.y < 0 ? (uv.y - int(uv.y) + 1) : (uv.y>1 ? uv.y - (int)uv.y : uv.y);
	float u = (float)(uv.x*(float)xSize - 0.5);
	float v = (float)(uv.y*(float)ySize - 0.5);

	//u = u * tex.size - 0.5;
	//v = v * tex.size - 0.5;
	int x = floor(u);
	int y = floor(v);
	float u_ratio = u - x;
	float v_ratio = v - y;
	float u_opposite = 1 - u_ratio;
	float v_opposite = 1 - v_ratio;

	texs[texId][(y * xSize) + x];
	if (x == 0 || x == xSize - 1 || y == 0 || y == ySize - 1)//!!!border
	{
		return texs[texId][(y * xSize) + x];
	}
	glm::vec3 result = (texs[texId][(y * xSize) + x] * u_opposite + texs[texId][(y * xSize) + x + 1] * u_ratio) * v_opposite +
		(texs[texId][((y + 1) * xSize) + x] * u_opposite + texs[texId][((y + 1) * xSize) + x + 1] * u_ratio) * v_ratio;
	return result;
}

__device__ glm::vec3 getTriangleSurfaceNormal(Triangle tri)
{
	//https://www.opengl.org/wiki/Calculating_a_Surface_Normal
	glm::vec3 p1 = tri.v[0].pos;
	glm::vec3 p2 = tri.v[1].pos;
	glm::vec3 p3 = tri.v[2].pos;

	glm::vec3 origN = glm::normalize(tri.v[0].nor + tri.v[1].nor + tri.v[1].nor);

	glm::vec3 u = p2 - p1;
	glm::vec3 v = p3 - p2;

	glm::vec3 n;
	n.x = u.y*v.z - u.z*v.y;
	n.y = u.z*v.x - u.x*v.z;
	n.z = u.x*v.y - u.y*v.x;
	if (glm::dot(n,origN)<0)
	{
		n = -n;
	}
	return glm::normalize(n);
}

__global__ void kernDispMapping(Triangle* primitives, int crntSize, glm::vec3** texs, glm::vec2* tInfo,float UVrepeat)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (texs == NULL || tInfo == NULL) return;
	if (index < crntSize)
	{
		for (int i = 0; i < 3; i++)
		{
			if (!primitives[index].v[i].DispAdded)
			{
				float disp = 0.1*glm::length(ColorInTexBilinear(0, texs, tInfo, glm::vec2(primitives[index].v[i].tex), UVrepeat));
				primitives[index].v[i].pos += (disp*primitives[index].v[i].nor);//!!! later : normal after displacement mapping.
				//!!! later !!! normal
				primitives[index].v[i].DispAdded = true;
			}
		}
		glm::vec3 normal = getTriangleSurfaceNormal(primitives[index]);
		primitives[index].v[0].nor = normal;
		primitives[index].v[1].nor = normal;
		primitives[index].v[2].nor = normal;
	}
}

__global__ void kernRasterizer(shadeControl sctrl,int w, int h, Fragment * depthbuffer, Triangle*primitives, int bufIdxSize, glm::vec3 lightWorld,glm::vec3 eyeWorld, glm::mat4 allMat, glm::vec3** texs, glm::vec2* tInfo,float UVrepeat)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < bufIdxSize / 3)
	{
		//Scanline
		glm::vec3 tri[3];
		tri[0] = primitives[index].v[0].winPos;
		tri[1] = primitives[index].v[1].winPos;
		tri[2] = primitives[index].v[2].winPos;

		glm::vec3 tex[3];
		tex[0] = primitives[index].v[0].tex;
		tex[1] = primitives[index].v[1].tex;
		tex[2] = primitives[index].v[2].tex;

		glm::vec3 norm[3];
		norm[0] = primitives[index].v[0].nor;
		norm[1] = primitives[index].v[1].nor;
		norm[2] = primitives[index].v[2].nor;
		//!!! currently linear . later interpolation
		glm::vec3 normal = glm::normalize( norm[0]+norm[1]+norm[2]);

		glm::vec3 cols[3];
		cols[0] = primitives[index].v[0].col;
		cols[1] = primitives[index].v[1].col;
		cols[2] = primitives[index].v[2].col;

		glm::vec3 color = glm::normalize(cols[0] + cols[1] + cols[2]);

		AABB triBox = getAABBForTriangle(tri);
		for (int x = triBox.min.x; x <= triBox.max.x; x++)
		{
			for (int y = triBox.min.y; y <= triBox.max.y; y++)
			{
				glm::vec3 bPoint = calculateBarycentricCoordinate(tri, glm::vec2(x, y));
				//!!! later line segment
				//!!! later color interpolation
				bool shade = false;
				bool OnEdge = isBarycentricCoordOnBounds(bPoint,1);
				bool InTri = isBarycentricCoordInBounds(bPoint);
				if (sctrl.Wireframe&&!sctrl.Color)
					shade = OnEdge;
				else
					shade = InTri;
				//if () //On triangle edges : Frame shading
				if (shade) // Inside triangle
				{
					if (x<0 || x>w || y<0 || y>h)
						continue;
					float crntDepth = getZAtCoordinate(bPoint, tri);
					crntDepth *= MAX_DEPTH;
					int orig = atomicMin(&(depthbuffer[x+y*w].depth), (int)crntDepth);
					if (orig >= crntDepth)
					//if (depthbuffer[x + y*w].depth==crntDepth)
					{
						normal = norm[0] * bPoint.x + norm[1] * bPoint.y + norm[2] * bPoint.z;
						color = cols[0] * bPoint.x + cols[1] * bPoint.y + cols[2] * bPoint.z;
						if (sctrl.Normal)		
							depthbuffer[x + y*w].color = normal;
						else{
							glm::vec3 Pos = tri[0] * bPoint.x + tri[1] * bPoint.y + tri[2] * bPoint.z;
							glm::vec3 uv = tex[0] * bPoint.x + tex[1] * bPoint.y + tex[2] * bPoint.z;
							//texture mapping !!! later : repeat, offset...
							if (texs != NULL &&tInfo != NULL && sctrl.Texture)
								color = ColorInTexBilinear(0, texs, tInfo, glm::vec2(uv), UVrepeat);
							glm::vec4 PosWorld = glm::inverse(allMat)* glm::vec4(Pos, 1);
							glm::vec3 lightDir = glm::normalize(lightWorld - glm::vec3(PosWorld));
							float ambient = 0.2*max(dot(glm::normalize(eyeWorld - glm::vec3(PosWorld)), normal), 0.0);
							float diffuse = max(dot(lightDir, normal), 0.0);
							if (sctrl.Wireframe && sctrl.Color && OnEdge)
								depthbuffer[x + y*w].color = glm::vec3(0, 0, 0.7);
							else
								depthbuffer[x + y*w].color = color*(ambient + (1.f - ambient)*diffuse);
						}
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
void rasterizeSetBuffers(obj * mesh, int TessLevel) {
	bufIdxSize = mesh->getBufIdxsize();
	int *bufIdx = mesh->getBufIdx();
	bufTexSize = mesh->getBufTexsize() / 3;
	float * bufTex = mesh->getBufTex();
	vertCount = mesh->getBufPossize() / 3;
	float *bufPos = mesh->getBufPos();
	float *bufNor = mesh->getBufNor();
	float *bufCol = mesh->getBufCol();

	//Copy materials to dev_textures

	int texSize = mesh->textureImages.size()*sizeof(glm::vec3 *);
	int texInfoSize = mesh->textureImages.size()*sizeof(glm::vec2);
	if (texSize > 0 && texInfoSize > 0)
	{
		cudaMalloc((void**)&dev_textures, texSize);
		cudaMalloc((void**)&dev_texInfo, texInfoSize);
		std::vector<glm::vec3*> tempImg;
		std::vector<glm::vec2> tempInfo;
		for (int i = 0; i < mesh->textureImages.size(); i++)
		{
			glm::vec3 * dev_img;
			int imgSize = mesh->textureImages[i].getSize()*sizeof(glm::vec3);
			cudaMalloc((void**)&dev_img, imgSize);
			cudaMemcpy(dev_img, mesh->textureImages[i].pixels, imgSize, cudaMemcpyHostToDevice);
			tempImg.push_back(dev_img);
			tempInfo.push_back(glm::vec2(mesh->textureImages[i].xSize, mesh->textureImages[i].ySize));
		}
		cudaMemcpy(dev_textures, tempImg.data(), texSize, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_texInfo, tempInfo.data(), texInfoSize, cudaMemcpyHostToDevice);
	}


	//
	cudaFree(dev_bufIdx);
	cudaMalloc(&dev_bufIdx, bufIdxSize * sizeof(int));
	cudaMemcpy(dev_bufIdx, bufIdx, bufIdxSize * sizeof(int), cudaMemcpyHostToDevice);

	VertexIn *bufVertex = new VertexIn[vertCount];
	for (int i = 0; i < vertCount; i++) {
		int j = i * 3;
		bufVertex[i].pos = glm::vec3(bufPos[j + 0], bufPos[j + 1], bufPos[j + 2]);
		bufVertex[i].nor = glm::vec3(bufNor[j + 0], bufNor[j + 1], bufNor[j + 2]);
		bufVertex[i].col = glm::vec3(bufCol[j + 0], bufCol[j + 1], bufCol[j + 2]);
		bufVertex[i].tex = glm::vec3(bufTex[j + 0], bufTex[j + 1], bufTex[j + 2]);
	}
	cudaFree(dev_bufVertex);
	cudaMalloc(&dev_bufVertex, vertCount * sizeof(VertexIn));
	cudaMemcpy(dev_bufVertex, bufVertex, vertCount * sizeof(VertexIn), cudaMemcpyHostToDevice);

	//!!!
	cudaFree(dev_bufVtxOut);
	cudaMalloc(&dev_bufVtxOut, vertCount * sizeof(VertexOut));

	cudaFree(dev_bufVtxOut_tess);
	int priCount = vertCount / 3;

	tessLevel = TessLevel;
	lastLevel = tessLevel;
	tessIncre = pow(4, tessLevel);
	cudaMalloc(&dev_bufVtxOut, vertCount * tessIncre* sizeof(VertexOut));
	priCount *= tessIncre;

	cudaFree(dev_primitives);
	cudaMalloc(&dev_primitives, priCount * sizeof(Triangle));
	cudaMemset(dev_primitives, 0, priCount * sizeof(Triangle));

	checkCUDAError("rasterizeSetBuffers");

	int bSize_vtx = 128;
	int bSize_pri = 128;
	dim3 gSize_vtx((vertCount + bSize_vtx - 1) / bSize_vtx);
	int priSize = bufIdxSize / 3;
	dim3 gSize_pri((priSize + bSize_pri - 1) / bSize_pri);
	
	kernVertexShader << <gSize_vtx, bSize_vtx >> >(vertCount, glm::mat4(), M_view, glm::mat4(), dev_bufVertex, dev_bufVtxOut, M_win );

	kernPrimitiveAssembly << <gSize_pri, bSize_pri >> >(dev_primitives, dev_bufIdx, bufIdxSize, dev_bufVtxOut, tessLevel, tessIncre);

}

/**
 * Perform rasterization.
 */
bool lastDisp = true;
float lastUVrepeat = 1;
void rasterize(uchar4 *pbo,glm::mat4 viewMat,glm::mat4 projMat,glm::vec3 eye,int TessLevel,shadeControl sCtrl) {
    int sideLength2d = 8;

	glm::vec3 light(0.3, 0.4, 0.5);

    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
                      (height - 1) / blockSize2d.y + 1);

	M_view = viewMat;
	glm::mat4 M_all = M_win*projMat * M_view * glm::mat4();

	int bSize_vtx = 128;
	int bSize_pri = 128;
	dim3 gSize_vtx((vertCount + bSize_vtx - 1) / bSize_vtx);
	int priSize = bufIdxSize / 3;
	dim3 gSize_pri((priSize + bSize_pri - 1) / bSize_pri);

	tessLevel = TessLevel;
	tessIncre = pow(4, tessLevel);

	kernBufInit << <blockCount2d, blockSize2d >> >(width, height, dev_depthbuffer, dev_framebuffer);

	//kernVertexShader << <gSize_vtx, bSize_vtx >> >(vertCount, glm::mat4(), M_view, projMat, dev_bufVertex, dev_bufVtxOut, M_win );
	if (TessLevel != lastLevel || lastDisp != sCtrl.DispMap || lastUVrepeat != sCtrl.UVrepeat)
	{
		lastDisp = sCtrl.DispMap;
		if (TessLevel > lastLevel)
		{
			cudaMalloc(&dev_primitives, priSize * tessIncre * sizeof(Triangle));
			//kernPrimitiveAssembly << <gSize_pri, bSize_pri >> >(dev_primitives, dev_bufIdx, bufIdxSize, dev_bufVtxOut, tessLevel, tessIncre, projMat * M_view * glm::mat4(), M_win);
		}
		kernPrimitiveAssembly << <gSize_pri, bSize_pri >> >(dev_primitives, dev_bufIdx, bufIdxSize, dev_bufVtxOut, tessLevel, tessIncre);
		lastLevel = tessLevel;

		int tempSize = bufIdxSize / 3;
		int tempIncre = tessIncre;
		for (int i = 0; i < tessLevel; i++)
		{
			kernTessellation_aftPri << <gSize_pri, bSize_pri >> >(dev_primitives, tempSize, tempIncre, projMat * M_view * glm::mat4(), M_win);
			tempSize *= 4;
			tempIncre /= 4;
			priSize = tempSize;
			gSize_pri = dim3((priSize + bSize_pri - 1) / bSize_pri);
		}
		//priSize /= 4;
		if (dev_textures != NULL && dev_texInfo != NULL && sCtrl.DispMap)
			kernDispMapping << <gSize_pri, bSize_pri >> >(dev_primitives, tempSize, dev_textures, dev_texInfo, sCtrl.UVrepeat);
	}
	gSize_pri = dim3((bufIdxSize*tessIncre/3+ bSize_pri - 1) / bSize_pri);
	kernPrimUpdate << <gSize_pri, bSize_pri >> >(dev_primitives, priSize, glm::mat4(), M_view, projMat, M_win);
	kernRasterizer << <gSize_pri, bSize_pri >> >(sCtrl, width, height, dev_depthbuffer, dev_primitives, bufIdxSize*tessIncre, light, eye, M_all, dev_textures, dev_texInfo, sCtrl.UVrepeat);

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

	cudaFree(dev_bufVtxOut);
	dev_bufVtxOut = NULL;

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

	cudaFree(dev_bufVtxOut_tess);
	dev_bufVtxOut_tess = NULL;

	cudaFree(dev_textures);
	//dev_textures = NULL;

	cudaFree(dev_texInfo);
	//dev_texInfo = NULL;

    checkCUDAError("rasterizeFree");
}

//tessellation & geometry : further reading...
//http://prideout.net/blog/?p=48
//http://www.cs.cmu.edu/afs/cs/academic/class/15418-s12/www/lectures/25_micropolygons.pdf
//http://www.eecs.berkeley.edu/~sequin/CS284/PROJ_12/Brandon/Smooth%20GPU%20Tessellation.pdf
//http://http.developer.nvidia.com/GPUGems2/gpugems2_chapter07.html
//http://perso.telecom-paristech.fr/~boubek/papers/PhongTessellation/PhongTessellation.pdf
