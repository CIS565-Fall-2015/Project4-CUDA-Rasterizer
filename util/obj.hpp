/**
 * @file      obj.hpp
 * @brief     An OBJ mesh loading library. Part of Yining Karl Li's OBJCORE.
 * @authors   Yining Karl Li
 * @date      2012
 * @copyright Yining Karl Li
 */

#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>

using namespace std;

class obj {
private:
    vector<glm::vec4> points;
    vector<vector<int> > faces;
    vector<vector<int> > facenormals;
    vector<vector<int> > facetextures;
    vector<float *> faceboxes;  //bounding boxes for each face are stored in vbo-format!
    vector<glm::vec4> normals;
    //vector<glm::vec4> texturecoords;
	vector<glm::vec2> texturecoords;
    int vbosize;
    int nbosize;
    int cbosize;
    int ibosize;
	int tbosize;
    float *vbo;
    float *nbo;
    float *cbo;
	float *tbo;
    int *ibo;

    float *boundingbox;
    float top;
    glm::vec3 defaultColor;
    float xmax;
    float xmin;
    float ymax;
    float ymin;
    float zmax;
    float zmin;
    bool maxminSet;

public:
	glm::vec3 diffuse_color;
	glm::vec3 ambient_color;
	glm::vec3 specular_color;
	float specular_exponent;
	std::string diffuse_texture_file;
	std::string specular_texture_file;
	int mDiffTexID;
	int mSpecTexID;

	int diffuse_width, diffuse_height;
	int specular_width, specular_height;
	std::vector<glm::vec3> diffuse_tex;
	std::vector<glm::vec3> specular_tex;

public:
    obj();
    ~obj();

    //-------------------------------
    //-------Mesh Operations---------
    //-------------------------------
    void buildBufPoss();
    void addPoint(glm::vec3);
    void addFace(vector<int>);
    void addNormal(glm::vec3);
    //void addTextureCoord(glm::vec3);
	void addTextureCoord(glm::vec2);
    void addFaceNormal(vector<int>);
    void addFaceTexture(vector<int>);
    void compareMaxMin(float, float, float);
    bool isConvex(vector<int>);
    void recenter();

    //-------------------------------
    //-------Get/Set Operations------
    //-------------------------------
    float *getBoundingBox();    //returns vbo-formatted bounding box
    float getTop();
    void setColor(glm::vec3);
    glm::vec3 getColor();
    float *getBufPos();
    float *getBufCol();
    float *getBufNor();
	float *getBufTex();
    int *getBufIdx();
    int getBufPossize();
    int getBufNorsize();
    int getBufIdxsize();
    int getBufColsize();
    vector<glm::vec4> *getPoints();
    vector<vector<int> > *getFaces();
    vector<vector<int> > *getFaceNormals();
    vector<vector<int> > *getFaceTextures();
    vector<glm::vec4> *getNormals();
    //vector<glm::vec4> *getTextureCoords();
	vector<glm::vec2> *getTextureCoords();
    vector<float *> *getFaceBoxes();



	
};
