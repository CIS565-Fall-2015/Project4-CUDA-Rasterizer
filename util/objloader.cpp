/**
 * @file      objloader.cpp
 * @brief     An OBJ mesh loading library. Part of Yining Karl Li's OBJCORE.
 * @authors   Yining Karl Li
 * @date      2012
 * @copyright Yining Karl Li
 */

#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>
#include <glm/glm.hpp>
#include "objloader.hpp"

using namespace std;

objLoader::objLoader(string filename, obj *newMesh) {

    geomesh = newMesh;
    cout << "Loading OBJ File: " << filename << endl;
    ifstream fp_in;
    char *fname = (char *)filename.c_str();
    fp_in.open(fname);
    if (fp_in.is_open()) {
        while (fp_in.good() ) {
            string line;
            getline(fp_in, line);
            if (line.empty()) {
                line = "42";
            }
            istringstream liness(line);
            if (line[0] == 'v' && line[1] == 't') {
                string v;
                string x;
                string y;
                string z;
                getline(liness, v, ' ');
                getline(liness, x, ' ');
                getline(liness, y, ' ');
                getline(liness, z, ' ');
                geomesh->addTextureCoord(glm::vec3(::atof(x.c_str()), ::atof(y.c_str()), ::atof(z.c_str())));
            } else if (line[0] == 'v' && line[1] == 'n') {
                string v;
                string x;
                string y;
                string z;
                getline(liness, v, ' ');
                getline(liness, x, ' ');
                getline(liness, y, ' ');
                getline(liness, z, ' ');
                geomesh->addNormal(glm::vec3(::atof(x.c_str()), ::atof(y.c_str()), ::atof(z.c_str())));
            } else if (line[0] == 'v') {
                string v;
                string x;
                string y;
                string z;
                getline(liness, v, ' ');
                getline(liness, x, ' ');
                getline(liness, y, ' ');
                getline(liness, z, ' ');
                geomesh->addPoint(glm::vec3(::atof(x.c_str()), ::atof(y.c_str()), ::atof(z.c_str())));
            } else if (line[0] == 'f') {
                string v;
                getline(liness, v, ' ');
                string delim1 = "//";
                string delim2 = "/";
                if (std::string::npos != line.find("//")) {
                    //std::cout << "Vertex-Normal Format" << std::endl;
                    vector<int> pointList;
                    vector<int> normalList;
                    while (getline(liness, v, ' ')) {
                        istringstream facestring(v);
                        string f;
                        getline(facestring, f, '/');
                        pointList.push_back(::atoi(f.c_str()) - 1);

                        getline(facestring, f, '/');
                        getline(facestring, f, ' ');
                        normalList.push_back(::atoi(f.c_str()) - 1);

                    }
                    geomesh->addFace(pointList);
                    geomesh->addFaceNormal(normalList);
                } else if (std::string::npos != line.find("/")) {
                    vector<int> pointList;
                    vector<int> normalList;
                    vector<int> texturecoordList;
                    while (getline(liness, v, ' ')) {
                        istringstream facestring(v);
                        string f;
                        int i = 0;
                        while (getline(facestring, f, '/')) {
                            if (i == 0) {
                                pointList.push_back(::atoi(f.c_str()) - 1);
                            } else if (i == 1) {
                                texturecoordList.push_back(::atoi(f.c_str()) - 1);
                            } else if (i == 2) {
                                normalList.push_back(::atoi(f.c_str()) - 1);
                            }
                            i++;
                        }
                    }
                    geomesh->addFace(pointList);
                    geomesh->addFaceNormal(normalList);
                    geomesh->addFaceTexture(texturecoordList);
                } else {
                    string v;
                    vector<int> pointList;
                    while (getline(liness, v, ' ')) {
                        pointList.push_back(::atoi(v.c_str()) - 1);
                    }
                    geomesh->addFace(pointList);
                    //std::cout << "Vertex Format" << std::endl;
                }
            }
        }
        cout << "Loaded " << geomesh->getFaces()->size() << " faces, " << geomesh->getPoints()->size() << " vertices from " << filename << endl;
    } else {
        cout << "ERROR: " << filename << " could not be found" << endl;
        exit(EXIT_FAILURE);
    }
	cout<<"finish"<<endl;
}

/*objLoader::objLoader(string fileName,obj* m){
	string input;
	geomesh=m;
	int index;
	ifstream inObj;
	inObj.open(fileName);
	while(inObj>>input){
		if(input=="v"){
			glm::vec3 vec;
			inObj>>vec.x;
			inObj>>vec.y;
			inObj>>vec.z;
			geomesh->addPoint(vec);
		}
		else if(input=="vn"){
			glm::vec3 vec;
			inObj>>vec.x;
			inObj>>vec.y;
			inObj>>vec.z;
			geomesh->addNormal(vec);
		}
		else if(input=="vt"){}
		else if(input=="f"){
			vector<int> ind,tex,norIdx;
			for(int j=0;j<3;j++){
				int i=0;
				inObj>>input;
				int count=0;
				index=0;
				while(input[count]<='9'&&input[count]>='0') count++;
				for(i=0;i<count;i++)
					index+=(input[i]-'0')*pow(10.0,count-1-i);
				ind.push_back(index-1);
				i++;

				if(i<input.size()){//texture
					int count=i;
					index=0;
					while(input[count]<='9'&&input[count]>='0') count++;
					for(;i<count;i++)
						index+=(input[i]-'0')*pow(10.0,count-1-i);
					tex.push_back(index-1);
					i++;
				}
				if(i<input.size()){//normal
					int count=i;
					index=0;
					while(input[count]<='9'&&input[count]>='0') count++;
					for(;i<count;i++)
						index+=(input[i]-'0')*pow(10.0,count-1-i);
					norIdx.push_back(index-1);
					i++;
				}
			}
			geomesh->addFace(ind);
			geomesh->addFaceTexture(tex);
			geomesh->addFaceNormal(norIdx);
		}
	}
	inObj.close();
	cout<<"Faces:"<<geomesh->getFaces()->size()<<endl;
	cout<<"Points:"<<geomesh->getPoints()->size()<<endl;
}*/

objLoader::~objLoader() {
}

obj *objLoader::getMesh() {
    return geomesh;
}
