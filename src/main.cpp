/**
 * @file      main.cpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"
#define RUN_MIN false

//-------------------------------
//-------------MAIN--------------
//-------------------------------

float theta = 0.78539816339f;// 1.57079632679f;
float phi = 0.0f;//2.35619449019f;
float zoom = 2.0f;
float fovy = 0.785398f;
glm::mat4 camMatrix;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: [obj file]" << endl;
        return 0;
    }

    obj *mesh = new obj();

    {
        objLoader loader(argv[1], mesh);
        mesh->buildBufPoss();
    }

    frame = 0;
    seconds = time(NULL);
    fpstracker = 0;

    // Launch CUDA/GL
    if (init(mesh)) {
        // GLFW main loop
        mainLoop();
    }

    return 0;
}

void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        runCuda();

        time_t seconds2 = time (NULL);

        if (seconds2 - seconds >= 1) {

            fps = fpstracker / (seconds2 - seconds);
            fpstracker = 0;
            seconds = seconds2;
        }

        string title = "CIS565 Rasterizer | " + utilityCore::convertIntToString((int)fps) + " FPS";
        glfwSetWindowTitle(window, title.c_str());

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);
        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

void runCuda() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    dptr = NULL;

    cudaGLMapBufferObject((void **)&dptr, pbo);
	// set up soooooome matrices!
	glm::mat4 ID = glm::mat4();

	//glm::mat4 cam = glm::mat4();
	//cam[0] = glm::vec4(-1.26755321f, 0.896295428f, -0.501001000f, -0.500000000);
	//cam[1] = glm::vec4(0.000000000f, 1.26755321f, 0.708522379f, 0.707106769f);
	//cam[2] = glm::vec4(-1.26755321f, -0.896295428f, 0.501001000f, 0.500000000f);
	//cam[3] = glm::vec4(0.000000000f, 0.0000f, 9.81981945f, 10.0f);

	glm::mat4 tf;
	tf = glm::translate(tf, glm::vec3(0.0f, 0.0f, 0.0f));
	if (RUN_MIN) minRasterizeFirstTry(dptr, tf, camMatrix);//cam);
	else {

		rasterize(dptr, camMatrix);
	}
    cudaGLUnmapBufferObject(pbo);

    frame++;
    fpstracker++;

}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init(obj *mesh) {
    glfwSetErrorCallback(errorCallback);
	computeCameraMatrix();
    if (!glfwInit()) {
        return false;
    }

    width = 816;
	height = 816;
    window = glfwCreateWindow(width, height, "CIS 565 Pathtracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();

    float cbo[] = {
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0
    };
	if (RUN_MIN) {
		minRasterizeSetBuffers(mesh->getBufIdxsize(), mesh->getBufIdx(),
			mesh->getBufPossize() / 3,
			mesh->getBufPos(), mesh->getBufNor(), mesh->getBufCol());
	}
	else {
		rasterizeSetBuffers(mesh->getBufIdxsize(), mesh->getBufIdx(),
			mesh->getBufPossize() / 3,
			mesh->getBufPos(), mesh->getBufNor(), mesh->getBufCol());
		rasterizeSetVariableBuffers();

		// add lights
		std::vector<glm::vec3> positions;
		positions.push_back(glm::vec3(10.0f, 10.0f, 0.0f));
		positions.push_back(glm::vec3(0.0f, -10.0f, 10.0f));

		std::vector<glm::vec3> ambient;
		ambient.push_back(glm::vec3(0.0f, 0.0f, 0.1f));
		ambient.push_back(glm::vec3(0.0f, 0.1f, 0.0f));

		std::vector<glm::vec3> diffuse;
		diffuse.push_back(glm::vec3(0.0f, 0.0f, 1.0f));
		diffuse.push_back(glm::vec3(0.0f, 0.3f, 0.0f));

		std::vector<glm::vec3> specular;
		specular.push_back(glm::vec3(0.0f, 1.0f, 1.0f));
		specular.push_back(glm::vec3(1.0f, 0.3f, 0.0f));

		addLights(positions, ambient, diffuse, specular);

        // add some transformations
        glm::mat4 ID = glm::mat4();
        std::vector<glm::mat4> transformations;
        transformations.push_back(ID);
		transformations.push_back(glm::translate(ID, glm::vec3(-0.4f, -0.5f, 0.05f)));
        //transformations.push_back(glm::translate(ID, glm::vec3(1.0f, 0.0f, 0.0f)));
		//transformations.push_back(glm::translate(ID, glm::vec3(-1.0f, 0.0f, 0.0f)) * 
		//	glm::scale(ID, glm::vec3(2.0f, 2.0f, 2.0f)));
        setupInstances(transformations);

		// set up tiling!
		setupTiling();

		// anti aliasing toggle
		//enableAA();
	}

    GLuint passthroughProgram;
    passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    return true;
}

void initPBO() {
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject(pbo);

}

void initCuda() {
    // Use device with highest Gflops/s
    cudaGLSetGLDevice(0);

    if (RUN_MIN) minRasterizeInit(width, height);
	else {
		rasterizeInit(width, height);
	}

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA,
                  GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}


GLuint initShader() {
    const char *attribLocations[] = { "Position", "Tex" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

//-------------------------------
//---------CLEANUP STUFF---------
//-------------------------------

void cleanupCuda() {
    if (pbo) {
        deletePBO(&pbo);
    }
    if (displayImage) {
        deleteTexture(&displayImage);
    }
}

void deletePBO(GLuint *pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGLUnregisterBufferObject(*pbo);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint *tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void shut_down(int return_code) {
	if (RUN_MIN) minRasterizeFree();
	else rasterizeFree();
    cudaDeviceReset();
#ifdef __APPLE__
    glfwTerminate();
#endif
    exit(return_code);
}

//------------------------------
//-------GLFW CALLBACKS---------
//------------------------------

void errorCallback(int error, const char *description) {
    fputs(description, stderr);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		//printf("theta %f phi %f zoom %f fovy %f\n", theta, phi, zoom, fovy);
		switch (key) {
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_DOWN:  
			theta += 0.1f;
			computeCameraMatrix();
			break;
		case GLFW_KEY_UP:    
			theta -= 0.1f;
			computeCameraMatrix();
			break;
		case GLFW_KEY_RIGHT: 
			phi += 0.1f;
			computeCameraMatrix();
			break;
		case GLFW_KEY_LEFT:
			phi -= 0.1f;
			computeCameraMatrix();
			break;
		case GLFW_KEY_Z:
			zoom += 0.5f;
			computeCameraMatrix();
			break;
		case GLFW_KEY_X:
			zoom -= 0.5f;
			computeCameraMatrix();
			break;
		case GLFW_KEY_G:
			fovy += 0.1f;
			computeCameraMatrix();
			break;
		case GLFW_KEY_F:
			fovy -= 0.1f;
			computeCameraMatrix();
			break;
		case GLFW_KEY_C:
			glm::vec3 cameraPos;
			cameraPos.x = zoom * sin(phi) * sin(theta);
			cameraPos.y = zoom * cos(theta);
			cameraPos.z = zoom * cos(phi) * sin(theta);
			cout << cameraPos.x << " " << cameraPos.y << " " << cameraPos.z << endl;
		}
	}
}



void computeCameraMatrix() {
    // Projection matrix : 45° Field of View, 1:1 ratio, display range : 0.1 unit <-> 100 units
	glm::mat4 projection = glm::perspective(fovy, (float)width / (float)height, 0.1f, 100.0f);
    // compute position: http://www.cs.cmu.edu/~barbic/camera.html
    if (theta < 0.0f) theta = 0.01f;
	if (theta > 3.141592653589) theta = 3.14;

    glm::vec3 cameraPos;
	cameraPos.x = zoom * sin(phi) * sin(theta);
	cameraPos.y = zoom * cos(theta);
	cameraPos.z = zoom * cos(phi) * sin(theta);
	//cout << "camera pos is " << cameraPos[0] << " " << cameraPos[1] << " " << cameraPos[2] << endl;
    // view matrix
    glm::mat4 view = glm::lookAt(
		cameraPos, // Camera position in World Space
        glm::vec3(0, 0, 0), // camera lookAt
        glm::vec3(0, 1, 0)  // Head is up
        );
	//cout << projection[0][0] << " " << projection[1][0] << " " << projection[2][0] << " " << projection[3][0] << endl;
	//cout << projection[0][1] << " " << projection[1][1] << " " << projection[2][1] << " " << projection[3][1] << endl;
	//cout << projection[0][2] << " " << projection[1][2] << " " << projection[2][2] << " " << projection[3][2] << endl;
	//cout << projection[0][3] << " " << projection[1][3] << " " << projection[2][3] << " " << projection[3][3] << endl;
	//cout << endl;
	//projection[2][3] = 1.0f;
	//cout << view[0][0] << " " << view[0][1] << " " << view[0][2] << " " << view[0][3] << endl;
	//cout << view[1][0] << " " << view[1][1] << " " << view[1][2] << " " << view[1][3] << endl;
	//cout << view[2][0] << " " << view[2][1] << " " << view[2][2] << " " << view[2][3] << endl;
	//cout << view[3][0] << " " << view[3][1] << " " << view[3][2] << " " << view[3][3] << endl;
	//cout << endl;
	//cout << projection[0][0] << " " << projection[1][0] << " " << projection[2][0] << " " << projection[3][0] << endl;
	//cout << projection[0][1] << " " << projection[1][1] << " " << projection[2][1] << " " << projection[3][1] << endl;
	//cout << projection[0][2] << " " << projection[1][2] << " " << projection[2][2] << " " << projection[3][2] << endl;
	//cout << projection[0][3] << " " << projection[1][3] << " " << projection[2][3] << " " << projection[3][3] << endl;
	//cout << endl;
	
	camMatrix = projection * view;
	
	//cout << camMatrix[0][0] << " " << camMatrix[1][0] << " " << camMatrix[2][0] << " " << camMatrix[3][0] << endl;
	//cout << camMatrix[0][1] << " " << camMatrix[1][1] << " " << camMatrix[2][1] << " " << camMatrix[3][1] << endl;
	//cout << camMatrix[0][2] << " " << camMatrix[1][2] << " " << camMatrix[2][2] << " " << camMatrix[3][2] << endl;
	//cout << camMatrix[0][3] << " " << camMatrix[1][3] << " " << camMatrix[2][3] << " " << camMatrix[3][3] << endl;
}

