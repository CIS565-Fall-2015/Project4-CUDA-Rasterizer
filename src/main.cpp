/**
 * @file      main.cpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"
#include <chrono>
#include "sceneStructs.h"
#include "glm/gtx/rotate_vector.hpp"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

MVP mvp;
bool useScanline = false;

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
	if (useScanline){
		rasterize(dptr);
	}
	else {
		rasterizeTile(dptr);
	}
    cudaGLUnmapBufferObject(pbo);

    frame++;
    fpstracker++;

}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void calculateMVP(MVP &mvp, bool flush){
	glm::vec3 camDirection = glm::normalize(mvp.camLookAt - mvp.camPosition);
	mvp.camUp = glm::cross(mvp.camRight, camDirection);
	mvp.view = glm::lookAt(mvp.camPosition, mvp.camLookAt, mvp.camUp);
	mvp.projection = glm::perspective(mvp.fov, 1.0f, -mvp.nearPlane, -mvp.farPlane);
	mvp.mvp = mvp.projection*mvp.view*mvp.model;
	if (flush){
		flushDepthBuffer();
	}
}

void calculateMVP(MVP &mvp){
	calculateMVP(mvp, true);
}

static void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos){
	if (mvp.initPos){
		mvp.mouseDownX = xpos;
		mvp.mouseDownY = ypos;
		mvp.initPos = false;
	}
	else {
		if (mvp.mouseLeftDown){
			// Rotate; move camera around lookAt
			glm::vec3 inverseLook = mvp.camPosOld - mvp.camLookAt;
			float r = glm::length(inverseLook);
			float shiftX = (mvp.mouseDownX - (float)xpos)*0.25f;
			float shiftY = (mvp.mouseDownY - (float)ypos)*0.25f;
			glm::vec3 newInverseLook = glm::rotateY(inverseLook, (float)atan2(shiftX/(TWO_PI*r), glm::length(inverseLook)));
			mvp.camRight = glm::normalize(glm::rotateY(glm::vec3(inverseLook.x, 0, inverseLook.z), (float)(PI / 2)));
			newInverseLook = glm::rotate(newInverseLook, (float)atan2(shiftY / (TWO_PI*r), glm::length(newInverseLook)), mvp.camRight);
			mvp.camPosition = mvp.camLookAt + newInverseLook;
			calculateMVP(mvp);
		}
		else if (mvp.mouseRightDown){
			// Pan; move camera along camera right & up
			float shiftX = (mvp.mouseDownX - (float)xpos)*0.01f;
			float shiftZ = ((float)ypos - mvp.mouseDownY)*0.01f;
			glm::vec3 newPos = mvp.camPosOld + mvp.camRight*shiftX;
			glm::vec3 look = mvp.camLookAt - newPos;
			newPos = newPos + mvp.camUp*shiftZ;
			look = mvp.camLookAt - newPos;
			if (glm::length(look) > 1){
				mvp.camPosition = newPos;
				mvp.camLookAt = mvp.camLookAtOld + newPos - mvp.camPosOld;
				/*
				mvp.camPosition = mvp.camPosOld + mvp.camRight*shiftX;
				glm::vec3 camDirection = glm::normalize(mvp.camLookAt - mvp.camPosition);
				mvp.camPosition = mvp.camPosition + camDirection*shiftZ;
				*/
				calculateMVP(mvp);
			}
		}
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS){
		mvp.mouseLeftDown = true;
		mvp.initPos = true;
		mvp.camLookAtOld = mvp.camLookAt;
		mvp.camPosOld = mvp.camPosition;
		mvp.camRightOld = mvp.camRight;
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS){
		mvp.mouseRightDown = true;
		mvp.initPos = true;
		mvp.camLookAtOld = mvp.camLookAt;
		mvp.camPosOld = mvp.camPosition;
	}
	else if (action == GLFW_RELEASE){
		mvp.mouseLeftDown = false;
		mvp.mouseRightDown = false;
		mvp.initPos = false;
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
	// Zoom
	if (yoffset > 0){
		glm::vec3 look = mvp.camLookAt - mvp.camPosition;
		if (glm::length(look)>1){
			glm::vec3 camDirection = glm::normalize(look);
			mvp.camPosition = mvp.camPosition + camDirection*0.1f;
			calculateMVP(mvp);
		}
	}
	else if (yoffset < 0){
		glm::vec3 camDirection = glm::normalize(mvp.camPosition - mvp.camLookAt);
		mvp.camPosition = mvp.camPosition + camDirection*0.1f;
		calculateMVP(mvp);
	}
}

bool init(obj *mesh) {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        return false;
    }

    width = 800;
    height = 800;
    window = glfwCreateWindow(width, height, "CIS 565 Pathtracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, keyCallback);
	// Mouse control
	glfwSetCursorPosCallback(window, cursor_pos_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize other stuff
	mvp.scissor.max = glm::vec2(width * 0.75, height * 0.75);
	mvp.scissor.min = glm::vec2(width * 0.25, height * 0.25);
	mvp.shadeMode = 0;

	mvp.camPosition = glm::vec3(0,0,3);
	mvp.camLookAt = glm::vec3(0, 0, 0);
	mvp.camRight = glm::vec3(1, 0, 0);
	mvp.fov = 45.0f;
	calculateMVP(mvp, false);

    initVAO();
    initTextures();
    initCuda();
    initPBO();

    float cbo[] = {
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0
    };
	rasterizeSetBuffers(mesh->getBufIdxsize(), mesh->getBufIdx(),
		mesh->getBufPossize() / 3,
		mesh->getBufPos(), mesh->getBufNor(), mesh->getBufCol());

	rasterizeTileInit();

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

	rasterizeInit(width, height, &mvp);

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

    //glUseProgram(program);
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
	rasterizeFree();
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
		switch (key) {
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_SPACE:
			mvp.camPosition = glm::vec3(0, 0, 3);
			mvp.camLookAt = glm::vec3(0, 0, 0);
			mvp.camRight = glm::vec3(1, 0, 0);
			mvp.fov = 45.0f;
			calculateMVP(mvp);
			break;
		case GLFW_KEY_S:
			mvp.doScissor = !mvp.doScissor;
			flushDepthBuffer();
			break;
		case GLFW_KEY_N:	// Shade normal
			mvp.shadeMode = 1;
			flushDepthBuffer();
			break;
		case GLFW_KEY_R:	// Reset shading to color shading
			mvp.shadeMode = 0;
			flushDepthBuffer();
			break;
		case GLFW_KEY_G:	// Geometry shader
			mvp.geomShading = !mvp.geomShading;
			flushDepthBuffer();
			break;
		case GLFW_KEY_P:	// Point shader
			mvp.pointShading = !mvp.pointShading;
			flushDepthBuffer();
			break;
		case GLFW_KEY_B:	// Backface culling
			mvp.culling = !mvp.culling;
			flushDepthBuffer();
			break;
		case GLFW_KEY_L:	// Pipeline switch
			useScanline = !useScanline;
			flushDepthBuffer();
			break;
		}
	}
}