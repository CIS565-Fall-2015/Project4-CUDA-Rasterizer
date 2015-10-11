/**
 * @file      main.cpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"
#include <cuda.h>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/rotate_vector.hpp>

static Cam cam;
static bool camIsMobile;
static glm::vec2 oldCursorPos;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

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
    rasterize(dptr, cam);
    cudaGLUnmapBufferObject(pbo);

    frame++;
    fpstracker++;

}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

bool init(obj *mesh) {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        return false;
    }

    width = 800;
    height = 800;

	cam.width = width;
	cam.height = height;
	cam.pos = glm::vec3(0.0f, 0.0f, 5.0f);
	cam.focus = glm::vec3(0.0f, 0.0f, 0.0f);
	cam.up = glm::vec3(0.0f, -1.0f, 0.0f);
	cam.fovy = 45.0f * glm::pi<float>() / 180.0f;
	cam.zNear = 0.1f;
	cam.zFar = 100.0f;
	cam.aspect = 1.0f;

    window = glfwCreateWindow(width, height, "CIS 565 Pathtracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
	glfwSetScrollCallback(window, scrollCallback);
	glfwSetCursorPosCallback(window, cursorCallback);
	glfwSetMouseButtonCallback(window, mouseCallback);

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
    rasterizeSetBuffers(mesh->getBufIdxsize(), mesh->getBufIdx(),
            mesh->getBufPossize() / 3,
            mesh->getBufPos(), mesh->getBufNor(), mesh->getBufCol());

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

    rasterizeInit(width, height);

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

	glm::vec3 camView = glm::normalize(cam.focus - cam.pos);
	glm::vec3 camUp = cam.up;
	glm::vec3 camRight = glm::cross(camView, camUp);

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
	}
	else if (key == GLFW_KEY_DOWN && action == GLFW_PRESS){
		cam.pos += camUp * 0.1f;
		cam.focus += camUp * 0.1f;
	}
	else if (key == GLFW_KEY_UP && action == GLFW_PRESS){
		cam.pos += camUp * -0.1f;
		cam.focus += camUp * -0.1f;
	}
	else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS){
		cam.pos += camRight * -0.1f;
		cam.focus += camRight * -0.1f;
	}
	else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS){
		cam.pos += camRight * 0.1f;
		cam.focus += camRight * 0.1f;
	}
	// Reset
	else if (key == GLFW_KEY_R && action == GLFW_PRESS){
		cam.pos = glm::vec3(0.0, 0.0, 4.0);
		cam.focus = glm::vec3(0.0);
	}
}

void scrollCallback(GLFWwindow *window, double x_offset, double y_offset){
	glm::vec3 camView = glm::normalize(cam.focus - cam.pos);
	cam.pos += camView * (float)y_offset;
	cam.focus += camView * (float)y_offset;
}

void mouseCallback(GLFWwindow *window, int button, int action, int mods){
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS){
		camIsMobile = true;
	}
	else {
		camIsMobile = false;
	}
}

void cursorCallback(GLFWwindow *window, double x_pos, double y_pos){
	glm::vec3 camView = glm::normalize(cam.focus - cam.pos);
	glm::vec3 camUp = cam.up;
	glm::vec3 camRight = glm::cross(camView, camUp);

	glm::vec3 rotatedView;

	if (camIsMobile){
		float x_diff = x_pos - oldCursorPos[0];
		float y_diff = y_pos - oldCursorPos[1];

		rotatedView = glm::rotate(camView, y_diff/100.0f, camRight);
		rotatedView = glm::rotate(rotatedView, x_diff/100.0f, camUp);
		cam.up = camUp;
		cam.focus = cam.pos + glm::normalize(rotatedView);
	}
	oldCursorPos[0] = x_pos;
	oldCursorPos[1] = y_pos;
}