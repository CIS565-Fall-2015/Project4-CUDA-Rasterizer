/**
 * @file      main.cpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"

#include <glm/gtx/rotate_vector.hpp>

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
    rasterize(dptr);
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
    window = glfwCreateWindow(width, height, "CIS 565 Pathtracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
	glfwSetCursorPosCallback(window, mouseMoveCallback);
	glfwSetMouseButtonCallback(window, mousePressCallback);
	glfwSetScrollCallback(window, mouseScrollCallback);

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

	scene = new Scene();
    rasterizeInit(width, height, scene);

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
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

void mouseMoveCallback(GLFWwindow *window, double x, double y) {
	if (!scene->mouseState.initialPositionsSet) {
		// Have to set a baseline position for our offsets to base off of
		scene->mouseState.x = x;
		scene->mouseState.y = y;
		scene->mouseState.initialPositionsSet = true;
	}
	else {
		if (scene->mouseState.leftPressed) {
			// Holding the left mouse rotates the camera.
			glm::vec3 inverseLookAt = scene->camera.position - scene->camera.lookAt;
			float xOffset = 0.005f * (scene->mouseState.x - (float)x);
			float yOffset = 0.005f * (scene->mouseState.y - (float)y);
			glm::vec3 tempInverseLookAt = glm::rotateY(inverseLookAt, glm::atan((float)(xOffset / 
				(TWO_PI * glm::length(inverseLookAt))), glm::length(inverseLookAt)));

			scene->camera.right = glm::normalize(glm::rotateY(glm::vec3(inverseLookAt.x, 0.0f, inverseLookAt.z), (float)(PI / 2.0)));
			scene->camera.position = scene->camera.lookAt + glm::rotate(tempInverseLookAt, glm::atan(yOffset /
				(float)(TWO_PI * glm::length(inverseLookAt)), glm::length(tempInverseLookAt)), scene->camera.right);
			scene->updateModelView();
		}
		else if (scene->mouseState.middlePressed) {
			// Holding middle will translate the camera
			float xOffset = 0.0001f * (scene->mouseState.x - (float)x);
			float yOffset = 0.0001f * ((float)y - scene->mouseState.y);
			glm::vec3 tempPosition = (scene->camera.position + scene->camera.right * xOffset) + (scene->camera.up * yOffset);
			glm::vec3 tempLookAt = scene->camera.lookAt - tempPosition;

			if (glm::length(tempLookAt) > 1.0f) {
				// Restrict amount user can pan to stop going through model
				scene->camera.lookAt = scene->camera.lookAt + tempPosition - scene->camera.position;
				scene->camera.position = tempPosition;
				scene->updateModelView();
			}
		}
	}
}

void mousePressCallback(GLFWwindow *window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		// Mark left mouse button as held
		scene->mouseState.leftPressed = true;
		scene->mouseState.initialPositionsSet = false;
	}
	else if (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS) {
		// Mark right mouse button as held
		scene->mouseState.middlePressed = true;
		scene->mouseState.initialPositionsSet = false;
	}
	else if (action == GLFW_RELEASE) {
		// Reset the flags for which button is held
		scene->mouseState.leftPressed = false;
		scene->mouseState.middlePressed = false;
		scene->mouseState.initialPositionsSet = true;
	}
}

void mouseScrollCallback(GLFWwindow *window, double x, double y) {
	if (y > 0.0) {
		glm::vec3 tempLookAt = scene->camera.lookAt - scene->camera.position;
		if (glm::length(tempLookAt) > 1.0) {
			// Stop user from zooming in too close and through/past the model
			scene->camera.position += glm::normalize(tempLookAt) * 0.1f;
			scene->updateModelView();
		}
	}
	else if (y < 0.0) {
		scene->camera.position = scene->camera.position + (glm::normalize(scene->camera.position - scene->camera.lookAt) * 0.1f);
		scene->updateModelView();
	}
}
