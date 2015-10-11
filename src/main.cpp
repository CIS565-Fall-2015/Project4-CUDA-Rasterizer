/**
 * @file      main.cpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"
#include "image.h"
#include <ctime>
#include "Scene.h"

static std::string startTimeString;
Scene *scene;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

std::string currentTimeString() {
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: [obj file]" << endl;
        return 0;
    }

    startTimeString = currentTimeString();

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
        mainLoop(mesh);
    }

    return 0;
}

void mainLoop(obj *mesh) {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
//        rasterizeInit(width, height);
//        setPrimitiveBuffer(mesh->getBufIdxsize());
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

void saveImage() {
    //float samples = iteration;
    // output image file
    image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = scene->imageColor[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix));// / samples);
        }
    }

    std::string filename = "Rasterize";
    std::ostringstream ss;
    ss << filename << "." << startTimeString;
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

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
    window = glfwCreateWindow(width, height, "SANCHIT GARG : CIS 565 Rasterizer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    //Mouse callbacks
    glfwSetCursorPosCallback(window, mouseCursorPosCallBack);
    glfwSetMouseButtonCallback(window, mouseButtonCallBack);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    scene = new Scene(width, height);

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

	if(action == GLFW_PRESS)
	{
		switch(key)
		{
			case GLFW_KEY_ESCAPE:
				delete(scene->imageColor);
				glfwSetWindowShouldClose(window, GL_TRUE);
				break;

			case GLFW_KEY_SPACE:
				saveImage();
				break;

			case GLFW_KEY_M:
				scene->updateRenderMode();
				break;

			case GLFW_KEY_F:
				scene->setDefaultCamera();
				break;

			case GLFW_KEY_A:
				scene->toggleAntiAliasing();
				break;

			case GLFW_KEY_B:
				scene->toggleBackFaceCulling();
				break;

			case GLFW_KEY_S:
//				scene->toggleScissorTest();
				break;
			default:
				break;
		}
	}
}

//Mouse Callbacks
//Reference: http://www.glfw.org/docs/latest/input.html#input_mouse_button
void mouseButtonCallBack(GLFWwindow *window, int button, int action, int mods)
{
	if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		scene->mouse.left = true;
		scene->mouse.clicked = true;
	}

	else if(button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
	{
		scene->mouse.right = true;
		scene->mouse.clicked = true;
	}

	else if(button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
	{
		scene->mouse.middle = true;
		scene->mouse.clicked = true;
	}

	else if(action == GLFW_RELEASE)
	{
		scene->mouse.dragging = false;
		scene->mouse.clicked = false;
		scene->mouse.left = false;
		scene->mouse.right = false;
		scene->mouse.middle = false;
	}
}

void mouseCursorPosCallBack(GLFWwindow *window, double x, double y)
{
	//Set the first click position
	//Second time the function is called, it means the user is dragging
	if(!scene->mouse.clicked)
	{
		scene->mouse.pos.x = float(x);
		scene->mouse.pos.y = float(y);
		scene->mouse.dragging = true;
	}

	else if(scene->mouse.dragging)
	{
		//Left Rotate
		if(scene->mouse.left)
		{
			glm::vec2 angle((float(y) - scene->mouse.pos.y), (float(x) - scene->mouse.pos.x));
			angle *= 0.003f;
			scene->rotateModel(glm::vec3(angle.x, angle.y ,0));
			scene->mouse.pos.x = x;
			scene->mouse.pos.y = y;
		}

		//Right Zoom
		else if(scene->mouse.right)
		{
			float movedX = (float(x) - scene->mouse.pos.x) * 0.003f;

			scene->moveModel(glm::vec3(0, 0, movedX));

			scene->mouse.pos.x = x;
			scene->mouse.pos.y = y;
		}

		//Middle Pan
		else if(scene->mouse.middle)
		{
			glm::vec2 move((float(x) - scene->mouse.pos.x), (scene->mouse.pos.y - float(y)));
			move *= 0.003f;
			scene->moveModel(glm::vec3(move.x,move.y,0));
			scene->mouse.pos.x = x;
			scene->mouse.pos.y = y;
		}
	}
}
