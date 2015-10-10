/**
 * @file      main.cpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/transform.hpp"
#include "GLFW/glfw3.h"


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

	delete mesh;
    return 0;
}

void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

		//MY
		setupCamera();

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

//translate x,y, angle, scale for keyboard operations
float scale = 0.15f;
//float scale = 0.05f;
float x_trans = 0.0f, y_trans = 0.0f, z_trans = -10.0f;
float x_angle = 0.0f, y_angle = 0.0f;


glm::mat4 M_model;
glm::mat4 M_view;
glm::mat4 M_perspective;

glm::mat4 inv_trans_M_view;	//for normal transformation


//lights



void runCuda() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    dptr = NULL;

	//int xpos, ypos;
	//glfwGetMousePos(&xpos, &ypos);

    cudaGLMapBufferObject((void **)&dptr, pbo);

	vertexShader(M_perspective * M_view * M_model, M_view*M_model, inv_trans_M_view);

    rasterize(dptr);



    cudaGLUnmapBufferObject(pbo);

    frame++;
    fpstracker++;

}

//MY
void setupCamera()
{

	//model-view
	//M_model = glm::mat4(1.0f);
	//M_model = glm::translate(M_model, glm::vec3(0, 0, z_trans));

	//x_angle, y_angle : radium
	M_model = glm::translate(glm::vec3(x_trans, y_trans, z_trans))
		* glm::rotate(x_angle, glm::vec3(1.0f, 0.0f, 0.0f))
		* glm::rotate(y_angle, glm::vec3(0.0f, 1.0f, 0.0f));
	
	M_view = glm::mat4(1.0f);

	//projection
	//left, right, bottom, top, near, far
	M_perspective = glm::frustum<float>(-scale * ((float)width) / ((float)height),
		scale * ((float)width / (float)height),
		-scale, scale, 1.0, 1000.0);

	inv_trans_M_view = glm::transpose(glm::inverse(M_view * M_model));
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

	//MY Mouse Control
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetCursorPosCallback(window, mouseMotionCallback);


	glfwSetScrollCallback(window,mouseWheelCallback);
	/////

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
			mesh->getFaces()->size(),
            mesh->getBufPos(), mesh->getBufNor(), mesh->getBufCol()
			, (mesh->getTextureCoords())->size() > 0, mesh->getBufTex());


	initTextureData(mesh->diffuse_tex.size() > 0, mesh->diffuse_width, mesh->diffuse_height, mesh->diffuse_tex.data(),
		mesh->specular_tex.size() > 0, mesh->specular_width, mesh->specular_height, mesh->specular_tex.data(),
		mesh->ambient_color,mesh->diffuse_color,mesh->specular_color,mesh->specular_exponent);




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
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
	else if (key == GLFW_KEY_W && action == GLFW_PRESS)
	{
		changeGeomMode();
	}
	else if (key == GLFW_KEY_S && action == GLFW_PRESS)
	{
		
		changeShaderMode( ) ;
	}
	
	
}

enum ControlState {NONE=0,ROTATE,TRANSLATE};
ControlState mouseState = NONE;
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		if (button == GLFW_MOUSE_BUTTON_LEFT)
		{
			mouseState = ROTATE;
		}
		else if (button == GLFW_MOUSE_BUTTON_RIGHT)
		{
			mouseState = TRANSLATE;
		}
		
	}
	else if (action == GLFW_RELEASE)
	{
		mouseState = NONE;
	}
	//printf("%d\n", mouseState);
}

double lastx = (double)width / 2;
double lasty = (double)height / 2;
void mouseMotionCallback(GLFWwindow* window, double xpos, double ypos)
{
	const double s_r = 0.01;
	const double s_t = 0.01;

	double diffx = xpos - lastx;
	double diffy = ypos - lasty;
	lastx = xpos;
	lasty = ypos;

	if (mouseState == ROTATE)
	{
		//rotate
		x_angle += (float)s_r * diffy;
		y_angle += (float)s_r * diffx;
	}
	else if (mouseState == TRANSLATE)
	{
		//translate
		x_trans += (float)(s_t * diffx);
		y_trans += (float)(-s_t * diffy);
	}
}

void mouseWheelCallback(GLFWwindow* window, double xoffset, double yoffset)
{
	const double s_s = 0.01;

	scale += (float)(-s_s * yoffset);
}
