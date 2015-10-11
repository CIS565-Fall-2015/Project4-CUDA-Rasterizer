/**
 * @file      main.cpp
 * @brief     Main file for CUDA rasterizer. Handles CUDA-GL interop for display.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"

//-------------------------------
//-------------MAIN--------------
//-------------------------------

glm::vec3 eye(0, 0, 0);

shadeControl sCtrl;

int tessLevel = 0;
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
glm::mat4 ViewMatrix;// = glm::mat4(-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 1);
glm::mat4 ProjectionMatrix = glm::mat4();
void runCuda() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    dptr = NULL;

    cudaGLMapBufferObject((void **)&dptr, pbo);
	rasterize(dptr, ViewMatrix, ProjectionMatrix,eye,tessLevel,sCtrl);
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
	//
	CalcViewPersMat(0,0);
	glfwSetCursorPosCallback(window, mouseMoveCallback);
	glfwSetMouseButtonCallback(window, mouseDownCallback);
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

	//vector<glm::vec4>* texts = mesh->getTextureCoords();
	
    rasterizeSetBuffers(mesh,tessLevel);

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
	if (key == GLFW_KEY_UP && action == GLFW_PRESS){
		tessLevel++;	
		printf("tessellation level: %d\n",tessLevel);
	}
	if (key == GLFW_KEY_DOWN && action == GLFW_PRESS){
		tessLevel--;
		if (tessLevel < 0)
			tessLevel = 0;
		printf("tessellation level: %d\n", tessLevel);
	}
	if (key == GLFW_KEY_0 && action == GLFW_PRESS)
	{
		printf("press '0': wireframe only\n");
		sCtrl.Wireframe = true;
		sCtrl.Color = false;
	}
	if (key == GLFW_KEY_1 && action == GLFW_PRESS)
	{
		printf("press '1': color\n");
		sCtrl.Color = true;
		sCtrl.Normal = false;
	}
	if (key == GLFW_KEY_W && action == GLFW_PRESS)
	{	
		if (sCtrl.Color)
		{
			sCtrl.Wireframe = !sCtrl.Wireframe;
			string t = sCtrl.Wireframe ? "on" : "off";
			printf("press 'w': wireframe %s\n", t);
		}
	}
	if (key == GLFW_KEY_T && action == GLFW_PRESS)
	{
		sCtrl.Texture = !sCtrl.Texture;
		string t = sCtrl.Texture ? "on" : "off";
		printf("press 'texture': texture %s\n", t);
	}
	if (key == GLFW_KEY_N && action == GLFW_PRESS)
	{
		sCtrl.Normal = !sCtrl.Normal;
		string t = sCtrl.Normal ? "on" : "off";
		printf("press 'n': normal debugging %s\n", t);
	}
	if (key == GLFW_KEY_D && action == GLFW_PRESS)
	{
		sCtrl.DispMap = !sCtrl.DispMap;
		string t = sCtrl.DispMap ? "on" : "off";
		printf("press 'd': displacement mapping %s\n", t);
	}
	if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
	{
		sCtrl.UVrepeat += 0.2;		
		printf("UV repeat : %2f \n", sCtrl.UVrepeat);
	}
	if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
	{
		sCtrl.UVrepeat -= 0.2;
		if (sCtrl.UVrepeat<0.2)
		{
			sCtrl.UVrepeat = 0.2;
		}
		printf("UV repeat : %2f \n", sCtrl.UVrepeat);
	}

}

float horizontalAngle = 0.04;
float verticalAngle = -2.5;
bool isMoving = false;
bool isRotating = false;
glm::vec3 center(0,0,0);
glm::vec3 direction;
glm::vec3 right_vec;
glm::vec3 up;
float x_lsPos;
float y_lsPos;
float lastTime;
float zoom = -5;
float FOV = PI/4;
//http://www.opengl-tutorial.org/beginners-tutorials/tutorial-6-keyboard-and-mouse/
//http://r3dux.org/2011/05/simple-opengl-keyboard-and-mouse-fps-controls/
//https://github.com/LWJGL/lwjgl3-wiki/wiki/2.6.3-Input-handling-with-GLFW

void CalcViewPersMat(float x_move, float y_move)
{
	// Direction : Spherical coordinates to Cartesian coordinates conversion
	direction = glm::vec3(
		cos(verticalAngle) * sin(horizontalAngle),
		sin(verticalAngle),
		cos(verticalAngle) * cos(horizontalAngle)
		);
	//direction = glm::normalize(direction);
	// Right vector
	right_vec = glm::vec3(
		sin(horizontalAngle - 3.14f / 2.0f),
		0,
		cos(horizontalAngle - 3.14f / 2.0f)
		);
	// Up vector : perpendicular to both direction and right
	up = glm::cross(right_vec, direction);

	//direction *= 2;
	//right_vec *= 2;
	//up *= 2;

	center += (x_move*right_vec);
	center -= (y_move*up);
	glm::vec3 position = center - direction;
	position += (zoom*direction);
	eye = position;
	ProjectionMatrix = glm::perspective(FOV, 1.f, -0.1f,- 100.0f);
	//ProjectionMatrix = glm::mat4();
	ViewMatrix = glm::lookAt(
		position,           // Camera is here
		center, // and looks here : at the same position, plus "direction"
		up                  // Head is up (set to 0,-1,0 to look upside-down)
		);
}

void mouseMoveCallback(GLFWwindow *window, double xpos, double ypos)
{
	float mouseSpeed = 0.04;
	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - lastTime);
	float xMove = 0;
	float yMove = 0;
	if (isRotating)
	{
		horizontalAngle -= mouseSpeed * deltaTime * float(xpos - x_lsPos);//!!!
		verticalAngle += mouseSpeed * deltaTime * float(ypos - y_lsPos);
		//printf("h:%2f, v:%2f\n", horizontalAngle, verticalAngle);
	}
	else if (isMoving)
	{
		xMove = mouseSpeed * deltaTime * float(xpos - x_lsPos);
		yMove = mouseSpeed * deltaTime * float(ypos - y_lsPos);
	}
	CalcViewPersMat(xMove, yMove);

	x_lsPos = xpos;
	y_lsPos = ypos;
	lastTime = currentTime;
}

void mouseDownCallback(GLFWwindow *window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
	{
		//right down : move center
		isMoving = true;
		isRotating = false;
	}
	else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		//left down : rotate
		isRotating = true;
		isMoving = false;
	}
	else if (action == GLFW_RELEASE)
	{
		isRotating = false;
		isMoving = false;
	}
}

void mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{

	zoom += yoffset*0.1;
	if (zoom >= 0)
	{
		zoom = 0;
	}
	//printf("zoom: %3f\n", zoom);
	//float zoom = yoffset*0.1;
	CalcViewPersMat(0, 0);
}