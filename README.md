CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Ziye Zhou
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

Representive Image
========================
![alt tag](https://github.com/ziyezhou-Jerry/Project4-CUDA-Rasterizer/blob/master/image/representive.png?raw=true)

Building the Pipeline
========================
###Vertex shader
In this stage, I apply the MVP (Model-View-Projection) Matrix on the input vertices, so that I can get their corresponding coordinates in normalized device coordinates (-1 to 1 in each dimension).

`vec3 out =vec3 ( MVP * vec4(in,1.f))`

###Primitive assembly
In this stage, I read the vertices index of triangle from the obj file and store them into the device vector of 'Triangle'

###Rasterization
This is the key part of this project. Basically, I parallelize with each primitive (triangle) to do the scanline algorithm. For each primitive, I first caculate AABB(Axis Alined Bounding Box) to get the scan region of latter procedure. If this region is outside the normalized device coordinates (-1 to 1 in each dimension), we can just ignore this primitive since it will not be shown on the screen. Ohterwise, we continue to the sampling and decide stage. In this stage, I simply pick the center of each pixel to do the sampling.
![alt tag](https://github.com/ziyezhou-Jerry/Project4-CUDA-Rasterizer/blob/master/image/sample_combine.png?raw=true)

If this smaple is in the range of the primitive (using the barycenteric coordinate to decide), I just assign the interpolated color and normal to the fragment buffer. But one thing to notice is that we need to know if this primitive is of lager depth value (in the front to be seen). In order to do this, I compare the current z value and the old z value in the fragment buffer. Since multiple primitives can write into the same fragment, we need to avoid the race condition here. To do this, I use a different boolean vector for each fragment. Before enter the critical section of writing into the buffer, we first check if this value is true, if it is true we are save to enter this section; otherwise, we need to wait until others leave the section. As soon as one primitive enter the critical section, I set the value of boolean to false to avoid others enter, and set it back to true when it leaves the critical section.

###Fragment shading
In this stage, I just use the Lambert equation to get the color of the surafce. Also, in order to debug the normal direction, I also implement the visualization of normal direction in this stage.
![alt tag](https://github.com/ziyezhou-Jerry/Project4-CUDA-Rasterizer/blob/master/image/debug_image.png?raw=true)

Extra Features
========================
### Mouse Interaction
I implemented the mouse interaction using the left buttion to change the pitch and head angle of the view direction, middle buttion to change the eye distance, right button to change the Lookat point.

### Transform with Key Interaction
I implemented the transform of obj with the key interaction. Button Up and Down for the scaling of the obj, Button W and S for translating along the Y-axis, Button A and D for translating along the X-axis, Button Z and X for translating along Z-axis.
