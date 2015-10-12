CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Ziye Zhou
* Tested on: Windows 7, i5-3210M @ 2.50GHz 4.00GB, GeForce GT 640M LE (Lenovo Laptop)

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

### Alti-Aliasing
I am using the subpixel sampling method to do the anti-aliasing, basically I do it like this:
![alt tag](https://github.com/ziyezhou-Jerry/Project4-CUDA-Rasterizer/blob/master/image/multiple_sample.png?raw=true)

I use these five sample point to decide the fraction of pixel occupied by the primitive, so that I can use the fraction to change the color assigned to the fragment. The compare of w and w/o anti-aliasing is like this:
![alt tag](https://github.com/ziyezhou-Jerry/Project4-CUDA-Rasterizer/blob/master/image/wo_anti_aliasing_new%20-%20Copy.png?raw=true)
![alt tag](https://github.com/ziyezhou-Jerry/Project4-CUDA-Rasterizer/blob/master/image/w_anti_aliasing_new%20-%20Copy.png?raw=true)

### Color-Interpolation
Here I test the interpolation on the simple triangle, with each vertex assign different color.
![alt tag](https://github.com/ziyezhou-Jerry/Project4-CUDA-Rasterizer/blob/master/image/color_interpolation.png?raw=true)

###Blending
Here I am using the simple blending scheme. I add one variable Alpha to denote the transparency of the material. For the color we see, the equation is simply:

`Color = (1-Alpha)*front_Color + Alpha*back_Color`
![alt tag](https://github.com/ziyezhou-Jerry/Project4-CUDA-Rasterizer/blob/master/image/bending_compare.png?raw=true)

More Results
========================
![alt tag](https://github.com/ziyezhou-Jerry/Project4-CUDA-Rasterizer/blob/master/image/4.png?raw=true)

Performace Analysis
========================
Since I haven't implamented the optimization part for the pipeline, I don't have a very numerical analysis of this program. But I do test the performance with and without the anti-liasing. By using the cow_smooth.obj, the difference is not obvious. I use the FPS to roughly compare the performance. I run the program 100 iterations and get the average result like this:

![alt tag](https://github.com/ziyezhou-Jerry/Project4-CUDA-Rasterizer/blob/master/image/excel.png?raw=true)

The difference is not obivious, and I think the reason may be the mesh is not so large and my sample point is just 5.

Future Work
========================
* Adding some more stages ( Geometry shader, etc.) into the pipeline
* Implementing some optimization methods


