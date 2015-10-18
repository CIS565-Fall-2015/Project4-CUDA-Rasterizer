CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Kangning (Gary) Li
* Tested on: Windows 10, i7-4790 @ 3.6GHz 16GB, GTX 970 4096MB (Personal)

![](img/AAAAAAAAAAAAAAA.png)

This repository contains a basic rasterization pipeline written in CUDA. This pipeline has the following rough stages, as seen in the rasterize() function in [src/rasterize.cu#L856](rasterize.cu):
* clear the fragment/depth buffer
* multiply all scenegraph transformation matrices by the view/projection matrix
* shade vertices
* perform primitive assembly
* use scanline rasterization to generate fragments. perform depth tests.
* shade the fragments
* copy the fragments to the frame buffer

This pipeline also supports instancing, anti-aliasing, and tile based rasterization. Enabling anti-aliasing and tile rasterization cause some notable changes to the pipeline, which will be explained below.

*Places to Tweak*
Most configurable settings are in [src/main.cpp](main.cpp), which handles sending delivering buffers of vertices, normals, indices, and matrices to the rasterizer. The [src/main.cpp#L16](top) of the file contains some variables that can be used to control the camera's starting state. Camera controls are spherical. The camera can also be controlled using the keyboard. The left, right, up and down arrow keys rotate the "sphere" of the camera, while the z and x keys control the "radius." Field of view can also be changed with f and g but may be unpredictable.
[src/main.cpp#L199](Line 199), if uncommented, activates tiled rendering.
[src/main.cpp$L202](Line 202) similarly toggles anti aliasing.
Adding modeling transformations to the [src/main.cpp#L181](transformations) vector allows adding additional instances.
Changing [src/rasterize.cpp#L912] from MSAA to FSAA will switch between multisample and full screen antialiasing.

*Instancing*
![](img/one_cow.png) ![](img/instanced_cows.png)

Instancing allows "drawing" one model (one set of vertices, indices, and normals - this rasterizer only supports rendering one model at a time) multiple times in a scene with different transformations. This rasterizer implements instancing by keeping a larger buffer for primitives and transformed vertices. The point of instancing isn't to increase performance on the GPU (the GPU is still drawing all the triangles) but to decrease the bandwidth consumption when preparing a scene by reducing the amount of data that needs to be sent from the host to the device. This improvement is demonstrated by the following charts:

![](img/charts/instancing/stack_comparison).png)

![](img/charts/instancing/single_cow.png) ![](img/charts/instancing/many_cows_host.png) ![](img/charts/instancing/many_cows_instanced.png)

In this comparison, 6 cows were rasterized by uploading an obj file with 6 cows in it (see [objs/many_cows.obj](many_cows.obj)) and separately by uploading data for a single cow and transformations for 6. Although the relevant pipeline section runtimes were about the same for both cases, the host-to-device transfer time differs dramatically.