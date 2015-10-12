CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Xinyue zhu
* Tested on: Windows 10, i7- @ 2.22GHz 22GB, GTX 960M

========================
## Description:
 <p>This is a rasterizer.</p>
It includes the following pipeline:<br/>
1)vertex transformation with camera movement</br>
2)Tessellation shader</br>
3)Bling-phong shader and blending</br>
4)rasterization brute force scan line->boundingbox scan line</br>
4)depth test</br>
the base color is normal map</br>
<img src="1.png"  width="400" height="400">



* A brief description of the project and the specific features you implemented.
* At least one screenshot of your project running.
* A 30 second or longer video of your project running.
* A performance analysis (described below).

### Performance Analysis
<img src="p1.png"  width="600" height="100">
<p>The CPU are all used for opengl related funtions. The computation mostly happened in GPU
<p> a breakdown of time spent in each pipeline stage 
<img src="pipe_line.png"  width="600" height="200">



