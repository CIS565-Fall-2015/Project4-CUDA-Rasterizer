CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Levi Cai
* Tested on: Windows 8, i7-5500U @ 2.4GHz, 12GB, NVidia GeForce 940M 2GB

Attack of the cowwwssss!

![](renders/cow_instancing.PNG)

### Graphics Pipeline

This is an implementation of a CUDA-based graphics pipeline with the following stages/features:

* Vertex shading
* Instancing
* Primitive Assembly
* Rasterization with Anti-Aliasing (FSAA using fixed-pattern supersampling)
* Fragment shading
* Fragment to Frame buffer transfer
* Mouse control
* Color interpolation on primitive surfaces

### Feature Demos

## Depth-Testing

Correct:

![](renders/depth_test_flower.PNG)

Incorrect:

![](renders/render_gone_wrong.PNG)

## Color Interpolation

![](renders/cow_color_interpolation.PNG)

## Instancing

5x cows with translation and rotations

![](renders/cow_instancing.PNG)

## Anti-Aliasing

3x super sampling results

![](renders/anti_aliasing_cow_3_v_1.PNG)

### Performance Analysis

##Comparison of size of triangles vs. FPS

This comparison illustrates the bottleneck of the rasterization portion of the pipeline.
As triangles get nearer to the camera (effectively, larger), then each thread must spend 
more time rasterizing. One possible method of reducing this is to compare the number of
primitives to be rasterized vs. the size of the primitives on screen. If the ratio of size to 
number is large, then instead of launching 1 thread per primitive, then launch one thread per
fragment and depth test sequentially that way.

![](renders/trisize_vs_fps.png)

Comparison of pipeline stages

![](renders/pie_chart.png)

Performance Effects of super-sampled Anti-Aliasing

It is quite clear that AAing in this manner is extremely costly as the samples per pixels increases.

![](renders/aa_vs_fps.png)
