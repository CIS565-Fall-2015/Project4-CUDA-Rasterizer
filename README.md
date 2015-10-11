CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Levi Cai
* Tested on: Windows 8, i7-5500U @ 2.4GHz, 12GB, NVidia GeForce 940M 2GB

![](renders/cow_instancing.PNG)
Attack of the cowwwssss!

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

5x cows
![](renders/cow_instancing.PNG)

## Anti-Aliasing

3x super sampling results
![](renders/anti_aliasing_cow_3_v_1.PNG)

### Performance Analysis

The performance analysis is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must have
performed at least one experiment on your code to investigate the positive or
negative effects on performance. 

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them. 

Provide summary of your optimizations (no more than one page), along with
tables and or graphs to visually explain any performance differences.

* Include a breakdown of time spent in each pipeline stage for a few different
  models. It is suggested that you use pie charts or 100% stacked bar charts.
* For optimization steps (like backface culling), include a performance
  comparison to show the effectiveness.
