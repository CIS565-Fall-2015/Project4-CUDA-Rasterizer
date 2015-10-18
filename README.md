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

This pipeline also supports instancing, anti-aliasing, and tile based rasterization. Anti-aliasing and tile rasterization cause some notable changes to the pipeline, which will be explained below.
