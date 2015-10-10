CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Bradley Crusco
* Tested on: Windows 10, i7-3770K @ 3.50GHz 16GB, 2 x GTX 980 4096MB (Personal Computer)

## Description
An interactive GPU accelerated rasterizer (Add here)
![](renders/dragon.png "Stanford Dragon")

## Pipeline

### Vertex Shading

### Primitive Assembly

### Backface Culling (Optional)

* **Overview**: Backface culling is a relatively simple procedure added to the pipeline as an option after the primitive assembly step. The step determines which primitives are facing the camera, and marks those that are not as not visible. After this has been done for all primitives, stream compaction is run on the resulting array of primitives and those that are not visible are removed from the pipeline. To determine if a primitive is facing the camera, we use the dot product between the vector from the camera to the model's position and the normal of the primitive.
* **Perfromance Impact**:
* 

### Scissor Test (Optional)
![](renders/dragon_scissor.png "Scissor Test Enabled on Stanford Dragon")

* **Overview**: The scissor test is another relatively simple stage added as an option to the pipeline. In the scene in the program, a rectangular portion of the screen can be defined as the bounds of this scissor clipping. Anything outside the bounds will be clipped from the scene.
* **Perfromance Impact**:
* 

### Rasterization

#### Triangles

#### Points

* **Overview**:
* **Perfromance Impact**:
* 

#### Lines
* **Overview**:
* **Perfromance Impact**:
* 

### Fragment Shading

#### Normal and Color Interpolation

* **Overview**:
* **Perfromance Impact**:
* 

### Fragments to Depth Buffer?

Fragment to framebuffer?

## Additional Features

### Mouse Controls

## Performance Analysis
