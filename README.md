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

For vertex shading we take input vertices from a vertex in buffer and transform them by applying the model-view projection matrix so they are orientated correctly within our scene. After being transformed they are sent to a vertex out buffer.

### Primitive Assembly

While this project supports the rendering of different primitive types (triangles, points, and lines, see below for details), that is achieved through manipulating the only true primtive supported, triangles. The vertices that were transformed in the previous stage and sent to the vertex out buffer are turned into triangle primitives in groups of three, each making up one vertex of the triangle. The total size of the primtives array we create will be a third the size of the vertex out buffer.

### Backface Culling (Optional)

* **Overview**: Backface culling is a relatively simple procedure added to the pipeline as an option after the primitive assembly step. The step determines which primitives are facing the camera, and marks those that are not as not visible. After this has been done for all primitives, stream compaction is run on the resulting array of primitives and those that are not visible are removed from the pipeline. To determine if a primitive is facing the camera, we use the dot product between the vector from the camera to the model's position and the normal of the primitive.
* **Perfromance Impact**:
* 

### Scissor Test (Optional)
![](renders/dragon_scissor.png "Scissor Test Enabled on Stanford Dragon")

* **Overview**: The scissor test is another relatively simple stage added as an option to the pipeline. In the scene in the program, a rectangular portion of the screen can be defined as the bounds of this scissor clipping. Anything outside the bounds will be clipped from the scene. Whether a primitive is to be displayed is determined by checking the maximum and minimum points on the primitive's bounding box and comparing those positions to the dimensions of the rectangle defining our culling area. If the max or min of the bounding box lays outside this area, the primitive is marked as not visible. Once we've run the test on all the primitives in the scene, stream compaction is used to remove the invalid primitives from the array.
* **Perfromance Impact**:
* 

### Rasterization

#### Triangles
![](renders/dragon_tri.png "Stanford Dragon Rendered Using Triangle Primitives (No Normal Interpolation)")

To render the basic rasterization primitive, the triangle, each GPU thread is responsible for one triangle. The bounding box for that triangle is retreived, and, through the scanline implementation, the thread loops over each pixel in the bounding box.

#### Normal and Color Interpolation
![](renders/cow_interp_comp.png "Cow With and Without Normal Interpolation")

* **Overview**: Implementing normal and color interpolation gives significantly more visually pleasing results, as can be seen above. Without the interpolation, models look obviously contructed of triangles. With interpolation, the models are smooth and provide the realistic effect we'd expect. To achieve these smooth models, the obj file must provide vertex normals. If not, the result will look like interpolation is disabled. The same for color, if not provided in the object file per vertex, the model will be one solid color. To calculate the interpolated results, we first calculate the bary centric coordinate. Once we do that and have determined that the coordinate is in bounds and the z position of our primitive passes the depth test, we interpolate by adding the sum of the product of the x, y, and z components of the bary centric coordiante and each of the three verticies, respectively.
* **Perfromance Impact**:
* 

#### Points
![](renders/dragon_points.png "Stanford Dragon Rendered Using Point Primitives")

* **Overview**: For this effect, the standard rasterization step of the pipeline is replaced with one to output points. Because we are only rendering a point and not an entire triangle, bary centric coordinates do not need to be calculated, nor do we have to interpolate the normals or colors across each verticy. Instead we just output the values for a single vertex to the depth buffer (I use the middle vertex at index one) and lead the others as zero. This vertex will be the point that is rendered to the screen.
* **Perfromance Impact**:
* 

#### Lines
![](renders/dragon_lines.png "Stanford Dragon Rendered Using Line Primitives")

* **Overview**: The implementation for lines is more complicated than the other two primitives. For ease of implementation, I render the line between the first and second verticies of a the triangle primitives. There are two situations that need to be handled when rendering these. The first is the simplest case, when the line to render is a straight vertical line. Here we loop between the min and end values in the y direction and render as we travel along. For all other lines we must calculate a Bresenham line, is a method of approximating the rasterization of a line. Otherwise it works just like rendering the vertical line, with the main difference being a loop along the x direction while using the slope of the line to calculate a y value in the correct position at each iteration of the loop, and z is determined using the depth test.
* **Perfromance Impact**:
* 

### Fragment Shading

The fragment shader updates the color value of a fragment in the depth buffer using a basic Lambert shader. It essentially adds lighting to the scene. As of this writing the implementation uses a single light source at located at the camera position to keep the object lit while the user moves the camera using the mouse controls.

### Fragments to Framebuffer
These fragments are finally output to the framebuffer, and from there displayed to the screen.

## Additional Features

### Mouse Controls

* **Overview**: 

## Performance Analysis
