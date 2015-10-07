CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Tongbo Sui
* Tested on: Windows 10, i5-3320M @ 2.60GHz 8GB, NVS 5400M 2GB (Personal)

## Video demo

## Pipeline overview

* Vertex shading with perspective transformation
* Primitive assembly with support for triangles read from buffers of index and vertex data.
* Geometry shader (optional)
  * `G` toggle geometry shader; shows vertex normals
  * Able to output a variable number of primitives per input primitive, optimized using stream compaction
  * Can output at most 8 primitives
  * Final result is trimmed via stream compaction
* Backface culling (optional)
  * `B` toggle
* Rasterization
  * Scan each pixel in the bounding box
    * Subject to primitive size in window coordinate
    * For the same object, camera being further away results in smaller object screen size; thus higher FPS
    * Camera very close results in each primitive covering big proportion of screen; thus each thread scans longer; thus lower FPS
    * However, pixel scanning is at least 5x faster than standard scanline
  * Scissor test (optional)
    * `S`: toggle scissor test
  * Window clipping
  * Depth test (using atomics for race avoidance)
  * Barycentric color interpolation
  * Support for rasterizing lines and points
      * Does **not** support vertex shading for such primitives; only rasterization
* Fragment shading
  * Lambert

## Misc features
* Mouse-based interactive camera support
  * `Left button` hold and drag: rotate
  * `Right button` hold and drag: pan
  * Scroll: zoom
  * `SPACE`: reset camera to default
* `N`: fragment shader will use normal vector as color vector; enable to see fragment normals
* `R`: reset to color shading (use fragment color for shaded color, instead of fragment normal)
* `P`: toggle point shader; only shows shaded vertices instead of all fragments
  * Not compatible with geometry shader because it will set all primitives to point; rasterization will still work, but both effects won't show at the same time

**IMPORTANT:**
For each extra feature, please provide the following brief analysis:

* Concise overview write-up of the feature.
* Performance impact of adding the feature (slower or faster).
* If you did something to accelerate the feature, what did you do and why?
* How might this feature be optimized beyond your current implementation?

### Performance Analysis

* Camera properties
  * Position `(0,0,3)`
  * LookAt `(0,0,0)`
  * FOV = 45.0 degrees

* Performance breakdown
  * Fragement shader time is almost fixed, since it's only dependent on the pixel count of the output window
  * Breakdown are core pipeline only
  * For the exact same camera properties described above, frame rate largely depends on the transformed size of the primitives, due to the current rasterization implementation
    * `suzanne.obj` and `flower.obj` see increased frame rate when camera moves away from the object, yielding smaller transformed primitive sizes

###### `cow.obj` performance breakdown

###### `suzanne.obj` performance breakdown

###### `suzanne.obj` FPS by camera distance

###### `flower.obj` performance breakdown

###### `flower.obj` FPS by camera distance

* Optimization (`cow.obj`)
  * Kernel: minor improvements (~10FPS)
    * Alter block size for different kernels to achieve higher warp count
    * Substitute fixed divisions with corresponding multiplications for marginal performance gain
    * Cache repetitive calculations; reorder executions to reduce execution dependency
  * Backface culling
    * Only useful when the object is big in window
      * Reduces rasterization time
    * Stream compaction overhead might be more significant and cancel out the benefit

###### Backface culling performance impact

## References

* Line segment intersection test
  * http://paulbourke.net/geometry/pointlineplane/
* Vertex shader transformation
  * http://www.songho.ca/opengl/gl_transform.html
* Bresenham's line algorithm
  * https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm