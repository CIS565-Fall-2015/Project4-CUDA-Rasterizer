CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Tongbo Sui
* Tested on: Windows 10, i5-3320M @ 2.60GHz 8GB, NVS 5400M 2GB (Personal)

## Video demo

## Pipeline overview

* Vertex shading with perspective transformation
* Primitive assembly with support for triangles read from buffers of index and vertex data.
* Geometry shader, able to output a variable number of primitives per input primitive, optimized using stream compaction
  * `G` toggle geometry shader; shows vertex normals
  * Can output at most 8 primitives
  * Final result is trimmed via stream compaction
* Rasterization
  * Scan each pixel in the bounding box
    * Subject to primitive size in window coordinate
    * For the same object, camera being further away results in smaller object screen size; thus higher FPS
    * Camera very close results in each primitive covering big proportion of screen; thus each thread scans longer; thus lower FPS
    * However, pixel scanning is at least 5x faster than standard scanline
  * Scissor test
    * `S`: toggle scissor test
  * Fragment clipping
  * Depth test (using atomics for race avoidance)
  * Barycentric color interpolation
  * Support for rasterizing lines and points
      * Does **not** support vertex shading for such primitives; only rasterization
* Fragment shading
  * Lambert
* Mouse-based interactive camera support
  * `Left button` hold and drag: rotate
  * `Right button` hold and drag: pan
  * Scroll: zoom
  * `SPACE`: reset camera to default
* Misc support features
  * `N`: fragment shader will use normal vector as color vector; enable to see fragment normals
  * `R`: reset to color shading (use fragment color for shaded color, instead of fragment normal)
  * `P`: toggle point shader; only shows shaded vertices instead of all fragments

**IMPORTANT:**
For each extra feature, please provide the following brief analysis:

* Concise overview write-up of the feature.
* Performance impact of adding the feature (slower or faster).
* If you did something to accelerate the feature, what did you do and why?
* How might this feature be optimized beyond your current implementation?

### Performance Analysis

Provide summary of your optimizations (no more than one page), along with
tables and or graphs to visually explain any performance differences.

* Include a breakdown of time spent in each pipeline stage for a few different
  models. It is suggested that you use pie charts or 100% stacked bar charts.
* For optimization steps (like backface culling), include a performance
  comparison to show the effectiveness.

* Baseline
  * `tri.obj`, 
  * Camera:
    * Position `(0,0,3)`
    * LookAt `(0,0,0)`
    * FOV = 45.0 degrees

* Optimization
  * Alter block size for different kernels to achieve higher warp count
    * Immediate benefit: ~90 FPS to ~130 FPS
  * Substitute fixed divisions with corresponding multiplications for marginal performance gain
    * Reduced register counts
    * Slight speed up
  * Cache repetitive calculations; reorder executions to reduce execution dependency
    * Minor speed up

## References

* Line segment intersection test
  * http://paulbourke.net/geometry/pointlineplane/
* Vertex shader transformation
  * http://www.songho.ca/opengl/gl_transform.html
* Bresenham's line algorithm
  * https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm