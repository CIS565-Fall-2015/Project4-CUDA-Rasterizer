CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Tongbo Sui
* Tested on: Windows 10, i5-3320M @ 2.60GHz 8GB, NVS 5400M 2GB (Personal)

## Video Demo

###### Tile-based render demo

###### Scanline demo
[![](img/Splash.png)](https://vimeo.com/141638182)

## Pipeline Overview (Tile-based) use `L` to switch between pipelines

## Pipeline Overview (Scanline)

* Vertex shading
  * Vertex shader with perspective transformation. Takes in vertices and transform the coordinates to window coordinates
* Primitive assembly
  * Assembles triangles from the list of vertices
  * Calculates AABB bounding box for each triangle
  * Initializes primitive properties for shading later
* Geometry shader (optional)
  * `G` toggle geometry shader
  * For each triangle's first vertex, draw a line to demonstrate the vertex's normal
  * Shader has the ability to output at most 8 primitives for each input primitive
  * Shader output is optimized via stream compaction, where unused storage is trimmed
  * Analysis
    * Performance impact: on average 2x slower, since the current shader outputs 2 primitives per input primitive
    * Optimization: stream compaction is used to trim down the output array size
    * Possible improvement: it might be better to simply refrain from using any compaction method, but direcly indexing the primitives during rasterization. This way the stream compaction overhead can be avoided. However it adds complexity to the next stage in the pipeline

###### Geometry shader. Hairs are vertex normals
![](img/geometry-shading.png)

* Backface culling (optional)
  * `B` toggle
  * Marks all backfaces, then remove them using stream compaction
  * Analysis
    * Performance impact: varying impact on performance (see below in Performance Analysis)
    * Optimization: stream compaction is used to trim down the output array size
    * Possible improvement: same as geometry shader. In this case there is actually strong evidence that stream compaction overhead might be too significant

###### Backface culling. Notice that backface normals are removed
![](img/culling.png)

* Rasterization
  * Rasterize input primitives. Uses scanline algorithm
    * Each thread takes care of one triangle, and scans every pixel inside the triangle's bounding box
    * Supports lines and points
        * Does **not** support vertex shading and assembly for such primitives; only rasterization
        * Analysis
          * Performance impact: no visible direct impact; however if point shader is enabled, the scene would render much faster since the rasterization logic is dramatically simplified
          * Optimization: various kernel optimization techniques, including reordering to reduce execution dependency, and removing cache variables for lower memory dependency and higher cache hit rate
          * Possible improvement: faster algorithm to rasterize the primitive
  * Sub-pipeline
    * Pre-clipping
      * Window clipping: remove fragments outside window by directly shrinking the scan bounding box
      * Scissor test (optional): same as window clipping, and further shrinks the scan bounding box
        * `S` toggle scissor test
        * Analysis
          * Performance impact: direct positive impact, but only when the object is partially inside the scissor box. Even in that case, the impact is small around ~10FPS
          * Optimization: clipping before scaning, instead of discarding fragment during scan, essentially decreases the amount of computation needed for clipped primitives
          * Possible improvement: may be a faster calculation for the scissor box clipping
    * Each thread scans each pixel in the triangle's bounding box
      * Subject to primitive size in window coordinate; the bigger, the slower
        * For the same object, camera being further away results in smaller object screen size; thus higher FPS
        * On the other hand, camera being very close results in each primitive covering big proportion of screen; thus each thread scans longer; thus lower FPS
      * The overhead of calculating and sorting intersection points per scanline cancels out the benefit that such points can shrink the scan range. Scanning every pixel is actually faster in general
    * Depth test
      * Directly testing the current calculated depth with the shallowest
      * Write to buffer if it wins the depth test, using atomics to avoid races
      * Use barycentric interpolation for color, position and normal. Thus it enables rendering of flat-shade objects, as well as smooth objects
      * Analysis on barycentric interpolatoin
        * Performance impact: no visible impact
        * Optimization: none
        * Possible improvement: none

###### Scissor test. Fragments outside the scissor box are clipped
![](img/scissor.png)

* Fragment shading
  * Simple Lambert shader
  * Each fragment is shaded with two fixed lights

###### Lambert shading with barycentric interpolation. Two lights are used to better demonstrate the effect
![](img/lambert.png)

## Misc Features
* Mouse-based interactive camera support
  * Controls
    * `Left button` hold and drag: rotate
    * `Right button` hold and drag: pan
    * Scroll: zoom
    * `SPACE`: reset camera to default
  * Performance impact: no visible impact
* `N`: fragment shader will use normal vector as color vector; enable to see fragment normals

###### Normal shading
![](img/normal.png)

* `R`: reset to color shading (use fragment color for shaded color, instead of fragment normal)
* `P`: toggle point shader; only shows shaded vertices instead of all fragments
  * Not compatible with geometry shader because it will set all primitives to point; rasterization will still work, but the two effects won't show at the same time

###### Point shading
![](img/point-shading.png)

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
  * For `cow.obj`, camera moving away is not as effective. This may due to the fact that the performance limit of current implementation has been reached

###### `cow.obj` performance breakdown
![](img/cow-perf.png)

###### `cow.obj` FPS by camera distance
![](img/cow-dist.png)

###### `suzanne.obj` performance breakdown
![](img/suzanne-perf.png)

###### `suzanne.obj` FPS by camera distance
![](img/suzanne-dist.png)

###### `flower.obj` performance breakdown
![](img/flower-perf.png)

###### `flower.obj` FPS by camera distance
![](img/flower-dist.png)

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
![](img/backface.png)

## References

* Line segment intersection test
  * http://paulbourke.net/geometry/pointlineplane/
* Vertex shader transformation
  * http://www.songho.ca/opengl/gl_transform.html
* Bresenham's line algorithm
  * https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
