CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Megan Moore
* Tested on:  Windows 7, i7-4770 @ 3.10GHz 32GB (Sig Lab)

**Summary:** 
In this project, I used CUDA to implement a simplified
rasterized graphics pipeline, similar to the OpenGL pipeline. I
implemented vertex shading, primitive assembly, rasterization, fragment shading,
and a framebuffer. 

* Vertex Shader
 * In the vertex shader, I took the local coordinates of each triangle and converted them into world coordinates.  To do this, I multiplied the local coordinates by the model, view, and projection matrix.  OpenGl is used to create the view and projection matrices, inputing the camera's location, where it is looking, along with the near and far clippinig planes.  Along with transforming the vertices, the normals were also transformed.  However, the normals were multiplied by the inverse transpose of the model matrix.  This is because the normals are vectors, not points like the vertices.  Thus, we have to transform them differently.
 
* Primitive assembly with support for triangles read from buffers of index and
  vertex data
 * The newly transformed vertices are passed into the primitive assembler.  The output of this function are all the triangles that are in the model.  Each vertex is passed into the primitve assembler, and based on it's index, it is placed into a new triangle.  The triangle struct holds three VertexOuts and a boolean that says whether it is backfacing or not.  After this implementation, I was able to get the output shown below. 

![](img/cow_triangles.png "Cow after primitive assembly")
 
* Rasterization
 * The rasterizer does most of the work.  Here, each triangle is passed into the function, and it loops through every fragment to determine if the triangle is in that fragment.  The depth test is done here, along with antialiasing.  With each fragment, we calculate the barycentric coordinates with respect to the given triangle.  These coordinates tell us whether the fragment is in the triangle.  Then, the depth test is done.  If the depth of the triangle at that fragment is lower than any other triangle at that fragment, then the fragment should show that particular triangle.  AtomicMin is used to check the depth.  This function is needed when using the GPU because all threads are being run at once, we need an uninterupted function, since multiple threads could be trying to access the fragment's depth value.  Including the depth test allowed for a much better output.  It changed the image below on the left to the image on the right.

![](img/cow_normals3.png "Cow no depth test") {width: 150px} ![](img/cow_normals_depth.png "Cow with depth test")

* Fragment shading
 * In the fragment shader, I applied a diffuse shader to the model.  There was a light source that was placed behind the camera.  Using the lights position, an ambient and diffuse term (created from the dot product of the surface normal and light vector), the new color of the surface was calculated.  The output of the fragment shader was a new triangle with the final output color.  The image below is the same cow image with the shader applied to it.

![](img/cow_frag_shader.png "Cow with fragment shader")

* A depth buffer for storing and depth testing fragments / Fragment to depth buffer writing (**with** atomics for race avoidance)
 * The depth buffer is part of the Fragment struct that is created during rasterization.  There are two values, a float depth and an int depth value.  This is because the AtomicMin function only allows for ints.  Thus, in order to compare the fragment depth's, avoiding race conditions, the floats must be turned into ints.  The value INT_MAX was used to do this.  This allowed me to convert the floats into ints with the highest possible accuracy.  The depth values were stored, and if two triangles were in the same fragment, the one with the smallest depth would be drawn.  When the fragments were passed into the fragment shader, the depth value was checked before applying any color to the fragment.  If the depth value was still equal to INT_MAX, that meant no triangle was in the fragment, and no color would be applied.  However, if the depth was less than that, it would draw the triangle that was in the fragment.  
 
* (Fragment shader) simple lighting scheme, such as Lambert or Blinn-Phong
 * A Phong shader was applied to the models.  The phong shader takes into account ambient, diffuse, and specular shading.  In this shader, the ambient term is .2, while the diffuse and specular term are determined by the lighting location and normal of the surface.  The image below shows the phong lighting with a red light.  

![](img/cow_red_light.png "Cow with a red light source and phong shading")


* (1.0) Tile-based pipeline.
* Additional pipeline stages.
   * (1.0) Tessellation shader.
   * (1.0) Geometry shader, able to output a variable number of primitives per
     input primitive, optimized using stream compaction (thrust allowed).
   * (0.5 **if not doing geometry shader**) Backface culling, optimized using
     stream compaction (thrust allowed).
   * (1.0) Transform feedback.
   * (0.5) Scissor test.
   * (0.5) Blending (when writing into framebuffer).
* (1.0) Instancing: draw one set of vertex data multiple times, each run
  through the vertex shader with a different ID.
* (0.5) Correct color interpolation between points on a primitive.
* (1.0) UV texture mapping with bilinear texture filtering and perspective
  correct texture coordinates.
* Support for rasterizing additional primitives:
   * (0.5) Lines or line strips.
   * (0.5) Points.
* (1.0) Anti-aliasing.
* (1.0) Occlusion queries.
* (1.0) Order-independent translucency using a k-buffer.
* (0.5) **Mouse**-based interactive camera support.


**IMPORTANT:**
For each extra feature, please provide the following brief analysis:

* Concise overview write-up of the feature.
* Performance impact of adding the feature (slower or faster).
* If you did something to accelerate the feature, what did you do and why?
* How might this feature be optimized beyond your current implementation?


## Base Code Tour

You will be working primarily in two files: `rasterize.cu`, and
`rasterizeTools.h`. Within these files, areas that you need to complete are
marked with a `TODO` comment. Areas that are useful to and serve as hints for
optional features are marked with `TODO (Optional)`. Functions that are useful
for reference are marked with the comment `CHECKITOUT`. **You should look at
all TODOs and CHECKITOUTs before starting!** There are not many.

* `src/rasterize.cu` contains the core rasterization pipeline. 
  * A few pre-made structs are included for you to use, but those marked with
    TODO will also be needed for a simple rasterizer. As with any part of the
    base code, you may modify or replace these as you see fit.

* `src/rasterizeTools.h` contains various useful tools
  * Includes a number of barycentric coordinate related functions that you may
    find useful in implementing scanline based rasterization.

* `util/utilityCore.hpp` serves as a kitchen-sink of useful functions.


## Rasterization Pipeline

Possible pipelines are described below. Pseudo-type-signatures are given.
Not all of the pseudocode arrays will necessarily actually exist in practice.

### First-Try Pipeline

This describes a minimal version of *one possible* graphics pipeline, similar
to modern hardware (DX/OpenGL). Yours need not match precisely.  To begin, try
to write a minimal amount of code as described here. Verify some output after
implementing each pipeline step. This will reduce the necessary time spent
debugging.

Start out by testing a single triangle (`tri.obj`).

* Clear the depth buffer with some default value.
* Vertex shading: 
  * `VertexIn[n] vs_input -> VertexOut[n] vs_output`
  * A minimal vertex shader will apply no transformations at all - it draws
    directly in normalized device coordinates (-1 to 1 in each dimension).
* Primitive assembly.
  * `VertexOut[n] vs_output -> Triangle[n/3] primitives`
  * Start by supporting ONLY triangles. For a triangle defined by indices
    `(a, b, c)` into `VertexOut` array `vo`, simply copy the appropriate values
    into a `Triangle` object `(vo[a], vo[b], vo[c])`.
* Rasterization.
  * `Triangle[n/3] primitives -> FragmentIn[m] fs_input`
  * A scanline implementation is simpler to start with.
* Fragment shading.
  * `FragmentIn[m] fs_input -> FragmentOut[m] fs_output`
  * A super-simple test fragment shader: output same color for every fragment.
    * Also try displaying various debug views (normals, etc.)
* Fragments to depth buffer.
  * `FragmentOut[m] -> FragmentOut[width][height]`
  * Results in race conditions - don't bother to fix these until it works!
  * Can really be done inside the fragment shader, if you call the fragment
    shader from the rasterization kernel for every fragment (including those
    which get occluded). **OR,** this can be done before fragment shading, which
    may be faster but means the fragment shader cannot change the depth.
* A depth buffer for storing and depth testing fragments.
  * `FragmentOut[width][height] depthbuffer`
  * An array of `fragment` objects.
  * At the end of a frame, it should contain the fragments drawn to the screen.
* Fragment to framebuffer writing.
  * `FragmentOut[width][height] depthbuffer -> vec3[width][height] framebuffer`
  * Simply copies the colors out of the depth buffer into the framebuffer
    (to be displayed on the screen).

### A Useful Pipeline

* Clear the depth buffer with some default value.
* Vertex shading: 
  * `VertexIn[n] vs_input -> VertexOut[n] vs_output`
  * Apply some vertex transformation (e.g. model-view-projection matrix using
    `glm::lookAt ` and `glm::perspective `).
* Primitive assembly.
  * `VertexOut[n] vs_output -> Triangle[n/3] primitives`
  * As above.
  * Other primitive types are optional.
* Rasterization.
  * `Triangle[n/3] primitives -> FragmentIn[m] fs_input`
  * You may choose to do a tiled rasterization method, which should have lower
    global memory bandwidth.
  * A scanline optimization: when rasterizing a triangle, only scan over the
    box around the triangle (`getAABBForTriangle`).
* Fragment shading.
  * `FragmentIn[m] fs_input -> FragmentOut[m] fs_output`
  * Add a shading method, such as Lambert or Blinn-Phong. Lights can be defined
    by kernel parameters (like GLSL uniforms).
* Fragments to depth buffer.
  * `FragmentOut[m] -> FragmentOut[width][height]`
  * Can really be done inside the fragment shader, if you call the fragment
    shader from the rasterization kernel for every fragment (including those
    which get occluded). **OR,** this can be done before fragment shading, which
    may be faster but means the fragment shader cannot change the depth.
    * This result in an optimization: it allows you to do depth tests before
     spending execution time in complex fragment shader code!
  * Handle race conditions! Since multiple primitives write fragments to the
    same fragment in the depth buffer, depth buffer locations must be locked
    while comparing the old and new fragment depths and (possibly) writing into
    it.
    * The `flower.obj` test file is good for testing race conditions.
* A depth buffer for storing and depth testing fragments.
  * `FragmentOut[width][height] depthbuffer`
  * An array of `fragment` objects.
  * At the end of a frame, it should contain the fragments drawn to the screen.
* Fragment to framebuffer writing.
  * `FragmentOut[width][height] depthbuffer -> vec3[width][height] framebuffer`
  * Simply copies the colors out of the depth buffer into the framebuffer
    (to be displayed on the screen).

This is a suggested sequence of pipeline steps, but you may choose to alter the
order of this sequence or merge entire kernels as you see fit.  For example, if
you decide that doing has benefits, you can choose to merge the vertex shader
and primitive assembly kernels, or merge the perspective transform into another
kernel. There is not necessarily a right sequence of kernels, and you may
choose any sequence that works.  Please document in your README what sequence
you choose and why.


## Resources

The following resources may be useful for this project:

* High-Performance Software Rasterization on GPUs:
  * [Paper (HPG 2011)](http://www.tml.tkk.fi/~samuli/publications/laine2011hpg_paper.pdf)
  * [Code](http://code.google.com/p/cudaraster/)
  * Note that looking over this code for reference with regard to the paper is
    fine, but we most likely will not grant any requests to actually
    incorporate any of this code into your project.
  * [Slides](http://bps11.idav.ucdavis.edu/talks/08-gpuSoftwareRasterLaineAndPantaleoni-BPS2011.pdf)
* The Direct3D 10 System (SIGGRAPH 2006) - for those interested in doing
  geometry shaders and transform feedback:
  * [Paper](http://dl.acm.org/citation.cfm?id=1141947)
  * [Paper, through Penn Libraries proxy](http://proxy.library.upenn.edu:2247/citation.cfm?id=1141947)
* Multi-Fragment Eﬀects on the GPU using the k-Buﬀer - for those who want to do
  order-independent transparency using a k-buffer:
  * [Paper](http://www.inf.ufrgs.br/~comba/papers/2007/kbuffer_preprint.pdf)
* FreePipe: A Programmable, Parallel Rendering Architecture for Efficient
  Multi-Fragment Effects (I3D 2010):
  * [Paper](https://sites.google.com/site/hmcen0921/cudarasterizer)
* Writing A Software Rasterizer In Javascript:
  * [Part 1](http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-1.html)
  * [Part 2](http://simonstechblog.blogspot.com/2012/04/software-rasterizer-part-2.html)


## Third-Party Code Policy

* Use of any third-party code must be approved by asking on our Google Group.
* If it is approved, all students are welcome to use it. Generally, we approve
  use of third-party code that is not a core part of the project. For example,
  for the path tracer, we would approve using a third-party library for loading
  models, but would not approve copying and pasting a CUDA function for doing
  refraction.
* Third-party code **MUST** be credited in README.md.
* Using third-party code without its approval, including using another
  student's code, is an academic integrity violation, and will, at minimum,
  result in you receiving an F for the semester.


## README

Replace the contents of this README.md in a clear manner with the following:

* A brief description of the project and the specific features you implemented.
* At least one screenshot of your project running.
* A 30 second or longer video of your project running.
* A performance analysis (described below).

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


## Submit

If you have modified any of the `CMakeLists.txt` files at all (aside from the
list of `SOURCE_FILES`), you must test that your project can build in Moore
100B/C. Beware of any build issues discussed on the Google Group.

1. Open a GitHub pull request so that we can see that you have finished.
   The title should be "Submission: YOUR NAME".
   * **ADDITIONALLY:**
     In the body of the pull request, include a link to your repository.
2. Send an email to the TA (gmail: kainino1+cis565@) with:
   * **Subject**: in the form of `[CIS565] Project N: PENNKEY`.
   * Direct link to your pull request on GitHub.
   * Estimate the amount of time you spent on the project.
   * If there were any outstanding problems, or if you did any extra
     work, *briefly* explain.
   * Feedback on the project itself, if any.
