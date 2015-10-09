CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3

siqi Huang Tested on: Windows 7, Inter(R) Core(TM) i7-4870 HQ CPU@ 2.5GHz; GeForce GT 750M(GK107) (Personal Computer)

Representative Images:
![](image/earth_sample_explain.bmp)
![](image/earth_sample.bmp)
![](image/dragon_onfire.bmp)
![](image/dragon_onfire_infog.bmp)

Video Demos:
[![ScreenShot](image/earth_screenshot.png)](https://youtu.be/fJt1fT1zZMo)
[![ScreenShot](image/dragon_screenshot.png)](https://youtu.be/PqhqiYVQujU)

PART O: Program Control
use R to rotate;
use A,D and W,S to move the light source
use Up and Down arrow to adjust the distance to the object
use Enter to start record images
use Backspace to end record images
use mouse scroll to adjust fovy angle
use mouse move to move around the object.

PART I: Camera Setup
For this project, there is not a very clear idea about the camera, but you still have to do model-view-projection transformation to get the basic vertex input. In this stage, I have used the interactive control from both keyboard and mouse input(). 

