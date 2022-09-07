# CUDA Visual Servoing
Spring 2019 Independent study with Shalom Ruben for game automation using Nvidia Jetson Nano. 

What is [visual servoing](https://en.wikipedia.org/wiki/Visual_servoing)?
> **Visual servoing**, also known as vision-based robot control and abbreviated VS, is a technique which uses feedback information extracted from a vision sensor (visual feedback]) to control the motion of a robot. 
## Setup
To build all targets, create a build directory and use cmake to configure and build the targets:
```
mkdir build && cd build
cmake ../
cmake --build .
```

Reference documentation for using CMake with CUDA:
https://developer.nvidia.com/blog/building-cuda-applications-cmake/