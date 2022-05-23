# IndependentStudy_SP2019
Spring 2019 Independent study with Shalom Ruben for game automation using Nvidia Jetson Nano. 

## Setup
To build all targets, create a build directory and use cmake to configure and build the targets:
```
mkdir build && cd build
cmake ../
cmake --build .
```

Reference documentation for using CMake with CUDA:
https://developer.nvidia.com/blog/building-cuda-applications-cmake/