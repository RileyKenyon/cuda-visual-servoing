# CUDA Visual Servoing
Spring 2019 Independent study with Shalom Ruben for game automation using Nvidia Jetson Nano. 

What is [visual servoing](https://en.wikipedia.org/wiki/Visual_servoing)?
> **Visual servoing**, also known as vision-based robot control and abbreviated VS, is a technique which uses feedback information extracted from a vision sensor (visual feedback]) to control the motion of a robot.


## Running
Examples running the main application
```bash
./Main -i media/pineapple.jpeg -o media/output/test_%02d.jpeg
./Main -i media/example.avi -o media/output/video.avi
```

## Setup
To build all targets, create a build directory and use cmake to configure and build the targets:
```
mkdir build && cd build
cmake ../
cmake --build .
```

## Conan
Setting up conan using a python virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install conan
```

For future reference, the process used to create the conan package with a sample import test
```
conan new <package-name>/<version> -t
```

Ammend the conanfile.py with the appropriate fields

Reference documentation for using CMake with CUDA:
https://developer.nvidia.com/blog/building-cuda-applications-cmake/

## Using the Nvidia Container Tookit
Need to define the environment variables for the container:
```
NVIDIA_VISIBLE_DEVICES="all"
NVIDIA_DRIVER_CAPABILITIES="compute,utility"
REPO_ROOT=<path-to-your-repo>
```

Using the image from nvidia:
```
cd docker
docker compose build
docker compose up -d
docker compose exec cuda-devel bash
```

The nvidia runtime is added here: `/etc/docker/daemon.json`


References
* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html
* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html
* https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda
* https://docs.docker.com/compose/gpu-support/
* https://github.com/compose-spec/compose-spec/blob/master/deploy.md#driver
* https://github.com/NVIDIA/nvidia-docker/issues/1643
* https://catalog.ngc.nvidia.com/orgs/nvidia/teams/k8s/containers/container-toolkit
* https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda/tags
* https://compose-spec.io/
* https://nvidia.github.io/libcudacxx/
* https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
