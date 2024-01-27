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

## Settting up with a Video 4 Linux device
Need to install and configure:

```bash
sudo apt install v4l2loopback-dkms v4l-utils
sudo modprobe v4l2loopback
scrcpy --v4l2-sink=/dev/video0 --no-display

# Test playback
ffplay -i /dev/video0 # FFMPEG
gst-launch-1.0 v4l2src ! xvimagesink # Gstreamer
```

## Building scrcpy from src
There is a temporary cmake target to build the application
```bash
cd extern
cmake -S . -B build
cmake --build build

# Set environment
export SCRCPY_SERVER_PATH=/root/repo/extern/scrcpy-server
./build/scrcpy_app --v4l2-sink=/dev/video0 --no-playback -Vverbose
```

## Running the application
On the host
```bash
# Start emulators and enable loopback
sudo modprobe v4l2loopback

# Bring up docker container
docker compose up -d
docker compose exec cuda-devel bash

# Build the application
cmake -S . -B build
cmake --build build -j$(nproc)

# Run the application
export SCRCPY_SERVER_PATH=/root/repo/extern/scrcpy-server
./build/apps/piano_tiles/PianoTiles -o output.avi
```

## WIP Running the app using a UI
Proved out sending commands from the [Simple DirectMedia Layer](https://wiki.libsdl.org/SDL2/FrontPage)
using the [touch event](https://wiki.libsdl.org/SDL2/SDL_TouchFingerEvent)
```bash
# Enable xhost locally
xhost + local:
```

## Without SDL or the scrcpy client
Implementing a C++ client that uses the same protocol allows us to send events to the server. To configure, copy the server over and forward the port:
```bash
adb push extern/scrcpy-server /data/local/tmp/scrcpy-server-manual.jar
adb forward tcp:1234 localabstract:scrcpy
adb shell CLASSPATH=/data/local/tmp/scrcpy-server-manual.jar     app_process / com.genymobile.scrcpy.Server 2.3.1     tunnel_forward=true audio=false control=true cleanup=false     raw_stream=false max_size=1920
```

## Encoding data for touch events
[Start the server](https://github.com/Genymobile/scrcpy/blob/master/app/src/adb/adb.c#L201-L206)
[Connect to the device](https://github.com/Genymobile/scrcpy/blob/master/app/src/server.c#L996)
https://github.com/Genymobile/scrcpy/blob/master/app/src/server.c#L548-L549
[Touch Action](https://github.com/Genymobile/scrcpy/blob/master/app/src/mouse_inject.c#L127)
[Input Events](https://github.com/Genymobile/scrcpy/blob/master/doc/develop.md#input-events-injection)
* Instead of SDL - we want to call this directly from our code using the [InputManager](https://github.com/Genymobile/scrcpy/blob/master/app/src/input_manager.h#L17)
* Ultimately I think we want to send a [virtual finger](https://github.com/Genymobile/scrcpy/blob/master/app/src/input_manager.c#L324C1-L324C24)
[Example controller](https://github.com/Genymobile/scrcpy/blob/master/app/src/controller.h)
[Example of serialization](https://github.com/Genymobile/scrcpy/blob/master/app/tests/test_control_msg_serialize.c#L115-L148)

References
* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html
* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
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
