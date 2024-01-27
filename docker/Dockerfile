ARG CUDA_DOCKER_TAG=13.0.2-devel-ubuntu22.04
FROM nvcr.io/nvidia/cuda:${CUDA_DOCKER_TAG} AS builder
SHELL [ "/usr/bin/bash", "-c" ]
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install \
    build-essential \
    vim \
    unzip \
    wget \
    cmake \
    git \
    libopencv-dev \
    v4l2loopback-dkms \
    v4l-utils \
    ffmpeg \
    libsdl2-2.0-0 \
    libusb-1.0-0 \
    libgstreamer1.0-dev \
    pkg-config \
    meson \
    ninja-build \
    libsdl2-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libusb-1.0-0-dev

# Custom adb version to match host
RUN mkdir -p /usr/local/android-sdk \ 
    && cd /usr/local/android-sdk/ \
    && wget https://dl.google.com/android/repository/platform-tools_r29.0.5-linux.zip -O platform-tools.zip \
    && unzip platform-tools.zip \
    && rm platform-tools.zip \
    && ln -s /usr/local/android-sdk/platform-tools/adb /usr/bin/adb

# Custom cmake version to have better CUDA support
WORKDIR /usr/local
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.6/cmake-3.18.6-Linux-x86_64.sh -O install.sh
RUN chmod +x install.sh
RUN bash ./install.sh --skip-license

# Add a non-root user to avoid permission issues with the host
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -rm -d /home/$USERNAME -s /bin/bash -g $USERNAME -G sudo -u $USER_UID $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

CMD ["/bin/bash"]