FROM nvcr.io/nvidia/cuda:11.3.1-devel-ubuntu20.04 as cuda-base
CMD [ "/bin/bash" ]
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install \
    build-essential \
    vim \
    cmake \
    libopencv-dev
RUN apt-get -y install wget
WORKDIR /usr/local
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.6/cmake-3.18.6-Linux-x86_64.sh -O install.sh
RUN chmod +x install.sh
RUN bash ./install.sh --skip-license
ENTRYPOINT ["tail","-f","/dev/null"]