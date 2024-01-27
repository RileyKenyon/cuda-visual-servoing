FROM nvcr.io/nvidia/cuda:11.3.1-base-ubuntu20.04 as cuda-base
CMD [ "/bin/bash" ]
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install \
    build-essential \
    vim \
    cmake \
    libopencv-dev
ENTRYPOINT ["tail","-f","/dev/null"]