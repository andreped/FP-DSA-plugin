# build docker image from nvidia/cuda image with CUDA 11.0
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

# This tells girder_worker to enable gpu if possible
LABEL com.nvidia.volumes.needed=nvidia_driver

ARG DEBIAN_FRONTEND=noninteractive

# maintainer of docker file
LABEL maintainer="André Pedersen andre.pedersen@sintef.no"

# set language, format, and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# install OpenGL
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y \
    apt-utils \
    curl \
    xz-utils \
    ocl-icd-libopencl1 \
    opencl-headers \
    clinfo \
    python3-pip \
    ;
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# install OpenGL
RUN apt-get install -y \
    libopengl0 \
    libgl1 \
    libusb-1.0-0-dev \
    libcairo2

# install convenience tools
RUN apt-get install -y \
    wget \
    vim \
    git \
    memcached

# install pyFAST dependencies
RUN pip3 install --default-timeout=1000 pyfast==4.6.0

# necessary for opencv in docker
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# install histomicstk necessary for communicating with HistomicsUI
RUN pip3 install histomicstk --find-links https://girder.github.io/large_image_wheels

# install additional dependencies useful for handling annotations and progress
RUN pip3 install opencv-python tqdm h5py

# set working directory and setup environment
ADD . /opt/fastpathology/dsa
WORKDIR /opt/fastpathology/dsa/cli
ENV PYTHONPATH=$PYTHONPATH:/opt/fastpathology/dsa

# copy fastpathology into working directory
COPY fastpathology .

# install fastpathology python package
RUN git clone https://github.com/andreped/FP-dsa-plugin.git \
  && cd FP-dsa-plugin/dsa/fastpathology/ \
  && pip install -e .

# test entrypoint
RUN python3 -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python3 -m slicer_cli_web.cli_list_entrypoint NucleiSegmentation --help
RUN python3 -m slicer_cli_web.cli_list_entrypoint FASTNucleiSegmentation --help
RUN python3 -m slicer_cli_web.cli_list_entrypoint FASTBachClassification --help

# define docker entrypoint
ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
