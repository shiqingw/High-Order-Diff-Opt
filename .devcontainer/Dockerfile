ARG UBUNTU_VER=22.04
# ARG CONDA_VER=latest
# ARG OS_TYPE=x86_64
# ARG OS_TYPE=aarch64

FROM ubuntu:${UBUNTU_VER}

# FROM ros:humble

# Use bash shell
SHELL ["/bin/bash", "--login", "-c"]

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive
ENV PATH="/usr/local/bin:${PATH}"

# System packages 
RUN apt-get update -y \
    && apt-get upgrade -y

RUN apt -y purge binutils \
    && apt -y remove make \
    && apt -y autoremove 
    
RUN apt-get install -y vim \
    net-tools \
    cmake \
    build-essential \
    git \
    liblapack-dev \
    libopenblas-dev \
    libeigen3-dev \
    libboost-all-dev \
    # libglib2.0-dev \
    python3-pip 
    # libpoco-dev

# Install minimal LaTeX packages for English
RUN apt-get install -y --no-install-recommends \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-base \
    cm-super \
    dvipng \
    && texhash

# Display
# RUN apt-get install -y -q --no-install-recommends \
#     libglvnd0 \
#     libgl1 \
#     libglx0 \
#     libegl1 \
#     libxext6 \
#     libx11-6 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Install the required Python packages
RUN pip3 install numpy \
    scipy==1.11.4 \
    matplotlib \
    proxsuite \
    pin \
    mujoco \
    osqp \
    ipykernel \
    "pybind11[global]" \
    sympy
    # empy==3.3.4 \
    # catkin_pkg \
    # lark \

# Install xtl
RUN git clone https://github.com/xtensor-stack/xtl.git \
    && cd xtl \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf xtl

# # Install xsimd
RUN git clone https://github.com/xtensor-stack/xsimd.git \
    && cd xsimd \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf xsimd

# Install xtensor
RUN git clone https://github.com/shiqingw/xtensor.git \
    && cd xtensor \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf xtensor

# Install xtensor-blas
RUN git clone https://github.com/shiqingw/xtensor-blas.git \
    && cd xtensor-blas \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf xtensor-blas

# Install xtensor-python
RUN git clone https://github.com/shiqingw/xtensor-python.git \
    && cd xtensor-python \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf xtensor-python

# Install scs
RUN git clone https://github.com/cvxgrp/scs.git \
    && cd scs \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX='/usr/local' .. \
    && make \
    && make install \
    && cd / \
    && rm -rf scs

# Install liegroups
RUN git clone https://github.com/utiasSTARS/liegroups.git \
    && cd liegroups \
    && pip install -e . 
    
# Install Scaling-Functions-Helper
RUN git clone https://github.com/shiqingw/Scaling-Functions-Helper.git\
    && cd Scaling-Functions-Helper \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install \
    && cd .. \
    && pip install -e .
    
# Install HOCBF-Helper
RUN git clone https://github.com/shiqingw/HOCBF-Helper.git\
    && cd HOCBF-Helper \
    && pip install -e .

# Clean up
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ~/.cache/pip

# Spin the container
CMD ["tail", "-f", "/dev/null"]