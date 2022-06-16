FROM nvidia/cudagl:11.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

ENV CUDNN_VERSION=8.0.5.39-1+cuda11.1

ARG python=3.8
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get update && apt-get install -y --allow-downgrades \
    --allow-change-held-packages --no-install-recommends \
    --allow-unauthenticated \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    wget \
    ca-certificates \
    libcudnn8=${CUDNN_VERSION} \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \ 
    libosmesa6 \ 
    libosmesa6-dev \ 
    patchelf \
    libglew-dev \
    libgl1-mesa-glx \
    xorg \
    openbox \
    tmux

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN /usr/bin/python -m pip install --upgrade pip

RUN adduser --disabled-password --gecos '' docker && \
    adduser docker sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p /.cache/pip
RUN mkdir -p /.local/share
RUN chown -R docker:docker /.cache/pip
RUN chown -R docker:docker /.local

USER docker 

WORKDIR /home/docker/

RUN chmod a+rwx /home/docker/ && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh && \
    bash Miniconda3-py39_4.9.2-Linux-x86_64.sh -b && rm Miniconda3-py39_4.9.2-Linux-x86_64.sh

ENV PATH /home/docker/miniconda3/bin:$PATH
RUN conda config --set allow_conda_downgrades true && conda config --env --set always_yes true && conda install conda=4.9.2 && conda install -c anaconda cudatoolkit=11.0

# Install pytorch for A100
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN mkdir /home/docker/.mujoco
COPY .mujoco /home/docker/.mujoco/

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r ./requirements.txt

## git clone before doing this. No need when can do with pip
# RUN mkdir /tmp/dmc2gym
# COPY dmc2gym /tmp/dmc2gym
# WORKDIR /tmp/dmc2gym
# RUN pip install -e .

USER docker
WORKDIR /home/docker/app

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/docker/.mujoco/mujoco210/bin"