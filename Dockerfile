FROM nvidia/cuda:11.0-base-ubuntu20.04

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
    openbox

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN /usr/bin/python -m pip install --upgrade pip

# Install pytorch
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install -r requirements.txt




