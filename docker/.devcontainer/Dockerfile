FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    wget \
    gedit \
    unzip \
    curl \
    git \
    ffmpeg \
    python-setuptools \
    lsb-release \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    nvidia-container-toolkit \
    lshw \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    #         #
    #         # Clean up
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src/ src/
ENV PYTHONPATH /app

ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute,display

RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && \
    apt-get update

ENV DEBIAN_FRONTEND=dialog

RUN useradd -ms /bin/bash lm

USER lm