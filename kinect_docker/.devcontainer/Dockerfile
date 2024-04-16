FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    # python3.10 \
    python-pip \
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
    # nvidia-container-toolkit \
    lshw \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    software-properties-common \
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

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod

COPY sources.list /etc/apt/sources.list
RUN apt-get update

RUN echo 'libk4a1.4 libk4a1.4/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' | debconf-set-selections

RUN echo 'libk4a1.4 libk4a1.4/accept-eula boolean true' | debconf-set-selections

RUN apt-get install -y k4a-tools
RUN apt-get install -y libk4a1.4-dev

COPY 99-k4a.rules /etc/udev/rules.d/99-k4a.rules

ENV DEBIAN_FRONTEND=dialog

RUN useradd -ms /bin/bash kinect

USER kinect