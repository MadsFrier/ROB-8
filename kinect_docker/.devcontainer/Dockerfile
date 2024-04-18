FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspaces/ROB-8/kinect_docker/

# install packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-setuptools \
    wget \
    gedit \
    unzip \
    curl \
    git \
    ffmpeg \
    v4l-utils \
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

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade Pillow

# copy
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY src/ src/
ENV PYTHONPATH /app

# install nvidia container essentials
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute,display
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && \
    apt-get update

# ------ Azure Kinect ------ #
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN apt-add-repository https://packages.microsoft.com/ubuntu/18.04/prod

COPY sources.list /etc/apt/sources.list
RUN apt-get update

RUN echo 'libk4a1.4 libk4a1.4/accepted-eula-hash string 0f5d5c5de396e4fee4c0753a21fee0c1ed726cf0316204edda484f08cb266d76' | debconf-set-selections

RUN echo 'libk4a1.4 libk4a1.4/accept-eula boolean true' | debconf-set-selections

RUN apt-get install -y k4a-tools
RUN apt-get install -y libk4a1.4-dev

COPY 99-k4a.rules /etc/udev/rules.d/99-k4a.rules

# ------------------------- #



# ---------- ROS ---------- #
# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA

# setup sources.list
RUN echo "deb http://snapshots.ros.org/melodic/final/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-snapshots.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO melodic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-ros-core=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-ros-base=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-perception=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*
# ------------------------- #



# -------- RTABMAP -------- #
# Install build dependencies
RUN apt-get update && \
    apt-get install -y git software-properties-common ros-melodic-rtabmap-ros && \
    apt-get remove -y ros-melodic-rtabmap* && \
    apt-get clean && rm -rf /var/lib/apt/lists/

WORKDIR /root/

# GTSAM
RUN add-apt-repository ppa:borglab/gtsam-release-4.0 -y
RUN apt-get update && apt install libgtsam-dev libgtsam-unstable-dev -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/

# MRPT # build failed: https://github.com/introlab/rtabmap/actions/runs/7259315306/job/19776230419
#RUN add-apt-repository ppa:joseluisblancoc/mrpt-stable -y
#RUN apt-get update && apt install libmrpt-poses-dev -y && \
#    apt-get clean && rm -rf /var/lib/apt/lists/

ARG TARGETPLATFORM
ENV TARGETPLATFORM=${TARGETPLATFORM:-linux/amd64}
RUN echo "I am building for $TARGETPLATFORM"

# cmake >=3.14 required
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then apt update && apt install -y wget && apt-get clean && rm -rf /var/lib/apt/lists/ && \
    wget -nv https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-aarch64.tar.gz   && \
    tar -xzf cmake-3.20.0-linux-aarch64.tar.gz  && \
    rm cmake-3.20.0-linux-aarch64.tar.gz  &&\
    ln -s ~/cmake-3.20.0-linux-aarch64/bin/cmake ~/cmake; fi
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then apt update && apt install -y wget && apt-get clean && rm -rf /var/lib/apt/lists/ && \
    wget -nv https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.tar.gz  && \
    tar -xzf cmake-3.20.0-linux-x86_64.tar.gz  && \
    rm cmake-3.20.0-linux-x86_64.tar.gz  &&\
    ln -s ~/cmake-3.20.0-linux-x86_64/bin/cmake ~/cmake; fi
    
#commit Aug 6 2020
#RUN apt-get update && apt install wget && apt-get clean && rm -rf /var/lib/apt/lists/
#RUN git clone https://github.com/laurentkneip/opengv.git && \
#    cd opengv && \
#    git checkout 91f4b19c73450833a40e463ad3648aae80b3a7f3 && \
#    wget https://gist.githubusercontent.com/matlabbe/a412cf7c4627253874f81a00745a7fbb/raw/accc3acf465d1ffd0304a46b17741f62d4d354ef/opengv_disable_march_native.patch && \
#    git apply opengv_disable_march_native.patch && \
#    mkdir build && \
#    cd build && \
#    ~/cmake -DCMAKE_BUILD_TYPE=Release .. && \
#    make -j$(nproc) && \
#    make install && \
#    cd && \
#    rm -r opengv

# Will be used to read/store databases on host
RUN mkdir -p /root/Documents/RTAB-Map

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Build RTAB-Map project
#RUN source /ros_entrypoint.sh && \
#    git clone https://github.com/introlab/rtabmap.git && \
#    cd rtabmap/build && \
#    ~/cmake -DWITH_OPENGV=ON .. && \
#    make -j$(nproc) && \
#    make install && \
#    cd ../.. && \
#    rm -rf rtabmap && \
#    ldconfig
# ------------------------- #

RUN apt update && apt upgrade -y

ENV DEBIAN_FRONTEND=dialog

RUN useradd -ms /bin/bash kinect
#RUN useradd -m kinect && echo "kinect:kinect" | chpasswd && adduser kinect sudo
#RUN usermod -a -G video kinect

USER kinect

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc

