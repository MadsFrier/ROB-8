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

# ---------- ROS ---------- #
RUN apt update && apt install locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8

RUN apt install software-properties-common -y
RUN add-apt-repository universe -y

RUN apt update && apt install curl -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN  apt update && apt upgrade -y

RUN apt install ros-humble-desktop -y

# setup entrypoint
COPY ./ros_entrypoint.sh /

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
# ------------------------- #



# -------- RTABMAP -------- #
# Install build dependencies
RUN apt-get update && \
    apt-get install -y git software-properties-common ros-humble-rtabmap-ros && \
    apt-get remove -y ros-humble-rtabmap* && \
    apt-get clean && rm -rf /var/lib/apt/lists/

WORKDIR /root/

# GTSAM
#RUN add-apt-repository ppa:borglab/gtsam-release-4.0 -y
#RUN apt-get update && apt install libgtsam-dev libgtsam-unstable-dev -y && \
#    apt-get clean && rm -rf /var/lib/apt/lists/

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
RUN apt-get update && apt install wget && apt-get clean && rm -rf /var/lib/apt/lists/
RUN git clone https://github.com/laurentkneip/opengv.git && \
    cd opengv && \
    git checkout 91f4b19c73450833a40e463ad3648aae80b3a7f3 && \
    wget https://gist.githubusercontent.com/matlabbe/a412cf7c4627253874f81a00745a7fbb/raw/accc3acf465d1ffd0304a46b17741f62d4d354ef/opengv_disable_march_native.patch && \
    git apply opengv_disable_march_native.patch && \
    mkdir build && \
    cd build && \
    ~/cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && \
    make install && \
    cd && \
    rm -r opengv

# Will be used to read/store databases on host
RUN mkdir -p /root/Documents/RTAB-Map

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Build RTAB-Map project
RUN source /ros_entrypoint.sh && \
    git clone https://github.com/introlab/rtabmap.git && \
    cd rtabmap/build && \
    ~/cmake -DWITH_OPENGV=ON .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf rtabmap && \
    ldconfig
# ------------------------- #

ENV DEBIAN_FRONTEND=dialog

RUN useradd -ms /bin/bash lm

USER lm