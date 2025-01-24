ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends git
    
# Install python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;
    
# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install netron numba==0.48.0 numpy==1.20.0 nuscenes_devkit==1.1.9 Pillow==9.5.0 pycuda==2022.1 pyquaternion==0.9.9 \
    scipy==1.7.3 setuptools==59.8.0 Shapely==1.8.4 tqdm==4.64.0 opencv-python==4.5.5.64 \
    yapf==0.40.1 onnx_graphsurgeon onnx spconv 

RUN apt-get install -y libgl1-mesa-glx

# Install PyTorch
RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 

# Install MMLab
ARG TORCH_CUDA_ARCH_LIST="7.5;6.1;8.6"
ENV FORCE_CUDA="1"

RUN cd / && \
    git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && git checkout v1.5.2 && \
    pip3 install -r requirements/optional.txt && \
    MMCV_WITH_OPS=1 pip3 install -e .

RUN cd / && \
    git clone https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection && git checkout v2.24.0 && \
    pip3 install -v -e .

RUN pip3 install mmsegmentation==0.24.0 