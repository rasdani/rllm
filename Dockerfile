ARG CUDA_VERSION="12.4.1"
ARG CUDNN_VERSION=""
ARG UBUNTU_VERSION="22.04"

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    git-lfs \
    build-essential \
    ninja-build \
    libaio-dev \
    pkg-config \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN git lfs install

# Setup user with args from compose file
ARG USER_ID
ARG GROUP_ID
ARG USERNAME

RUN groupadd -g ${GROUP_ID} ${USERNAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir -p /home/${USERNAME}/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/${USERNAME}/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

ENV PATH="/home/${USERNAME}/miniconda3/bin:${PATH}"

# Create Python environment
ARG ENV_NAME
ARG PYTHON_VERSION
RUN conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
ENV PATH="/home/${USERNAME}/miniconda3/envs/${ENV_NAME}/bin:${PATH}"

# Set working directory
WORKDIR /workspace
RUN chown ${USERNAME}:${USERNAME} /workspace

# Install PyTorch and other dependencies
ARG PYTORCH_VERSION="2.5.1"
ARG CUDA="124"
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Create directories for cache and data
RUN mkdir -p /home/${USERNAME}/.cache/huggingface && \
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cache

# Switch to the non-root user
USER ${USERNAME}

# Initialize conda for the user
RUN conda init bash && \
    echo "conda activate ${ENV_NAME}" >> /home/${USERNAME}/.bashrc

# Set default command
CMD ["/bin/bash"] 
