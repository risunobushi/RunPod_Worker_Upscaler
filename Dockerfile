# Stage 1: Base image with common dependencies
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Add deadsnakes PPA for Python 3.12 and install Python, git and other base tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3.12-distutils \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    # Link python3.12 to python immediately
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    # Clean lists for this layer
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12 using get-pip.py method
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py \
    # Link pip now that it's installed
    && ln -sf /usr/local/bin/pip /usr/bin/pip

# Create and set permissions for ControlNet Aux caching
RUN mkdir -p /tmp/ckpts && chmod -R 777 /tmp/ckpts

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install comfy-cli
RUN pip install comfy-cli

# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --cuda-version 12.6 --nvidia --version 0.3.26

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install runpod
RUN pip install runpod requests

# Install PyTorch with CUDA 12.6 support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Install other required python packages that were previously in the large install list
RUN pip install accelerate==1.6.0 numba scikit-image onnxruntime-gpu yacs

# Install performance optimization packages
RUN pip install flash_attn triton

# Copy the custom model paths configuration BEFORE ComfyUI potentially reads defaults
# Also, rename the example file first to avoid potential conflicts
RUN mv extra_model_paths.yaml.example extra_model_paths.yaml.example.bak || true
COPY src/extra_model_paths.yaml .

# Support for the network volume
# ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add scripts
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Optionally copy the snapshot file
ADD *snapshot*.json /

# Restore the snapshot to install custom nodes
RUN /restore_snapshot.sh

# Start container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base as downloader

# ARG HUGGINGFACE_ACCESS_TOKEN # No longer needed as no models are downloaded in this stage

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories for models that will be copied to the final stage
# No models will be downloaded here; they are expected to be on the network volume.
RUN mkdir -p models/unet models/clip models/vae

# Ensure there's no empty continuation line before the next stage
# Stage 3: Final image
FROM base as final

# Reverted: Copy the original config file from the base stage
COPY --from=base /comfyui/extra_model_paths.yaml /comfyui/

# Debug: List contents of /comfyui to verify copy
RUN echo "--- Listing /comfyui contents during build (final stage) ---" && ls -lA /comfyui

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start container
CMD ["/start.sh"]
