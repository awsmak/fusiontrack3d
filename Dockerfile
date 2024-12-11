# Start from NVIDIA's CUDA image to support GPU operations
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /workspace

# Install system essentials - now using Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    unzip \
    # GUI and X11 support
    python3-tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.10 the default python version
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip to its latest version
RUN python -m pip install --upgrade pip

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Command to run when container starts
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]