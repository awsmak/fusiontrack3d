# Start from NVIDIA's CUDA image - this gives us GPU support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables to make installation smoother
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set up our working directory - this will be our project's home inside the container
WORKDIR /workspace

# Install system essentials - these are like the basic tools we need
RUN apt-get update && apt-get install -y     python3.9     python3-pip     python3.9-dev     git     wget     && rm -rf /var/lib/apt/lists/*

# Make Python 3.9 the default python version
RUN ln -sf /usr/bin/python3.9 /usr/bin/python

# Upgrade pip - this ensures we have the latest package installer
RUN python -m pip install --upgrade pip

# Copy requirements first - this helps with Docker's caching mechanism
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Set up Jupyter Notebook as our development interface
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]
