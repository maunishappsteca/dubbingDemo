# Base image with CUDA 11.8 and Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Create directories for models and outputs
RUN mkdir -p models

# Copy files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.1.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 
RUN pip install --no-cache-dir -r requirements.txt

# Run the app
CMD ["python", "app.py"]