# Base image with CUDA 11.8 and Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    MODEL_CACHE_DIR=/app/models \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRRARY_PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_NO_CACHE_DIR=off

# Install system dependencies first (use Ubuntu's Python 3.10)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    git \
    ca-certificates \
    ffmpeg \
    libsndfile1 \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Create model cache directory
RUN mkdir -p /app/models

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with retries and timeouts
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=100 --retries=5 \
    torch==2.0.1+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir --timeout=100 --retries=5 -r requirements.txt

# Copy application code
COPY app.py .

# Verify model cache directory is accessible
RUN python -c "\
import os; \
cache_dir = '/app/models'; \
print(f'Model cache directory: {cache_dir}'); \
print(f'Directory exists: {os.path.exists(cache_dir)}'); \
print(f'Directory writable: {os.access(cache_dir, os.W_OK)}'); \
test_file = os.path.join(cache_dir, 'test_permissions.txt'); \
with open(test_file, 'w') as f: \
    f.write('test'); \
os.remove(test_file); \
print('✅ Model cache directory is accessible and writable');"

# Clean up
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* ~/.cache/pip

# Run app
CMD ["python", "app.py"]