# Base image with CUDA 11.8 and Python 3.10
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.10 as default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Set environment variables
ENV MODEL_CACHE_DIR=/app/models
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models

# Create model cache directory
RUN mkdir -p /app/models

# Copy files
COPY requirements.txt .
COPY app.py .

# Install Python dependencies with compatible versions
RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

# Install accelerate for better model loading
RUN pip install --no-cache-dir accelerate>=0.20.0

# Pre-download the SeamlessM4Tv2 model during build
RUN python -c "\
import os; \
from huggingface_hub import snapshot_download; \
model_name = 'facebook/seamless-m4t-v2-large'; \
cache_dir = os.getenv('MODEL_CACHE_DIR', '/app/models'); \
print(f'Downloading {model_name} to {cache_dir}'); \
try: \
    snapshot_download( \
        repo_id=model_name, \
        local_dir=os.path.join(cache_dir, 'seamless-m4t-v2-large'), \
        local_dir_use_symlinks=False, \
        resume_download=True, \
        max_workers=4, \
        ignore_patterns=['*.msgpack', '*.h5', '*.ot', '*.bin.index.json'] \
    ); \
    print('✅ SeamlessM4Tv2 model downloaded successfully'); \
    # Verify files were downloaded \
    import os; \
    model_path = os.path.join(cache_dir, 'seamless-m4t-v2-large'); \
    if os.path.exists(model_path): \
        files = os.listdir(model_path); \
        print(f'Downloaded files: {files}'); \
    else: \
        print('❌ Model directory not found'); \
except Exception as e: \
    print(f'❌ Model download failed: {e}'); \
    print('Will download during runtime instead');"

# Verify model files exist
RUN ls -la /app/models/ && \
    if [ -d "/app/models/seamless-m4t-v2-large" ]; then \
        echo "✅ Model directory exists"; \
        ls -la /app/models/seamless-m4t-v2-large/; \
    else \
        echo "❌ Model directory not found"; \
    fi

# Run the app
CMD ["python", "app.py"]