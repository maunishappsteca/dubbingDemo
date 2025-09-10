FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    MODEL_CACHE_DIR=/app/models \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models \
    PYTHONUNBUFFERED=1

# Install system dependencies including git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common wget curl git ca-certificates \
    ffmpeg libsndfile1 python3.10 python3.10-dev python3.10-distutils python3.10-venv && \
    rm -rf /var/lib/apt/lists/*

# Set up Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN mkdir -p /app/models

COPY requirements.txt .
COPY app.py .

# Install packages in a single command to avoid layer issues
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Starts the app
CMD ["python", "app.py"]