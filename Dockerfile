FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    MODEL_CACHE_DIR=/app/models \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common wget curl git ca-certificates \
    ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils python3.10-venv && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10 && \
    ln -sf /usr/bin/python3 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN mkdir -p /app/models

COPY requirements.txt .
COPY app.py .


RUN pip install --upgrade pip
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt


# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir torch==2.0.1+cu118 torchaudio==2.0.2+cu118 \
#         --index-url https://download.pytorch.org/whl/cu118 \
#         --extra-index-url https://pypi.org/simple && \
#     pip install --no-cache-dir -r requirements.txt --extra-index-url https://pypi.org/simple

# Pre-pull model into cache (optional but helps avoid slow startup)
# RUN python -c "from transformers import AutoProcessor, SeamlessM4Tv2Model; \
#     AutoProcessor.from_pretrained('facebook/seamless-m4t-v2-large', cache_dir='/app/models'); \
#     SeamlessM4Tv2Model.from_pretrained('facebook/seamless-m4t-v2-large', cache_dir='/app/models')"

#Starts the app
CMD ["python", "app.py"]
