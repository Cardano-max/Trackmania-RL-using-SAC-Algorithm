FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install tmrl and rollout dependencies
RUN pip install --upgrade pip && \
    pip install tmrl==0.7.1 --no-deps && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install numpy pandas gymnasium rtgym pyyaml wandb requests opencv-python-headless pyautogui pyinstrument tlspyo chardet packaging mss pillow

COPY docker/entrypoints /app/entrypoints
COPY tmrl_templates /app/tmrl_templates
COPY scripts /app/scripts

RUN mkdir -p /TmrlData/config && \
    mkdir -p /TmrlData/logs && \
    mkdir -p /TmrlData/weights && \
    mkdir -p /TmrlData/dataset

VOLUME ["/TmrlData"]