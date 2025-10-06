FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    xvfb \
    x11vnc \
    fluxbox \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    aiohttp \
    numpy \
    pydantic \
    python-multipart

# Create app directory
WORKDIR /app

# Copy environment server
COPY environment_server.py /app/
COPY requirements.txt /app/ 2>/dev/null || echo "fastapi\nuvicorn[standard]\naiohttp\nnumpy\npydantic" > /app/requirements.txt

# Install any additional requirements
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create data directory
RUN mkdir -p /data

# Expose ports
EXPOSE 8080 5900

# Start VNC server and application
CMD ["python3", "/app/environment_server.py"]