FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    aiohttp \
    numpy \
    pydantic \
    python-multipart \
    websockets

# Create app directory
WORKDIR /app

# Copy viewer server
COPY viewer_server.py /app/

# Create requirements file
RUN echo "fastapi\nuvicorn[standard]\naiohttp\nnumpy\npydantic\nwebsockets" > /app/requirements.txt

# Install any additional requirements
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create directories
RUN mkdir -p /data /app/static

# Expose port
EXPOSE 3000

# Start application
CMD ["python3", "/app/viewer_server.py"]