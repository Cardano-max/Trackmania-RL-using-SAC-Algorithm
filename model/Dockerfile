FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    fastapi \
    uvicorn[standard] \
    aiohttp \
    numpy \
    pydantic \
    python-multipart

# Create app directory
WORKDIR /app

# Copy model server
COPY model_server.py /app/
COPY requirements.txt /app/ 2>/dev/null || echo "torch\nfastapi\nuvicorn[standard]\naiohttp\nnumpy\npydantic" > /app/requirements.txt

# Install any additional requirements
RUN pip install --no-cache-dir -r /app/requirements.txt

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 8081

# Start application
CMD ["python3", "/app/model_server.py"]