# ==============================================================================
# MOSAICX Docker Image
# Self-contained with Ollama and LLM models for medical document extraction
# ==============================================================================
# 
# This Dockerfile creates a fully self-contained MOSAICX environment with:
#   - MOSAICX CLI and Python library
#   - Ollama inference server
#   - Pre-downloaded LLM models (gpt-oss:120b and gpt-oss:20b)
#
# Usage:
#   docker build -t mosaicx:latest .
#   docker run -it --gpus all -v /path/to/data:/data mosaicx:latest extract /data/doc.pdf --schema /data/schema.py
#
# ==============================================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="DIGIT-X Lab <lalith.shiyam@med.uni-muenchen.de>"
LABEL org.opencontainers.image.title="MOSAICX"
LABEL org.opencontainers.image.description="Medical cOmputational Suite for Advanced Intelligent eXtraction"
LABEL org.opencontainers.image.version="1.4.2"
LABEL org.opencontainers.image.source="https://github.com/DIGIT-X-Lab/MOSAICX"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Ollama configuration
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_MODELS=/root/.ollama/models

# MOSAICX configuration
ENV MOSAICX_LOG_DIR=/root/.mosaicx/logs

# ==============================================================================
# System Dependencies
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    wget \
    git \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    zstd \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ==============================================================================
# Install Ollama
# ==============================================================================
RUN curl -fsSL https://ollama.com/install.sh | sh

# ==============================================================================
# Install MOSAICX
# ==============================================================================
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY mosaicx/ ./mosaicx/

# Install MOSAICX
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ==============================================================================
# Download Models (done at build time for self-contained image)
# ==============================================================================
# Note: This creates a large image (~80GB+) but makes it fully self-contained
# To skip model download at build time, comment out the RUN command below
# and models will be pulled on first run instead

RUN echo "Starting Ollama server for model download..." \
    && ollama serve & \
    sleep 5 \
    && echo "Pulling gpt-oss:20b model..." \
    && ollama pull gpt-oss:20b \
    && echo "Pulling gpt-oss:120b model..." \
    && ollama pull gpt-oss:120b \
    && echo "Models downloaded successfully!" \
    && pkill ollama || true

# ==============================================================================
# Entrypoint Script
# ==============================================================================
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ==============================================================================
# Runtime Configuration
# ==============================================================================
WORKDIR /data
VOLUME ["/data", "/root/.mosaicx", "/root/.ollama"]

EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
