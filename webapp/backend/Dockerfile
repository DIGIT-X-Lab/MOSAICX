# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install webapp requirements first (for Docker cache)
COPY webapp/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install MOSAICX in development mode
# Copy the entire MOSAICX project (context is parent directory)
COPY . /mosaicx-source
RUN cd /mosaicx-source && pip install -e .

# Copy webapp backend code
COPY webapp/backend/ .

# Create directories for uploads and schemas
RUN mkdir -p /tmp/mosaicx_uploads /tmp/mosaicx_schemas

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]