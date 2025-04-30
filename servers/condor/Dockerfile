FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy server code and model implementation
COPY server.py .
COPY model/ ./model/

# Set environment variables for the model
ENV CONDOR_MODEL_SIZE="40b"  # Options: "7b" or "40b"
ENV CONDOR_PORT=8001
ENV CONDOR_HOST="0.0.0.0"
ENV CONDOR_MAX_LENGTH=2048
ENV CONDOR_TEMPERATURE=0.7

# Expose the port
EXPOSE 8001

# Command to run the server
CMD ["python3", "server.py"] 