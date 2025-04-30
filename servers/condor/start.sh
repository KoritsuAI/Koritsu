#!/bin/bash

# Start script for Condor server

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print banner
echo -e "${GREEN}"
echo "=============================================="
echo "   Condor LLM Server - Startup Script"
echo "=============================================="
echo -e "${NC}"

# Check for docker and docker-compose
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check if NVIDIA drivers and runtime are available
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected. Using GPU acceleration.${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}No NVIDIA GPU detected. The model will run on CPU (very slow).${NC}"
    GPU_AVAILABLE=false
fi

# Generate model weights if they don't exist
echo "Checking for model weights..."
if [ ! -f "./model/weights/condor_7b_weights.pt" ] && [ ! -f "./model/weights/condor_40b_weights.pt" ]; then
    echo "Generating model weights (this will only happen once)..."
    cd model/weights
    python3 generate_weights.py --model_size 7b --fp16
    cd ../..
    echo -e "${GREEN}Model weights generated successfully.${NC}"
fi

# Start the server
echo "Starting Condor server..."
docker-compose up -d

# Wait for server to be ready
echo "Waiting for server to be ready..."
attempt=1
max_attempts=30
while [ $attempt -le $max_attempts ]; do
    echo -n "."
    if curl -s http://localhost:8001/health | grep -q "healthy"; then
        echo
        echo -e "${GREEN}Server is ready!${NC}"
        echo 
        echo "You can now use the Condor model with the following settings:"
        echo "API endpoint: http://localhost:8001/v1/chat/completions"
        echo
        echo "To use with Koritsu, set the following environment variables:"
        echo "  DEFAULT_MODEL_PROVIDER=CONDOR"
        echo "  CONDOR_API_URL=http://localhost:8001"
        echo "  CONDOR_MODEL_NAME=condor-40b"
        echo
        echo "To stop the server, run: docker-compose down"
        exit 0
    fi
    
    sleep 2
    attempt=$((attempt + 1))
done

echo
echo -e "${YELLOW}Server did not become ready in the expected time."
echo "You can check the logs with: docker-compose logs -f${NC}"
exit 1 