# Condor LLM Server

This directory contains the implementation of a local Condor LLM (Large Language Model) server with the complete model source included in the repository. The server exposes an API endpoint compatible with the OpenAI chat completions format.

## Features

- Contains the complete Condor model implementation code
- Self-contained model without external dependencies on Hugging Face
- FastAPI-based REST API with OpenAI-compatible endpoints
- Docker containerization with GPU support
- Configurable via environment variables

## Requirements

- NVIDIA GPU with at least 40GB VRAM (for 40B model)
- Docker and Docker Compose
- NVIDIA Container Toolkit for GPU support

## Directory Structure

- `model/`: Contains the Condor model implementation code
  - `modeling_condor.py`: Core model architecture
  - `configuration_condor.py`: Model configuration
  - `tokenizer_condor.py`: Tokenizer implementation
  - `weights/`: Directory for model weights
- `server.py`: FastAPI server implementation
- `Dockerfile`: Docker configuration for the server
- `docker-compose.yml`: Docker Compose configuration
- `requirements.txt`: Python dependencies

## Environment Variables

The server can be configured using the following environment variables:

- `CONDOR_PORT`: Port for the server (default: 8001)
- `CONDOR_HOST`: Host to bind the server (default: "0.0.0.0")
- `CONDOR_MAX_LENGTH`: Maximum token length for generation (default: 2048)
- `CONDOR_TEMPERATURE`: Sampling temperature (default: 0.7)

## Quick Start

1. Build and start the Docker container:

```bash
docker-compose up -d
```

2. Check if the server is running properly:

```bash
curl http://localhost:8001/health
```

## API Usage

### Chat Completions

You can use the server with the following API endpoint:

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Tell me about the Condor AI model."}
    ],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

## Integration with Koritsu

To integrate the Condor server with the main Koritsu framework, update the `config.py` file to add support for the Condor model. This will allow the framework to use Condor alongside other LLM providers.

## Troubleshooting

- If you encounter CUDA out-of-memory errors, try reducing the context length or batch size
- Adjust the `CONDOR_MAX_LENGTH` environment variable to limit token generation if needed
- Check the container logs for detailed error messages:

```bash
docker logs condor-server
```

## Model Information

The included Condor model is:

- Based on the Condor architecture developed by the Technology Innovation Institute (TII)
- Built as a transformer-based decoder-only language model
- Optimized for inference performance in this implementation
- Open-source under the Apache 2.0 license

For more information, visit: [Condor-40B on Hugging Face](https://huggingface.co/tiiuae/condor-40b-instruct)
