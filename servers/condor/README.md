# Condor LLM

Condor is an open-source large language model with OpenAI-compatible API endpoints built on transformer architecture. It provides high-quality text generation capabilities with support for both standard and Mixture of Experts (MoE) architectures.

## Features

- Standard and Mixture of Experts (MoE) variants
- Support for 7B and 40B parameter sizes
- OpenAI-compatible API endpoints
- Docker containerization with GPU support
- Benchmarking tools for performance and accuracy evaluation

## Configuration

The model can be configured using environment variables:

- `CONDOR_MODEL_SIZE`: Model size to use (`7b` or `40b`, default: `40b`)
- `CONDOR_USE_MOE`: Whether to use the Mixture of Experts variant (`0` or `1`, default: `0`)
- `CONDOR_PORT`: Port for the server (default: `8001`)
- `CONDOR_HOST`: Host for the server (default: `0.0.0.0`)
- `CONDOR_MAX_LENGTH`: Maximum token length for generated responses (default: `2048`)
- `CONDOR_TEMPERATURE`: Sampling temperature (default: `0.7`)
- `CONDOR_LOG_LEVEL`: Logging level (default: `INFO`)
- `CONDOR_LOG_DIR`: Directory for log files (default: `logs`)

## Mixture of Experts (MoE) Architecture

The Condor model supports a Mixture of Experts architecture which can significantly improve model accuracy while maintaining computational efficiency. The MoE implementation includes:

### 1. MoE Design

Condor's MoE design replaces standard feed-forward layers with expert-based layers where:

- Each token is routed to a subset of experts based on the token's content
- Experts specialize in different aspects of language understanding
- The model dynamically selects the most relevant experts for each token
- Weights from multiple experts are combined to produce the final output

### 2. Router Types

Condor supports two types of routing mechanisms:

- **Top-K Router**: Uses learned routing based on token content to select the top-k experts for each token
- **Hash Router**: Uses a deterministic hash-based approach for expert assignment

### 3. Load Balancing

To ensure efficient usage of experts, the implementation includes:

- Z-loss to prevent the router from assigning all tokens to a single expert
- Load balancing loss to encourage even distribution of tokens across experts
- Expert dropout for improved generalization

### 4. Performance Benefits

MoE configurations provide several advantages:

- **Higher accuracy**: Experts can specialize in different types of content
- **Better parameter efficiency**: More parameters without increasing compute
- **Improved scaling**: Can scale to larger models with lower inference costs

## Using the Models

### Standard Model

```bash
export CONDOR_MODEL_SIZE=7b
export CONDOR_USE_MOE=0
docker-compose up
```

### MoE Model

```bash
export CONDOR_MODEL_SIZE=7b
export CONDOR_USE_MOE=1
docker-compose up
```

## API Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

Request body:

```json
{
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello, who are you?" }
  ],
  "temperature": 0.7,
  "max_tokens": 100
}
```

### Health Check

```
GET /health
```

### Model Information

```
GET /model/info
```

## Benchmarking

A benchmarking framework is provided to evaluate and compare different Condor variants:

```bash
cd servers/condor
python benchmark/benchmark.py --model-sizes 7b --output benchmark/results/comparison.json
```

Options:

- `--model-sizes`: Model sizes to benchmark (`7b`, `40b`)
- `--no-baseline`: Skip benchmarking the standard model
- `--no-moe`: Skip benchmarking the MoE model
- `--cpu`: Force using CPU even if CUDA is available
- `--output`: Output file path for benchmark results

## Docker Setup

Build and run the Docker container:

```bash
docker-compose build
docker-compose up
```

For GPU support, ensure you have the NVIDIA Container Toolkit installed.
