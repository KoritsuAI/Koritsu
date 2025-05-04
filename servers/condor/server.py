"""
Condor LLM Server Integration

This module implements a server for the Condor LLM using
our local model implementation.
"""

import os
import torch
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn
import logging

# Import our local model implementation
from model import CondorConfig, CondorTokenizer, CondorForCausalLM
from utils.logging_utils import setup_logger

# Set up logging
logger = setup_logger(
    name="condor_server",
    log_level=os.getenv("CONDOR_LOG_LEVEL", "INFO"),
    console_output=True,
    file_output=True
)

# Configure FastAPI app
app = FastAPI(title="Condor LLM Server", description="API for Condor language model inference")

# Model configuration
MODEL_SIZE = os.getenv("CONDOR_MODEL_SIZE", "40b")  # Options: "7b", "40b"
USE_MOE = os.getenv("CONDOR_USE_MOE", "0") == "1"  # Enable MoE if set to 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = int(os.getenv("CONDOR_MAX_LENGTH", "2048"))
TEMPERATURE = float(os.getenv("CONDOR_TEMPERATURE", "0.7"))

# Pydantic models for request/response
class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (system, user, or assistant)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of messages in the conversation")
    temperature: Optional[float] = Field(TEMPERATURE, description="Sampling temperature")
    max_tokens: Optional[int] = Field(MAX_LENGTH, description="Maximum number of tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    top_p: Optional[float] = Field(0.95, description="Top-p sampling parameter")
    top_k: Optional[int] = Field(50, description="Top-k sampling parameter")
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty")

class ChatResponse(BaseModel):
    content: str = Field(..., description="Generated text response")
    model: str = Field(..., description="Model identifier")

class ModelInfo(BaseModel):
    model_size: str = Field(..., description="Model size (7b or 40b)")
    use_moe: bool = Field(..., description="Whether the model uses Mixture of Experts")
    device: str = Field(..., description="Device the model is running on")
    max_length: int = Field(..., description="Maximum length of sequences")
    active: bool = Field(..., description="Whether the model is loaded and active")

# Global variables for model and tokenizer
tokenizer = None
model = None

@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer on startup."""
    global tokenizer, model
    
    model_type = "moe" if USE_MOE else "base"
    model_name = f"condor-{MODEL_SIZE}-{model_type}"
    logger.info(f"Loading Condor model: {model_name}")
    logger.info(f"Using device: {DEVICE}")
    
    try:
        # Initialize configuration based on model size and type
        if MODEL_SIZE == "7b":
            if USE_MOE:
                logger.info("Using Mixture of Experts configuration")
                config = CondorConfig.condor_7b_moe_config()
            else:
                config = CondorConfig.condor_7b_config()
        else:
            if USE_MOE:
                logger.info("Using Mixture of Experts configuration")
                config = CondorConfig.condor_40b_moe_config()
            else:
                config = CondorConfig.condor_40b_config()
        
        # Initialize tokenizer
        tokenizer = CondorTokenizer.from_pretrained("model/weights")
        
        # Initialize model with our configuration
        model = CondorForCausalLM(config)
        
        # Move model to appropriate device
        if DEVICE == "cuda":
            # Use dtype bfloat16 if available for better performance/memory usage
            dtype = torch.bfloat16 if torch.cuda.is_available() and hasattr(torch, 'bfloat16') else torch.float16
            model = model.to(DEVICE).to(dtype)
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def generate_chat_completion(request: ChatRequest):
    """Generate a response from the Condor model based on chat messages."""
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Format messages into a prompt
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        
        prompt += "Assistant: "
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048 - request.max_tokens
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate response
        logger.info(f"Generating response with temp={request.temperature}")
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=len(inputs["input_ids"][0]) + request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0.0,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                num_return_sequences=1,
            )
        
        # Decode generated tokens
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
        
        # Remove the prompt from the response
        response_text = generated_text[len(prompt):].strip()
        
        model_type = "moe" if USE_MOE else "base"
        model_name = f"condor-{MODEL_SIZE}-{model_type}"
        
        return ChatResponse(
            content=response_text,
            model=model_name
        )
    
    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model and tokenizer:
        model_type = "moe" if USE_MOE else "base"
        return {"status": "healthy", "model": f"condor-{MODEL_SIZE}-{model_type}"}
    return {"status": "not_ready"}

@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get information about the loaded model."""
    return ModelInfo(
        model_size=MODEL_SIZE,
        use_moe=USE_MOE,
        device=DEVICE,
        max_length=MAX_LENGTH,
        active=model is not None and tokenizer is not None
    )

if __name__ == "__main__":
    port = int(os.getenv("CONDOR_PORT", "8001"))
    host = os.getenv("CONDOR_HOST", "0.0.0.0")
    logger.info(f"Starting Condor server on {host}:{port}")
    uvicorn.run(app, host=host, port=port) 