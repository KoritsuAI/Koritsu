"""
Condor LLM Server Integration

This module implements a server for the Condor LLM using
our local model implementation.
"""

import os
import torch
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging

# Import our local model implementation
from model import CondorConfig, CondorTokenizer, CondorForCausalLM

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configure FastAPI app
app = FastAPI(title="Condor LLM Server", description="API for Condor language model inference")

# Model configuration
MODEL_SIZE = os.getenv("CONDOR_MODEL_SIZE", "40b")  # Options: "7b", "40b"
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

class ChatResponse(BaseModel):
    content: str = Field(..., description="Generated text response")
    model: str = Field(..., description="Model identifier")

# Global variables for model and tokenizer
tokenizer = None
model = None

@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer on startup."""
    global tokenizer, model
    
    model_name = f"condor-{MODEL_SIZE}"
    logger.info(f"Loading Condor model: {model_name}")
    logger.info(f"Using device: {DEVICE}")
    
    try:
        # Initialize configuration based on model size
        if MODEL_SIZE == "7b":
            config = CondorConfig.condor_7b_config()
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
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                num_return_sequences=1,
            )
        
        # Decode generated tokens
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
        
        # Remove the prompt from the response
        response_text = generated_text[len(prompt):].strip()
        
        return ChatResponse(
            content=response_text,
            model=f"condor-{MODEL_SIZE}"
        )
    
    except Exception as e:
        logger.error(f"Error generating completion: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model and tokenizer:
        return {"status": "healthy", "model": f"condor-{MODEL_SIZE}"}
    return {"status": "not_ready", "model": f"condor-{MODEL_SIZE}"}

if __name__ == "__main__":
    port = int(os.getenv("CONDOR_PORT", "8001"))
    host = os.getenv("CONDOR_HOST", "0.0.0.0")
    logger.info(f"Starting Condor server on {host}:{port}")
    uvicorn.run(app, host=host, port=port) 