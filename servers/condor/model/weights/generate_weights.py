"""
Generate random weights for Condor model.

This script generates random weights for demonstration purposes.
In a real implementation, these weights would be trained or converted from a pretrained model.
"""

import os
import torch
import numpy as np
from typing import Dict, Optional

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


def generate_random_weights(
    model_size: str = "7b",
    output_dir: str = ".",
    fp16: bool = True,
) -> None:
    """
    Generate random weights for a Condor model.
    
    Args:
        model_size: Size of the model ("7b" or "40b")
        output_dir: Directory to save weights
        fp16: Whether to save weights in fp16 format
    """
    print(f"Generating random weights for Condor-{model_size}...")
    
    # Set model parameters based on size
    if model_size == "7b":
        vocab_size = 65024
        hidden_size = 4544
        num_hidden_layers = 32
        num_attention_heads = 71
        num_kv_heads = 1
    else:  # 40b
        vocab_size = 65024
        hidden_size = 8192
        num_hidden_layers = 60
        num_attention_heads = 128
        num_kv_heads = 8
    
    # Create dictionary to store weights
    state_dict = {}
    
    # Word embeddings
    print("Generating token embeddings...")
    state_dict["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size).div_(np.sqrt(hidden_size))
    
    # Tied LM head
    state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    
    # Generate weights for each layer
    print(f"Generating weights for {num_hidden_layers} layers...")
    for layer_idx in range(num_hidden_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        
        # Layer norm
        state_dict[layer_prefix + "input_layernorm.weight"] = torch.ones(hidden_size)
        state_dict[layer_prefix + "input_layernorm.bias"] = torch.zeros(hidden_size)
        
        # Additional layer norm for non-parallel attention models
        if model_size == "40b":
            state_dict[layer_prefix + "post_attention_layernorm.weight"] = torch.ones(hidden_size)
            state_dict[layer_prefix + "post_attention_layernorm.bias"] = torch.zeros(hidden_size)
        
        # Self-attention
        attn_prefix = layer_prefix + "self_attention."
        
        # Query projection
        state_dict[attn_prefix + "q_proj.weight"] = torch.randn(
            num_attention_heads * (hidden_size // num_attention_heads), 
            hidden_size
        ).div_(np.sqrt(hidden_size))
        
        # Key projection
        state_dict[attn_prefix + "k_proj.weight"] = torch.randn(
            num_kv_heads * (hidden_size // num_kv_heads), 
            hidden_size
        ).div_(np.sqrt(hidden_size))
        
        # Value projection
        state_dict[attn_prefix + "v_proj.weight"] = torch.randn(
            num_kv_heads * (hidden_size // num_kv_heads), 
            hidden_size
        ).div_(np.sqrt(hidden_size))
        
        # Output projection
        state_dict[attn_prefix + "o_proj.weight"] = torch.randn(
            hidden_size, 
            num_attention_heads * (hidden_size // num_attention_heads)
        ).div_(np.sqrt(hidden_size))
        
        # MLP
        mlp_prefix = layer_prefix + "mlp."
        
        # MLP up projection
        state_dict[mlp_prefix + "dense_h_to_4h.weight"] = torch.randn(
            4 * hidden_size, 
            hidden_size
        ).div_(np.sqrt(hidden_size))
        
        # MLP down projection
        state_dict[mlp_prefix + "dense_4h_to_h.weight"] = torch.randn(
            hidden_size, 
            4 * hidden_size
        ).div_(np.sqrt(4 * hidden_size))
    
    # Final layer norm
    state_dict["model.norm.weight"] = torch.ones(hidden_size)
    state_dict["model.norm.bias"] = torch.zeros(hidden_size)
    
    # Convert to FP16 if requested
    if fp16:
        print("Converting weights to FP16...")
        for key in state_dict:
            state_dict[key] = state_dict[key].to(torch.float16)
    
    # Save weights
    output_path = os.path.join(output_dir, f"condor_{model_size}_weights.pt")
    torch.save(state_dict, output_path)
    print(f"Weights saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate random weights for Condor model")
    parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "40b"], help="Model size")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--fp16", action="store_true", help="Save weights in FP16 format")
    
    args = parser.parse_args()
    generate_random_weights(args.model_size, args.output_dir, args.fp16) 