"""
Condor model implementation.

This is a simplified version of the Condor model architecture.
"""

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .configuration_condor import CondorConfig


def _make_causal_mask(input_ids_shape, dtype):
    """
    Create a causal mask for the attention mechanism.
    
    The mask ensures that the model can only attend to previous tokens.
    """
    bsz, seq_len = input_ids_shape
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min)
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    
    if bsz > 1:
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
    
    return mask.unsqueeze(1)  # [bsz, 1, seq_len, seq_len]


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
    
    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached or self.cos_cached is None or self.sin_cached is None:
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]
    
    def apply_rotary_emb(self, q, k, cos, sin):
        # Reshape for applying rotary embeddings
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class CondorAttention(nn.Module):
    """
    Multi-head attention mechanism for Condor model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_kv_heads if hasattr(config, "num_kv_heads") else self.num_heads
        self.kv_head_dim = self.hidden_size // self.num_kv_heads
        
        # Query, key, value projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.kv_head_dim, bias=config.bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.kv_head_dim, bias=config.bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
        
        # Initialize rotary embeddings if not using ALiBi
        if not config.alibi:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                base=config.rope_theta,
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the attention mechanism.
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        
        # Use past key values if provided
        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat([past_key, key_states], dim=-2)
            value_states = torch.cat([past_value, value_states], dim=-2)
        
        # Save key and value states for future use if use_cache is True
        if use_cache:
            present = (key_states, value_states)
        else:
            present = None
        
        # Apply rotary position embeddings if not using ALiBi
        if not self.config.alibi:
            kv_seq_len = key_states.shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = self.rotary_emb.apply_rotary_emb(query_states, key_states, cos, sin)
        
        # If using multi-query attention, repeat keys and values for all heads
        if self.num_kv_heads < self.num_heads:
            key_states = key_states.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            value_states = value_states.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply ALiBi if enabled
        if self.config.alibi:
            alibi = self._build_alibi_tensor(batch_size, seq_length, key_states.shape[-2], query_states.device)
            attn_weights = attn_weights + alibi
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape back to batch_size x seq_length x hidden_size
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        # Project back to hidden size
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present
    
    def _build_alibi_tensor(self, batch_size, seq_length, key_length, device):
        """
        Build ALiBi (Attention with Linear Biases) tensor.
        
        Based on the paper "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
        """
        # Build slopes for each head
        num_heads = self.num_heads
        slopes = torch.Tensor(self._get_slopes(num_heads)).to(device)
        
        # Build distance matrix
        context_position = torch.arange(seq_length, device=device)[:, None]
        memory_position = torch.arange(key_length, device=device)[None, :]
        relative_position = memory_position - context_position
        relative_position = torch.abs(relative_position).unsqueeze(0).unsqueeze(0)
        
        # Scale by slopes for each head and expand to batch size
        alibi = slopes.view(1, num_heads, 1, 1) * relative_position
        return alibi.expand(batch_size, -1, -1, -1)
    
    def _get_slopes(self, n):
        """Get slopes for ALiBi."""
        def get_slopes_power_of_2(n):
            start = 2**(-(2**-(math.log2(n)-3)))
            ratio = start
            return [start * ratio**i for i in range(n)]
        
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(n - closest_power_of_2)


class CondorMLP(nn.Module):
    """
    Multi-layer Perceptron for Condor model.
    """
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        
        # Condor typically uses a 4x multiple for intermediate size
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size, bias=config.bias)
        self.act = nn.GELU()
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size, bias=config.bias)
    
    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = self.act(x)
        x = self.dense_4h_to_h(x)
        return x


class CondorDecoderLayer(nn.Module):
    """
    Decoder layer for Condor model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # LayerNorm
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Self-attention
        self.self_attention = CondorAttention(config)
        
        # Fully-connected feed-forward network
        self.mlp = CondorMLP(config)
        
        # For older Condor variants, we have two LayerNorms
        if not config.parallel_attn:
            self.post_attention_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_epsilon)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the decoder layer.
        """
        residual = hidden_states
        
        # Apply layernorm (input)
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply self-attention
        attention_output, present = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        
        if self.config.parallel_attn:
            # Parallel attention and MLP
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + attention_output + hidden_states
        else:
            # Sequential attention and MLP
            hidden_states = residual + attention_output
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        
        return hidden_states, present


class CondorModel(nn.Module):
    """
    Condor base model.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Word embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([CondorDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        
        # Final LayerNorm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass for the Condor model.
        """
        batch_size, seq_length = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal attention mask if none provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)
        
        # Convert 2D mask to 4D mask for attention
        attention_mask_4d = _make_causal_mask(input_ids.shape, hidden_states.dtype).to(hidden_states.device)
        if attention_mask is not None:
            attention_mask_4d = attention_mask_4d + attention_mask.unsqueeze(1).unsqueeze(2).to(attention_mask_4d.dtype)
        
        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Output containers
        presents = [] if use_cache else None
        
        # Process through decoder layers
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask_4d,
                layer_past=layer_past,
                use_cache=use_cache,
            )
            
            if use_cache:
                presents.append(present)
        
        # Apply final LayerNorm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, presents


class CondorForCausalLM(nn.Module):
    """
    Condor model for causal language modeling.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base model
        self.model = CondorModel(config)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights of token embeddings and LM head
        self.lm_head.weight = self.model.embed_tokens.weight
    
    @classmethod
    def from_pretrained(cls, model_path: str, config=None) -> "CondorForCausalLM":
        """
        Load a model from a pretrained checkpoint.
        
        In a real implementation, this would load weights from files.
        Here we just create a model with the configuration.
        """
        if config is None:
            if os.path.isdir(model_path):
                config = CondorConfig.from_pretrained(model_path)
            else:
                # Default to 40B config
                config = CondorConfig.condor_40b_config()
        
        model = cls(config)
        
        # In a real implementation, load weights here
        # model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
        
        return model
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, any]:
        """
        Prepare inputs for generation.
        """
        # Only keep the last token for generation
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        }
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for causal language modeling.
        """
        # Get transformer outputs
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        
        hidden_states = transformer_outputs[0]
        
        # Apply LM head to get logits
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return (loss, lm_logits, transformer_outputs[1]) if loss is not None else (lm_logits, transformer_outputs[1])
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 20,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
    ) -> torch.LongTensor:
        """
        Generate text using the model.
        
        This is a simplified implementation of text generation with Condor model.
        """
        # Ensure batch size is at least 1
        batch_size = input_ids.shape[0]
        
        # Expand input tensors for multiple return sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
            batch_size = batch_size * num_return_sequences
        
        # Initialize generated sequences with input_ids
        generated_tokens = input_ids.clone()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Track past key values for faster generation
        past_key_values = None
        
        # Generation loop
        for _ in range(max_length - input_ids.shape[1]):
            # Prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                generated_tokens,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
            
            # Forward pass to get logits
            outputs = self.forward(**model_inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            past_key_values = outputs[1] if isinstance(outputs, tuple) and len(outputs) > 1 else None
            
            # Get the logits for the next token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for batch_idx in range(batch_size):
                    for token_idx in set(generated_tokens[batch_idx].tolist()):
                        next_token_logits[batch_idx, token_idx] /= repetition_penalty
            
            # Apply sampling
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][:, -1, None]
                    next_token_logits.masked_fill_(indices_to_remove, -float("Inf"))
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits.masked_fill_(indices_to_remove, -float("Inf"))
                
                # Convert logits to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Append new tokens to the sequence
            generated_tokens = torch.cat([generated_tokens, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask for new token
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)], dim=-1
            )
        
        return generated_tokens


# Create a function to easily load the model
def AutoModelForCausalLM_from_pretrained(model_name_or_path: str, **kwargs) -> CondorForCausalLM:
    """Helper function to load a model with the given name or path."""
    config = CondorConfig.from_pretrained(model_name_or_path)
    return CondorForCausalLM.from_pretrained(model_name_or_path, config=config, **kwargs) 