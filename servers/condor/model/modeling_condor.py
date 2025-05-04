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
from .moe import CondorMoELayer, MoEConfig


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
    MLP for Condor model.
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense_in = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=config.bias)
        self.dense_out = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=config.bias)
        self.act_fn = nn.GELU()
    
    def forward(self, x):
        x = self.dense_in(x)
        x = self.act_fn(x)
        x = self.dense_out(x)
        return x


class CondorDecoderLayer(nn.Module):
    """
    Decoder layer for Condor model.
    """
    
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.hidden_size = config.hidden_size
        self.self_attn = CondorAttention(config)
        
        # Determine if this layer should use MoE
        self.use_moe = (
            config.use_moe and 
            layer_idx is not None and 
            layer_idx % config.moe_layer_frequency == 0
        )
        
        if self.use_moe:
            # Initialize MoE layer
            self.moe = CondorMoELayer(
                config, 
                config.moe_config, 
                config.hidden_size, 
                4 * config.hidden_size
            )
        else:
            # Standard MLP 
            self.mlp = CondorMLP(config)
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Only use a second layer norm if not using parallel attention
        if not config.parallel_attn:
            self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for the decoder layer.
        """
        residual = hidden_states
        
        # Apply layer normalization before self-attention
        hidden_states = self.ln_1(hidden_states)
        
        # Self-attention
        attn_outputs, present = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        
        if self.config.parallel_attn:
            # In parallel attention, we apply both self-attention and MLP/MoE to the same input
            # Then we add both outputs to the residual
            
            # MoE or MLP branch
            if self.use_moe:
                moe_outputs, aux_loss_dict = self.moe(hidden_states, training=training)
                hidden_states = residual + attn_outputs + moe_outputs
                return hidden_states, present, aux_loss_dict
            else:
                mlp_outputs = self.mlp(hidden_states)
                hidden_states = residual + attn_outputs + mlp_outputs
                return hidden_states, present, None
        else:
            # In sequential attention, we add the self-attention output to the residual
            # Then apply another layer norm before the MLP/MoE
            hidden_states = residual + attn_outputs
            
            residual = hidden_states
            hidden_states = self.ln_2(hidden_states)
            
            # MoE or MLP branch
            if self.use_moe:
                moe_outputs, aux_loss_dict = self.moe(hidden_states, training=training)
                hidden_states = residual + moe_outputs
                return hidden_states, present, aux_loss_dict
            else:
                mlp_outputs = self.mlp(hidden_states)
                hidden_states = residual + mlp_outputs
                return hidden_states, present, None


class CondorModel(nn.Module):
    """
    Base model for Condor.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            CondorDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for the model.
        """
        batch_size, seq_length = input_ids.shape
        
        # Get token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask if attention mask is not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=hidden_states.device)
        
        # Convert attention mask to format expected by the model
        # This creates a mask of shape [batch_size, 1, seq_length, seq_length]
        # with -inf values for positions that should not be attended to
        extended_attention_mask = _make_causal_mask(
            input_ids_shape=input_ids.shape,
            dtype=hidden_states.dtype,
        )
        extended_attention_mask = extended_attention_mask.to(hidden_states.device)
        
        # If past_key_values are provided, only attend to the new tokens
        if past_key_values is not None:
            # past_length is the length of past tokens
            past_length = past_key_values[0][0].size(2)
            
            # Only process the new tokens
            extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
        else:
            past_length = 0
        
        # Initialize present_key_values if use_cache is True
        present_key_values = [] if use_cache else None
        
        # Initialize dictionary to collect MoE auxiliary losses
        all_aux_losses = {}
        
        # Process through each decoder layer
        for idx, layer in enumerate(self.layers):
            # Pass through the layer
            # The past_key_values for this layer are at position idx
            layer_past = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=extended_attention_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                training=training,
            )
            
            # Unpack layer outputs
            if len(layer_outputs) == 3:
                hidden_states, present, aux_losses = layer_outputs
            else:
                hidden_states, present = layer_outputs
                aux_losses = None
            
            # Collect present key values for caching
            if use_cache:
                present_key_values.append(present)
            
            # Collect auxiliary losses if any
            if aux_losses is not None:
                for loss_name, loss_value in aux_losses.items():
                    if loss_name in all_aux_losses:
                        all_aux_losses[loss_name] += loss_value
                    else:
                        all_aux_losses[loss_name] = loss_value
        
        # Apply final layer normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states, present_key_values, all_aux_losses


class CondorForCausalLM(nn.Module):
    """
    Condor model for causal language modeling.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = CondorModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head
        self.lm_head.weight = self.model.embed_tokens.weight
    
    @classmethod
    def from_pretrained(cls, model_path: str, config=None) -> "CondorForCausalLM":
        """
        Load a pretrained model.
        
        Args:
            model_path: Path to the pretrained model directory
            config: Model configuration (optional)
            
        Returns:
            Loaded model
        """
        # Load configuration if not provided
        if config is None:
            config = CondorConfig.from_pretrained(model_path)
        
        # Create model with configuration
        model = cls(config)
        
        # Load weights if they exist
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict)
        
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
        
        Args:
            input_ids: Input token IDs
            past_key_values: Past key/value states for attention
            attention_mask: Attention mask
            
        Returns:
            Dictionary of prepared inputs
        """
        # Only keep the last token for inference if past_key_values are provided
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
            # Update attention mask for the new token
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True)
        }
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        training: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key/value states for attention
            labels: Labels for computing the masked language modeling loss
            use_cache: Whether to use cache for incremental decoding
            
        Returns:
            Tuple containing:
                - logits: Output logits
                - past_key_values: Updated key/value states (if use_cache=True)
                - loss: Language modeling loss (if labels are provided)
        """
        # Get hidden states from the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            training=training,
        )
        
        hidden_states, present_key_values, aux_losses = outputs
        
        # Project hidden states to vocabulary
        logits = self.lm_head(hidden_states)
        
        loss = None
        # Calculate language modeling loss if labels are provided
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # Add auxiliary losses if any
            if aux_losses:
                for loss_name, aux_loss in aux_losses.items():
                    loss = loss + aux_loss
        
        return (loss, logits, present_key_values) if loss is not None else (logits, present_key_values)
    
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
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            top_k: Top k sampling parameter
            top_p: Top p sampling parameter
            repetition_penalty: Penalty for repeating tokens
            num_return_sequences: Number of sequences to return
            
        Returns:
            Generated token IDs
        """
        # Ensure input_ids is on the correct device
        input_ids = input_ids.to(self.lm_head.weight.device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.device)
        
        # Set generation parameters
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        
        # Expand input_ids for num_return_sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
            batch_size *= num_return_sequences
        
        # Initialize past key values
        past_key_values = None
        
        # Initialize token tracking for repetition penalty
        prev_tokens = input_ids.clone()
        
        # Loop until max_length or EOS token
        while cur_len < max_length:
            # Prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True
            )
            
            # Forward pass
            outputs = self(
                **model_inputs,
                training=False,
            )
            
            # Get logits and updated past key values
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                logits, past_key_values = outputs[:2]
            else:
                logits = outputs
                past_key_values = None
            
            # Get logits for the next token
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(prev_tokens[i].tolist()):
                        if token_id in next_token_logits[i]:
                            next_token_logits[i, token_id] /= repetition_penalty
            
            # Set scores for some unwanted tokens to -inf
            # For example, you might want to disable generation of special tokens or padding
            
            # Sample from the distribution or take the argmax
            if do_sample:
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits.fill_(float("-inf"))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep the first token above the threshold
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        next_token_logits[i, indices_to_remove] = float("-inf")
                
                # Convert logits to probabilities
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Add the next tokens to the generated sequence
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask if necessary
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
                )
            
            # Update prev_tokens for repetition penalty
            prev_tokens = torch.cat([prev_tokens, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update current length
            cur_len += 1
        
        return input_ids


def AutoModelForCausalLM_from_pretrained(model_name_or_path: str, **kwargs) -> CondorForCausalLM:
    """
    Helper function to load a pretrained model.
    
    Args:
        model_name_or_path: Path to the pretrained model directory
        
    Returns:
        Loaded model
    """
    return CondorForCausalLM.from_pretrained(model_name_or_path, **kwargs) 