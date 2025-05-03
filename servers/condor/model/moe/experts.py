"""
Mixture of Experts (MoE) implementation for Condor model.

This module provides the core components for Mixture of Experts:
- Expert networks
- Router networks
- MoE layer implementation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod

from .moe_config import MoEConfig


class ExpertRouter(nn.Module, ABC):
    """
    Abstract base class for expert routers.
    
    Expert routers determine which experts should process each token.
    """
    
    def __init__(self, config: MoEConfig, hidden_size: int):
        """
        Initialize the router.
        
        Args:
            config: MoE configuration
            hidden_size: Hidden dimension size of the input
        """
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_experts = config.num_experts
        self.num_selected_experts = config.num_selected_experts
    
    @abstractmethod
    def forward(
        self, 
        hidden_states: torch.Tensor,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route hidden states to experts.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Tuple containing:
                - expert_weights: Weights for each selected expert [batch_size*seq_len, num_selected_experts]
                - expert_indices: Indices of selected experts [batch_size*seq_len, num_selected_experts]
                - router_logits: Raw router logits [batch_size*seq_len, num_experts]
                - aux_loss_dict: Dictionary containing auxiliary losses
        """
        pass


class TopKRouter(ExpertRouter):
    """
    Top-K router implementation that selects top-k experts based on routing probabilities.
    """
    
    def __init__(self, config: MoEConfig, hidden_size: int):
        """
        Initialize the Top-K router.
        
        Args:
            config: MoE configuration
            hidden_size: Hidden dimension size
        """
        super().__init__(config, hidden_size)
        
        # Router projection to get expert scores
        self.router = nn.Linear(hidden_size, self.num_experts, bias=False)
        # Initialize router weights
        self._initialize_router_weights()
    
    def _initialize_router_weights(self):
        """Initialize router weights with normal distribution."""
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
    
    def _compute_routing_probabilities(
        self, 
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the routing probabilities for all experts.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Router logits [batch_size*seq_len, num_experts]
        """
        # Get the router logits
        router_logits = self.router(hidden_states)
        
        # Reshape for routing
        batch_size, seq_len, _ = hidden_states.shape
        router_logits = router_logits.view(-1, self.num_experts)
        
        return router_logits
    
    def _compute_router_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute router z-loss to prevent router from assigning high probability to a single expert.
        
        Args:
            router_logits: Router logits [batch_size*seq_len, num_experts]
            
        Returns:
            Router z-loss
        """
        # Compute the mean of the router logits
        mean_logits = router_logits.mean(dim=0)
        # Compute the variance of the router logits
        var_logits = (router_logits ** 2).mean(dim=0) - mean_logits ** 2
        # Compute the z-loss to encourage a uniform distribution over experts
        router_z_loss = var_logits.mean() * self.config.router_z_loss_weight
        
        return router_z_loss
    
    def _compute_load_balancing_loss(
        self, 
        router_probs: torch.Tensor, 
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing loss to prevent over-utilization of certain experts.
        
        Args:
            router_probs: Router probabilities [batch_size*seq_len, num_experts]
            expert_indices: Indices of selected experts [batch_size*seq_len, num_selected_experts]
            
        Returns:
            Load balancing loss
        """
        # Compute the mean probability of selecting each expert
        num_tokens = router_probs.shape[0]
        expert_count = torch.zeros(self.num_experts, device=router_probs.device)
        
        # For each token, add the probability for each selected expert
        for i in range(self.num_selected_experts):
            for token_idx, expert_idx in enumerate(expert_indices[:, i]):
                expert_count[expert_idx] += router_probs[token_idx, expert_idx]
        
        # Normalize by the number of tokens
        expert_fraction = expert_count / num_tokens
        
        # Compute coefficient of variation (std/mean)
        expert_mean = expert_fraction.mean()
        expert_std = expert_fraction.std()
        coefficient_of_variation = expert_std / (expert_mean + 1e-8)
        
        # We want to minimize the coefficient of variation
        # This encourages a more uniform distribution of tokens across experts
        load_balancing_loss = coefficient_of_variation * self.config.aux_loss_weight
        
        return load_balancing_loss
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        training: bool = False,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route hidden states to experts.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            training: Whether the model is in training mode
            
        Returns:
            Tuple containing:
                - expert_weights: Weights for each selected expert [batch_size*seq_len, num_selected_experts]
                - expert_indices: Indices of selected experts [batch_size*seq_len, num_selected_experts]
                - router_logits: Raw router logits [batch_size*seq_len, num_experts]
                - aux_loss_dict: Dictionary containing auxiliary losses
        """
        # Initialize auxiliary loss dictionary
        aux_loss_dict = {}
        
        # Reshape input for routing
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, self.hidden_size)
        
        # Get router logits
        router_logits = self._compute_routing_probabilities(hidden_states)
        
        # Calculate router z-loss if required
        if self.config.use_aux_loss and training:
            router_z_loss = self._compute_router_z_loss(router_logits)
            aux_loss_dict["router_z_loss"] = router_z_loss
        
        # Apply jitter noise during training to improve generalization
        if training and self.config.jitter_noise > 0:
            router_logits += torch.randn_like(router_logits) * self.config.jitter_noise
        
        # Convert logits to probabilities using softmax
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Get top-k expert indices and scores
        expert_weights, expert_indices = torch.topk(
            router_probs, 
            self.num_selected_experts, 
            dim=-1
        )
        
        # Calculate load balancing loss if required
        if self.config.use_aux_loss and self.config.balance_experts and training:
            load_balancing_loss = self._compute_load_balancing_loss(router_probs, expert_indices)
            aux_loss_dict["load_balancing_loss"] = load_balancing_loss
        
        # Normalize the expert weights to sum to 1
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        return expert_weights, expert_indices, router_logits, aux_loss_dict


class HashRouter(ExpertRouter):
    """
    Hash-based router that assigns tokens to experts using a hash function.
    
    This is a simpler, deterministic alternative to learned routing that
    doesn't require training but can still provide good performance.
    """
    
    def __init__(self, config: MoEConfig, hidden_size: int):
        """
        Initialize the hash router.
        
        Args:
            config: MoE configuration
            hidden_size: Hidden dimension size
        """
        super().__init__(config, hidden_size)
        
        # Linear projection used to create hash inputs
        self.hash_proj = nn.Linear(hidden_size, config.num_selected_experts * 32, bias=True)
        self._initialize_hash_weights()
    
    def _initialize_hash_weights(self):
        """Initialize hash projection weights."""
        nn.init.normal_(self.hash_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.hash_proj.bias)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        training: bool = False,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Determine expert assignment using a hash function.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            training: Whether the model is in training mode
            
        Returns:
            Tuple containing:
                - expert_weights: Equal weights for each selected expert [batch_size*seq_len, num_selected_experts]
                - expert_indices: Indices of selected experts [batch_size*seq_len, num_selected_experts]
                - router_logits: Dummy router logits (all zeros) [batch_size*seq_len, num_experts]
                - aux_loss_dict: Empty dictionary as hash routing has no auxiliary losses
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, self.hidden_size)
        
        # Project hidden states to get hash inputs
        hash_inputs = self.hash_proj(hidden_states_reshaped)
        
        # Reshape to [batch_size*seq_len, num_selected_experts, 32]
        hash_inputs = hash_inputs.view(-1, self.num_selected_experts, 32)
        
        # Binarize the inputs to create "random" bit patterns
        hash_bits = (hash_inputs > 0).long()
        
        # Compute expert indices for each token and position
        expert_indices = torch.zeros(
            batch_size * seq_len, 
            self.num_selected_experts, 
            dtype=torch.long, 
            device=hidden_states.device
        )
        
        # For each selected expert position
        for i in range(self.num_selected_experts):
            # Sum the bits and mod by number of experts to get expert index
            expert_indices[:, i] = hash_bits[:, i].sum(dim=-1) % self.num_experts
        
        # Ensure each token gets different experts by adjusting repeats
        for token_idx in range(batch_size * seq_len):
            # Find unique experts for this token
            unique_experts = set()
            for i in range(self.num_selected_experts):
                expert_idx = expert_indices[token_idx, i].item()
                # If this expert is already selected, choose another one
                attempt = 0
                while expert_idx in unique_experts and attempt < self.num_experts:
                    expert_idx = (expert_idx + 1) % self.num_experts
                    attempt += 1
                
                unique_experts.add(expert_idx)
                expert_indices[token_idx, i] = expert_idx
        
        # Equal weights for each selected expert
        expert_weights = torch.ones_like(expert_indices, dtype=torch.float) / self.num_selected_experts
        
        # Create dummy router logits (all zeros)
        router_logits = torch.zeros(
            batch_size * seq_len, 
            self.num_experts, 
            device=hidden_states.device
        )
        
        # No auxiliary losses for hash routing
        aux_loss_dict = {}
        
        return expert_weights, expert_indices, router_logits, aux_loss_dict


class CondorExpert(nn.Module):
    """
    Implementation of a single expert in the Mixture of Experts layer.
    
    Each expert is a feed-forward network similar to the MLP in transformer models.
    """
    
    def __init__(self, config, hidden_size: int, intermediate_size: int):
        """
        Initialize the expert.
        
        Args:
            config: MoE configuration
            hidden_size: Size of hidden states
            intermediate_size: Size of intermediate layer
        """
        super().__init__()
        self.config = config
        self.dense_in = nn.Linear(hidden_size, intermediate_size)
        self.dense_out = nn.Linear(intermediate_size, hidden_size)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.expert_dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the expert."""
        # Xavier initialization for the linear layers
        nn.init.xavier_uniform_(self.dense_in.weight)
        nn.init.xavier_uniform_(self.dense_out.weight)
        nn.init.zeros_(self.dense_in.bias)
        nn.init.zeros_(self.dense_out.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the expert.
        
        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            
        Returns:
            Processed hidden states [batch_size, hidden_size]
        """
        # Up-project, apply activation function, and dropout
        intermediate = self.act_fn(self.dense_in(hidden_states))
        intermediate = self.dropout(intermediate)
        
        # Down-project back to hidden size
        output = self.dense_out(intermediate)
        output = self.dropout(output)
        
        return output


class CondorMoELayer(nn.Module):
    """
    Mixture of Experts layer for Condor model.
    
    This layer routes tokens to different experts based on routing scores.
    """
    
    def __init__(
        self, 
        config, 
        moe_config: MoEConfig, 
        hidden_size: int, 
        intermediate_size: int
    ):
        """
        Initialize the MoE layer.
        
        Args:
            config: Model configuration
            moe_config: MoE configuration
            hidden_size: Size of hidden states
            intermediate_size: Size of intermediate layer
        """
        super().__init__()
        self.config = config
        self.moe_config = moe_config
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Create the router based on the configuration
        if moe_config.router_type == "top-k":
            self.router = TopKRouter(moe_config, hidden_size)
        elif moe_config.router_type == "hash":
            self.router = HashRouter(moe_config, hidden_size)
        else:
            raise ValueError(f"Unknown router type: {moe_config.router_type}")
        
        # Create the experts
        self.experts = nn.ModuleList([
            CondorExpert(moe_config, hidden_size, intermediate_size)
            for _ in range(moe_config.num_experts)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon if hasattr(config, "layer_norm_epsilon") else 1e-5)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        training: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for the MoE layer.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            training: Whether the model is in training mode
            
        Returns:
            Tuple containing:
                - output_hidden_states: Processed hidden states [batch_size, seq_len, hidden_size]
                - aux_loss_dict: Dictionary of auxiliary losses
        """
        # Apply layer normalization before routing
        normalized_hidden_states = self.layer_norm(hidden_states)
        
        # Get original shape
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Route tokens to experts
        expert_weights, expert_indices, router_logits, aux_loss_dict = self.router(
            normalized_hidden_states,
            training=training
        )
        
        # Reshape for expert processing
        flat_hidden_states = normalized_hidden_states.view(-1, hidden_size)
        
        # Prepare to accumulate expert outputs
        final_output = torch.zeros_like(flat_hidden_states)
        
        # Process tokens through their assigned experts
        for expert_idx in range(self.moe_config.num_experts):
            # Find all tokens assigned to this expert
            # We do this for each expert position (in case a token uses the same expert multiple times)
            for k in range(self.moe_config.num_selected_experts):
                # Get masks for tokens that use this expert at position k
                expert_mask = (expert_indices[:, k] == expert_idx)
                
                if not expert_mask.any():
                    continue
                
                # Get hidden states for these tokens
                expert_input = flat_hidden_states[expert_mask]
                
                # Process through the expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # Get the corresponding weights
                expert_weight = expert_weights[expert_mask, k].unsqueeze(-1)
                
                # Add the weighted output to the final output
                final_output[expert_mask] += expert_weight * expert_output
        
        # Reshape back to original dimensions
        output_hidden_states = final_output.view(batch_size, seq_len, hidden_size)
        
        # Add the residual connection
        output_hidden_states = output_hidden_states + hidden_states
        
        return output_hidden_states, aux_loss_dict 