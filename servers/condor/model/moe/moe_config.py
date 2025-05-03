"""
Configuration classes for Mixture of Experts in Condor model.
"""

from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass


@dataclass
class MoEConfig:
    """
    Configuration class for Mixture of Experts.
    
    This class contains all the parameters needed to configure
    the Mixture of Experts layers in the Condor model.
    """
    
    # Number of experts in the MoE layer
    num_experts: int = 8
    
    # Number of experts to be selected for each token
    num_selected_experts: int = 2
    
    # Router type: "top-k" or "hash"
    router_type: str = "top-k"
    
    # Whether to use auxiliary loss for load balancing
    use_aux_loss: bool = True
    
    # Weight of the auxiliary load balancing loss
    aux_loss_weight: float = 0.01
    
    # Expert dropout rate
    expert_dropout: float = 0.1
    
    # Router z-loss weight to prevent router from assigning high probability to single expert
    router_z_loss_weight: float = 0.001
    
    # Whether to jitter the expert selection during training
    jitter_noise: float = 0.0
    
    # Capacity factor for dynamic routing
    capacity_factor: float = 1.25
    
    # Whether to balance the experts (prevents the same expert from being used too much)
    balance_experts: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.router_type not in ["top-k", "hash"]:
            raise ValueError(f"Invalid router type: {self.router_type}, must be 'top-k' or 'hash'")
        
        if self.num_selected_experts > self.num_experts:
            raise ValueError(f"Selected experts ({self.num_selected_experts}) cannot exceed total experts ({self.num_experts})")
        
        if self.capacity_factor < 1.0:
            raise ValueError(f"Capacity factor must be >= 1.0, got {self.capacity_factor}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MoEConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict) 