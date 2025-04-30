"""
Configuration class for Condor model.
"""

import os
from typing import Dict, List, Optional, Union


class CondorConfig:
    """
    Configuration class for Condor model.
    
    This configuration class contains all the necessary parameters to define and 
    initialize a Condor language model.
    """
    
    model_type: str = "condor"
    
    def __init__(
        self,
        vocab_size: int = 65024,
        hidden_size: int = 4544,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 71,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_kv_heads: Optional[int] = None,
        alibi: bool = False,
        new_decoder_architecture: bool = False,
        multi_query: bool = True,
        parallel_attn: bool = True,
        bias: bool = False,
        max_position_embeddings: Optional[int] = None,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[Dict[str, Union[str, float]]] = None,
        sliding_window: Optional[int] = None,
        attention_bias: bool = False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_attention_heads
        self.alibi = alibi
        self.new_decoder_architecture = new_decoder_architecture
        self.multi_query = multi_query
        self.parallel_attn = parallel_attn
        self.bias = bias
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.attention_bias = attention_bias
        
        # Additional configuration parameters from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "CondorConfig":
        """
        Load a configuration from a pretrained model directory or weights file.
        """
        config_path = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(config_path):
            # In a real implementation, you would load JSON here
            # For this example, we'll use predefined configurations
            if "7b" in model_name_or_path.lower():
                return cls.condor_7b_config(**kwargs)
            else:
                return cls.condor_40b_config(**kwargs)
        else:
            # Default to 40B config
            return cls.condor_40b_config(**kwargs)
    
    @classmethod
    def condor_7b_config(cls, **kwargs) -> "CondorConfig":
        """
        Returns the configuration for Condor-7B model.
        """
        return cls(
            vocab_size=65024,
            hidden_size=4544,
            num_hidden_layers=32,
            num_attention_heads=71,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            num_kv_heads=1,  # Multi-query attention
            alibi=False,
            new_decoder_architecture=True,
            multi_query=True,
            parallel_attn=True,
            bias=False,
            **kwargs
        )
    
    @classmethod
    def condor_40b_config(cls, **kwargs) -> "CondorConfig":
        """
        Returns the configuration for Condor-40B model.
        """
        return cls(
            vocab_size=65024,
            hidden_size=8192,
            num_hidden_layers=60,
            num_attention_heads=128,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            num_kv_heads=8,  # Multi-head attention with 8 key-value heads
            alibi=False,
            new_decoder_architecture=True,
            multi_query=False,
            parallel_attn=True,
            bias=False,
            **kwargs
        )
    
    def to_dict(self) -> Dict[str, any]:
        """
        Convert the configuration to a dictionary.
        """
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith("__") and not callable(getattr(self, attr))
        } 