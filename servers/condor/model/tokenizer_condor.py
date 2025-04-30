"""
Tokenizer implementation for Condor models.
"""

import os
import json
import numpy as np
import regex as re
from typing import Dict, List, Optional, Tuple, Union


class CondorTokenizer:
    """
    Tokenizer for Condor models.
    
    This is a basic implementation of a BPE tokenizer similar to the one used by Condor models.
    In a full implementation, this would load the vocabulary and merges from model files.
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<pad>",
        add_prefix_space: bool = False,
    ):
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.add_prefix_space = add_prefix_space
        
        # In a real implementation, we would load vocabulary and merges from files
        # For simplicity, we use a smaller dummy vocabulary
        self.vocab = self._init_dummy_vocab()
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        
        # Special token ids
        self.unk_token_id = self.vocab.get(unk_token, 0)
        self.bos_token_id = self.vocab.get(bos_token, 1)
        self.eos_token_id = self.vocab.get(eos_token, 2)
        self.pad_token_id = self.vocab.get(pad_token, 3)
        
        # Regex for basic tokenization
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def _init_dummy_vocab(self) -> Dict[str, int]:
        """
        Initialize a small dummy vocabulary for demonstration purposes.
        
        In a real implementation, this would load from a vocabulary file.
        """
        # Start with special tokens
        vocab = {
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "<pad>": 3,
        }
        
        # Add some common english words and characters
        tokens = ["the", "of", "and", "in", "to", "a", "is", "that", "for", "it", "with", "as", "was", "on"]
        tokens += [chr(i) for i in range(ord('a'), ord('z')+1)]  # a-z
        tokens += [chr(i) for i in range(ord('A'), ord('Z')+1)]  # A-Z
        tokens += [str(i) for i in range(10)]  # 0-9
        
        # Add common punctuation
        tokens += [",", ".", "!", "?", ":", ";", "-", "'", "\"", "(", ")", "[", "]", "{", "}", "/", "\\"]
        
        # Add to vocabulary
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        
        return vocab
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "CondorTokenizer":
        """
        Load a tokenizer from a pretrained model directory.
        
        In a real implementation, this would load vocabulary and merges files.
        """
        vocab_file = os.path.join(model_name_or_path, "vocab.json")
        merges_file = os.path.join(model_name_or_path, "merges.txt")
        
        if os.path.exists(vocab_file) and os.path.exists(merges_file):
            return cls(vocab_file=vocab_file, merges_file=merges_file, **kwargs)
        else:
            return cls(**kwargs)
    
    def encode(self, text: str) -> List[int]:
        """
        Tokenize text and convert to token ids.
        
        In a real implementation, this would apply BPE encoding.
        Here we do a simple split on common boundaries.
        """
        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text
        
        # Basic tokenization
        tokens = re.findall(self.pat, text)
        
        # Convert to token ids
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.unk_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token ids back to text.
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.ids_to_tokens:
                tokens.append(self.ids_to_tokens[token_id])
            else:
                tokens.append(self.unk_token)
        
        # Simple space joining - in a real tokenizer this would be more complex
        text = "".join(tokens)
        
        # Clean up tokenized spaces
        text = text.replace("<pad>", "")
        
        return text
    
    def __call__(
        self, 
        text: Union[str, List[str]],
        padding: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, List]:
        """
        Main tokenization method.
        
        Args:
            text: Text or list of texts to tokenize
            padding: Whether to pad sequences to max_length
            max_length: Maximum sequence length
            truncation: Whether to truncate sequences to max_length
            return_tensors: Format of returned tensors ("pt", "tf", or None)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if isinstance(text, str):
            text = [text]
        
        input_ids = []
        for t in text:
            ids = self.encode(t)
            
            if truncation and max_length and len(ids) > max_length:
                ids = ids[:max_length]
            
            input_ids.append(ids)
        
        # Get maximum length if padding is enabled
        if padding:
            if max_length:
                max_len = max_length
            else:
                max_len = max(len(ids) for ids in input_ids)
            
            # Pad sequences
            for i in range(len(input_ids)):
                padding_length = max_len - len(input_ids[i])
                if padding_length > 0:
                    input_ids[i] = input_ids[i] + [self.pad_token_id] * padding_length
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [[1] * len(ids) for ids in input_ids]
        if padding:
            for i in range(len(attention_mask)):
                padding_length = max_len - len(attention_mask[i])
                if padding_length > 0:
                    attention_mask[i] = attention_mask[i] + [0] * padding_length
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            try:
                import torch
                result = {k: torch.tensor(v) for k, v in result.items()}
            except ImportError:
                pass
        elif return_tensors == "tf":
            try:
                import tensorflow as tf
                result = {k: tf.convert_to_tensor(v) for k, v in result.items()}
            except ImportError:
                pass
        elif return_tensors == "np":
            result = {k: np.array(v) for k, v in result.items()}
        
        return result 