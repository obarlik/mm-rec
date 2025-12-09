"""
OpenAI-Compatible Tokenizer (tiktoken)
Wrapper for OpenAI tokenizer support
"""

import os
from typing import List, Optional, Union
import warnings

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    warnings.warn("tiktoken not installed. Install with: pip install tiktoken")


class OpenAITokenizer:
    """
    OpenAI-compatible tokenizer using tiktoken.
    Supports GPT-4, GPT-3.5, and other OpenAI models.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        vocab_size: Optional[int] = None
    ):
        """
        Initialize OpenAI tokenizer.
        
        Args:
            model_name: OpenAI model name (gpt-4, gpt-3.5-turbo, cl100k_base, etc.)
            vocab_size: Vocabulary size (auto-detected from model_name)
        """
        if not TIKTOKEN_AVAILABLE:
            raise ImportError(
                "tiktoken is required for OpenAI tokenizer. "
                "Install with: pip install tiktoken"
            )
        
        self.model_name = model_name
        
        # Map model names to encoding names
        encoding_map = {
            "gpt-4": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "gpt-3.5": "cl100k_base",
            "text-davinci-003": "p50k_base",
            "text-davinci-002": "p50k_base",
            "text-davinci-001": "r50k_base",
            "text-curie-001": "r50k_base",
            "text-babbage-001": "r50k_base",
            "text-ada-001": "r50k_base",
        }
        
        encoding_name = encoding_map.get(model_name, "cl100k_base")
        
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
            self.vocab_size = self.encoding.n_vocab
        except Exception as e:
            warnings.warn(f"Failed to load {encoding_name}, using cl100k_base")
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.vocab_size = self.encoding.n_vocab
        
        # Special tokens
        self.eos_token_id = self.encoding.eot_token  # End of text
        self.pad_token_id = self.eos_token_id  # Use EOS as pad
        self.unk_token_id = None  # tiktoken doesn't have UNK
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Add special tokens (EOS)
            max_length: Maximum length
            truncation: Truncate if too long
            padding: Pad if too short
        
        Returns:
            List of token IDs
        """
        tokens = self.encoding.encode(text)
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        if max_length:
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
            elif padding and len(tokens) < max_length:
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
        
        Returns:
            Decoded text
        """
        if skip_special_tokens:
            # Filter out pad/eos tokens
            token_ids = [t for t in token_ids if t != self.pad_token_id and t != self.eos_token_id]
        
        return self.encoding.decode(token_ids)
    
    def __call__(self, text: str, **kwargs) -> List[int]:
        """Callable interface."""
        return self.encode(text, **kwargs)
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self._vocab_size
    
    @vocab_size.setter
    def vocab_size(self, value: int):
        """Set vocabulary size."""
        self._vocab_size = value


class SimpleTokenizer:
    """
    Fallback simple tokenizer if tiktoken is not available.
    Uses character-level hashing.
    """
    
    def __init__(self, vocab_size: int = 32000):
        """
        Initialize simple tokenizer.
        
        Args:
            vocab_size: Vocabulary size
        """
        self.vocab_size = vocab_size
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = 0
        self.unk_token_id = 1
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False
    ) -> List[int]:
        """Encode text using character-level hashing."""
        tokens = []
        for char in text:
            token_id = hash(char) % (self.vocab_size - 1)  # Reserve last for EOS
            tokens.append(token_id)
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        if max_length:
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
            elif padding and len(tokens) < max_length:
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode tokens (not perfect, but works for training)."""
        # Simple tokenizer can't perfectly decode
        return f"[{len(token_ids)} tokens]"


def get_tokenizer(model_name: str = "gpt-4", vocab_size: Optional[int] = None):
    """
    Get tokenizer (OpenAI if available, else simple).
    
    Args:
        model_name: Model name
        vocab_size: Vocabulary size
    
    Returns:
        Tokenizer instance
    """
    if TIKTOKEN_AVAILABLE:
        return OpenAITokenizer(model_name=model_name, vocab_size=vocab_size)
    else:
        warnings.warn("Using SimpleTokenizer (install tiktoken for OpenAI compatibility)")
        return SimpleTokenizer(vocab_size=vocab_size or 32000)

