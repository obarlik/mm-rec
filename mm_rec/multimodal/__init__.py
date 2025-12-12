"""
Multi-Modal Content Processor Foundation
Prepares MM-Rec for future multi-modal expansion while maintaining text-only focus

This module provides the foundation for multi-modal I/O following OpenAI's content format:
- Text-only mode (current): Direct text processing
- Multi-modal mode (future): Support for images, audio, code, etc.

Architecture is designed to be extensible without breaking existing text-only workflows.
"""

import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ContentItem:
    """Single content item (text, image, etc.)"""
    type: str  # "text", "image_url", "audio", etc.
    content: Any  # Actual content (string, tensor, URL, etc.)
    metadata: Optional[Dict] = None


class ModalityEmbedder(nn.Module):
    """Base class for modality-specific embedders."""
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, content: Any) -> torch.Tensor:
        """
        Convert modality-specific content to embeddings.
        
        Args:
            content: Modality-specific input
        
        Returns:
            Embeddings [batch, seq_len, output_dim]
        """
        raise NotImplementedError


class TextEmbedder(ModalityEmbedder):
    """Text embedding (current implementation)."""
    
    def __init__(self, vocab_size: int, output_dim: int):
        super().__init__(output_dim)
        self.embedding = nn.Embedding(vocab_size, output_dim)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len] token IDs
        
        Returns:
            [batch, seq_len, output_dim] embeddings
        """
        return self.embedding(token_ids)


class ImageEmbedder(ModalityEmbedder):
    """
    Image embedding (placeholder for future).
    
    Future implementation will use:
    - CLIP for vision encoding
    - Patch-based tokenization
    - Learnable projection to model_dim
    """
    
    def __init__(self, output_dim: int, image_encoder_dim: int = 768):
        super().__init__(output_dim)
        # Placeholder: will integrate CLIP or similar
        self.projection = nn.Linear(image_encoder_dim, output_dim)
        print("⚠️  ImageEmbedder is a placeholder. Integrate CLIP for production.")
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: [batch, num_patches, encoder_dim] from vision encoder
        
        Returns:
            [batch, num_patches, output_dim] embeddings
        """
        return self.projection(image_features)


class MultiModalContentProcessor:
    """
    Process OpenAI-style multi-modal content.
    
    Supports:
    - Text-only (current): Simple string input
    - Multi-modal (future): List of content items with type field
    
    Example OpenAI format:
        content = [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "..."}}
        ]
    """
    
    def __init__(
        self,
        text_embedder: TextEmbedder,
        tokenizer,
        image_embedder: Optional[ImageEmbedder] = None
    ):
        self.text_embedder = text_embedder
        self.tokenizer = tokenizer
        self.image_embedder = image_embedder
    
    def process_content(
        self,
        content: Union[str, List[Dict]],
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Process content in OpenAI format.
        
        Args:
            content: Either:
                - String (text-only mode)
                - List of dicts with 'type' field (multi-modal mode)
            device: Target device
        
        Returns:
            Embeddings tensor [batch, seq_len, model_dim]
        """
        if isinstance(content, str):
            # Text-only mode (current)
            return self._process_text(content, device)
        
        # Multi-modal mode (future)
        return self._process_multimodal(content, device)
    
    def _process_text(self, text: str, device: torch.device) -> torch.Tensor:
        """Process text-only content."""
        token_ids = self.tokenizer.encode(text, max_length=None, truncation=False)
        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        return self.text_embedder(token_tensor)
    
    def _process_multimodal(
        self,
        content_list: List[Dict],
        device: torch.device
    ) -> torch.Tensor:
        """
        Process multi-modal content.
        
        Future implementation will:
        1. Parse each content item by type
        2. Route to appropriate embedder
        3. Concatenate embeddings
        """
        embeddings = []
        
        for item in content_list:
            content_type = item.get("type")
            
            if content_type == "text":
                # Text content
                text = item.get("text", "")
                emb = self._process_text(text, device)
                embeddings.append(emb)
            
            elif content_type == "image_url":
                # Image content (future)
                if self.image_embedder is None:
                    raise NotImplementedError(
                        "Image processing not yet implemented. "
                        "Provide ImageEmbedder with vision encoder."
                    )
                
                # Future: Load image from URL or base64
                # image_url = item["image_url"]["url"]
                # image_features = load_and_encode_image(image_url)
                # emb = self.image_embedder(image_features)
                # embeddings.append(emb)
                
                raise NotImplementedError("Image processing coming soon")
            
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
        
        # Concatenate along sequence dimension
        return torch.cat(embeddings, dim=1)


# Example usage (future):
"""
# Initialize
text_embedder = TextEmbedder(vocab_size=100256, output_dim=512)
image_embedder = ImageEmbedder(output_dim=512)  # With CLIP integration
processor = MultiModalContentProcessor(text_embedder, tokenizer, image_embedder)

# Process multi-modal input
content = [
    {"type": "text", "text": "Describe this image:"},
    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
]
embeddings = processor.process_content(content)

# Feed to MM-Rec model
output = model(embeddings)
"""
