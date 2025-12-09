"""
Supervised Fine-Tuning (SFT) Trainer for OpenAI-Compatible Training
Handles chat format, loss masking, and OpenAI API compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from ..data.chat_format import ChatFormatter, ChatMessage
from ..tokenizers.openai_tokenizer import get_tokenizer


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    model_name: str = "gpt-4"
    max_length: int = 2048
    only_predict_assistant: bool = True
    label_smoothing: float = 0.0
    ignore_index: int = -100


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for OpenAI-compatible chat models.
    Handles proper loss masking for chat format.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[SFTConfig] = None
    ):
        """
        Initialize SFT trainer.
        
        Args:
            model: MM-Rec model instance
            tokenizer: Tokenizer (OpenAI or compatible)
            config: SFT configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SFTConfig()
        self.chat_formatter = ChatFormatter()
    
    def prepare_chat_input(
        self,
        messages: List[ChatMessage],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare chat input for training.
        
        Args:
            messages: List of ChatMessage objects
            device: Device to place tensors
        
        Returns:
            (input_ids, attention_mask, labels) tuple
        """
        # Format messages
        input_text, target_text = self.chat_formatter.create_training_pairs(
            messages,
            only_assistant=self.config.only_predict_assistant
        )
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            input_text,
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        )
        
        # Create labels (only predict assistant responses)
        if self.config.only_predict_assistant:
            # Find where assistant response starts
            assistant_token = self.chat_formatter.assistant_token
            assistant_start = input_text.find(assistant_token)
            
            if assistant_start >= 0:
                # Tokenize target separately to get labels
                target_ids = self.tokenizer.encode(
                    target_text,
                    max_length=self.config.max_length,
                    truncation=True,
                    padding=False
                )
                
                # Create labels: -100 for non-assistant tokens, token_ids for assistant
                labels = [-100] * len(input_ids)
                # Replace with target_ids where assistant starts
                # (simplified - in practice, need to find exact position)
                if len(target_ids) <= len(input_ids):
                    labels[-len(target_ids):] = target_ids
            else:
                labels = [-100] * len(input_ids)
        else:
            # Predict all tokens
            labels = input_ids.copy()
        
        # Convert to tensors
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        labels = torch.tensor([labels], dtype=torch.long, device=device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return input_ids, attention_mask, labels
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute SFT loss with proper masking.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            labels: Labels [batch, seq_len] (-100 for ignored tokens)
            attention_mask: Attention mask [batch, seq_len]
        
        Returns:
            Loss tensor
        """
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute loss (ignoring -100 labels)
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=self.config.ignore_index,
            label_smoothing=self.config.label_smoothing
        )
        
        return loss
    
    def train_step(
        self,
        messages: List[ChatMessage],
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            messages: List of ChatMessage objects
            optimizer: Optimizer
            device: Device
        
        Returns:
            Dictionary with loss and metrics
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Prepare input
        input_ids, attention_mask, labels = self.prepare_chat_input(messages, device)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Compute loss
        loss = self.compute_loss(logits, labels, attention_mask)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'perplexity': torch.exp(loss).item()
        }


class ChatCompletionAPI:
    """
    OpenAI Chat Completion API-compatible inference.
    """
    
    def __init__(self, model: nn.Module, tokenizer):
        """
        Initialize chat completion API.
        
        Args:
            model: MM-Rec model instance
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.chat_formatter = ChatFormatter()
    
    def create(
        self,
        messages: List[Dict],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 1.0,
        device: torch.device = torch.device('cpu')
    ) -> Dict:
        """
        Create chat completion (OpenAI API format).
        
        Args:
            messages: List of message dicts [{"role": "user", "content": "..."}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling
            device: Device
        
        Returns:
            OpenAI-compatible response dict
        """
        # Convert to ChatMessage objects
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        
        # Format input
        input_text = self.chat_formatter.format_messages(chat_messages)
        
        # Tokenize
        input_ids = self.tokenizer.encode(input_text, max_length=2048, truncation=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            generated_ids = self._generate(
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        
        # Extract assistant response
        assistant_response = self._extract_assistant_response(generated_text)
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": assistant_response
                },
                "finish_reason": "stop"
            }]
        }
    
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> torch.Tensor:
        """Generate tokens (simplified - implement proper sampling)."""
        # Simplified generation - in practice, use proper sampling
        generated = input_ids.clone()
        
        for _ in range(max_tokens):
            logits = self.model(generated)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-p sampling (simplified)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return generated
    
    def _extract_assistant_response(self, text: str) -> str:
        """Extract assistant response from generated text."""
        assistant_token = self.chat_formatter.assistant_token
        if assistant_token in text:
            parts = text.split(assistant_token)
            if len(parts) > 1:
                return parts[-1].strip()
        return text.strip()

