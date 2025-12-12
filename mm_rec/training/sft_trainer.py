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
    max_length: Optional[int] = 2048  # None = infinite context
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
            # Tokenize full input to find assistant token position
            full_input_ids = self.tokenizer.encode(
                input_text,
                max_length=self.config.max_length * 2,  # Allow longer to find position
                truncation=False,
                padding=False
            )
            
            # Tokenize assistant token to find its position
            assistant_token = self.chat_formatter.assistant_token
            assistant_token_ids = self.tokenizer.encode(
                assistant_token,
                max_length=10,
                truncation=False,
                padding=False
            )
            
            # Find assistant token position in input
            assistant_pos = -1
            if len(assistant_token_ids) > 0:
                for i in range(len(full_input_ids) - len(assistant_token_ids) + 1):
                    if full_input_ids[i:i+len(assistant_token_ids)] == assistant_token_ids:
                        assistant_pos = i + len(assistant_token_ids)  # Start after token
                        break
            
            # Create labels: -100 for non-assistant tokens, token_ids for assistant
            labels = [-100] * len(input_ids)
            
            if assistant_pos >= 0:
                # Tokenize target (assistant response) separately
                target_ids = self.tokenizer.encode(
                    target_text,
                    max_length=self.config.max_length,
                    truncation=True,
                    padding=False
                )
                
                # Map assistant position to truncated input_ids
                # Find where assistant starts in truncated input_ids
                truncated_input_ids = input_ids[:len(input_ids)]
                
                # Simple approach: if assistant is in input, predict next tokens
                # Find assistant token in truncated input
                assistant_in_truncated = False
                for i in range(len(truncated_input_ids) - len(assistant_token_ids) + 1):
                    if list(truncated_input_ids[i:i+len(assistant_token_ids)]) == assistant_token_ids:
                        assistant_start_idx = i + len(assistant_token_ids)
                        # Set labels for assistant response (next tokens)
                        remaining_len = len(input_ids) - assistant_start_idx
                        if remaining_len > 0 and len(target_ids) > 0:
                            # Use target_ids for labels
                            label_len = min(remaining_len, len(target_ids))
                            labels[assistant_start_idx:assistant_start_idx+label_len] = target_ids[:label_len]
                            assistant_in_truncated = True
                        break
                
                # Fallback: if assistant not found, predict last part
                if not assistant_in_truncated and len(target_ids) > 0:
                    label_len = min(len(input_ids), len(target_ids))
                    labels[-label_len:] = target_ids[:label_len]
            else:
                # No assistant token found, predict all tokens
                labels = input_ids.copy()
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
        
        # Filter out ignored labels before computing loss
        valid_mask = shift_labels != self.config.ignore_index
        if valid_mask.sum() == 0:
            # No valid labels, return zero loss
            return torch.tensor(0.0, device=shift_logits.device, requires_grad=True)
        
        # Get valid logits and labels
        valid_logits = shift_logits[valid_mask]
        valid_labels = shift_labels[valid_mask]
        
        # Clamp logits for numerical stability
        valid_logits = torch.clamp(valid_logits, min=-50.0, max=50.0)
        
        # Compute loss (ignoring -100 labels)
        loss = F.cross_entropy(
            valid_logits.unsqueeze(0) if valid_logits.dim() == 1 else valid_logits,
            valid_labels.unsqueeze(0) if valid_labels.dim() == 0 else valid_labels,
            label_smoothing=self.config.label_smoothing,
            reduction='mean'
        )
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            # Return small positive loss instead of NaN
            loss = torch.tensor(1e-6, device=shift_logits.device, requires_grad=True)
        
        return loss
    
    def train_step(
        self,
        messages: List[ChatMessage],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            messages: List of ChatMessage objects
            optimizer: Optimizer
            device: Device
            verbose: Print detailed progress
        
        Returns:
            Dictionary with loss and metrics
        """
        import sys
        
        if verbose:
            print("  📝 Preparing input...", flush=True, file=sys.stderr)
        
        self.model.train()
        optimizer.zero_grad()
        
        # Prepare input
        input_ids, attention_mask, labels = self.prepare_chat_input(messages, device)
        
        valid_labels_count = (labels != -100).sum().item()
        if verbose:
            print(f"  ✅ Input prepared: shape={input_ids.shape}, valid_labels={valid_labels_count}/{labels.numel()}", flush=True, file=sys.stderr)
        
        # Forward pass
        if verbose:
            print("  🔄 Running forward pass...", flush=True, file=sys.stderr)
        
        logits = self.model(input_ids)
        
        if verbose:
            print(f"  ✅ Forward done: logits shape={logits.shape}", flush=True, file=sys.stderr)
        
        # Compute loss
        if verbose:
            print("  📊 Computing loss...", flush=True, file=sys.stderr)
        
        loss = self.compute_loss(logits, labels, attention_mask)
        
        if verbose:
            print(f"  ✅ Loss computed: {loss.item():.6f}", flush=True, file=sys.stderr)
        
        # Backward
        if verbose:
            print("  ⬅️  Running backward pass...", flush=True, file=sys.stderr)
        
        loss.backward()
        
        if verbose:
            # Check gradients
            total_grad_norm = 0.0
            param_count = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
                    param_count += 1
            total_grad_norm = total_grad_norm ** (1. / 2)
            print(f"  ✅ Backward done: grad_norm={total_grad_norm:.4f}, params_with_grad={param_count}", flush=True, file=sys.stderr)
        
        optimizer.step()
        
        if verbose:
            print("  ✅ Optimizer step completed", flush=True, file=sys.stderr)
        
        return {
            'loss': loss.item(),
            'perplexity': torch.exp(loss).item() if loss.item() < 10 else float('inf')
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
        device: torch.device = torch.device('cpu'),
        **kwargs
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
        
        # Tokenize (no hardcoded limit - use model's capability)
        input_ids = self.tokenizer.encode(input_text, max_length=None, truncation=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Streaming Mode
        if kwargs.get('stream', False):
            return self._create_stream(
                input_ids, 
                max_tokens=max_tokens, 
                temperature=temperature, 
                top_p=top_p
            )
        
        # Standard Mode with Explainability
        self.model.eval()
        with torch.no_grad():
            generated_ids, token_logprobs = self._generate(
                input_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                return_logprobs=True
            )
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        
        # Extract assistant response
        assistant_response = self._extract_assistant_response(generated_text)
        
        # Calculate Confidence (Explainability)
        avg_confidence = torch.exp(token_logprobs).mean().item() if token_logprobs is not None else 0.0
        
        return {
            "id": "chatcmpl-mmrec",
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": assistant_response
                },
                "finish_reason": "stop",
                "logprobs": {
                    "average_token_confidence": f"{avg_confidence:.2%}",
                    "token_logprobs": token_logprobs.tolist() if token_logprobs is not None else []
                }
            }],
            "usage": {
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": len(generated_ids[0]),
                "total_tokens": len(input_ids[0]) + len(generated_ids[0])
            }
        }
    
    def _create_stream(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float
    ):
        """Generator for streaming responses."""
        import time
        self.model.eval()
        
        generated = input_ids.clone()
        
        # 1. Yield Role
        yield {
            "id": "chatcmpl-mmrec",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        }
        
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.model(generated)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                # Append and Yield
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                token_text = self.tokenizer.decode([next_token.item()])
                
                yield {
                    "id": "chatcmpl-mmrec",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}],
                    "confidence": probs[next_token].item()
                }
        
        # Final finish
        yield {
            "id": "chatcmpl-mmrec",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
    
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int,
        temperature: float,
        top_p: float,
        return_logprobs: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate tokens with optional logprobs."""
        generated = input_ids.clone()
        logprobs_list = [] if return_logprobs else None
        
        for _ in range(max_tokens):
            logits = self.model(generated)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Save probabilities before sampling if needed
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
                probs = F.softmax(next_token_logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            if return_logprobs:
                token_logprob = torch.log(probs[next_token]).item()
                logprobs_list.append(token_logprob)
            
            # Stop at EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Return generated sequence (excluding input)
        generated_only = generated[:, input_ids.shape[1]:]
        
        if return_logprobs:
            return generated_only, torch.tensor(logprobs_list)
        return generated_only, None
    
    def _extract_assistant_response(self, text: str) -> str:
        """Extract assistant response from generated text."""
        assistant_token = self.chat_formatter.assistant_token
        if assistant_token in text:
            parts = text.split(assistant_token)
            if len(parts) > 1:
                return parts[-1].strip()
        return text.strip()

