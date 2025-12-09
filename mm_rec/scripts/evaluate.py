#!/usr/bin/env python3
"""
MM-Rec Model Evaluation Script

Evaluates trained model on test data and computes metrics:
- Perplexity (PPL)
- Loss
- Token-level accuracy
- Generation quality (sample outputs)
"""

import torch
import torch.nn.functional as F
import argparse
import sys
import os
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig
from mm_rec.data.chat_format import ChatMessage
from mm_rec.data.convert_to_chat import load_chat_data


class ModelEvaluator:
    """Evaluate MM-Rec model on test data."""
    
    def __init__(
        self,
        model: MMRec100M,
        tokenizer,
        device: torch.device,
        max_length: int = 512
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()
    
    def evaluate_conversation(
        self,
        messages: List[ChatMessage],
        trainer: SFTTrainer
    ) -> Dict[str, float]:
        """Evaluate a single conversation."""
        with torch.no_grad():
            # Prepare input
            input_ids, attention_mask, labels = trainer.prepare_chat_input(
                messages, self.device
            )
            
            # Forward pass
            logits, _ = self.model(input_ids)
            
            # Compute loss
            loss = trainer.compute_loss(logits, labels, attention_mask)
            
            # Compute perplexity
            perplexity = math.exp(loss.item()) if loss.item() < 10 else float('inf')
            
            # Compute token-level accuracy
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Filter valid labels
            valid_mask = shift_labels != -100
            if valid_mask.sum() > 0:
                valid_logits = shift_logits[valid_mask]
                valid_labels = shift_labels[valid_mask]
                predictions = valid_logits.argmax(dim=-1)
                accuracy = (predictions == valid_labels).float().mean().item()
            else:
                accuracy = 0.0
            
            return {
                'loss': loss.item(),
                'perplexity': perplexity,
                'accuracy': accuracy,
                'num_tokens': valid_mask.sum().item() if valid_mask.sum() > 0 else 0
            }
    
    def evaluate_dataset(
        self,
        conversations: List[List[Dict[str, str]]],
        trainer: SFTTrainer,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate on entire dataset."""
        if max_samples:
            conversations = conversations[:max_samples]
        
        total_loss = 0.0
        total_perplexity = 0.0
        total_accuracy = 0.0
        total_tokens = 0
        num_valid = 0
        
        print(f"\n📊 Evaluating {len(conversations)} conversations...")
        
        for i, conv_data in enumerate(tqdm(conversations, desc="Evaluating")):
            try:
                messages = [
                    ChatMessage(role=msg["role"], content=msg["content"])
                    for msg in conv_data
                ]
                
                metrics = self.evaluate_conversation(messages, trainer)
                
                if metrics['num_tokens'] > 0:
                    total_loss += metrics['loss'] * metrics['num_tokens']
                    total_perplexity += metrics['perplexity'] * metrics['num_tokens']
                    total_accuracy += metrics['accuracy'] * metrics['num_tokens']
                    total_tokens += metrics['num_tokens']
                    num_valid += 1
            except Exception as e:
                print(f"⚠️ Error evaluating conversation {i}: {e}")
                continue
        
        if total_tokens == 0:
            return {
                'avg_loss': 0.0,
                'avg_perplexity': float('inf'),
                'avg_accuracy': 0.0,
                'total_tokens': 0,
                'num_valid': 0
            }
        
        return {
            'avg_loss': total_loss / total_tokens,
            'avg_perplexity': total_perplexity / total_tokens,
            'avg_accuracy': total_accuracy / total_tokens,
            'total_tokens': total_tokens,
            'num_valid': num_valid,
            'total_conversations': len(conversations)
        }
    
    def generate_sample(
        self,
        messages: List[ChatMessage],
        max_new_tokens: int = 50,
        temperature: float = 0.7
    ) -> str:
        """Generate a sample response."""
        # Format input
        from mm_rec.data.chat_format import format_messages_as_chat
        input_text = format_messages_as_chat(messages)
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding=False
        )
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # Generate
        generated_ids = input_ids.clone()
        self.model.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits, _ = self.model(generated_ids)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if max length reached
                if generated_ids.shape[1] >= self.max_length:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        return generated_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate MM-Rec Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, default="./data/chat_data_real.jsonl",
                        help="Path to test data JSONL file")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--expert_dim", type=int, default=256,
                        help="Expert dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--ffn_dim", type=int, default=1024,
                        help="FFN dimension")
    parser.add_argument("--vocab_size", type=int, default=100277,
                        help="Vocabulary size")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda)")
    parser.add_argument("--generate_samples", action="store_true",
                        help="Generate sample outputs")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of generation samples")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("="*80)
    print("MM-Rec Model Evaluation")
    print("="*80)
    print(f"🖥️  Device: {device}")
    print(f"📦 Checkpoint: {args.checkpoint}")
    print(f"📊 Test data: {args.test_data}")
    
    # Load test data
    print("\n📦 Loading test data...")
    conversations = load_chat_data(args.test_data)
    if not conversations:
        print("❌ No conversations loaded!")
        return 1
    
    print(f"✅ Loaded {len(conversations)} conversations")
    
    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...")
    tokenizer = get_tokenizer(model_name="gpt-4", vocab_size=args.vocab_size)
    print(f"✅ Tokenizer ready (vocab={tokenizer.vocab_size})")
    
    # Initialize model
    print("\n🤖 Initializing model...")
    model = MMRec100M(
        vocab_size=tokenizer.vocab_size,
        expert_dim=args.expert_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim
    ).to(device)
    print(f"✅ Model initialized ({model.get_num_params():,} params)")
    
    # Load checkpoint
    print(f"\n💾 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Checkpoint loaded")
    
    # Initialize trainer
    config = SFTConfig(
        model_name="gpt-4",
        max_length=args.max_length,
        only_predict_assistant=True
    )
    trainer = SFTTrainer(model, tokenizer, config)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, tokenizer, device, max_length=args.max_length)
    
    # Evaluate
    print("\n" + "="*80)
    results = evaluator.evaluate_dataset(
        conversations,
        trainer,
        max_samples=args.max_samples
    )
    
    # Print results
    print("\n" + "="*80)
    print("📊 Evaluation Results")
    print("="*80)
    print(f"   Conversations evaluated: {results['num_valid']}/{results['total_conversations']}")
    print(f"   Total tokens: {results['total_tokens']:,}")
    print(f"\n📈 Metrics:")
    print(f"   Average Loss: {results['avg_loss']:.4f}")
    print(f"   Average Perplexity: {results['avg_perplexity']:.2f}")
    print(f"   Average Accuracy: {results['avg_accuracy']*100:.2f}%")
    
    # Generate samples
    if args.generate_samples:
        print("\n" + "="*80)
        print("🎨 Sample Generations")
        print("="*80)
        
        for i in range(min(args.num_samples, len(conversations))):
            conv_data = conversations[i]
            messages = [
                ChatMessage(role=msg["role"], content=msg["content"])
                for msg in conv_data
            ]
            
            print(f"\n📝 Sample {i+1}:")
            print(f"   Input: {messages[-1].content[:100]}...")
            
            try:
                generated = evaluator.generate_sample(messages, max_new_tokens=50)
                print(f"   Generated: {generated[-200:]}...")
            except Exception as e:
                print(f"   ⚠️ Generation failed: {e}")
    
    print("\n" + "="*80)
    print("✅ Evaluation completed!")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

