#!/usr/bin/env python3
"""
MM-Rec Trained Model Evaluation Script
Eğitilmiş modeli yükler, inference testi yapar ve text generation örnekleri üretir
"""

import torch
import torch.nn.functional as F
import argparse
import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.model import MMRecModel
from mm_rec.data.text_data_loader import SimpleCharacterTokenizer
from mm_rec.training.evaluation import compute_perplexity, compute_accuracy


def load_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[MMRecModel, dict]:
    """Checkpoint'ten modeli yükle"""
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Model config'i al
    model_config = checkpoint.get('model_config', {})
    if not model_config:
        raise ValueError("Model config not found in checkpoint")
    
    # Model oluştur
    model = MMRecModel(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    print(f"✅ Model loaded successfully!")
    print(f"   Step: {checkpoint.get('step', 'N/A')}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Loss: {checkpoint.get('loss', 'N/A')}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, checkpoint


def generate_text(
    model: MMRecModel,
    tokenizer: SimpleCharacterTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: str = 'cpu'
) -> str:
    """Text generation yap"""
    model.eval()
    
    # Prompt'u tokenize et
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    
    generated = prompt_tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length - len(prompt_tokens)):
            # Forward pass
            logits = model(input_ids)
            
            # Son token'ın logits'ini al
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling (eğer belirtilmişse)
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            # Softmax ve sampling
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            
            # Yeni token'ı input'a ekle
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]], device=device, dtype=torch.long)
            ], dim=1)
            
            # EOS token kontrolü (eğer varsa)
            eos_id = tokenizer.char_to_id.get('<EOS>', -1)
            if next_token == eos_id:
                break
    
    # Decode et
    generated_text = tokenizer.decode(generated)
    return generated_text


def evaluate_on_text(
    model: MMRecModel,
    tokenizer: SimpleCharacterTokenizer,
    text: str,
    device: str = 'cpu'
) -> dict:
    """Tek bir text üzerinde değerlendirme yap"""
    model.eval()
    
    # Tokenize et
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return {"error": "Text too short"}
    
    # Input ve target hazırla
    input_ids = torch.tensor([tokens[:-1]], device=device, dtype=torch.long)
    targets = torch.tensor([tokens[1:]], device=device, dtype=torch.long)
    
    with torch.no_grad():
        # Forward pass
        logits = model(input_ids)
        
        # Loss hesapla
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )
        
        # Perplexity
        perplexity = compute_perplexity(loss.item())
        
        # Accuracy
        accuracy = compute_accuracy(logits, targets)
    
    return {
        "loss": loss.item(),
        "perplexity": perplexity,
        "accuracy": accuracy,
        "num_tokens": len(tokens)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MM-Rec model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/tiny/final_checkpoint.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Path to vocabulary file (optional)"
    )
    parser.add_argument(
        "--test-text",
        type=str,
        default=None,
        help="Test text for evaluation"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox",
        help="Prompt for text generation"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (optional)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔍 MM-Rec Model Evaluation")
    print("=" * 70)
    print()
    
    # Device kontrolü
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = "cpu"
    
    # Model yükle
    model, checkpoint = load_model(args.checkpoint, device)
    
    # Tokenizer oluştur ve vocabulary'yi yeniden oluştur
    vocab_size = checkpoint['model_config'].get('vocab_size', 5000)
    tokenizer = SimpleCharacterTokenizer(vocab_size=vocab_size)
    
    # Eğitim sırasında kullanılan sample corpus'u yükle ve vocabulary'yi oluştur
    sample_corpus_path = Path(project_root) / "checkpoints" / "sample_corpus.txt"
    if sample_corpus_path.exists():
        print(f"📚 Loading sample corpus to rebuild vocabulary...")
        with open(sample_corpus_path, 'r', encoding='utf-8') as f:
            sample_text = f.read()
        tokenizer.build_vocab([sample_text])
        print(f"✅ Vocabulary rebuilt: {tokenizer.next_id} tokens")
    else:
        print(f"⚠️  Sample corpus not found, using empty vocabulary")
    
    print()
    print("=" * 70)
    print("📊 Model Evaluation")
    print("=" * 70)
    print()
    
    # Test text ile değerlendirme
    if args.test_text:
        print(f"📝 Evaluating on test text: {args.test_text[:50]}...")
        results = evaluate_on_text(model, tokenizer, args.test_text, device)
        
        if "error" not in results:
            print(f"   Loss: {results['loss']:.4f}")
            print(f"   Perplexity: {results['perplexity']:.2f}")
            print(f"   Accuracy: {results['accuracy']:.2%}")
            print(f"   Tokens: {results['num_tokens']}")
        else:
            print(f"   Error: {results['error']}")
        print()
    
    # Text generation
    print("=" * 70)
    print("✍️  Text Generation")
    print("=" * 70)
    print()
    
    print(f"Prompt: '{args.prompt}'")
    print(f"Max Length: {args.max_length}")
    print(f"Temperature: {args.temperature}")
    if args.top_k:
        print(f"Top-K: {args.top_k}")
    print()
    
    try:
        generated = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        
        print("Generated Text:")
        print("-" * 70)
        print(generated)
        print("-" * 70)
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 70)
    print("✅ Evaluation completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
