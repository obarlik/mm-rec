#!/usr/bin/env python3
"""
OpenAI-Compatible Demo Script
Quick demonstration of OpenAI chat format training
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig, ChatCompletionAPI
from mm_rec.data.chat_format import ChatMessage

def main():
    print("="*80)
    print("MM-Rec OpenAI-Compatible Demo")
    print("="*80)
    
    device = torch.device('cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Initialize tokenizer
    print("\n🔤 Initializing tokenizer...")
    try:
        tokenizer = get_tokenizer(model_name="gpt-4", vocab_size=100256)
        print(f"✅ Tokenizer: {tokenizer.__class__.__name__} (vocab_size={tokenizer.vocab_size})")
    except Exception as e:
        print(f"⚠️ Using fallback tokenizer: {e}")
        from mm_rec.tokenizers.openai_tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer(vocab_size=32000)
        print(f"✅ Fallback tokenizer initialized (vocab_size={tokenizer.vocab_size})")
    
    # Initialize model (smaller for demo)
    print("\n🤖 Initializing model...")
    model = MMRec100M(
        vocab_size=tokenizer.vocab_size,
        expert_dim=128,  # Smaller for demo
        num_layers=4,   # Fewer layers
        num_heads=4,
        ffn_dim=512
    ).to(device)
    
    print(f"✅ Model initialized ({model.get_num_params():,} parameters)")
    
    # Initialize trainer
    print("\n🎓 Initializing SFT Trainer...")
    config = SFTConfig(max_length=128, only_predict_assistant=True)  # Short sequences
    trainer = SFTTrainer(model, tokenizer, config)
    print("✅ Trainer initialized")
    
    # Test chat format
    print("\n📝 Testing chat format...")
    messages = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Hello!"),
        ChatMessage(role="assistant", content="Hi! How can I help?")
    ]
    
    formatter = trainer.chat_formatter
    formatted = formatter.format_messages(messages)
    print("✅ Chat format test:")
    print(f"   Input: {formatted[:100]}...")
    
    # Test tokenization
    print("\n🔤 Testing tokenization...")
    tokens = tokenizer.encode("Hello, world!", max_length=20)
    decoded = tokenizer.decode(tokens)
    print(f"✅ Tokenization test:")
    print(f"   Tokens: {len(tokens)}")
    print(f"   Decoded: {decoded[:50]}")
    
    # Test model forward (no training)
    print("\n🧪 Testing model forward pass...")
    try:
        input_text = "Hello"
        input_ids = tokenizer.encode(input_text, max_length=10, truncation=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
        
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {input_tensor.shape}")
        print(f"   Logits shape: {logits.shape}")
        print(f"   Vocabulary size: {logits.shape[-1]}")
    except Exception as e:
        print(f"⚠️ Forward pass failed: {e}")
    
    print("\n" + "="*80)
    print("✅ Demo completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Install tiktoken: pip install tiktoken")
    print("   2. Run full training: python3 -m mm_rec.scripts.train_openai --create_sample_data")
    print("   3. See OPENAI_COMPATIBILITY.md for details")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

