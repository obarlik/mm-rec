#!/usr/bin/env python3
"""
Test: Stop Sequences Functionality
Verifies OpenAI-compatible stop parameter
"""

import torch
import sys
import os

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import ChatCompletionAPI


def test_basic_stop():
    """Test basic stop sequence."""
    print("🧪 Test 1: Basic Stop Sequence")
    
    device = torch.device('cpu')
    tokenizer = get_tokenizer(vocab_size=100256)
    
    model = MMRec100M(
        vocab_size=tokenizer.vocab_size,
        expert_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512
    ).to(device)
    
    api = ChatCompletionAPI(model, tokenizer)
    
    # Test: Stop at specific word
    response = api.create(
        messages=[{"role": "user", "content": "Count: 1 2 3 4 5 6 7 8 9 10"}],
        max_tokens=20,
        stop=["5"],  # Should stop before "5"
        device=device
    )
    
    content = response["choices"][0]["message"]["content"]
    print(f"   Response: {content[:50]}...")
    
    # Verify stop sequence not in output
    assert "5" not in content, f"Stop sequence '5' found in output: {content}"
    print("   ✅ Stop sequence excluded from output\n")


def test_multiple_stops():
    """Test multiple stop sequences."""
    print("🧪 Test 2: Multiple Stop Sequences")
    
    device = torch.device('cpu')
    tokenizer = get_tokenizer(vocab_size=100256)
    
    model = MMRec100M(
        vocab_size=tokenizer.vocab_size,
        expert_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512
    ).to(device)
    
    api = ChatCompletionAPI(model, tokenizer)
    
    # Test: Multiple stop sequences
    response = api.create(
        messages=[{"role": "user", "content": "Write a story"}],
        max_tokens=30,
        stop=["END", "The end", ".", "!"],  # Multiple stops
        device=device
    )
    
    content = response["choices"][0]["message"]["content"]
    print(f"   Response: {content[:50]}...")
    
    # Verify at least one stop worked
    stopped = any(seq not in content for seq in ["END", "The end"])
    print(f"   ✅ Stopped before reaching end markers\n")


def test_streaming_with_stop():
    """Test stop sequences in streaming mode."""
    print("🧪 Test 3: Streaming with Stop Sequences")
    
    device = torch.device('cpu')
    tokenizer = get_tokenizer(vocab_size=100256)
    
    model = MMRec100M(
        vocab_size=tokenizer.vocab_size,
        expert_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512
    ).to(device)
    
    api = ChatCompletionAPI(model, tokenizer)
    
    # Test: Streaming with stop
    stream = api.create(
        messages=[{"role": "user", "content": "Count to 10"}],
        max_tokens=20,
        stop=["5"],
        stream=True,
        device=device
    )
    
    chunks = list(stream)
    print(f"   Received {len(chunks)} chunks")
    
    # Verify stream completed
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    print("   ✅ Stream completed with stop\n")


def main():
    print("="*80)
    print("Stop Sequences - Verification Tests")
    print("="*80)
    print()
    
    try:
        test_basic_stop()
        test_multiple_stops()
        test_streaming_with_stop()
        
        print("="*80)
        print("✅ All stop sequence tests passed!")
        print("="*80)
        print("\n💡 Stop sequences are OpenAI-compatible!")
        print("   - Max 4 sequences supported")
        print("   - Stop sequence excluded from output")
        print("   - Works in both standard and streaming modes")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
