#!/usr/bin/env python3
"""
🔗 INTEGRATION TEST SUITE 🔗
-----------------------------
Tests that all components work together seamlessly.

Focus: Cross-component interactions, not individual mechanisms.
"""

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from mm_rec.models.mmrec_100m import MMRec100M
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import ChatCompletionAPI, SFTTrainer, SFTConfig
from mm_rec.streaming_input import StreamingInputProcessor


def print_test(name):
    print(f"\n{'='*80}")
    print(f"  {name}")
    print('='*80)


def test_streaming_with_stop_sequences():
    """Integration: Streaming output + Stop sequences."""
    print_test("Test 1: Streaming + Stop Sequences")
    
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
    
    # Test: Streaming should respect stop sequences
    stream = api.create(
        messages=[{"role": "user", "content": "Count: 1 2 3 4 5 6 7"}],
        max_tokens=20,
        stop=["5"],
        stream=True,
        device=device
    )
    
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
        if "content" in chunk["choices"][0]["delta"]:
            print(f"  Chunk: {chunk['choices'][0]['delta']['content']}", end="", flush=True)
    
    print()
    
    # Verify: Stream completed and stopped
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    print("  ✅ Streaming + Stop Sequences: INTEGRATED")


def test_infinite_context_with_sparse_routing():
    """Integration: Infinite context + Sparse routing."""
    print_test("Test 2: Infinite Context + Sparse Routing")
    
    device = torch.device('cpu')
    vocab_size = 100256
    
    # Use base MMRecModel with sparse enabled
    from mm_rec.model import MMRecModel
    model = MMRecModel(
        vocab_size=vocab_size,
        model_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        use_sparse=True,
        num_experts=32
    ).to(device)
    
    # Test: Variable-length sequences with sparse routing
    test_lengths = [512, 1024, 2048]
    
    for seq_len in test_lengths:
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        
        # Forward with sparse routing
        output = model(input_ids, router_threshold=0.5)
        
        # Verify: Output shape correct
        assert output.shape == (1, seq_len, vocab_size)
        print(f"  ✅ {seq_len} tokens with sparse routing: OK")
    
    print("  ✅ Infinite Context + Sparse Routing: INTEGRATED")


def test_sft_training_with_all_mechanisms():
    """Integration: SFT training + All mechanisms."""
    print_test("Test 3: SFT Training + All Mechanisms")
    
    device = torch.device('cpu')
    tokenizer = get_tokenizer(vocab_size=100256)
    
    # Use base MMRecModel
    from mm_rec.model import MMRecModel
    model = MMRecModel(
        vocab_size=tokenizer.vocab_size,
        model_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        use_sparse=True,
        num_experts=32,
        use_dpg=True,
        use_hem=True
    ).to(device)
    
    # Create SFT trainer
    config = SFTConfig(max_length=512)
    trainer = SFTTrainer(model, tokenizer, config)
    
    # Sample conversation (as ChatMessage objects)
    from mm_rec.data.chat_format import ChatMessage
    messages = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!")
    ]
    
    # Training step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Use train_step (correct API)
    result = trainer.train_step(messages, optimizer, device, verbose=False)
    
    print(f"  Training loss: {result['loss']:.4f}")
    
    # Verify: Gradients flowed through all mechanisms
    assert model.blocks[0].ffn.expert_weights.grad is not None
    assert model.blocks[0].W_gamma_up.weight.grad is not None
    
    print("  ✅ SFT Training + All Mechanisms: INTEGRATED")


def test_explainability_with_streaming():
    """Integration: Explainability (logprobs) + Streaming."""
    print_test("Test 4: Explainability + Streaming")
    
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
    
    # Test: Streaming with confidence scores
    stream = api.create(
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=5,
        stream=True,
        device=device
    )
    
    confidences = []
    for chunk in stream:
        if "confidence" in chunk:
            confidences.append(chunk["confidence"])
    
    # Verify: Got confidence scores
    assert len(confidences) > 0
    print(f"  Received {len(confidences)} confidence scores")
    print("  ✅ Explainability + Streaming: INTEGRATED")


def test_threshold_control_with_infinite_context():
    """Integration: Uncertainty threshold + Infinite context."""
    print_test("Test 5: Threshold Control + Infinite Context")
    
    device = torch.device('cpu')
    vocab_size = 100256
    
    from mm_rec.model import MMRecModel
    model = MMRecModel(
        vocab_size=vocab_size,
        model_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        use_sparse=True,
        num_experts=32
    ).to(device)
    
    # Test: Long sequence with different thresholds
    seq_len = 1024
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
    
    # Low threshold (more experts)
    out_low = model(input_ids, router_threshold=0.1)
    
    # High threshold (fewer experts)
    out_high = model(input_ids, router_threshold=0.9)
    
    # Verify: Different thresholds produce different outputs
    diff = (out_low - out_high).abs().sum().item()
    assert diff > 0
    
    print(f"  Output difference: {diff:.2f}")
    print("  ✅ Threshold Control + Infinite Context: INTEGRATED")


def test_streaming_input_with_generation():
    """Integration: Streaming input processor + Generation."""
    print_test("Test 6: Streaming Input + Generation")
    
    device = torch.device('cpu')
    tokenizer = get_tokenizer(vocab_size=100256)
    
    model = MMRec100M(
        vocab_size=tokenizer.vocab_size,
        expert_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512
    ).to(device)
    
    # Create streaming processor
    from mm_rec.streaming_input import StreamChunk
    processor = StreamingInputProcessor(model, tokenizer, device=device)
    
    # Simulate streaming input with StreamChunk objects
    chunks = ["Hello ", "world, ", "this is ", "a test."]
    
    for chunk_text in chunks:
        chunk = StreamChunk(text=chunk_text, timestamp=0.0)
        processor.add_chunk(chunk)
    
    # Generate response
    response = next(processor.generate(max_tokens=10, stream_output=False))
    
    print(f"  Generated: {response[:50]}...")
    print("  ✅ Streaming Input + Generation: INTEGRATED")


def test_end_to_end_chat_pipeline():
    """Integration: Full chat pipeline (input → process → output)."""
    print_test("Test 7: End-to-End Chat Pipeline")
    
    device = torch.device('cpu')
    tokenizer = get_tokenizer(vocab_size=100256)
    
    # Use base MMRecModel for full feature support
    from mm_rec.model import MMRecModel
    model = MMRecModel(
        vocab_size=tokenizer.vocab_size,
        model_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        use_sparse=True,
        num_experts=32
    ).to(device)
    
    api = ChatCompletionAPI(model, tokenizer)
    
    # Multi-turn conversation
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    
    # Test: Full pipeline with all features
    response = api.create(
        messages=conversation,
        max_tokens=20,
        temperature=0.7,
        top_p=0.9,
        stop=["END"],
        device=device
    )
    
    # Verify: Response structure
    assert "choices" in response
    assert "message" in response["choices"][0]
    assert "logprobs" in response["choices"][0]
    assert "usage" in response
    
    print(f"  Response: {response['choices'][0]['message']['content'][:50]}...")
    print(f"  Confidence: {response['choices'][0]['logprobs']['average_token_confidence']}")
    print("  ✅ End-to-End Chat Pipeline: INTEGRATED")


def main():
    print("\n" + "🔗"*40)
    print("  MM-REC INTEGRATION TEST SUITE")
    print("🔗"*40)
    
    tests = [
        test_streaming_with_stop_sequences,
        test_infinite_context_with_sparse_routing,
        test_sft_training_with_all_mechanisms,
        test_explainability_with_streaming,
        test_threshold_control_with_infinite_context,
        test_streaming_input_with_generation,
        test_end_to_end_chat_pipeline
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("  INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"\n  Total Tests: {len(tests)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Success Rate: {passed/len(tests)*100:.1f}%\n")
    
    if failed == 0:
        print("  🏆 ALL INTEGRATIONS VERIFIED")
        print("  ✅ Components work together seamlessly")
        print("  ✅ No integration issues detected\n")
        return 0
    else:
        print("  ⚠️  INTEGRATION ISSUES DETECTED")
        print(f"  ❌ {failed} integration(s) failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
