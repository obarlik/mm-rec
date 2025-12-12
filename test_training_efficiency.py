#!/usr/bin/env python3
"""
📚 TRAINING EFFICIENCY TEST 📚
------------------------------
Tests if the model can learn effectively:
- Fast convergence
- Low final loss
- No overfitting
- Gradient flow
"""

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from mm_rec.model import MMRecModel
from mm_rec.tokenizers.openai_tokenizer import get_tokenizer
from mm_rec.training.sft_trainer import SFTTrainer, SFTConfig
from mm_rec.data.chat_format import ChatMessage


def test_simple_pattern_learning():
    """Test: Can model learn a simple pattern?"""
    print("="*80)
    print("Test 1: Simple Pattern Learning")
    print("="*80)
    
    device = torch.device('cpu')
    tokenizer = get_tokenizer(vocab_size=100256)
    
    # Small model for fast training
    model = MMRecModel(
        vocab_size=tokenizer.vocab_size,
        model_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        use_sparse=True,
        num_experts=16
    ).to(device)
    
    # Simple pattern: "Hello" -> "World"
    training_data = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="World")
    ]
    
    config = SFTConfig(max_length=128)
    trainer = SFTTrainer(model, tokenizer, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Train for multiple epochs
    losses = []
    print("\nTraining on simple pattern (Hello -> World):")
    
    for epoch in range(20):
        result = trainer.train_step(training_data, optimizer, device, verbose=False)
        losses.append(result['loss'])
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: loss={result['loss']:.4f}, ppl={result['perplexity']:.2f}")
    
    # Check convergence
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Improvement:  {improvement:.1f}%")
    
    if improvement > 50:
        print("  ✅ Model learned the pattern (>50% improvement)")
        return True, losses
    else:
        print("  ❌ Model failed to learn (<50% improvement)")
        return False, losses


def test_multi_pattern_learning():
    """Test: Can model learn multiple patterns without forgetting?"""
    print("\n" + "="*80)
    print("Test 2: Multi-Pattern Learning (No Catastrophic Forgetting)")
    print("="*80)
    
    device = torch.device('cpu')
    tokenizer = get_tokenizer(vocab_size=100256)
    
    model = MMRecModel(
        vocab_size=tokenizer.vocab_size,
        model_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        use_sparse=True,
        num_experts=16
    ).to(device)
    
    # Multiple patterns
    patterns = [
        [ChatMessage(role="user", content="Hello"), ChatMessage(role="assistant", content="World")],
        [ChatMessage(role="user", content="Good"), ChatMessage(role="assistant", content="Morning")],
        [ChatMessage(role="user", content="Thank"), ChatMessage(role="assistant", content="You")]
    ]
    
    config = SFTConfig(max_length=128)
    trainer = SFTTrainer(model, tokenizer, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nTraining on 3 patterns:")
    
    all_losses = {i: [] for i in range(len(patterns))}
    
    for epoch in range(30):
        # Train on all patterns
        for i, pattern in enumerate(patterns):
            result = trainer.train_step(pattern, optimizer, device, verbose=False)
            all_losses[i].append(result['loss'])
        
        if epoch % 10 == 0:
            avg_loss = sum(all_losses[i][-1] for i in range(len(patterns))) / len(patterns)
            print(f"  Epoch {epoch:2d}: avg_loss={avg_loss:.4f}")
    
    # Check if all patterns learned
    print("\n  Final losses per pattern:")
    all_learned = True
    for i, pattern in enumerate(patterns):
        initial = all_losses[i][0]
        final = all_losses[i][-1]
        improvement = (initial - final) / initial * 100
        
        user_msg = pattern[0].content
        print(f"    Pattern '{user_msg}': {final:.4f} ({improvement:.1f}% improvement)")
        
        if improvement < 30:
            all_learned = False
    
    if all_learned:
        print("  ✅ All patterns learned without forgetting")
        return True
    else:
        print("  ❌ Some patterns not learned (catastrophic forgetting?)")
        return False


def test_gradient_flow_stability():
    """Test: Are gradients stable (no explosion/vanishing)?"""
    print("\n" + "="*80)
    print("Test 3: Gradient Flow Stability")
    print("="*80)
    
    device = torch.device('cpu')
    tokenizer = get_tokenizer(vocab_size=100256)
    
    model = MMRecModel(
        vocab_size=tokenizer.vocab_size,
        model_dim=128,
        num_layers=4,  # Deeper model
        num_heads=4,
        ffn_dim=512,
        use_sparse=True,
        num_experts=16
    ).to(device)
    
    training_data = [
        ChatMessage(role="user", content="Test gradient flow"),
        ChatMessage(role="assistant", content="Gradients should be stable")
    ]
    
    config = SFTConfig(max_length=128)
    trainer = SFTTrainer(model, tokenizer, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nChecking gradient norms across layers:")
    
    grad_norms = []
    
    for epoch in range(10):
        result = trainer.train_step(training_data, optimizer, device, verbose=False)
        
        # Collect gradient norms per layer
        layer_grads = []
        for i, block in enumerate(model.blocks):
            if block.ffn.expert_weights.grad is not None:
                grad_norm = block.ffn.expert_weights.grad.norm().item()
                layer_grads.append(grad_norm)
        
        grad_norms.append(layer_grads)
        
        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: layer_grads={[f'{g:.2e}' for g in layer_grads]}")
    
    # Check for gradient explosion/vanishing
    final_grads = grad_norms[-1]
    max_grad = max(final_grads)
    min_grad = min(final_grads)
    
    print(f"\n  Max gradient: {max_grad:.2e}")
    print(f"  Min gradient: {min_grad:.2e}")
    print(f"  Ratio: {max_grad/min_grad:.2f}x")
    
    if max_grad < 1e3 and min_grad > 1e-6:
        print("  ✅ Gradients stable (no explosion/vanishing)")
        return True
    else:
        print("  ⚠️  Gradient instability detected")
        return False


def test_sparse_expert_utilization():
    """Test: Are sparse experts being utilized effectively?"""
    print("\n" + "="*80)
    print("Test 4: Sparse Expert Utilization")
    print("="*80)
    
    device = torch.device('cpu')
    tokenizer = get_tokenizer(vocab_size=100256)
    
    model = MMRecModel(
        vocab_size=tokenizer.vocab_size,
        model_dim=128,
        num_layers=2,
        num_heads=4,
        ffn_dim=512,
        use_sparse=True,
        num_experts=16
    ).to(device)
    
    # Different types of inputs
    diverse_data = [
        [ChatMessage(role="user", content="Math: 2+2="), ChatMessage(role="assistant", content="4")],
        [ChatMessage(role="user", content="Code: def hello():"), ChatMessage(role="assistant", content="print('hi')")],
        [ChatMessage(role="user", content="Story: Once upon"), ChatMessage(role="assistant", content="a time")]
    ]
    
    config = SFTConfig(max_length=128)
    trainer = SFTTrainer(model, tokenizer, config)
    
    print("\nChecking expert activation patterns:")
    
    # Hook to capture expert usage
    expert_usage = {i: 0 for i in range(16)}
    
    def hook_fn(module, input, output):
        if len(output) == 3:
            _, expert_indices, _ = output
            for idx in expert_indices[0].tolist():
                expert_usage[idx] = expert_usage.get(idx, 0) + 1
    
    model.blocks[0].ffn.router.register_forward_hook(hook_fn)
    
    # Run inference on diverse inputs
    model.eval()
    with torch.no_grad():
        for data in diverse_data:
            input_ids, _, _ = trainer.prepare_chat_input(data, device)
            _ = model(input_ids)
    
    # Analyze expert usage
    active_experts = sum(1 for count in expert_usage.values() if count > 0)
    total_experts = len(expert_usage)
    utilization = active_experts / total_experts * 100
    
    print(f"\n  Active experts: {active_experts}/{total_experts}")
    print(f"  Utilization: {utilization:.1f}%")
    print(f"  Expert usage: {dict(sorted(expert_usage.items(), key=lambda x: x[1], reverse=True)[:5])}")
    
    if utilization > 30:  # At least 30% of experts used
        print("  ✅ Good expert utilization (diverse specialization)")
        return True
    else:
        print("  ⚠️  Low expert utilization (may need more diverse data)")
        return False


def main():
    print("\n" + "📚"*40)
    print("  MM-REC TRAINING EFFICIENCY TEST")
    print("📚"*40)
    
    results = {}
    
    # Run tests
    results['pattern_learning'], losses = test_simple_pattern_learning()
    results['multi_pattern'] = test_multi_pattern_learning()
    results['gradient_stability'] = test_gradient_flow_stability()
    results['expert_utilization'] = test_sparse_expert_utilization()
    
    # Summary
    print("\n" + "="*80)
    print("  TRAINING EFFICIENCY SUMMARY")
    print("="*80)
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n  Tests Passed: {passed}/{total}")
    print(f"  Success Rate: {passed/total*100:.1f}%\n")
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test_name.replace('_', ' ').title()}")
    
    if all(results.values()):
        print("\n  🏆 TRAINING IS EFFICIENT")
        print("  ✅ Model learns quickly")
        print("  ✅ No catastrophic forgetting")
        print("  ✅ Stable gradients")
        print("  ✅ Good expert utilization\n")
        return 0
    else:
        print("\n  ⚠️  TRAINING ISSUES DETECTED")
        print("  Review failed tests above\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
