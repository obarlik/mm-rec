"""
MM-Rec Model Benchmark Script
Comprehensive performance and memory benchmarking for different sequence lengths.

This script validates the core promise of MM-Rec:
- O(N log N) computational complexity
- O(M) memory complexity (where M << N)
- Efficient handling of 32K+ sequence lengths
"""

import torch
import torch.nn as nn
import time
import sys
import os
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.model import MMRecModel
from mm_rec.core.memory_state import MemoryState


def get_device() -> torch.device:
    """Get available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  CUDA not available, using CPU (benchmark will be slower)")
    return device


def create_model(device: torch.device, model_dim: int = 1024, num_layers: int = 2) -> MMRecModel:
    """
    Create MMRecModel with production-like dimensions.
    
    Args:
        device: Target device
        model_dim: Model dimension (default: 1024 for benchmark)
        num_layers: Number of layers (default: 2 for benchmark)
    
    Returns:
        MMRecModel instance
    """
    print(f"\n📦 Creating MMRecModel...")
    print(f"   model_dim: {model_dim}")
    print(f"   num_layers: {num_layers}")
    
    model = MMRecModel(
        vocab_size=1000,  # Smaller vocab for benchmark
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=8,
        num_memories=16,
        mem_dim=256,
        max_seq_len=32768,  # Support up to 32K
        dropout=0.0  # No dropout for benchmark
    )
    
    model = model.to(device)
    model.train()  # Training mode for backward pass
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


def run_benchmark(
    model: MMRecModel,
    seq_len: int,
    batch_size: int = 1,
    iterations: int = 10,
    warmup: int = 3,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Run benchmark for a specific sequence length.
    
    Args:
        model: MMRecModel instance
        seq_len: Sequence length to test
        batch_size: Batch size (default: 1)
        iterations: Number of iterations (default: 10)
        warmup: Number of warmup iterations (default: 3)
        device: Target device
    
    Returns:
        Dictionary with benchmark results:
        - forward_time_ms: Average forward pass time (ms)
        - backward_time_ms: Average backward pass time (ms)
        - total_time_ms: Average total time (ms)
        - tokens_per_second: Throughput (tokens/s)
        - max_memory_mb: Maximum GPU memory usage (MB)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Create input
    vocab_size = model.vocab_size
    input_ids = torch.randint(
        0, vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long
    )
    
    # Create targets for loss
    targets = torch.randint(
        0, vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long
    )
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Timing events (GPU) or time.time() (CPU)
    use_cuda_events = device.type == 'cuda'
    
    forward_times = []
    backward_times = []
    total_times = []
    
    print(f"   Running {iterations} iterations (warmup: {warmup})...")
    
    for i in range(iterations):
        # Clear gradients
        model.zero_grad()
        
        # Forward pass timing
        if use_cuda_events:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
        else:
            start_time = time.time()
        
        # Forward pass
        logits = model(input_ids)  # [batch, seq_len, vocab_size]
        
        if use_cuda_events:
            end_event.record()
            torch.cuda.synchronize()
            forward_time = start_event.elapsed_time(end_event)  # milliseconds
        else:
            end_time = time.time()
            forward_time = (end_time - start_time) * 1000  # convert to ms
        
        # Compute loss
        batch_size_actual, seq_len_actual, vocab_size_actual = logits.shape
        logits_flat = logits.view(-1, vocab_size_actual)
        targets_flat = targets.view(-1)
        loss = loss_fn(logits_flat, targets_flat)
        
        # Backward pass timing
        if use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
        else:
            start_time = time.time()
        
        # Backward pass
        loss.backward()
        
        if use_cuda_events:
            end_event.record()
            torch.cuda.synchronize()
            backward_time = start_event.elapsed_time(end_event)  # milliseconds
        else:
            end_time = time.time()
            backward_time = (end_time - start_time) * 1000  # convert to ms
        
        total_time = forward_time + backward_time
        
        # Skip warmup iterations
        if i >= warmup:
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            total_times.append(total_time)
        
        # Progress indicator
        if (i + 1) % max(1, iterations // 5) == 0:
            print(f"      Iteration {i+1}/{iterations} completed")
    
    # Calculate averages (excluding warmup)
    avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
    avg_backward = sum(backward_times) / len(backward_times) if backward_times else 0
    avg_total = sum(total_times) / len(total_times) if total_times else 0
    
    # Calculate throughput
    tokens_per_second = (seq_len * batch_size) / (avg_total / 1000) if avg_total > 0 else 0
    
    # Get maximum memory usage
    if device.type == 'cuda':
        max_memory_bytes = torch.cuda.max_memory_allocated()
        max_memory_mb = max_memory_bytes / (1024 ** 2)
    else:
        # CPU memory tracking is more complex, skip for now
        max_memory_mb = 0.0
    
    return {
        'forward_time_ms': avg_forward,
        'backward_time_ms': avg_backward,
        'total_time_ms': avg_total,
        'tokens_per_second': tokens_per_second,
        'max_memory_mb': max_memory_mb
    }


def print_results_table(results: List[Dict[str, any]]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "="*100)
    print("MM-Rec Model Benchmark Results")
    print("="*100)
    print(f"{'Seq Len (N)':<12} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<15} {'Tokens/s':<15} {'Memory (MB)':<15}")
    print("-"*100)
    
    for result in results:
        seq_len = result['seq_len']
        forward = result['forward_time_ms']
        backward = result['backward_time_ms']
        total = result['total_time_ms']
        tokens_per_sec = result['tokens_per_second']
        memory = result['max_memory_mb']
        
        print(f"{seq_len:<12} {forward:<15.2f} {backward:<15.2f} {total:<15.2f} {tokens_per_sec:<15.2f} {memory:<15.2f}")
    
    print("="*100)


def analyze_complexity(results: List[Dict[str, any]]) -> None:
    """
    Analyze computational and memory complexity from benchmark results.
    
    Validates O(N log N) computational complexity and O(M) memory complexity.
    """
    print("\n" + "="*100)
    print("Complexity Analysis")
    print("="*100)
    
    if len(results) < 2:
        print("⚠️  Need at least 2 data points for complexity analysis")
        return
    
    # Extract data
    seq_lens = [r['seq_len'] for r in results]
    total_times = [r['total_time_ms'] for r in results]
    memory_usage = [r['max_memory_mb'] for r in results]
    
    print("\n📊 Computational Complexity Analysis:")
    print(f"{'Seq Len':<12} {'Time (ms)':<15} {'N log N':<15} {'Ratio':<15}")
    print("-"*60)
    
    # Calculate N log N for each sequence length
    import math
    for i, (seq_len, time_ms) in enumerate(zip(seq_lens, total_times)):
        n_log_n = seq_len * math.log2(seq_len) if seq_len > 1 else seq_len
        if i > 0:
            prev_seq_len = seq_lens[i-1]
            prev_time = total_times[i-1]
            prev_n_log_n = prev_seq_len * math.log2(prev_seq_len) if prev_seq_len > 1 else prev_seq_len
            
            # Ratio of actual time increase vs expected N log N increase
            time_ratio = time_ms / prev_time if prev_time > 0 else 0
            n_log_n_ratio = n_log_n / prev_n_log_n if prev_n_log_n > 0 else 0
            complexity_ratio = time_ratio / n_log_n_ratio if n_log_n_ratio > 0 else 0
            
            print(f"{seq_len:<12} {time_ms:<15.2f} {n_log_n:<15.2f} {complexity_ratio:<15.2f}")
        else:
            print(f"{seq_len:<12} {time_ms:<15.2f} {n_log_n:<15.2f} {'-':<15}")
    
    print("\n💾 Memory Complexity Analysis:")
    print(f"{'Seq Len':<12} {'Memory (MB)':<15} {'Growth':<15}")
    print("-"*45)
    
    for i, (seq_len, mem_mb) in enumerate(zip(seq_lens, memory_usage)):
        if i > 0:
            prev_seq_len = seq_lens[i-1]
            prev_mem = memory_usage[i-1]
            mem_growth = mem_mb / prev_mem if prev_mem > 0 else 0
            seq_growth = seq_len / prev_seq_len if prev_seq_len > 0 else 0
            
            # Memory should grow sub-linearly (O(M) where M << N)
            print(f"{seq_len:<12} {mem_mb:<15.2f} {mem_growth:.2f}x (seq: {seq_growth:.2f}x)")
        else:
            print(f"{seq_len:<12} {mem_mb:<15.2f} {'-':<15}")
    
    # Summary
    print("\n📈 Summary:")
    if len(seq_lens) >= 2:
        # Check if time growth is close to N log N
        first_time = total_times[0]
        last_time = total_times[-1]
        first_seq = seq_lens[0]
        last_seq = seq_lens[-1]
        
        time_ratio = last_time / first_time if first_time > 0 else 0
        expected_n_log_n_ratio = (last_seq * math.log2(last_seq)) / (first_seq * math.log2(first_seq)) if first_seq > 1 else 0
        
        print(f"   Time growth: {time_ratio:.2f}x (expected ~{expected_n_log_n_ratio:.2f}x for O(N log N))")
        
        # Check memory growth
        first_mem = memory_usage[0]
        last_mem = memory_usage[-1]
        mem_ratio = last_mem / first_mem if first_mem > 0 else 0
        seq_ratio = last_seq / first_seq if first_seq > 0 else 0
        
        print(f"   Memory growth: {mem_ratio:.2f}x (sequence growth: {seq_ratio:.2f}x)")
        print(f"   Memory efficiency: {'✅ Good' if mem_ratio < seq_ratio else '⚠️  Needs optimization'}")
    
    print("="*100)


def main():
    """Main benchmark execution."""
    print("="*100)
    print("MM-Rec Model Benchmark")
    print("Validating O(N log N) computational complexity and O(M) memory complexity")
    print("="*100)
    
    # Device setup
    device = get_device()
    
    # Model configuration (production-like but smaller for benchmark)
    model_dim = 1024
    num_layers = 2
    
    # Create model
    model = create_model(device, model_dim=model_dim, num_layers=num_layers)
    
    # Sequence lengths to test
    # Short (Control): [128, 512]
    # Medium (Current Max): [4096, 8192]
    # Long (Target): [16384, 32768]
    test_seq_lengths = [128, 512, 4096, 8192, 16384, 32768]
    
    # Adjust iterations based on sequence length
    def get_iterations(seq_len: int) -> int:
        if seq_len <= 512:
            return 10
        elif seq_len <= 8192:
            return 5
        else:
            return 3  # Fewer iterations for very long sequences
    
    print(f"\n🧪 Benchmark Configuration:")
    print(f"   Sequence lengths: {test_seq_lengths}")
    print(f"   Batch size: 1")
    print(f"   Iterations: Adaptive (10 for short, 5 for medium, 3 for long)")
    print(f"   Warmup: 3 iterations")
    
    # Run benchmarks
    results = []
    
    for seq_len in test_seq_lengths:
        print(f"\n{'='*100}")
        print(f"Testing sequence length: {seq_len}")
        print(f"{'='*100}")
        
        iterations = get_iterations(seq_len)
        
        try:
            result = run_benchmark(
                model=model,
                seq_len=seq_len,
                batch_size=1,
                iterations=iterations,
                warmup=3,
                device=device
            )
            
            result['seq_len'] = seq_len
            results.append(result)
            
            print(f"\n✅ Completed: {seq_len} tokens")
            print(f"   Forward: {result['forward_time_ms']:.2f} ms")
            print(f"   Backward: {result['backward_time_ms']:.2f} ms")
            print(f"   Total: {result['total_time_ms']:.2f} ms")
            print(f"   Throughput: {result['tokens_per_second']:.2f} tokens/s")
            print(f"   Memory: {result['max_memory_mb']:.2f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️  Out of memory for sequence length {seq_len}")
                print(f"   Skipping longer sequences...")
                break
            else:
                raise
        except Exception as e:
            print(f"❌ Error testing sequence length {seq_len}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Print results table
    if results:
        print_results_table(results)
        
        # Complexity analysis
        analyze_complexity(results)
        
        print("\n✅ Benchmark completed successfully!")
    else:
        print("\n❌ No benchmark results collected")


if __name__ == "__main__":
    main()

