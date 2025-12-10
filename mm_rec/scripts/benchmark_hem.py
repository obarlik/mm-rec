"""
HEM (Mekanizma 1) Performance Benchmark
Gerçek performans ölçümleri: HEM aktif vs pasif karşılaştırması
"""

import torch
import torch.nn as nn
import time
import sys
import os
from typing import Dict, List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mm_rec.model import MMRecModel
from mm_rec.blocks.mm_rec_block import MMRecBlock


def get_device() -> torch.device:
    """Get available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
    else:
        device = torch.device('cpu')
        print("⚠️  CUDA not available, using CPU")
    return device


def benchmark_single_block(
    block: MMRecBlock,
    x: torch.Tensor,
    state,
    iterations: int = 50,
    warmup: int = 10,
    device: torch.device = None
) -> Dict[str, float]:
    """Benchmark single MMRecBlock."""
    if device is None:
        device = next(block.parameters()).device
    
    # Clear cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    for _ in range(warmup):
        _ = block(x, state)
    
    # Timing
    use_cuda_events = device.type == 'cuda'
    times = []
    
    for i in range(iterations):
        if use_cuda_events:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_time = time.time()
        
        _ = block(x, state)
        
        if use_cuda_events:
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
        else:
            elapsed = (time.time() - start_time) * 1000
        
        times.append(elapsed)
    
    # Get memory
    if device.type == 'cuda':
        max_memory_bytes = torch.cuda.max_memory_allocated()
        max_memory_mb = max_memory_bytes / (1024 ** 2)
    else:
        max_memory_mb = 0.0
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min(times),
        'max_time_ms': max(times),
        'max_memory_mb': max_memory_mb
    }


def count_kernel_launches(model: MMRecModel, x: torch.Tensor) -> int:
    """
    Kernel launch sayısını tahmin et (gerçek ölçüm zor, bu yüzden tahmin).
    HEM durumuna göre farklı sayıda matmul operasyonu var.
    """
    # Bu basit bir tahmin - gerçek kernel launch sayısı CUDA profiler ile ölçülmeli
    num_layers = model.num_layers
    
    # HEM aktif: 1 matmul per block
    # HEM pasif: 6 matmul per block (Q, K, V, Z, P, E)
    # Ayrıca diğer operasyonlar (norm, attention, ffn, vb.)
    
    # Bu sadece projeksiyon matmul'larını sayıyor
    # Gerçek kernel launch sayısı daha fazla (norm, activation, vb.)
    return num_layers  # HEM için tahmin


def benchmark_hem_comparison(
    model_dim: int = 1024,
    num_layers: int = 2,
    seq_len: int = 2048,
    batch_size: int = 4,
    iterations: int = 50
) -> Dict[str, any]:
    """HEM aktif vs pasif karşılaştırması."""
    device = get_device()
    
    print(f"\n{'='*80}")
    print(f"HEM Performance Benchmark")
    print(f"{'='*80}")
    print(f"Model Config: dim={model_dim}, layers={num_layers}, seq_len={seq_len}, batch={batch_size}")
    
    # Create input
    vocab_size = 1000
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    results = {}
    
    # Test HEM Pasif (Orijinal)
    print(f"\n📊 Testing HEM Pasif (Orijinal - 6 ayrı matmul)...")
    try:
        model_no_hem = MMRecModel(
            vocab_size=vocab_size,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=8,
            max_seq_len=seq_len,
            use_hem=False  # HEM pasif
        ).to(device)
        model_no_hem.eval()
        
        # Create memory states
        from mm_rec.core.memory_state import MemoryState
        memory_states = [model_no_hem.create_memory_state(batch_size, device) for _ in range(num_layers)]
        
        # Benchmark single block
        x = model_no_hem.embedding(input_ids)
        block = model_no_hem.blocks[0]
        state = memory_states[0]
        
        block_result = benchmark_single_block(block, x, state, iterations=iterations, device=device)
        
        # Full model benchmark
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        use_cuda_events = device.type == 'cuda'
        forward_times = []
        
        for i in range(iterations):
            if use_cuda_events:
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()
            
            _ = model_no_hem(input_ids, memory_states=memory_states)
            
            if use_cuda_events:
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event)
            else:
                elapsed = (time.time() - start_time) * 1000
            
            if i >= 10:  # Skip warmup
                forward_times.append(elapsed)
        
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        
        if device.type == 'cuda':
            max_memory_bytes = torch.cuda.max_memory_allocated()
            max_memory_mb = max_memory_bytes / (1024 ** 2)
        else:
            max_memory_mb = 0.0
        
        results['no_hem'] = {
            'block_latency_ms': block_result['avg_time_ms'],
            'block_std_ms': block_result['std_time_ms'],
            'model_forward_ms': avg_forward,
            'max_memory_mb': max_memory_mb,
            'tokens_per_second': (seq_len * batch_size) / (avg_forward / 1000) if avg_forward > 0 else 0
        }
        
        print(f"   ✅ Block Latency: {block_result['avg_time_ms']:.2f} ± {block_result['std_time_ms']:.2f} ms")
        print(f"   ✅ Model Forward: {avg_forward:.2f} ms")
        print(f"   ✅ Throughput: {results['no_hem']['tokens_per_second']:.2f} tokens/s")
        print(f"   ✅ Memory: {max_memory_mb:.2f} MB")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['no_hem'] = None
    
    # Test HEM Aktif
    print(f"\n📊 Testing HEM Aktif (Fused Kernel - 1 matmul)...")
    try:
        model_hem = MMRecModel(
            vocab_size=vocab_size,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=8,
            max_seq_len=seq_len,
            use_hem=True  # HEM aktif
        ).to(device)
        model_hem.eval()
        
        # Create memory states
        memory_states = [model_hem.create_memory_state(batch_size, device) for _ in range(num_layers)]
        
        # Benchmark single block
        x = model_hem.embedding(input_ids)
        block = model_hem.blocks[0]
        state = memory_states[0]
        
        block_result = benchmark_single_block(block, x, state, iterations=iterations, device=device)
        
        # Full model benchmark
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        forward_times = []
        
        for i in range(iterations):
            if use_cuda_events:
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = time.time()
            
            _ = model_hem(input_ids, memory_states=memory_states)
            
            if use_cuda_events:
                end_event.record()
                torch.cuda.synchronize()
                elapsed = start_event.elapsed_time(end_event)
            else:
                elapsed = (time.time() - start_time) * 1000
            
            if i >= 10:  # Skip warmup
                forward_times.append(elapsed)
        
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        
        if device.type == 'cuda':
            max_memory_bytes = torch.cuda.max_memory_allocated()
            max_memory_mb = max_memory_bytes / (1024 ** 2)
        else:
            max_memory_mb = 0.0
        
        results['hem'] = {
            'block_latency_ms': block_result['avg_time_ms'],
            'block_std_ms': block_result['std_time_ms'],
            'model_forward_ms': avg_forward,
            'max_memory_mb': max_memory_mb,
            'tokens_per_second': (seq_len * batch_size) / (avg_forward / 1000) if avg_forward > 0 else 0
        }
        
        print(f"   ✅ Block Latency: {block_result['avg_time_ms']:.2f} ± {block_result['std_time_ms']:.2f} ms")
        print(f"   ✅ Model Forward: {avg_forward:.2f} ms")
        print(f"   ✅ Throughput: {results['hem']['tokens_per_second']:.2f} tokens/s")
        print(f"   ✅ Memory: {max_memory_mb:.2f} MB")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results['hem'] = None
    
    # Comparison
    if results.get('no_hem') and results.get('hem'):
        print(f"\n{'='*80}")
        print(f"📈 HEM Performance Comparison")
        print(f"{'='*80}")
        
        no_hem = results['no_hem']
        hem = results['hem']
        
        # Block latency improvement
        block_improvement = ((no_hem['block_latency_ms'] - hem['block_latency_ms']) / no_hem['block_latency_ms']) * 100
        print(f"\nBlock Latency:")
        print(f"   Orijinal: {no_hem['block_latency_ms']:.2f} ms")
        print(f"   HEM:      {hem['block_latency_ms']:.2f} ms")
        print(f"   İyileştirme: {block_improvement:.2f}% azalma")
        
        # Model forward improvement
        model_improvement = ((no_hem['model_forward_ms'] - hem['model_forward_ms']) / no_hem['model_forward_ms']) * 100
        print(f"\nModel Forward:")
        print(f"   Orijinal: {no_hem['model_forward_ms']:.2f} ms")
        print(f"   HEM:      {hem['model_forward_ms']:.2f} ms")
        print(f"   İyileştirme: {model_improvement:.2f}% azalma")
        
        # Throughput improvement
        throughput_improvement = ((hem['tokens_per_second'] - no_hem['tokens_per_second']) / no_hem['tokens_per_second']) * 100
        print(f"\nThroughput:")
        print(f"   Orijinal: {no_hem['tokens_per_second']:.2f} tokens/s")
        print(f"   HEM:      {hem['tokens_per_second']:.2f} tokens/s")
        print(f"   İyileştirme: {throughput_improvement:.2f}% artış")
        
        # Memory comparison
        memory_diff = hem['max_memory_mb'] - no_hem['max_memory_mb']
        memory_diff_pct = (memory_diff / no_hem['max_memory_mb']) * 100 if no_hem['max_memory_mb'] > 0 else 0
        print(f"\nMemory:")
        print(f"   Orijinal: {no_hem['max_memory_mb']:.2f} MB")
        print(f"   HEM:      {hem['max_memory_mb']:.2f} MB")
        print(f"   Fark:     {memory_diff:+.2f} MB ({memory_diff_pct:+.2f}%)")
        
        results['comparison'] = {
            'block_latency_improvement_pct': block_improvement,
            'model_forward_improvement_pct': model_improvement,
            'throughput_improvement_pct': throughput_improvement,
            'memory_diff_mb': memory_diff,
            'memory_diff_pct': memory_diff_pct
        }
    
    return results


def main():
    """Main benchmark execution."""
    print("="*80)
    print("HEM (Mekanizma 1) Performance Benchmark")
    print("Gerçek performans ölçümleri: HEM aktif vs pasif")
    print("="*80)
    
    # Test configurations
    configs = [
        {'model_dim': 512, 'num_layers': 2, 'seq_len': 1024, 'batch_size': 2},
        {'model_dim': 1024, 'num_layers': 2, 'seq_len': 2048, 'batch_size': 4},
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n{'#'*80}")
        print(f"Testing Configuration: {config}")
        print(f"{'#'*80}")
        
        try:
            results = benchmark_hem_comparison(**config, iterations=30)
            results['config'] = config
            all_results.append(results)
        except Exception as e:
            print(f"❌ Error in configuration {config}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 Summary")
    print(f"{'='*80}")
    
    for result in all_results:
        if result.get('comparison'):
            comp = result['comparison']
            config = result['config']
            print(f"\nConfig: {config}")
            print(f"   Block Latency İyileştirme: {comp['block_latency_improvement_pct']:.2f}%")
            print(f"   Model Forward İyileştirme: {comp['model_forward_improvement_pct']:.2f}%")
            print(f"   Throughput İyileştirme: {comp['throughput_improvement_pct']:.2f}%")
            print(f"   Memory Fark: {comp['memory_diff_pct']:+.2f}%")


if __name__ == "__main__":
    main()


