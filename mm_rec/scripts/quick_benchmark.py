"""
Hızlı CPU Benchmark - Küçük model ile gerçek ölçümler
"""

import torch
import torch.nn as nn
import time
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from mm_rec.model import MMRecModel
    from mm_rec.blocks.mm_rec_block import MMRecBlock
    from mm_rec.core.memory_state import MemoryState
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def quick_benchmark():
    """Hızlı benchmark - küçük model."""
    print("="*80)
    print("MM-Rec CPU Benchmark (Gerçek Ölçümler)")
    print("="*80)
    
    device = torch.device('cpu')
    print(f"\n🔧 Device: CPU")
    print(f"   PyTorch version: {torch.__version__}")
    
    # Küçük config
    config = {
        'vocab_size': 1000,
        'model_dim': 256,
        'num_layers': 2,
        'num_heads': 4,
        'seq_len': 128,
        'batch_size': 1
    }
    
    print(f"\n📊 Configuration:")
    print(f"   model_dim: {config['model_dim']}")
    print(f"   num_layers: {config['num_layers']}")
    print(f"   seq_len: {config['seq_len']}")
    print(f"   batch_size: {config['batch_size']}")
    
    try:
        # Create model
        print(f"\n   Creating model...")
        model = MMRecModel(
            vocab_size=config['vocab_size'],
            model_dim=config['model_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            max_seq_len=config['seq_len']
        ).to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total Parameters: {total_params:,}")
        
        # Create input
        input_ids = torch.randint(0, config['vocab_size'], (config['batch_size'], config['seq_len']), device=device)
        
        # Warmup
        print(f"\n   Warmup...")
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids)
        
        # Benchmark single block
        print(f"\n   Benchmarking single block...")
        x = model.embedding(input_ids)
        block = model.blocks[0]
        batch_size, seq_len, _ = x.shape
        state = model.create_memory_state(batch_size, device)
        
        block_times = []
        for i in range(20):
            start = time.time()
            with torch.no_grad():
                _ = block(x, state)
            elapsed = (time.time() - start) * 1000  # ms
            if i >= 5:  # Skip first 5
                block_times.append(elapsed)
        
        avg_block = sum(block_times) / len(block_times)
        print(f"      Block Latency: {avg_block:.2f} ms (avg of {len(block_times)} runs)")
        
        # Benchmark full model
        print(f"\n   Benchmarking full model...")
        model_times = []
        for i in range(15):
            start = time.time()
            with torch.no_grad():
                _ = model(input_ids)
            elapsed = (time.time() - start) * 1000  # ms
            if i >= 3:  # Skip first 3
                model_times.append(elapsed)
        
        avg_model = sum(model_times) / len(model_times)
        tokens_per_sec = (config['seq_len'] * config['batch_size']) / (avg_model / 1000)
        per_layer = avg_model / config['num_layers']
        
        print(f"      Model Forward: {avg_model:.2f} ms (avg of {len(model_times)} runs)")
        print(f"      Per-Layer: {per_layer:.2f} ms")
        print(f"      Throughput: {tokens_per_sec:.2f} tokens/s")
        
        # Results
        results = {
            'device': 'CPU',
            'pytorch_version': torch.__version__,
            'config': config,
            'total_params': total_params,
            'block_latency_ms': avg_block,
            'model_forward_ms': avg_model,
            'per_layer_latency_ms': per_layer,
            'tokens_per_second': tokens_per_sec
        }
        
        print(f"\n{'='*80}")
        print(f"✅ Benchmark Completed")
        print(f"{'='*80}")
        print(f"Block Latency: {avg_block:.2f} ms")
        print(f"Model Forward: {avg_model:.2f} ms")
        print(f"Per-Layer: {per_layer:.2f} ms")
        print(f"Throughput: {tokens_per_sec:.2f} tokens/s")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = quick_benchmark()
    if results:
        import json
        with open('quick_benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: quick_benchmark_results.json")


