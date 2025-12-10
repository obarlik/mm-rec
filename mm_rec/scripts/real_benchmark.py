"""
Gerçek Performans Ölçümleri - Mevcut Kod ile
HEM ve UBÖO mekanizmaları henüz implement edilmemiş olsa bile,
mevcut kodun performansını ölçerek baseline oluşturuyoruz.
"""

import torch
import torch.nn as nn
import time
import sys
import os
from typing import Dict, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from mm_rec.model import MMRecModel
    from mm_rec.blocks.mm_rec_block import MMRecBlock
    from mm_rec.core.memory_state import MemoryState
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Lütfen gerekli bağımlılıkları yükleyin: pip install -r requirements.txt")
    sys.exit(1)


def get_device_info() -> Dict:
    """Get device information."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device': None,
        'device_name': None,
        'memory_gb': None,
        'compute_capability': None
    }
    
    if torch.cuda.is_available():
        info['device'] = torch.device('cuda')
        info['device_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info['memory_gb'] = props.total_memory / 1e9
        info['compute_capability'] = f"{props.major}.{props.minor}"
    else:
        info['device'] = torch.device('cpu')
        info['device_name'] = "CPU"
    
    return info


def benchmark_block_forward(
    block: MMRecBlock,
    x: torch.Tensor,
    state: MemoryState,
    iterations: int = 50,
    warmup: int = 10,
    device: torch.device = None
) -> Dict[str, float]:
    """Benchmark single block forward pass."""
    if device is None:
        device = next(block.parameters()).device
    
    # Clear cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
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
        
        with torch.no_grad():
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
        'max_memory_mb': max_memory_mb,
        'times': times  # For detailed analysis
    }


def benchmark_model_forward(
    model: MMRecModel,
    input_ids: torch.Tensor,
    iterations: int = 30,
    warmup: int = 5,
    device: torch.device = None
) -> Dict[str, float]:
    """Benchmark full model forward pass."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Clear cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Create memory states
    batch_size, seq_len = input_ids.shape
    memory_states = [model.create_memory_state(batch_size, device) for _ in range(model.num_layers)]
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids, memory_states=memory_states)
    
    # Timing
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
        
        with torch.no_grad():
            _ = model(input_ids, memory_states=memory_states)
        
        if use_cuda_events:
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
        else:
            elapsed = (time.time() - start_time) * 1000
        
        forward_times.append(elapsed)
    
    # Get memory
    if device.type == 'cuda':
        max_memory_bytes = torch.cuda.max_memory_allocated()
        max_memory_mb = max_memory_bytes / (1024 ** 2)
    else:
        max_memory_mb = 0.0
    
    avg_forward = sum(forward_times) / len(forward_times)
    std_forward = (sum((t - avg_forward) ** 2 for t in forward_times) / len(forward_times)) ** 0.5
    
    # Calculate throughput
    batch_size, seq_len = input_ids.shape
    tokens_per_second = (seq_len * batch_size) / (avg_forward / 1000) if avg_forward > 0 else 0
    
    return {
        'forward_time_ms': avg_forward,
        'std_time_ms': std_forward,
        'min_time_ms': min(forward_times),
        'max_time_ms': max(forward_times),
        'tokens_per_second': tokens_per_second,
        'max_memory_mb': max_memory_mb,
        'times': forward_times
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'total_mb': total * 2 / (1024 ** 2),  # Assuming BF16 (2 bytes)
        'trainable_mb': trainable * 2 / (1024 ** 2)
    }


def main():
    """Main benchmark execution."""
    print("="*80)
    print("MM-Rec Gerçek Performans Ölçümleri")
    print("="*80)
    
    # Device info
    device_info = get_device_info()
    print(f"\n🔧 Device Information:")
    print(f"   CUDA Available: {device_info['cuda_available']}")
    print(f"   Device: {device_info['device_name']}")
    if device_info['cuda_available']:
        print(f"   Memory: {device_info['memory_gb']:.2f} GB")
        print(f"   Compute Capability: {device_info['compute_capability']}")
    
    device = device_info['device']
    
    # Test configurations
    # Küçük config'lerle başla (bellek sınırlamaları için)
    configs = [
        {
            'name': 'Small',
            'vocab_size': 1000,
            'model_dim': 512,
            'num_layers': 2,
            'num_heads': 8,
            'seq_len': 512,
            'batch_size': 2
        },
        {
            'name': 'Medium',
            'vocab_size': 1000,
            'model_dim': 1024,
            'num_layers': 2,
            'num_heads': 8,
            'seq_len': 1024,
            'batch_size': 2
        },
    ]
    
    all_results = []
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"📊 Testing Configuration: {config['name']}")
        print(f"{'='*80}")
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
            
            # Count parameters
            params = count_parameters(model)
            print(f"   Total Parameters: {params['total']:,} ({params['total_mb']:.2f} MB)")
            print(f"   Trainable Parameters: {params['trainable']:,} ({params['trainable_mb']:.2f} MB)")
            
            # Create input
            input_ids = torch.randint(
                0, config['vocab_size'],
                (config['batch_size'], config['seq_len']),
                device=device,
                dtype=torch.long
            )
            
            # Benchmark single block
            print(f"\n   Benchmarking single block...")
            x = model.embedding(input_ids)
            block = model.blocks[0]
            batch_size, seq_len, _ = x.shape
            state = model.create_memory_state(batch_size, device)
            
            block_result = benchmark_block_forward(
                block, x, state,
                iterations=30,
                warmup=5,
                device=device
            )
            
            print(f"      Block Latency: {block_result['avg_time_ms']:.2f} ± {block_result['std_time_ms']:.2f} ms")
            print(f"      Block Memory: {block_result['max_memory_mb']:.2f} MB")
            
            # Benchmark full model
            print(f"\n   Benchmarking full model...")
            model_result = benchmark_model_forward(
                model, input_ids,
                iterations=20,
                warmup=3,
                device=device
            )
            
            print(f"      Model Forward: {model_result['forward_time_ms']:.2f} ± {model_result['std_time_ms']:.2f} ms")
            print(f"      Throughput: {model_result['tokens_per_second']:.2f} tokens/s")
            print(f"      Model Memory: {model_result['max_memory_mb']:.2f} MB")
            
            # Per-layer latency estimate
            per_layer_latency = model_result['forward_time_ms'] / config['num_layers']
            print(f"      Estimated Per-Layer: {per_layer_latency:.2f} ms")
            
            result = {
                'config': config,
                'params': params,
                'block': block_result,
                'model': model_result,
                'per_layer_latency_ms': per_layer_latency
            }
            all_results.append(result)
            
            # Cleanup
            del model
            del input_ids
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   ❌ Out of memory for configuration {config['name']}")
                print(f"      Skipping...")
            else:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📈 Summary Results")
    print(f"{'='*80}")
    
    for result in all_results:
        config = result['config']
        print(f"\n{config['name']} Configuration:")
        print(f"   Block Latency: {result['block']['avg_time_ms']:.2f} ms")
        print(f"   Model Forward: {result['model']['forward_time_ms']:.2f} ms")
        print(f"   Per-Layer: {result['per_layer_latency_ms']:.2f} ms")
        print(f"   Throughput: {result['model']['tokens_per_second']:.2f} tokens/s")
        print(f"   Memory: {result['model']['max_memory_mb']:.2f} MB")
    
    # Save results to file
    import json
    results_file = os.path.join(project_root, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = []
        for result in all_results:
            json_result = {
                'config': result['config'],
                'params': result['params'],
                'block': {
                    'avg_time_ms': result['block']['avg_time_ms'],
                    'std_time_ms': result['block']['std_time_ms'],
                    'max_memory_mb': result['block']['max_memory_mb']
                },
                'model': {
                    'forward_time_ms': result['model']['forward_time_ms'],
                    'std_time_ms': result['model']['std_time_ms'],
                    'tokens_per_second': result['model']['tokens_per_second'],
                    'max_memory_mb': result['model']['max_memory_mb']
                },
                'per_layer_latency_ms': result['per_layer_latency_ms']
            }
            json_results.append(json_result)
        
        json.dump({
            'device_info': device_info,
            'results': json_results
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    print(f"\n⚠️  NOT: HEM ve UBÖO mekanizmaları henüz implement edilmemiş.")
    print(f"   Bu ölçümler mevcut (baseline) kodun performansını gösteriyor.")
    print(f"   HEM ve UBÖO implement edildikten sonra tekrar ölçüm yapılmalı.")


if __name__ == "__main__":
    main()


