"""
Memory Profiler for MM-Rec Model
Detects O(N²) memory growth and tracks memory usage across sequence lengths
"""

import torch
import torch.cuda
from typing import Dict, List, Optional, Tuple
import warnings


class MemoryProfiler:
    """
    Memory profiler to detect O(N²) memory growth and identify memory bottlenecks.
    
    Usage:
        profiler = MemoryProfiler()
        with profiler.track("attention_scores"):
            scores = compute_attention_scores(...)
        profiler.report()
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_snapshots: Dict[str, List[Tuple[int, float]]] = {}
        self.current_operation: Optional[str] = None
        self.sequence_lengths: List[int] = []
        
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        else:
            return 0.0  # CPU memory tracking not implemented
    
    def track(self, operation_name: str):
        """Context manager to track memory usage for an operation."""
        return MemoryTracker(self, operation_name)
    
    def snapshot(self, operation_name: str, seq_len: int):
        """Take a memory snapshot for a specific operation and sequence length."""
        if operation_name not in self.memory_snapshots:
            self.memory_snapshots[operation_name] = []
        
        memory_mb = self._get_memory_mb()
        self.memory_snapshots[operation_name].append((seq_len, memory_mb))
        
        if seq_len not in self.sequence_lengths:
            self.sequence_lengths.append(seq_len)
    
    def analyze_complexity(self, operation_name: str) -> str:
        """
        Analyze memory complexity for an operation.
        
        Returns:
            "O(1)", "O(N)", "O(N²)", or "UNKNOWN"
        """
        if operation_name not in self.memory_snapshots:
            return "UNKNOWN"
        
        snapshots = self.memory_snapshots[operation_name]
        if len(snapshots) < 2:
            return "UNKNOWN"
        
        # Sort by sequence length
        snapshots_sorted = sorted(snapshots, key=lambda x: x[0])
        
        # Check growth pattern
        seq_lens = [s[0] for s in snapshots_sorted]
        memory_mbs = [s[1] for s in snapshots_sorted]
        
        # Calculate growth ratios
        if len(snapshots_sorted) >= 2:
            # Check if memory grows quadratically
            for i in range(1, len(snapshots_sorted)):
                seq_ratio = seq_lens[i] / seq_lens[i-1]
                mem_ratio = memory_mbs[i] / memory_mbs[i-1] if memory_mbs[i-1] > 0 else 0
                
                # O(N²) detection: memory grows ~seq_ratio²
                if seq_ratio > 1.5 and mem_ratio > seq_ratio * 1.5:
                    return "O(N²)"
                # O(N) detection: memory grows ~seq_ratio
                elif seq_ratio > 1.5 and 0.8 * seq_ratio < mem_ratio < 1.5 * seq_ratio:
                    return "O(N)"
        
        return "UNKNOWN"
    
    def report(self, verbose: bool = True) -> Dict[str, str]:
        """
        Generate memory complexity report.
        
        Returns:
            Dictionary mapping operation names to complexity strings
        """
        report = {}
        
        for operation_name in self.memory_snapshots:
            complexity = self.analyze_complexity(operation_name)
            report[operation_name] = complexity
            
            if verbose:
                snapshots = self.memory_snapshots[operation_name]
                if complexity == "O(N²)":
                    warnings.warn(
                        f"⚠️ CRITICAL: {operation_name} shows O(N²) memory growth!\n"
                        f"   This will cause OOM for long sequences (100K+).\n"
                        f"   Memory snapshots: {snapshots}",
                        RuntimeWarning,
                        stacklevel=2
                    )
                elif complexity == "O(N)":
                    if verbose:
                        print(f"✓ {operation_name}: O(N) memory growth (acceptable)")
                else:
                    if verbose:
                        print(f"? {operation_name}: {complexity} complexity")
        
        return report
    
    def clear(self):
        """Clear all memory snapshots."""
        self.memory_snapshots.clear()
        self.sequence_lengths.clear()


class MemoryTracker:
    """Context manager for tracking memory usage."""
    
    def __init__(self, profiler: MemoryProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        self.start_memory: float = 0.0
    
    def __enter__(self):
        if self.profiler.device.type == 'cuda':
            torch.cuda.synchronize()
        self.start_memory = self.profiler._get_memory_mb()
        self.profiler.current_operation = self.operation_name
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler.device.type == 'cuda':
            torch.cuda.synchronize()
        end_memory = self.profiler._get_memory_mb()
        memory_delta = end_memory - self.start_memory
        
        # Store snapshot (we need seq_len from context, so this is approximate)
        # In practice, call profiler.snapshot() explicitly with seq_len
        if self.profiler.current_operation:
            # Try to infer seq_len from current operation context
            # This is a placeholder - actual usage should call snapshot() explicitly
            pass
        
        self.profiler.current_operation = None


def profile_memory_growth(
    model: torch.nn.Module,
    sequence_lengths: List[int],
    batch_size: int = 1,
    vocab_size: int = 10000,
    device: Optional[torch.device] = None
) -> Dict[str, str]:
    """
    Profile memory growth across different sequence lengths.
    
    Args:
        model: MM-Rec model to profile
        sequence_lengths: List of sequence lengths to test (e.g., [16K, 32K, 64K])
        batch_size: Batch size for testing
        vocab_size: Vocabulary size
        device: Device to run on
    
    Returns:
        Dictionary mapping operation names to complexity strings
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    profiler = MemoryProfiler(device=device)
    
    print(f"\n🔬 Memory Profiling: Testing {len(sequence_lengths)} sequence lengths")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}\n")
    
    for seq_len in sequence_lengths:
        print(f"  Testing seq_len={seq_len}...")
        
        # Clear cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Generate input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Take snapshot before forward pass
        profiler.snapshot("before_forward", seq_len)
        
        try:
            # Forward pass
            with profiler.track("forward_pass"):
                logits = model(input_ids)
            
            # Take snapshot after forward pass
            profiler.snapshot("after_forward", seq_len)
            
            # Get peak memory
            if device.type == 'cuda':
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"    Peak memory: {peak_memory_mb:.2f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    ❌ OOM at seq_len={seq_len}")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                break
            else:
                raise
    
    # Generate report
    print("\n📊 Memory Complexity Analysis:")
    report = profiler.report(verbose=True)
    
    return report

