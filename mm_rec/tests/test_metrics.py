"""
Test Metrics Collection Infrastructure
Centralized metrics collection for all tests without polluting test code.
"""

import time
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import threading


@dataclass
class TestMetrics:
    """Metrics for a single test execution."""
    test_name: str
    test_file: str
    duration: float
    memory_peak_mb: Optional[float] = None
    memory_allocated_mb: Optional[float] = None
    gpu_memory_peak_mb: Optional[float] = None
    gpu_memory_allocated_mb: Optional[float] = None
    num_parameters: Optional[int] = None
    sequence_length: Optional[int] = None
    batch_size: Optional[int] = None
    chunk_size: Optional[int] = None
    throughput_tokens_per_sec: Optional[float] = None
    status: str = "passed"  # passed, failed, skipped, timeout
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Thread-safe metrics collector for tests."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("test_metrics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: List[TestMetrics] = []
        self.lock = threading.Lock()
        self.current_test: Optional[str] = None
        self.test_start_time: Optional[float] = None
        
    def start_test(self, test_name: str, test_file: str):
        """Mark the start of a test."""
        with self.lock:
            self.current_test = test_name
            self.test_start_time = time.time()
    
    def record_metric(self, key: str, value: Any, test_name: Optional[str] = None):
        """Record a metric for the current test."""
        with self.lock:
            if test_name is None:
                test_name = self.current_test
            if test_name is None:
                return  # No active test
            
            # Find or create metrics entry
            metric = None
            for m in self.metrics:
                if m.test_name == test_name:
                    metric = m
                    break
            
            if metric is None:
                # Create new metric entry
                metric = TestMetrics(
                    test_name=test_name,
                    test_file="",
                    duration=0.0
                )
                self.metrics.append(metric)
            
            # Set the metric value
            if hasattr(metric, key):
                setattr(metric, key, value)
            else:
                metric.metadata[key] = value
    
    def end_test(self, test_name: str, status: str = "passed", error_message: Optional[str] = None):
        """Mark the end of a test and finalize metrics."""
        with self.lock:
            if test_name != self.current_test:
                return  # Test name mismatch
            
            # Find or create metrics entry
            metric = None
            for m in self.metrics:
                if m.test_name == test_name:
                    metric = m
                    break
            
            if metric is None:
                metric = TestMetrics(
                    test_name=test_name,
                    test_file="",
                    duration=0.0
                )
                self.metrics.append(metric)
            
            # Update final metrics
            if self.test_start_time:
                metric.duration = time.time() - self.test_start_time
            metric.status = status
            metric.error_message = error_message
            metric.timestamp = time.time()
            
            # Reset current test
            self.current_test = None
            self.test_start_time = None
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        try:
            import torch
            stats = {}
            
            # CPU memory
            if torch.cuda.is_available():
                stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
                stats['gpu_memory_peak_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
            
            # System memory (requires psutil)
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                stats['memory_peak_mb'] = mem_info.rss / (1024**2)
            except ImportError:
                pass
            
            return stats
        except Exception:
            return {}
    
    def record_memory(self, test_name: Optional[str] = None):
        """Record current memory statistics."""
        stats = self.get_memory_stats()
        for key, value in stats.items():
            self.record_metric(key, value, test_name)
    
    def save_metrics(self, filename: Optional[str] = None):
        """Save all collected metrics to JSON file."""
        with self.lock:
            if filename is None:
                timestamp = int(time.time())
                filename = f"test_metrics_{timestamp}.json"
            
            filepath = self.output_dir / filename
            
            # Convert metrics to dict
            metrics_dict = {
                'timestamp': time.time(),
                'total_tests': len(self.metrics),
                'metrics': [asdict(m) for m in self.metrics]
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            return filepath
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all metrics."""
        with self.lock:
            if not self.metrics:
                return {}
            
            durations = [m.duration for m in self.metrics if m.duration > 0]
            statuses = defaultdict(int)
            for m in self.metrics:
                statuses[m.status] += 1
            
            summary = {
                'total_tests': len(self.metrics),
                'status_counts': dict(statuses),
                'duration_stats': {
                    'total': sum(durations),
                    'mean': sum(durations) / len(durations) if durations else 0,
                    'min': min(durations) if durations else 0,
                    'max': max(durations) if durations else 0,
                }
            }
            
            # GPU memory stats
            gpu_memories = [m.gpu_memory_peak_mb for m in self.metrics if m.gpu_memory_peak_mb]
            if gpu_memories:
                summary['gpu_memory_stats'] = {
                    'mean_mb': sum(gpu_memories) / len(gpu_memories),
                    'max_mb': max(gpu_memories),
                    'min_mb': min(gpu_memories),
                }
            
            return summary


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        output_dir = os.getenv('TEST_METRICS_DIR', 'test_metrics')
        _metrics_collector = MetricsCollector(output_dir=output_dir)
    return _metrics_collector


def record_test_metric(key: str, value: Any):
    """Convenience function to record a metric."""
    collector = get_metrics_collector()
    collector.record_metric(key, value)


def record_memory_stats():
    """Convenience function to record memory statistics."""
    collector = get_metrics_collector()
    collector.record_memory()


def save_test_metrics(filename: Optional[str] = None):
    """Convenience function to save metrics."""
    collector = get_metrics_collector()
    return collector.save_metrics(filename)

