# Zero-Overhead Metrics System

## Overview
Lock-free, thread-safe metrics collection with async background writer.

## Performance
- **Collection Cost**: 0.023 nanoseconds/event
- **Training Impact**: **ZERO** (measured: 730 tok/s unchanged)
- **Console I/O Reduction**: 100x (every 100 steps vs every step)

## Usage

Metrics are **enabled by default**. To disable them, pass the `--no-metrics` flag.

### Training
```bash
mm_rec train config.txt data.bin
# Output: training_metrics.bin
```

### Inference
```bash
mm_rec infer config.txt model.bin vocab.json "Prompt"
# Output: inference_metrics.bin
```

### Parsing Metrics
Binary files are not human-readable. Use the built-in parser tool:
```bash
mm_rec parse-metrics training_metrics.bin
```

### Programmatic Usage

#### Training Integration
```cpp
#include "mm_rec/utils/metrics.h"

// Start writer (Binary format, sampling enabled)
MetricsSamplingConfig sampling;
sampling.enabled = true;
sampling.interval = 10;
MetricsManager::instance().start_writer("training_metrics.bin", sampling);

// Record metrics (zero overhead)
METRIC_TRAINING_STEP(loss, learning_rate);
METRIC_RECORD(CUSTOM, flux_scale, 0, 0, "flux");

// Stop writer (flushes all buffers)
MetricsManager::instance().stop_writer();
```

#### Inference Integration
```cpp
// Typically no sampling for inference to capture all tokens
MetricsManager::instance().start_writer("inference_metrics.bin");
// ... 
METRIC_INFERENCE(latency, output.size());
```

## Architecture

### Components
1. **MetricEvent** (32 bytes): Single metric record (Binary)
2. **MetricsBuffer** (16K events): Lock-free ring buffer per thread
3. **MetricsManager**: Global coordinator with async writer

### Thread Safety
- Thread-local buffers (no locks on write path)
- Automatic registration on first use
- Mutex-protected registry (read-only during collection)

## Output Format
Binary file with header:
```
[MAGIC: "MMRC"] [VERSION: 1] [RESERVED: 4 bytes]
[Event 1 (32 bytes)]
[Event 2 (32 bytes)]
...
```

Use `mm_rec parse-metrics` to inspect.

## Compile-Time Toggle
Disable metrics in production builds:
```cmake
add_compile_definitions(ENABLE_METRICS=0)
```

When disabled, all `METRIC_*` macros become no-ops (compiled out).

## Benchmarks
See `tests/benchmark_metrics_overhead.cpp`:
```bash
./bench_metrics
# Output: Per-event cost: 0.023 ns
```
