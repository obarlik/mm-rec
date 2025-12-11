# MM-Rec Associative Scan - Testing Guide

## Hızlı Test

### 1. Gereksinimleri Yükleyin

```bash
# PyTorch yükleyin (CPU veya CUDA)
pip install torch

# GPU için Triton (opsiyonel, CPU fallback kullanılabilir)
pip install triton
```

### 2. Testi Çalıştırın

```bash
# Basit test
python test_associative_scan.py

# Veya modül olarak
python -m mm_rec.core.associative_scan_triton
```

## Test Senaryoları

### Senaryo 1: CUDA ile (GPU)
```bash
# CUDA varsa otomatik olarak GPU kullanır
python test_associative_scan.py
```

**Beklenen Çıktı:**
```
✓ Using CUDA/Triton implementation
✓ Test PASSED!
```

### Senaryo 2: CPU Fallback (CUDA yok)
```bash
# CUDA yoksa otomatik olarak CPU fallback kullanır
python test_associative_scan.py
```

**Beklenen Çıktı:**
```
⚠ Using CPU fallback (CUDA not available)
✓ Test PASSED!
```

### Senaryo 3: Manuel CPU Fallback
```python
from mm_rec.core import associative_scan_exponential_cpu_fallback
import torch

gamma = torch.rand(2, 8, 128, 64, dtype=torch.float32)
result = associative_scan_exponential_cpu_fallback(gamma)
```

## Test Detayları

### Forward Pass Testi
- **Girdi**: Random gamma değerleri [0.05, 0.95] aralığında
- **Referans**: `torch.cumprod()` ile sequential implementasyon
- **Tolerans**: Max difference < 1e-3

### Gradient Testi
- **Yöntem**: Finite difference ile gradient doğrulama
- **Tolerans**: Max gradient difference < 1e-2

## Sorun Giderme

### PyTorch Bulunamıyor
```bash
pip install torch
```

### Triton Bulunamıyor
```bash
# GPU için
pip install triton

# CPU fallback kullanılacaksa gerekli değil
```

### CUDA Hatası
- CPU fallback otomatik olarak devreye girer
- Test yine de çalışmalıdır

## Performans Notları

- **CPU Fallback**: Sequential implementasyon, O(N) complexity
- **Triton (GPU)**: Parallel implementasyon, O(log N) depth
- **Uzun sekanslar**: GPU'da çok daha hızlı (32K+ tokens)

## Örnek Kullanım

```python
from mm_rec.core import associative_scan_exponential
import torch

# Input: [BATCH, HEADS, SEQ_LEN, D_HEAD]
gamma = torch.rand(2, 8, 1024, 128, dtype=torch.float32)

# Compute cumulative product
result = associative_scan_exponential(gamma)

# Gradient computation
result.sum().backward()
```

