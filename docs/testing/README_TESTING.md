# Test Etme Rehberi

## Hızlı Başlangıç

### 1. PyTorch Kurulumu

```bash
# CPU versiyonu (her zaman çalışır)
pip install torch

# Veya CUDA ile (GPU varsa)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. Test Çalıştırma

```bash
# Basit test (CPU fallback kullanır)
python quick_test.py

# Tam test suite
python test_associative_scan.py
```

## Test Senaryoları

### ✅ CUDA Varsa
- Otomatik olarak GPU/Triton kullanır
- Daha hızlı performans
- Paralel işleme

### ✅ CUDA Yoksa
- Otomatik olarak CPU fallback kullanır
- Sequential implementasyon
- Yine de doğru sonuçlar verir

## Beklenen Çıktı

```
============================================================
MM-Rec Associative Scan Exponential - Test Suite
============================================================

🔧 System Info:
  CUDA available: False
  PyTorch version: 2.x.x
  Triton available: False

⚠ Note: Using CPU fallback implementation

============================================================
Test 1: Forward Pass Correctness
============================================================
⚠ Using CPU fallback (CUDA not available or use_cpu_fallback=True)

📊 Test Results:
  Max difference: 0.000123
  Mean difference: 0.000045
  Relative difference: 0.000012
✓ Test PASSED! (max_diff 0.000123 < tolerance 0.001)

============================================================
Test 2: Gradient Computation
============================================================
⚠ Testing gradients with CPU fallback

📊 Gradient Test Results:
  Max gradient difference: 0.001234
  Mean gradient difference: 0.000456
✓ Gradient test PASSED!

============================================================
📋 Summary
============================================================
  Forward test: ✓ PASSED
  Gradient test: ✓ PASSED

🎉 All tests passed!
```

## Sorun Giderme

### "No module named 'torch'"
```bash
pip install torch
```

### "No module named 'triton'"
- Sorun değil! CPU fallback otomatik kullanılır
- GPU için: `pip install triton`

### Test Başarısız Olursa
- Tolerance değerlerini kontrol edin
- Numerical precision farkları normal olabilir
- CPU ve GPU sonuçları arasında küçük farklar olabilir

## Manuel Test

```python
from mm_rec.core import associative_scan_exponential_cpu_fallback
import torch

# Test input
gamma = torch.rand(2, 8, 128, 64, dtype=torch.float32)

# Compute
result = associative_scan_exponential_cpu_fallback(gamma)

# Compare with reference
ref = torch.cumprod(gamma, dim=2)
diff = torch.abs(result - ref).max()
print(f"Max difference: {diff.item():.6e}")
```

