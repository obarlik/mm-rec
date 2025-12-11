# MM-Rec Kurulum Rehberi

## Hızlı Kurulum

### 1. PyTorch Kurulumu

```bash
# CPU versiyonu (her zaman çalışır)
pip install torch

# Veya CUDA 11.8 ile (GPU varsa)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Veya CUDA 12.1 ile
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Triton Kurulumu (Opsiyonel - GPU için)

```bash
# GPU hızlandırma için
pip install triton

# CPU fallback kullanılacaksa gerekli değil
```

### 3. Test

```bash
# Hızlı test
python quick_test.py

# Tam test suite
python test_associative_scan.py
```

## Kurulum Sonrası Kontrol

```python
import torch
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# MM-Rec import test
from mm_rec.core import associative_scan_exponential_cpu_fallback
print("✓ MM-Rec imported successfully")
```

## Sistem Gereksinimleri

- **Python**: 3.8+
- **PyTorch**: 2.0+ (CPU veya CUDA)
- **Triton**: 2.0+ (opsiyonel, GPU için)
- **CUDA**: 11.8+ (opsiyonel, GPU için)

## Sorun Giderme

### PyTorch Kurulum Hatası
```bash
# pip güncelleyin
pip install --upgrade pip

# Tekrar deneyin
pip install torch
```

### Import Hatası
```bash
# Modül yolunu kontrol edin
python -c "import sys; print(sys.path)"

# Gerekirse PYTHONPATH ekleyin
export PYTHONPATH="${PYTHONPATH}:/home/onur/workspace/mm-rec"
```

