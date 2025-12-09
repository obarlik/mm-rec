# MM-Rec Extension Gereksinimleri

## 🎯 KRİTİK KURAL: Extension'lar ZORUNLU

**CPU modunda C++ extension'lar olmadan fallback modda çalışmak YASAKTIR.**

---

## ✅ Extension Durumu

### Gerekli Extension'lar

1. **`mm_rec_scan_cpu`** (ZORUNLU - CPU)
   - Associative scan için optimize edilmiş C++ implementation
   - SIMD/AVX/OpenMP optimizasyonları
   - **Fallback YOK** - Extension yoksa RuntimeError

2. **`mm_rec_cpp_cpu`** (ZORUNLU - CPU)
   - MMRecBlock için optimize edilmiş C++ implementation
   - Sequential loop optimizasyonları
   - **Fallback YOK** - Extension yoksa RuntimeError

3. **`mm_rec_cpp_cuda`** (Opsiyonel - GPU)
   - CUDA kernel'leri (GPU varsa)

---

## 🔧 Extension Kontrolü

### Kontrol Komutu

```bash
python mm_rec/tests/check_extensions.py
```

**Beklenen Çıktı (CPU modunda):**
```
mm_rec_cpp_cpu:
  ✅ Yüklendi
  Path: ...
  SHA256: ...

mm_rec_scan_cpu:
  ✅ Yüklendi
  Path: ...
  SHA256: ...
```

**Eğer extension yüklenemezse:**
```
mm_rec_cpp_cpu:
  ❌ Yüklenemedi: libc10.so: cannot open shared object file
```

---

## 🚨 Hata Durumları ve Çözümler

### 1. libc10.so Hatası

**Hata:**
```
libc10.so: cannot open shared object file: No such file or directory
```

**Çözüm:**
```bash
# PyTorch library path'i bul
python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"

# LD_LIBRARY_PATH ayarla
export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

# Extension'ı yeniden derle
cd mm_rec/cpp && python setup.py build_ext --inplace
```

**Otomatik Çözüm:**
Extension'lar artık otomatik olarak PyTorch kütüphanelerini preload ediyor (`check_extensions.py` ve `associative_scan_triton.py`).

---

### 2. Extension Bulunamıyor

**Hata:**
```
No module named 'mm_rec_scan_cpu'
```

**Çözüm:**
```bash
cd mm_rec/cpp
python setup.py build_ext --inplace
```

**Kontrol:**
```bash
python mm_rec/tests/check_extensions.py
```

---

## 📋 Kod İçinde Extension Kontrolü

### 1. `associative_scan_exponential_cpu_fallback`

**Önceki (YANLIŞ):**
```python
try:
    import mm_rec_scan_cpu
    return mm_rec_scan_cpu.associative_scan_exponential_cpu(gamma)
except ImportError:
    # Fallback to Python - YASAK!
    pass
```

**Şimdi (DOĞRU):**
```python
try:
    import mm_rec_scan_cpu
    return mm_rec_scan_cpu.associative_scan_exponential_cpu(gamma)
except ImportError as e:
    raise RuntimeError(
        f"❌ CRITICAL: C++ extension 'mm_rec_scan_cpu' is REQUIRED!\n"
        f"   Error: {e}\n"
        f"   Solution: cd mm_rec/cpp && python setup.py build_ext --inplace"
    ) from e
```

---

### 2. `MMRecBlock` - CPU Mode

**Önceki (YANLIŞ):**
```python
if not torch.cuda.is_available():
    # CPU fallback - YASAK!
    cumprod_t = associative_scan_exponential_cpu_fallback(gamma_t_reshaped)
```

**Şimdi (DOĞRU):**
```python
if not torch.cuda.is_available():
    try:
        cumprod_t = associative_scan_exponential_cpu_fallback(gamma_t_reshaped)
    except RuntimeError as e:
        raise RuntimeError(
            f"❌ CRITICAL: C++ extension required for CPU mode!\n"
            f"   {str(e)}"
        ) from e
```

---

### 3. `pretrain.py` - Training Script

**Zorunlu Kontrol:**
```python
# On CPU, C++ extension is REQUIRED - NO FALLBACK ALLOWED
if device.type == 'cpu' and not cpp_available:
    print("❌ FATAL: Pre-training CANNOT start without C++ extension on CPU!")
    print("   Fallback mode is DISABLED for performance and correctness.")
    return 1
```

---

## 🧪 Test Senaryoları

### Senaryo 1: Extension Yüklü

```bash
python mm_rec/tests/check_extensions.py
# ✅ mm_rec_scan_cpu: Yüklendi
# ✅ mm_rec_cpp_cpu: Yüklendi
```

### Senaryo 2: Extension Yok

```python
# Extension'ı geçici olarak kaldır
import sys
if 'mm_rec_scan_cpu' in sys.modules:
    del sys.modules['mm_rec_scan_cpu']

# Test et
from mm_rec.core.associative_scan_triton import associative_scan_exponential_cpu_fallback
x = torch.rand(2, 8, 128, 64, dtype=torch.bfloat16)
y = associative_scan_exponential_cpu_fallback(x)
# ❌ RuntimeError: C++ extension is REQUIRED!
```

---

## 📊 Extension Versiyon ve Build Kontrolü

### Versiyon Kontrolü

Extension'lar şu bilgileri içerir:
- **Path**: Dosya yolu
- **Size**: Dosya boyutu (bytes)
- **mtime**: Son değiştirilme zamanı (build zamanı)
- **SHA256**: Build parmak izi
- **__version__**: Versiyon numarası (varsa)

### Build Kontrolü

```bash
# Extension'ları kontrol et
python mm_rec/tests/check_extensions.py

# SHA256 ve mtime'ı karşılaştır
# Eğer farklıysa, extension yeniden derlenmiş demektir
```

---

## ✅ Özet

1. **Extension'lar ZORUNLU**: CPU modunda extension olmadan çalışmak YASAK
2. **Fallback YOK**: Python fallback implementasyonu kaldırıldı
3. **Hata Mesajları**: Extension yoksa açıklayıcı hata mesajları
4. **Otomatik Preload**: libc10.so sorunu otomatik çözülüyor
5. **Kontrol Aracı**: `check_extensions.py` ile durum kontrolü

---

## 🚀 Kullanım

### Extension'ları Derle

```bash
cd mm_rec/cpp
python setup.py build_ext --inplace
```

### Extension'ları Kontrol Et

```bash
python mm_rec/tests/check_extensions.py
```

### Eğitimi Başlat

```bash
python -m mm_rec.scripts.pretrain --device cpu
# Extension yoksa hata verir ve çıkar
```

---

**SONUÇ**: Extension'lar artık **ZORUNLU** ve **fallback modu YOK**! ✅

