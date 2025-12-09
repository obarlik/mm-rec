# MM-Rec Test Mekanizmaları ve Timeout'lar

## ✅ Test Mekanizmaları Kontrolü

### Testler Eğitilmiş Model Varsaymıyor

**Tüm testler mekanizmaları kontrol eder, eğitilmiş model varsaymaz:**

1. **Random Initialization**: Tüm modeller random weight'lerle başlatılır
2. **Random Input**: Test input'ları `torch.rand()` veya `torch.randint()` ile oluşturulur
3. **Mechanism Validation**: Testler forward pass, gradient flow, memory state gibi mekanizmaları kontrol eder
4. **No Pretrained Weights**: Hiçbir test checkpoint veya pretrained weight yüklemez

### Örnekler

**✅ DOĞRU - Mekanizma Kontrolü:**
```python
def test_mm_rec_block_forward(self):
    # Random initialization
    block = MMRecBlock(model_dim=256, num_heads=8)
    
    # Random input
    x = torch.randn(batch_size, seq_len, model_dim)
    
    # Test mechanism (forward pass works)
    output, state = block(x, state)
    self.assertIsNotNone(output)
```

**❌ YANLIŞ - Eğitilmiş Model Varsayımı:**
```python
def test_mm_rec_block_forward(self):
    # Pretrained weights - YANLIŞ!
    model.load_state_dict(torch.load('pretrained.pth'))
    
    # Test with real data - YANLIŞ!
    input_ids = tokenizer.encode("Hello world")
```

---

## ⏱️ Test Timeout'ları

### Timeout Stratejisi

**pytest-timeout plugin kullanılıyor:**

1. **Global Timeout**: `pytest.ini`'de 60 saniye (default)
2. **Test-Specific Timeout**: `@pytest.mark.timeout(seconds)` ile override

### Timeout Kategorileri

| Kategori | Timeout | Testler |
|----------|---------|---------|
| **Hızlı** | 5-10s | Component tests, short sequences |
| **Orta** | 30-60s | Medium sequences, gradient tests |
| **Uzun** | 60-120s | Long sequences, numerical stability |
| **Çok Uzun** | 120-300s | 32K tests, gradcheck |

---

## 📋 Test Timeout'ları

### Component Tests (`test_components.py`)

**Default**: 60s (pytest.ini)

- `test_memory_bank_initialization`: ~0.01s
- `test_memory_state_initialization`: ~0.01s
- `test_mdi_forward_pass`: ~0.01s
- `test_mm_rec_block_forward`: ~0.09s

**Not**: Component testleri hızlı, timeout gerekmez.

---

### Associative Scan Tests (`test_associative_scan_validation.py`)

**Test-Specific Timeouts:**

- `test_short_sequence` (128 tokens): **Default 60s** (yeterli)
- `test_medium_sequence` (1024 tokens): **10s timeout**
- `test_long_sequence` (8192 tokens): **30s timeout**
- `test_hybrid_precision`: **Default 60s**
- `test_numerical_stability`: **Default 60s**

---

### Gradient Tests (`test_gradients.py`)

**Test-Specific Timeouts:**

- `test_mm_rec_model_gradcheck`: **300s (5 min)** - Çok uzun sürer
- `test_backward_pass_completes`: **Default 60s**
- `test_numerical_stability_long_sequence`: **60s timeout**
- `test_gradient_flow_through_components`: **Default 60s**
- `test_multiple_forward_backward_passes`: **Default 60s**

---

### 32K Sequence Tests (`test_32k_sequence.py`)

**Test-Specific Timeouts:**

- `test_32k_forward_pass`: **120s (2 min)**
- `test_32k_with_memory_states`: **120s (2 min)**
- `test_32k_chunking`: **180s (3 min)** - Multiple chunk sizes

---

## 🔧 Timeout Kullanımı

### pytest.ini (Global)

```ini
timeout = 60  # Default timeout for all tests
timeout_method = thread
```

### Test-Specific Timeout

```python
import pytest

class TestExample(unittest.TestCase):
    @pytest.mark.timeout(10)  # 10 second timeout
    def test_fast_operation(self):
        # Fast test
        pass
    
    @pytest.mark.timeout(300)  # 5 minute timeout
    @pytest.mark.slow
    def test_slow_operation(self):
        # Slow test
        pass
```

---

## 🚀 Test Çalıştırma

### Hızlı Testler (Timeout'lar dahil)

```bash
# Hızlı testler (timeout'lar otomatik)
pytest -m "not slow and not long" -v

# Timeout bilgisi ile
pytest -m "not slow and not long" -v --timeout=10
```

### Tüm Testler (Timeout'lar dahil)

```bash
# Tüm testler (her test kendi timeout'una sahip)
pytest -v

# Timeout bilgisi ile
pytest -v --durations=20
```

### Timeout Hatalarını Görmek

```bash
# Timeout hatalarını detaylı göster
pytest -v --tb=short --timeout=60

# Timeout olan testleri liste
pytest --timeout=60 -v | grep -i timeout
```

---

## ⚠️ Timeout Best Practices

### 1. Timeout Değerleri

**Kural:** Timeout, testin normal süresinin **3-5 katı** olmalı.

- Normal süre: 10s → Timeout: 30-50s
- Normal süre: 30s → Timeout: 90-150s

### 2. Timeout Method

**`thread` method kullanılıyor:**
- Daha güvenilir (signal method bazı durumlarda çalışmaz)
- Thread-safe
- Windows'ta da çalışır

### 3. Timeout Hataları

**Timeout olduğunda:**
```
FAILED mm_rec/tests/test_32k_sequence.py::Test32KSequence::test_32k_forward_pass
TimeoutError: Test exceeded 120 seconds
```

**Çözüm:**
1. Test optimizasyonu (daha hızlı çalıştır)
2. Timeout artırma (gerekirse)
3. Test'i `@pytest.mark.skip` ile atla (geçici)

---

## 📊 Timeout Özeti

### Mevcut Timeout'lar

| Test Dosyası | Test | Timeout |
|--------------|------|---------|
| `test_components.py` | Tümü | 60s (default) |
| `test_associative_scan_validation.py` | `test_medium_sequence` | 10s |
| `test_associative_scan_validation.py` | `test_long_sequence` | 30s |
| `test_gradients.py` | `test_mm_rec_model_gradcheck` | 300s |
| `test_gradients.py` | `test_numerical_stability_long_sequence` | 60s |
| `test_32k_sequence.py` | `test_32k_forward_pass` | 120s |
| `test_32k_sequence.py` | `test_32k_with_memory_states` | 120s |
| `test_32k_sequence.py` | `test_32k_chunking` | 180s |

### Timeout Coverage

- ✅ **Hızlı testler**: Default 60s (yeterli)
- ✅ **Orta testler**: 10-30s (test-specific)
- ✅ **Uzun testler**: 60-120s (test-specific)
- ✅ **Çok uzun testler**: 180-300s (test-specific)

---

## ✅ Kontrol Listesi

- [x] Tüm testler random initialization kullanıyor
- [x] Hiçbir test pretrained weight yüklemiyor
- [x] Tüm testler mekanizmaları kontrol ediyor
- [x] pytest-timeout plugin kurulu
- [x] pytest.ini'de global timeout var (60s)
- [x] Uzun testlerde test-specific timeout var
- [x] Timeout method: thread (güvenilir)

---

**Son Güncelleme:** Timeout'lar tüm testlere eklendi, mekanizma kontrolü doğrulandı.

