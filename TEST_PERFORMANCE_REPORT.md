# MM-Rec Test Performans Raporu

## 📊 Test Süre Analizi

### Hızlı Testler (< 1 saniye)

**Test Dosyası:** `test_components.py`
- ✅ Tüm testler: **0.25 saniye** (11 test)
- En uzun: `test_mm_rec_block_forward` (0.09s)
- En kısa: Çoğu test (< 0.01s)

**Test Süreleri:**
```
0.09s  test_mm_rec_block_forward
0.07s  test_end_to_end_flow
0.01s  test_mdi_forward_pass
<0.01s Diğer testler
```

---

### Orta Süreli Testler (1-5 saniye)

**Test Dosyası:** `test_associative_scan_validation.py`

**Sequence Length Bazlı Süreler:**
- ✅ **128 tokens**: 0.050 saniye (hızlı)
- ✅ **1024 tokens**: 0.016 saniye (hızlı)
- ⚠️ **8192 tokens**: 0.128 saniye (orta)
- ⚠️ **32768 tokens**: 0.303 saniye (orta)

**Not:** Sequence uzunluğu arttıkça süre artıyor, ancak hala kabul edilebilir.

---

### Uzun Süren Testler (> 5 saniye)

#### 1. **32K Sequence Tests** (`test_32k_sequence.py`)

**Markers:** `@pytest.mark.long`, `@pytest.mark.slow`

**Testler:**
- `test_32k_forward_pass`: 32K token forward pass
- `test_32k_with_memory_states`: 32K token + memory states
- `test_32k_chunking`: Farklı chunk size'ları test et

**Tahmini Süre:** 10-30 saniye (model boyutuna göre)

**Atlanma:**
```bash
pytest -m "not long"
```

---

#### 2. **Gradient Tests** (`test_gradients.py`)

**Markers:** `@pytest.mark.slow`

**Uzun Süren Testler:**
- `test_mm_rec_model_gradcheck`: **ÇOK UZUN** (gradcheck finite difference kullanır)
  - Tahmini: 30-120 saniye
  - Gradcheck her parametre için finite difference hesaplar
- `test_numerical_stability_long_sequence`: Orta-uzun (512 tokens)
  - Tahmini: 5-15 saniye

**Atlanma:**
```bash
pytest -m "not slow"
```

---

#### 3. **Long Sequence Associative Scan** (`test_associative_scan_validation.py`)

**Markers:** `@pytest.mark.slow`, `@pytest.mark.long`

**Testler:**
- `test_long_sequence`: 8192 tokens
- `test_medium_sequence`: 1024 tokens (orta)

**Tahmini Süreler:**
- 8192 tokens: 0.1-1 saniye
- 32768 tokens: 0.3-3 saniye

---

## 🎯 Test Kategorileri

### Hızlı Testler (Günlük Geliştirme)

```bash
# Sadece hızlı testler
pytest -m "not slow and not long"
```

**Kapsam:**
- Component tests (MemoryState, MDI, HDS, Attention, MMRecBlock)
- Short sequence tests (128-1024 tokens)
- Basic gradient tests

**Süre:** ~1-5 saniye

---

### Orta Süreli Testler (CI/CD)

```bash
# Hızlı + orta testler
pytest -m "not long"
```

**Kapsam:**
- Tüm hızlı testler
- Medium sequence tests (1024-8192 tokens)
- Gradient stability tests

**Süre:** ~10-30 saniye

---

### Tam Test Suite (Release)

```bash
# Tüm testler
pytest
```

**Kapsam:**
- Tüm testler (hızlı + orta + uzun)
- 32K sequence tests
- Full gradient checks

**Süre:** ~1-5 dakika

---

## 📋 Test Marker'ları

### Mevcut Marker'lar

1. **`@pytest.mark.slow`**: Uzun süren testler
   - Gradient checks
   - Long sequence tests (8192+)
   - Numerical stability tests

2. **`@pytest.mark.long`**: Çok uzun sequence testleri
   - 32K sequence tests
   - Very long sequence tests (32768+)

3. **`@pytest.mark.gpu`**: GPU gerektiren testler
   - CUDA-specific tests
   - Triton kernel tests

4. **`@pytest.mark.cpu`**: CPU-only testler
   - CPU fallback tests
   - C++ extension tests

5. **`@pytest.mark.extension`**: Extension gerektiren testler
   - C++ extension tests
   - Extension validation tests

---

## 🚀 Kullanım Örnekleri

### Günlük Geliştirme

```bash
# Sadece hızlı testler
pytest -m "not slow and not long" -v

# Belirli bir test dosyası
pytest mm_rec/tests/test_components.py -v
```

### CI/CD Pipeline

```bash
# Orta süreli testler (hızlı + orta)
pytest -m "not long" --durations=10

# Süre raporu ile
pytest --durations=10
```

### Release Öncesi

```bash
# Tüm testler
pytest -v --durations=20

# Sadece uzun testler
pytest -m "slow or long" -v
```

---

## ⚠️ Performans İpuçları

### 1. Test Timeout

**pytest.ini'de ayarlandı:**
```ini
timeout = 300  # 5 dakika
timeout_method = thread
```

**Uzun süren testler timeout ile sonlandırılır.**

---

### 2. Test Paralelleştirme

```bash
# Paralel test çalıştırma (pytest-xdist gerekli)
pytest -n auto  # Otomatik core sayısı
pytest -n 4     # 4 paralel işlem
```

**Not:** Bazı testler state paylaştığı için paralel çalışmayabilir.

---

### 3. Test Seçici Çalıştırma

```bash
# Belirli bir test
pytest mm_rec/tests/test_components.py::TestMMRecBlock::test_mm_rec_block_forward

# Belirli bir kategori
pytest mm_rec/tests/test_components.py -k "test_memory"

# Marker bazlı
pytest -m slow
pytest -m "not slow"
```

---

## 📊 Özet

### Test Süreleri (Tahmini)

| Kategori | Süre | Test Sayısı |
|----------|------|-------------|
| Hızlı | < 5s | ~20 test |
| Orta | 5-30s | ~10 test |
| Uzun | 30s-5min | ~5 test |

### Öneriler

1. **Günlük Geliştirme:** `pytest -m "not slow and not long"`
2. **CI/CD:** `pytest -m "not long"`
3. **Release:** `pytest` (tüm testler)

### En Uzun Süren Testler

1. **`test_mm_rec_model_gradcheck`**: 30-120 saniye
2. **`test_32k_forward_pass`**: 10-30 saniye
3. **`test_32k_chunking`**: 20-60 saniye (3 farklı chunk size)
4. **`test_numerical_stability_long_sequence`**: 5-15 saniye

---

## 🔧 Optimizasyon Önerileri

### 1. Test Parametrelerini Azalt

**Mevcut:**
- 32K testleri: 32768 tokens
- Long sequence: 8192 tokens

**Öneri:**
- CI/CD'de: 8192 tokens (32K yerine)
- Release'de: 32768 tokens

### 2. Test Skip Mekanizması

```python
@pytest.mark.skipif(
    os.environ.get("SKIP_SLOW_TESTS") == "1",
    reason="Skipping slow tests in CI"
)
def test_slow():
    ...
```

### 3. Test Caching

```bash
# pytest-cache eklentisi ile
pytest --cache-clear
pytest --cache-show
```

---

**Son Güncelleme:** Test süreleri gerçek çalıştırmalardan ölçüldü (CPU modunda).

