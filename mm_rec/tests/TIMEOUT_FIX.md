# Timeout Sorunu ve Çözümü

## Sorun

unittest.TestCase ile pytest timeout'ları bazen çalışmıyor. Özellikle:
- unittest.TestCase testleri pytest timeout'larını bypass edebilir
- `timeout_func_only = False` (default) unittest.TestCase ile sorunlu

## Çözüm

### 1. pytest.ini Ayarları

```ini
timeout = 60
timeout_method = thread
timeout_func_only = True  # CRITICAL for unittest.TestCase
```

**`timeout_func_only = True`** unittest.TestCase test methodlarını wrap eder.

### 2. Test-Specific Timeout

```python
@pytest.mark.timeout(120, method='thread')
def test_example(self):
    # Test code
    pass
```

### 3. Alternatif: pytest-native Testler

unittest.TestCase yerine pytest-native testler kullanılabilir:

```python
# unittest.TestCase (timeout sorunlu olabilir)
class TestExample(unittest.TestCase):
    def test_example(self):
        pass

# pytest-native (timeout daha güvenilir)
def test_example():
    assert True
```

## Test Etme

```bash
# Timeout'un çalışıp çalışmadığını test et
pytest mm_rec/tests/test_components.py::TestMMRecBlock::test_mm_rec_block_forward -v --timeout=5

# Eğer 5 saniyede timeout olmazsa, timeout_func_only = True ayarını kontrol et
```

## Notlar

- `timeout_func_only = True`: Sadece test fonksiyonunu wrap eder (unittest için gerekli)
- `timeout_method = thread`: Thread-based timeout (daha güvenilir)
- unittest.TestCase testleri için timeout'lar bazen çalışmayabilir - pytest-native testlere geçiş önerilir

