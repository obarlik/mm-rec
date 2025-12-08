# MM-Rec Test Kılavuzu

## Hızlı Test Çalıştırma

### Tüm Testler (Hızlı Mod)
```bash
# Tüm testleri çalıştır (progress mesajları ile)
python3 -m unittest discover mm_rec.tests -v

# Sadece gradient testleri
python3 -m unittest mm_rec.tests.test_gradients -v

# Sadece component testleri
python3 -m unittest mm_rec.tests.test_components -v
```

### Tekil Testler
```bash
# Backward pass testi (hızlı)
python3 -m unittest mm_rec.tests.test_gradients.TestGradients.test_backward_pass_completes -v

# Gradient flow testi
python3 -m unittest mm_rec.tests.test_gradient_flow_detailed.TestGradientFlowDetailed.test_identify_parameters_without_gradients -v

# Numerical stability (512 tokens - hızlı)
python3 -m unittest mm_rec.tests.test_gradients.TestGradients.test_numerical_stability_long_sequence -v
```

## Test Süreleri

### Hızlı Testler (< 1 saniye)
- `test_backward_pass_completes`
- `test_gradient_flow_through_components`
- `test_identify_parameters_without_gradients`

### Orta Süreli Testler (1-10 saniye)
- `test_numerical_stability_long_sequence` (512 tokens)
- `test_multiple_forward_backward_passes` (3 iterasyon)

### Uzun Süreli Testler (30-60 saniye)
- `test_mm_rec_model_gradcheck` (gradcheck çok yavaş)

## Progress Mesajları

Tüm testler artık progress mesajları gösteriyor:
```
[Progress] Forward pass çalışıyor...
[Progress] ✓ Forward pass tamamlandı
[Progress] Backward pass başlıyor...
[Progress] ✓ Backward pass tamamlandı
```

## Test Parametreleri

### Varsayılan (Hızlı)
- `seq_len`: 64 tokens
- `batch_size`: 2
- `model_dim`: 64
- `num_layers`: 1

### Uzun Dizi (Orta Hız)
- `seq_len`: 512 tokens (8192'den düşürüldü)
- `batch_size`: 1
- `model_dim`: 128
- `num_layers`: 2

## Uzun Testleri Atlama

Eğer gradcheck testi çok uzun sürüyorsa, atlayabilirsiniz:
```bash
# Gradcheck hariç tüm testler
python3 -m unittest discover mm_rec.tests -v -k "not gradcheck"
```

## Sorun Giderme

### Test Çok Uzun Sürüyor
1. `long_seq_config` içindeki `seq_len` değerini kontrol edin (512 olmalı)
2. Sadece hızlı testleri çalıştırın
3. Gradcheck testini atlayın

### Progress Mesajları Görünmüyor
- `-v` (verbose) flag'ini kullanın
- Test output'unu kontrol edin

### Memory Hatası
- Batch size'ı küçültün
- Seq len'i azaltın
- Model boyutunu küçültün

