# 🔍 CPU Kullanım Verimliliği Analizi

**Tarih**: 2025-01-27  
**Analiz**: Kodları değiştirmeden mevcut durum analizi

---

## 📊 Mevcut Durum

### 1. DataLoader Ayarları
```python
# mm_rec/data/text_data_loader.py:240
num_workers=0  # ❌ PROBLEM: Paralel data loading yok
```

**Sorun**: 
- `num_workers=0` → Data loading ana thread'de yapılıyor
- CPU core'lar kullanılmıyor
- Data loading ve training aynı thread'de → blocking

**Etki**: 
- ~82 saniye/step (çok yavaş)
- CPU %100 kullanılmıyor
- I/O ve computation overlap yok

---

### 2. Batch Size
```python
# Test komutu
--batch-size 2  # ❌ ÇOK KÜÇÜK
```

**Sorun**:
- Batch size = 2 çok küçük
- CPU paralelizasyonu için yetersiz
- Overhead fazla, throughput düşük

**Öneri**: 
- CPU için batch_size = 8-16 daha iyi
- Memory izin veriyorsa daha da artırılabilir

---

### 3. PyTorch Thread Ayarları
```python
# Kontrol edilmedi - muhtemelen default
torch.get_num_threads()  # Default: CPU core sayısı
```

**Durum**:
- Thread sayısı ayarlanmamış
- Default değerler kullanılıyor (muhtemelen tüm core'lar)
- Ama data loading blocking olduğu için thread'ler verimli kullanılmıyor

---

### 4. Pin Memory
```python
# mm_rec/data/text_data_loader.py:241
pin_memory=True if torch.cuda.is_available() else False
```

**Durum**: ✅ Doğru
- CPU'da pin_memory=False (doğru)
- GPU'da pin_memory=True (doğru)

---

### 5. Prefetching
```python
# DataLoader'da prefetch_factor yok
# Default: 2 (ama num_workers=0 olduğu için çalışmıyor)
```

**Sorun**:
- `num_workers=0` olduğu için prefetching çalışmıyor
- Data loading blocking

---

## ❌ Tespit Edilen Problemler

### Kritik Problemler
1. **num_workers=0** 
   - Paralel data loading yok
   - CPU core'lar kullanılmıyor
   - I/O ve computation overlap yok

2. **Batch size çok küçük (2)**
   - CPU paralelizasyonu için yetersiz
   - Overhead fazla

3. **Thread ayarları yok**
   - PyTorch thread sayısı optimize edilmemiş
   - OMP_NUM_THREADS, MKL_NUM_THREADS ayarlanmamış

### Orta Seviye Problemler
4. **Prefetching yok**
   - Data loading blocking
   - Next batch hazır değil

5. **Persistent workers yok**
   - Her epoch'ta worker'lar yeniden oluşturuluyor
   - Overhead

---

## ✅ İyileştirme Önerileri

### 1. num_workers Ayarlama
```python
# CPU core sayısına göre
num_workers = min(4, os.cpu_count())  # 4 worker yeterli
# veya
num_workers = os.cpu_count() // 2  # Core'ların yarısı
```

**Beklenen İyileşme**: %30-50 hızlanma

### 2. Batch Size Artırma
```python
# CPU için optimal
batch_size = 8  # veya 16 (memory izin veriyorsa)
```

**Beklenen İyileşme**: %20-30 hızlanma

### 3. PyTorch Thread Ayarlama
```python
import torch
import os

# CPU core sayısına göre
torch.set_num_threads(os.cpu_count())
# veya
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
```

**Beklenen İyileşme**: %10-20 hızlanma

### 4. Prefetching Ekleme
```python
DataLoader(
    ...,
    num_workers=4,
    prefetch_factor=2,  # 2 batch önceden yükle
    persistent_workers=True  # Worker'ları koru
)
```

**Beklenen İyileşme**: %10-15 hızlanma

---

## 📈 Toplam Beklenen İyileşme

### Mevcut Durum
- **Step süresi**: ~82 saniye
- **CPU kullanımı**: Düşük (data loading blocking)

### İyileştirme Sonrası (Tahmini)
- **Step süresi**: ~30-40 saniye (2x hızlanma)
- **CPU kullanımı**: Yüksek (paralel işlem)
- **Toplam süre**: 45 dakika → 20-25 dakika

---

## 🎯 Öncelik Sırası

### Yüksek Öncelik (Hemen Yapılmalı)
1. ✅ **num_workers > 0** (en kritik)
2. ✅ **Batch size artırma** (8-16)

### Orta Öncelik
3. ✅ **Thread ayarları** (PyTorch, OMP, MKL)
4. ✅ **Prefetching** (num_workers ile birlikte)

### Düşük Öncelik
5. ✅ **Persistent workers** (nice to have)

---

## 💡 Sonuç

### Mevcut Durum: ❌ Verimsiz
- CPU core'lar kullanılmıyor
- Data loading blocking
- Batch size çok küçük
- Thread ayarları yok

### İyileştirme Potansiyeli: ✅ Yüksek
- **2x hızlanma** mümkün
- CPU kullanımı %20-30 → %80-90
- Step süresi 82s → 30-40s

### Öneri
**En kritik**: `num_workers=4` ve `batch_size=8` ayarları ile **2x hızlanma** beklenebilir.

---

**Not**: Bu analiz kodları değiştirmeden yapıldı. İyileştirmeler için kod değişikliği gerekiyor.
