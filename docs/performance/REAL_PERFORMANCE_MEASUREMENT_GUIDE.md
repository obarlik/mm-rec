# MM-Rec Gerçek Performans Ölçümleri - Kullanım Kılavuzu

**ÖNEMLİ**: Bu doküman, gerçek performans ölçümleri yapmak için hazırlanmıştır. HEM ve UBÖO mekanizmaları henüz kod tabanına entegre edilmemiş olabilir, bu durumda önce mevcut kodun baseline performansını ölçeceğiz.

---

## 🔍 Mevcut Durum Kontrolü

### HEM (Mekanizma 1) Durumu

**Kontrol Komutu**:
```bash
cd /home/onur/workspace/mm-rec
grep -r "use_hem\|W_fused\|fused" mm_rec/ --include="*.py"
```

**Beklenen Çıktı**:
- Eğer HEM implement edilmişse: `use_hem`, `W_fused`, `fused_output` gibi terimler görülmeli
- Eğer implement edilmemişse: Sadece dokümantasyon dosyalarında görülür

### UBÖO (Mekanizma 3) Durumu

**Kontrol Komutu**:
```bash
cd /home/onur/workspace/mm-rec
grep -r "use_uboo\|planning_error\|L_Aux\|auxiliary" mm_rec/ --include="*.py"
```

**Beklenen Çıktı**:
- Eğer UBÖO implement edilmişse: `use_uboo`, `planning_error`, `L_Aux` gibi terimler görülmeli
- Eğer implement edilmemişse: Sadece dokümantasyon dosyalarında görülür

---

## 📊 Gerçek Performans Ölçümleri

### 1. Baseline Ölçümleri (Mevcut Kod)

**Script**: `mm_rec/scripts/real_benchmark.py`

**Çalıştırma**:
```bash
cd /home/onur/workspace/mm-rec
python mm_rec/scripts/real_benchmark.py
```

**Ölçülen Metrikler**:
- Block latency (ms)
- Model forward time (ms)
- Throughput (tokens/s)
- Memory usage (MB)
- Per-layer latency estimate

**Çıktı**: `benchmark_results.json`

### 2. HEM Karşılaştırması (Eğer Implement Edilmişse)

**Script**: `mm_rec/scripts/benchmark_hem.py`

**Çalıştırma**:
```bash
cd /home/onur/workspace/mm-rec
python mm_rec/scripts/benchmark_hem.py
```

**Ölçülen Metrikler**:
- HEM pasif: Block latency, throughput, memory
- HEM aktif: Block latency, throughput, memory
- İyileştirme yüzdesi

### 3. UBÖO Eğitim Testi (Eğer Implement Edilmişse)

**Script**: `mm_rec/scripts/train_uboo_test.py` (oluşturulacak)

**Çalıştırma**:
```bash
cd /home/onur/workspace/mm-rec
python mm_rec/scripts/train_uboo_test.py
```

**Ölçülen Metrikler**:
- Convergence steps (UBÖO vs baseline)
- Final perplexity
- Training stability (loss variance)
- Memory overhead

---

## 🛠️ Gereksinimler

### Python Ortamı

```bash
# Virtual environment oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt
```

### CUDA (GPU için)

```bash
# CUDA sürümünü kontrol et
nvcc --version

# PyTorch CUDA desteğini kontrol et
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

---

## 📝 Ölçüm Sonuçlarını Dokümana Ekleme

### Adım 1: Ölçümleri Çalıştır

```bash
# Baseline ölçümleri
python mm_rec/scripts/real_benchmark.py > baseline_results.txt 2>&1

# HEM karşılaştırması (eğer implement edilmişse)
python mm_rec/scripts/benchmark_hem.py > hem_results.txt 2>&1
```

### Adım 2: Sonuçları Analiz Et

```bash
# JSON sonuçlarını oku
cat benchmark_results.json | python -m json.tool
```

### Adım 3: PERFORMANCE_AND_DEPENDENCIES.md'yi Güncelle

Ölçülen gerçek değerleri dokümana ekle:

```markdown
## 1. Performans Doğrulaması

### 1.1 HEM (Mekanizma 1) - Gerçek Ölçümler

**Test Tarihi**: [TARİH]
**GPU**: [GPU MODEL]
**CUDA**: [CUDA VERSION]

| Metrik | Orijinal | HEM | İyileştirme |
|--------|----------|-----|-------------|
| Block Latency | [GERÇEK DEĞER] ms | [GERÇEK DEĞER] ms | [GERÇEK DEĞER]% |
| Throughput | [GERÇEK DEĞER] tokens/s | [GERÇEK DEĞER] tokens/s | [GERÇEK DEĞER]% |
```

---

## ⚠️ Önemli Notlar

1. **HEM ve UBÖO Henüz Implement Edilmemiş Olabilir**
   - Kod tabanında `use_hem` ve `use_uboo` parametreleri yoksa, bu mekanizmalar henüz entegre edilmemiştir
   - Bu durumda sadece baseline ölçümleri yapılabilir
   - HEM ve UBÖO implement edildikten sonra tekrar ölçüm yapılmalı

2. **Gerçek Ölçümler için GPU Gerekli**
   - CPU'da ölçümler çok yavaş olacaktır
   - GPU'da ölçümler daha anlamlı sonuçlar verecektir

3. **Warmup Önemli**
   - İlk birkaç iterasyon GPU'yu ısıtır, bu yüzden warmup iterasyonları atlanmalı
   - Script'lerde warmup iterasyonları otomatik olarak atlanır

4. **Memory Sınırlamaları**
   - Büyük modeller için OOM (Out of Memory) hatası alınabilir
   - Bu durumda model boyutunu veya batch size'ı küçültün

---

## 🔄 Sonraki Adımlar

1. **HEM ve UBÖO Implement Et** (eğer henüz yapılmadıysa)
   - `HEM_INTEGRATION_CODE.md` ve `UBOO_INTEGRATION_CODE.md` dosyalarındaki kodları kullan
   - `mm_rec/blocks/mm_rec_block.py` ve `mm_rec/model.py` dosyalarını güncelle

2. **Gerçek Ölçümleri Yap**
   - Baseline ölçümleri
   - HEM karşılaştırması
   - UBÖO eğitim testi

3. **Dokümanı Güncelle**
   - `PERFORMANCE_AND_DEPENDENCIES.md` dosyasındaki hayali değerleri gerçek değerlerle değiştir

---

**Hazırlayan**: MM-Rec Performance Team  
**Tarih**: 2025-01-27


