# MM-Rec 100M CPU Eğitim Rehberi

**Versiyon**: 1.0  
**Tarih**: 2025-12-08  
**Hedef**: CPU'da otomatik veri indirme ve eğitim

---

## 🎯 Genel Bakış

Bu rehber, MM-Rec 100M modelini **CPU'da** eğitmek için gereken tüm adımları içerir. Sistem otomatik olarak:
1. İnternetten veri indirir (WikiText, Code datasets)
2. Veriyi işler ve hazırlar
3. CPU'da eğitime başlar

---

## 🚀 Hızlı Başlangıç

### En Basit Yöntem (Tam Otomatik)

```bash
python3 -m mm_rec.scripts.auto_train
```

Bu komut:
- ✅ Veriyi otomatik indirir
- ✅ Veriyi işler
- ✅ CPU'da eğitime başlar
- ✅ Checkpoint'leri kaydeder

### Adım Adım Yöntem

**1. Veri İndirme**:
```bash
python3 -m mm_rec.data.download_data \
    --output_dir ./data \
    --text_samples 500 \
    --code_samples 500
```

**2. CPU Eğitimi**:
```bash
python3 -m mm_rec.scripts.train_cpu \
    --stage stage1 \
    --batch_size 2 \
    --data_dir ./data \
    --download_data  # Veri yoksa otomatik indirir
```

---

## 📥 Veri İndirme

### Desteklenen Kaynaklar

**Text Data**:
- **WikiText**: Wikipedia articles
- **The Pile**: Large text corpus
- **C4**: Common Crawl dataset

**Code Data**:
- **The Stack**: Large code dataset
- **Python Code**: Python-specific code
- **CodeSearchNet**: Code search dataset

### Veri İndirme Komutları

**Sadece Text**:
```bash
python3 -m mm_rec.data.download_data \
    --text_samples 1000 \
    --no_code
```

**Sadece Code**:
```bash
python3 -m mm_rec.data.download_data \
    --code_samples 1000 \
    --no_wikitext
```

**Her İkisi**:
```bash
python3 -m mm_rec.data.download_data \
    --text_samples 1000 \
    --code_samples 1000
```

### İndirilen Veri Formatı

```
data/
├── text/
│   └── wikitext.jsonl    # Her satır bir JSON string
└── code/
    └── code.jsonl        # Her satır bir JSON string
```

**Format**:
```json
"Bu bir örnek metin..."
"def example_function(): ..."
```

---

## 🖥️ CPU Eğitimi

### CPU Optimizasyonları

**Otomatik Optimizasyonlar**:
- ✅ Tüm CPU thread'leri kullanılır
- ✅ MKL optimizasyonları aktif
- ✅ Küçük batch size (memory için)
- ✅ Kısa sekanslar (512 tokens)
- ✅ Gradient checkpointing (memory için)

**Performans İpuçları**:
```python
# CPU thread sayısını ayarla
torch.set_num_threads(os.cpu_count())

# MKL kullan
import mkl
mkl.set_num_threads(os.cpu_count())
```

### Eğitim Komutları

**Stage 1 (Local Consistency)**:
```bash
python3 -m mm_rec.scripts.train_cpu \
    --stage stage1 \
    --batch_size 2 \
    --checkpoint_dir ./checkpoints_cpu \
    --data_dir ./data
```

**Tüm Stage'ler**:
```bash
python3 -m mm_rec.scripts.train_cpu \
    --stage all \
    --batch_size 2 \
    --data_dir ./data
```

**Synthetic Data ile (Hızlı Test)**:
```bash
python3 -m mm_rec.scripts.train_cpu \
    --stage stage1 \
    --use_synthetic \
    --batch_size 2
```

### Beklenen Performans (CPU)

**Süre** (Stage 1, 1000 steps):
- Modern CPU (16 cores): ~2-4 saat
- Older CPU (8 cores): ~4-8 saat

**Memory**:
- Model: ~400 MB (FP32)
- Training: ~2-4 GB RAM
- Data: ~100-500 MB

**Throughput**:
- ~0.1-0.5 tokens/second (GPU: ~1000+ tokens/second)
- CPU eğitimi GPU'dan **1000x daha yavaş**

---

## 📊 Monitoring

### CPU Training Monitoring

```python
# Training loop içinde
for step in range(num_steps):
    # ... training code ...
    
    # CPU utilization
    import psutil
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.percent}%")
```

### Loss Tracking

```python
# Loss'u dosyaya kaydet
with open("loss_log.txt", "a") as f:
    f.write(f"{step},{loss.item()}\n")
```

---

## ⚙️ Konfigürasyon

### CPU-Friendly Ayarlar

**Batch Size**:
- CPU için: 1-2 (memory için)
- GPU için: 4-8

**Sequence Length**:
- CPU için: 512 (hızlı iterasyon)
- GPU için: 8192+

**Max Steps**:
- CPU için: 1000 (test için)
- GPU için: 5000-10000

**Checkpoint Interval**:
- CPU için: 50 (sık kaydet)
- GPU için: 100-200

### Örnek Konfigürasyon

```bash
python3 -m mm_rec.scripts.train_cpu \
    --stage stage1 \
    --batch_size 1 \
    --checkpoint_interval 25 \
    --max_samples 100 \
    --use_synthetic  # Hızlı test için
```

---

## 🔧 Sorun Giderme

### Sorun 1: Veri İndirme Başarısız

**Hata**: `ConnectionError` veya `Timeout`

**Çözüm**:
```bash
# Retry with timeout
python3 -m mm_rec.data.download_data \
    --text_samples 100 \
    --code_samples 100

# Veya synthetic data kullan
python3 -m mm_rec.scripts.train_cpu --use_synthetic
```

### Sorun 2: CPU Memory Yetersiz

**Hata**: `RuntimeError: out of memory`

**Çözüm**:
```bash
# Batch size'ı düşür
--batch_size 1

# Sequence length'i düşür
# (train_cpu.py içinde seq_len=256 yap)

# Max samples'ı azalt
--max_samples 50
```

### Sorun 3: Çok Yavaş

**Sorun**: Eğitim çok yavaş

**Çözüm**:
- Synthetic data kullan (hızlı test)
- Max steps'i azalt (100-200)
- Sadece Stage 1'i çalıştır

### Sorun 4: Veri Formatı Hatası

**Hata**: `JSONDecodeError` veya `KeyError`

**Çözüm**:
```bash
# Veriyi yeniden indir
rm -rf ./data
python3 -m mm_rec.data.download_data

# Veya synthetic data kullan
python3 -m mm_rec.scripts.train_cpu --use_synthetic
```

---

## 📝 Örnek Kullanım Senaryoları

### Senaryo 1: Hızlı Test (Synthetic)

```bash
# 5 dakikada test
python3 -m mm_rec.scripts.train_cpu \
    --stage stage1 \
    --use_synthetic \
    --batch_size 2 \
    --max_samples 10
```

### Senaryo 2: Gerçek Veri ile Eğitim

```bash
# 1. Veri indir
python3 -m mm_rec.data.download_data \
    --text_samples 500 \
    --code_samples 500

# 2. Eğit
python3 -m mm_rec.scripts.train_cpu \
    --stage stage1 \
    --data_dir ./data \
    --batch_size 2
```

### Senaryo 3: Tam Otomatik

```bash
# Tek komutla her şey
python3 -m mm_rec.scripts.auto_train
```

---

## 🎓 Best Practices

### 1. CPU Eğitimi İçin

- ✅ Küçük batch size kullan (1-2)
- ✅ Kısa sekanslar (512 tokens)
- ✅ Sık checkpoint (her 25-50 step)
- ✅ Synthetic data ile test et
- ✅ Monitoring'i aktif tut

### 2. Veri İndirme İçin

- ✅ İnternet bağlantısını kontrol et
- ✅ Disk alanını kontrol et (500MB+)
- ✅ Hugging Face token gerekebilir (büyük dataset'ler için)
- ✅ İlk indirmede az sample al (test için)

### 3. Production İçin

- ⚠️ CPU eğitimi **production için önerilmez**
- ✅ GPU kullan (1000x daha hızlı)
- ✅ Distributed training (multi-GPU)
- ✅ Cloud GPU (AWS, GCP, Azure)

---

## 📊 Beklenen Sonuçlar

### CPU Training Metrics

**Stage 1** (1000 steps, CPU):
- Süre: ~2-4 saat
- Loss: 8-10 → 3-4
- Memory: ~2-4 GB
- CPU Usage: ~80-100%

**GPU Training** (karşılaştırma):
- Süre: ~15-30 dakika
- Loss: 8-10 → 2-3
- Memory: ~8-16 GB
- GPU Usage: ~90-100%

---

## 🚀 Hızlı Komutlar

### Tam Otomatik
```bash
python3 -m mm_rec.scripts.auto_train
```

### Veri İndir + Eğit
```bash
python3 -m mm_rec.scripts.train_cpu \
    --download_data \
    --stage stage1
```

### Synthetic Data ile Test
```bash
python3 -m mm_rec.scripts.train_cpu \
    --use_synthetic \
    --stage stage1 \
    --batch_size 2
```

---

## ⚠️ Önemli Notlar

1. **CPU Eğitimi Yavaştır**: GPU'dan 1000x daha yavaş
2. **Test İçin**: CPU eğitimi test ve geliştirme için uygundur
3. **Production**: Production eğitimi için GPU kullanın
4. **Memory**: CPU'da memory sınırlı, batch size küçük tutun
5. **Veri**: İnternet bağlantısı gereklidir

---

**Son Güncelleme**: 2025-12-08  
**Hazırlayan**: MM-Rec Development Team

