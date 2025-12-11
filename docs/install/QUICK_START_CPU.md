# MM-Rec 100M - CPU Eğitim Hızlı Başlangıç

**Tek Komutla Başla**: İnternetten veri indirip CPU'da eğit!

---

## 🚀 Tek Komut (Tam Otomatik)

```bash
python3 -m mm_rec.scripts.auto_train
```

Bu komut:
1. ✅ İnternetten veri indirir (WikiText, Code datasets)
2. ✅ Veriyi işler ve hazırlar
3. ✅ CPU'da eğitime başlar
4. ✅ Checkpoint'leri kaydeder

**Süre**: ~2-4 saat (Stage 1, 1000 steps)

---

## 📦 Kurulum

### 1. Bağımlılıkları Yükle

```bash
# CPU eğitim için ek bağımlılıklar
pip install requests datasets huggingface-hub psutil tqdm
```

veya

```bash
pip install -r requirements_cpu.txt
```

### 2. Test Et

```bash
# Import test
python3 -c "from mm_rec.data.download_data import DataDownloader; print('✅ OK')"
```

---

## 📥 Veri İndirme (Manuel)

Eğer sadece veri indirmek istiyorsanız:

```bash
python3 -m mm_rec.data.download_data \
    --output_dir ./data \
    --text_samples 500 \
    --code_samples 500
```

**İndirilen Veriler**:
- `./data/text/wikitext.jsonl` - Text data
- `./data/code/code.jsonl` - Code data

---

## 🖥️ CPU Eğitimi (Manuel)

Veri indirildikten sonra:

```bash
python3 -m mm_rec.scripts.train_cpu \
    --stage stage1 \
    --batch_size 2 \
    --data_dir ./data \
    --checkpoint_dir ./checkpoints_cpu
```

**Parametreler**:
- `--stage`: stage1, stage2, stage3, veya all
- `--batch_size`: 1-2 (CPU için)
- `--data_dir`: Veri dizini
- `--use_synthetic`: Synthetic data kullan (hızlı test)

---

## ⚡ Hızlı Test (Synthetic Data)

GPU yoksa ve hızlı test yapmak istiyorsanız:

```bash
python3 -m mm_rec.scripts.train_cpu \
    --stage stage1 \
    --use_synthetic \
    --batch_size 2
```

**Süre**: ~10-30 dakika (1000 steps)

---

## 📊 Beklenen Sonuçlar

**CPU Training** (Stage 1, 1000 steps):
- Süre: ~2-4 saat
- Loss: 8-10 → 3-4
- Memory: ~2-4 GB RAM
- CPU Usage: ~80-100%

**Checkpoint'ler**:
- `./checkpoints_cpu/checkpoint_stage1_step_*.pt`

---

## 🔧 Sorun Giderme

### Veri İndirme Başarısız

```bash
# Synthetic data kullan
python3 -m mm_rec.scripts.train_cpu --use_synthetic
```

### Memory Yetersiz

```bash
# Batch size'ı düşür
python3 -m mm_rec.scripts.train_cpu --batch_size 1
```

### Çok Yavaş

```bash
# Max steps'i azalt (test için)
# train_cpu.py içinde max_steps=100 yap
```

---

## 📖 Detaylı Dokümantasyon

- **CPU_TRAINING_GUIDE.md**: Kapsamlı CPU eğitim rehberi
- **TRAINING_GUIDE_100M.md**: Genel eğitim rehberi

---

## ⚠️ Önemli Notlar

1. **CPU Eğitimi Yavaştır**: GPU'dan 1000x daha yavaş
2. **Test İçin**: CPU eğitimi test ve geliştirme için uygundur
3. **Production**: Production eğitimi için GPU kullanın
4. **İnternet**: Veri indirme için internet bağlantısı gereklidir

---

**Hazır!** Tek komutla başlayın:
```bash
python3 -m mm_rec.scripts.auto_train
```

