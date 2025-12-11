# 🎯 Gerçek Dataset ile Tiny Model Eğitimi - Plan

**Tarih**: 2025-01-27  
**Hedef**: Gerçek veriyle gerçek tiny model eğitimi

---

## 📊 Mevcut Durum

### Tamamlanan
- ✅ Tiny model eğitildi (sample corpus ile)
- ✅ Loss: 8.6465 → 0.8179 (%90.5 iyileşme)
- ✅ Checkpoint'ler kaydedildi
- ✅ Training scripti hazır (`train_base_model.py`)
- ✅ Data loader gerçek dataset desteği var

### Eksikler
- ⚠️ Gerçek dataset yok (sample corpus kullanıldı)
- ⚠️ Validation set yok (sample corpus nedeniyle)
- ⚠️ Dataset indirme scripti eksik

---

## 🚀 Gerçek Dataset Seçenekleri

### 1. Tiny Shakespeare (Başlangıç İçin - Küçük)
- **Boyut**: ~1MB
- **Kullanım**: İlk gerçek dataset eğitimi için
- **İndirme**: Otomatik (script ile)
- **Avantaj**: Hızlı indirme, küçük boyut, gerçek text data
- **Dezavantaj**: Çok küçük, daha büyük dataset'ler daha iyi sonuç verir

### 2. OpenWebText (Gerçek Eğitim İçin)
- **Boyut**: ~8GB (compressed)
- **Kullanım**: Gerçek pre-training için
- **İndirme**: Manuel (talimatlar var)
- **Avantaj**: Gerçekçi, büyük dataset
- **Dezavantaj**: İndirme zamanı, büyük boyut

### 3. C4 (Colossal Clean Crawled Corpus)
- **Boyut**: ~750GB (compressed)
- **Kullanım**: Büyük ölçekli eğitim için
- **İndirme**: TensorFlow Datasets
- **Avantaj**: Çok büyük, temiz dataset
- **Dezavantaj**: Çok büyük, indirme zamanı

### 4. Wikipedia (Orta Boyut)
- **Boyut**: ~20GB (compressed)
- **Kullanım**: Orta ölçekli eğitim için
- **İndirme**: Manuel (dumps.wikimedia.org)
- **Avantaj**: Orta boyut, kaliteli içerik
- **Dezavantaj**: İndirme zamanı

---

## 📋 Adım Adım Plan

### Adım 1: Dataset Hazırlama (Şimdi)

**Seçenek A: Tiny Shakespeare (İlk Gerçek Dataset)**
```bash
python mm_rec/scripts/prepare_real_dataset.py \
    --download_tiny_shakespeare \
    --output_dir ./data/real/tiny_shakespeare \
    --val_split 0.1
```

**Seçenek B: Kendi Dosyanız**
```bash
python mm_rec/scripts/prepare_real_dataset.py \
    --input_file /path/to/your/text.txt \
    --output_dir ./data/real/custom \
    --val_split 0.1
```

**Seçenek C: Dizin**
```bash
python mm_rec/scripts/prepare_real_dataset.py \
    --input_dir /path/to/text/files \
    --output_dir ./data/real/custom \
    --val_split 0.1
```

### Adım 2: Dataset Kontrolü

```bash
# Dosyaları kontrol et
ls -lh ./data/real/tiny_shakespeare/
# train.txt ve val.txt olmalı

# Boyutları kontrol et
wc -c ./data/real/tiny_shakespeare/train.txt
wc -c ./data/real/tiny_shakespeare/val.txt
```

### Adım 3: Gerçek Dataset ile Eğitim

```bash
python mm_rec/scripts/train_base_model.py \
    --config tiny \
    --data_dir ./data/real/tiny_shakespeare \
    --epochs 10 \
    --batch_size 4 \
    --seq_len 512 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --save_best_model \
    --early_stopping_patience 5
```

**Farklar (Sample Corpus'tan)**:
- ✅ `--data_dir` parametresi eklendi
- ✅ Validation set otomatik oluşturulacak
- ✅ Best model mekanizması çalışacak
- ✅ Early stopping çalışacak

---

## 🎯 Beklenen Sonuçlar

### Sample Corpus vs Gerçek Dataset

| Özellik | Sample Corpus | Gerçek Dataset |
|---------|---------------|----------------|
| **Loss Düşüşü** | %90.5 (çok hızlı) | %70-80 (daha gerçekçi) |
| **Validation** | ❌ Yok | ✅ Var |
| **Best Model** | ❌ Yok | ✅ Var |
| **Early Stopping** | ❌ Çalışmıyor | ✅ Çalışacak |
| **Overfitting Kontrolü** | ❌ Yok | ✅ Var |
| **Gerçekçilik** | ⚠️ Düşük | ✅ Yüksek |

### Gerçek Dataset Avantajları
1. ✅ **Validation set**: Overfitting kontrolü
2. ✅ **Best model**: En iyi checkpoint seçimi
3. ✅ **Early stopping**: Gereksiz eğitim önleme
4. ✅ **Gerçekçi loss**: Daha gerçekçi metrikler
5. ✅ **Daha iyi model**: Gerçek data ile öğrenme

---

## 📊 Eğitim Karşılaştırması

### Sample Corpus Eğitimi (Önceki)
- **Loss**: 8.6465 → 0.8179 (%90.5)
- **Validation**: ❌ Yok
- **Best Model**: ❌ Yok
- **Early Stopping**: ❌ Çalışmıyor
- **Süre**: ~33 dakika (3 epoch)

### Gerçek Dataset Eğitimi (Yeni)
- **Loss**: Beklenen: 8.0 → 1.5-2.0 (%75-80)
- **Validation**: ✅ Var
- **Best Model**: ✅ Var
- **Early Stopping**: ✅ Çalışacak
- **Süre**: Dataset boyutuna bağlı

---

## 💡 Öneriler

### İlk Gerçek Dataset Eğitimi İçin
1. **Tiny Shakespeare kullan**: İlk gerçek dataset eğitimi için
2. **10 epoch eğit**: Validation loss'u izle
3. **Best model kaydet**: Early stopping çalışacak

### Gerçek Eğitim İçin
1. **OpenWebText indir**: Gerçek pre-training için
2. **Daha fazla epoch**: 50-100 epoch
3. **Daha büyük batch**: GPU varsa batch_size=8-16
4. **Learning rate tuning**: Validation loss'a göre

---

## 🚀 Hızlı Başlangıç

### 1. Gerçek Dataset Hazırla (Tiny Shakespeare - Gerçek Text Data)
```bash
cd /home/onur/workspace/mm-rec
source venv/bin/activate
python mm_rec/scripts/prepare_real_dataset.py \
    --download_tiny_shakespeare \
    --output_dir ./data/real/tiny_shakespeare
```

### 2. Eğitimi Başlat
```bash
python mm_rec/scripts/train_base_model.py \
    --config tiny \
    --data_dir ./data/real/tiny_shakespeare \
    --epochs 10 \
    --batch_size 4 \
    --seq_len 512
```

### 3. İzle
- Validation loss'u izle
- Best model otomatik kaydedilecek
- Early stopping çalışacak

---

## 📝 Notlar

### Dataset Boyutları
- **Tiny Shakespeare**: ~1MB (ilk gerçek dataset eğitimi için)
- **OpenWebText**: ~8GB (gerçek eğitim için)
- **C4**: ~750GB (büyük ölçekli için)

### Validation Split
- **Önerilen**: 0.1 (10% validation)
- **Küçük dataset**: 0.2 (20% validation)
- **Büyük dataset**: 0.05 (5% validation)

### Epoch Sayısı
- **İlk gerçek eğitim**: 10 epoch
- **Gerçek eğitim**: 50-100 epoch
- **Early stopping**: 5-10 patience

---

## ✅ Sonuç

**Gerçek dataset ile eğitim hazır!**

1. ✅ Dataset hazırlama scripti oluşturuldu
2. ✅ Training scripti gerçek dataset desteği var
3. ✅ Validation set otomatik oluşturulacak
4. ✅ Best model mekanizması çalışacak
5. ✅ Early stopping çalışacak

**Sonraki Adım**: Dataset hazırla ve eğitimi başlat!

---

**Tarih**: 2025-01-27  
**Durum**: ✅ Hazır - Gerçek dataset ile eğitim başlatılabilir


