# Kaliteli Eğitim Rehberi

**Tarih**: 2025-01-27  
**Amaç**: En küçük modelin bile kaliteli ve sağlam temellerle eğitilmesi

---

## 🎯 Kaliteli Eğitim Prensipleri

### 1. Gerçek Data Kullanımı
- ❌ **Önceki**: Simüle edilmiş random data
- ✅ **Şimdi**: Gerçek text data (character-level tokenization)
- ✅ Sample corpus ile başla, gerçek dataset'e geç

### 2. Validation ve Evaluation
- ✅ Validation split (10% default)
- ✅ Perplexity metrikleri
- ✅ Accuracy metrikleri
- ✅ Early stopping (patience: 5 epochs)

### 3. Best Model Saving
- ✅ En iyi validation loss'a göre model kaydetme
- ✅ Checkpoint'ler: step-based + best model

### 4. Proper Hyperparameters
- ✅ Learning rate: 3e-4 (standart)
- ✅ Warmup steps: 100
- ✅ Weight decay: 0.1
- ✅ Gradient clipping: 1.0

---

## 📋 Yeni Özellikler

### 1. Gerçek Text Data Loader

**Dosya**: `mm_rec/data/text_data_loader.py`

**Özellikler**:
- Character-level tokenization
- Sliding window sequence generation
- Train/validation split
- Sample corpus oluşturma

**Kullanım**:
```python
from mm_rec.data.text_data_loader import create_data_loaders

train_loader, val_loader, tokenizer = create_data_loaders(
    train_texts=train_texts,
    val_texts=val_texts,
    vocab_size=5000,
    seq_len=512,
    batch_size=4
)
```

### 2. Evaluation Metrikleri

**Dosya**: `mm_rec/training/evaluation.py`

**Metrikler**:
- **Loss**: Cross-entropy loss
- **Perplexity**: exp(loss)
- **Accuracy**: Token-level accuracy

**Kullanım**:
```python
from mm_rec.training.evaluation import evaluate_model, print_evaluation_metrics

val_metrics = evaluate_model(
    model=model,
    data_loader=val_loader,
    criterion=criterion,
    device=device
)
print_evaluation_metrics(val_metrics)
```

### 3. Early Stopping

**Özellikler**:
- Validation loss'a göre early stopping
- Patience: 5 epochs (default)
- Best model otomatik kaydedilir

---

## 🚀 Kullanım

### Temel Eğitim (Sample Corpus ile)

```bash
python mm_rec/scripts/train_base_model.py \
    --config tiny \
    --epochs 10 \
    --batch-size 4 \
    --seq-len 256 \
    --use-sample-corpus
```

### Gerçek Dataset ile

```bash
# 1. Dataset hazırla (text dosyaları bir dizinde)
mkdir -p data/train
# text dosyalarını data/train/ dizinine koy

# 2. Eğitimi başlat
python mm_rec/scripts/train_base_model.py \
    --config tiny \
    --data-dir data/train \
    --epochs 20 \
    --val-split 0.1 \
    --early-stopping-patience 5 \
    --save-best-model
```

### Parametreler

- `--config`: Model konfigürasyonu (tiny, mini, small, etc.)
- `--data-dir`: Dataset dizini (None = sample corpus)
- `--use-sample-corpus`: Sample corpus kullan (default: True)
- `--val-split`: Validation split ratio (default: 0.1)
- `--early-stopping-patience`: Early stopping patience (default: 5)
- `--save-best-model`: Best model kaydet (default: True)

---

## 📊 Çıktılar

### Checkpoint'ler

1. **Step-based checkpoints**: `checkpoints/{config}/checkpoint_step_{step}.pt`
   - Her 100 step'te bir kaydedilir
   - Training devam ederken kullanılır

2. **Best model**: `checkpoints/{config}/best_model.pt`
   - En iyi validation loss'a sahip model
   - Early stopping veya final checkpoint olarak kullanılır

3. **Final checkpoint**: `checkpoints/{config}/final_checkpoint.pt`
   - Tüm epoch'lar tamamlandığında kaydedilir

### Log Çıktıları

```
📚 Preparing data...
✅ Using sample corpus: checkpoints/sample_corpus.txt
✅ Vocabulary built: 150 tokens
✅ Train dataset: 1000 sequences
✅ Validation dataset: 100 sequences

🚀 Training started...

Epoch 1/10: 100%|████████| 250/250 [00:30<00:00, loss=8.234, lr=3.00e-04]

📊 Epoch 1 completed - Avg Loss: 8.234

Validation Metrics:
  Loss: 7.891
  Perplexity: 2654.23
  Accuracy: 12.34%
  Batches: 25

💾 Best model saved: checkpoints/tiny/best_model.pt
```

---

## ✅ Kalite Kontrol Listesi

### Eğitim Öncesi
- [ ] Dataset hazır (gerçek text veya sample corpus)
- [ ] Validation split belirlendi
- [ ] Hyperparameter'lar ayarlandı
- [ ] Model konfigürasyonu doğru

### Eğitim Sırasında
- [ ] Loss düşüyor mu?
- [ ] Validation loss training loss'tan düşük mü?
- [ ] Perplexity makul değerlerde mi? (çok yüksek değil)
- [ ] Accuracy artıyor mu?
- [ ] Early stopping çalışıyor mu?

### Eğitim Sonrası
- [ ] Best model kaydedildi mi?
- [ ] Final checkpoint kaydedildi mi?
- [ ] Evaluation metrikleri raporlandı mı?
- [ ] Model test edilebilir durumda mı?

---

## 🔍 Sorun Giderme

### Loss Çok Yüksek
- Learning rate'i düşür (1e-4)
- Warmup steps'i artır (200)
- Batch size'ı artır

### Loss Düşmüyor
- Model çok küçük olabilir (config'i kontrol et)
- Data yeterli değil (daha fazla text ekle)
- Learning rate çok düşük (3e-4'e çıkar)

### Validation Loss Artıyor (Overfitting)
- Dropout'u artır (0.2)
- Weight decay'ı artır (0.2)
- Early stopping patience'ı azalt (3)

### Memory Hatası
- Batch size'ı azalt (2)
- Sequence length'i azalt (256)
- Gradient checkpointing kullan

---

## 📈 İyileştirme Önerileri

### Kısa Vadede
1. ✅ Gerçek text data loader (tamamlandı)
2. ✅ Validation ve evaluation (tamamlandı)
3. ✅ Early stopping (tamamlandı)
4. ⏳ Gerçek dataset entegrasyonu (OpenWebText, etc.)

### Orta Vadede
1. Word-level tokenization (BPE, SentencePiece)
2. Daha fazla evaluation metrikleri (BLEU, ROUGE)
3. Learning rate scheduling iyileştirmeleri
4. Mixed precision training

### Uzun Vadede
1. Distributed training desteği
2. Advanced data augmentation
3. Curriculum learning
4. Multi-task learning

---

**Hazırlayan**: MM-Rec Training Team  
**Tarih**: 2025-01-27  
**Durum**: ✅ Kaliteli eğitim altyapısı hazır
