# MM-Rec Model Evaluation Guide

## Eğitim Sonrası Beklenenler

### 1. Loss Trend Analizi

Eğitim sırasında loss değerlerinin azalması beklenir:

```
✅ İyi Trend:
   Step 1: Loss = 11.59
   Step 2: Loss = 11.62 (küçük artış normal)
   Step 3: Loss = 11.45 (azalma başladı!)
   Step 4: Loss = 11.37 (devam ediyor)
   
📊 Beklenen Pattern:
   - İlk adımlarda loss yüksek (11-12 arası)
   - Yavaş yavaş azalma (10-11 arası)
   - Uzun eğitimde 8-10 arası beklenir
```

### 2. Checkpoint Dosyaları

Eğitim sonunda checkpoint'ler kaydedilir:

```
checkpoints/
├── checkpoint_step_100.pt
├── checkpoint_step_200.pt
└── checkpoint_step_500.pt
```

Her checkpoint içerir:
- `model_state_dict`: Model ağırlıkları
- `optimizer_state_dict`: Optimizer durumu
- `scheduler_state_dict`: LR scheduler durumu
- `step`: Eğitim adımı
- `loss`: Son loss değeri
- `metrics`: Özet metrikler

### 3. Test Verisi Analizi

**Mevcut Test Verisi:**
- Dosya: `data/chat_data_real.jsonl`
- Format: JSONL (her satır bir konuşma)
- İçerik: 1400 konuşma
- Yapı: `{"messages": [{"role": "system/user/assistant", "content": "..."}]}`

**Örnek Konuşma:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
  ]
}
```

### 4. Evaluation Metrikleri

Model değerlendirmesi için şu metrikler hesaplanır:

#### a) Perplexity (PPL)
- **Tanım**: Model'in tahmin belirsizliği
- **İyi Değer**: 10-50 arası (daha düşük = daha iyi)
- **Başlangıç**: ~100,000 (random)
- **Eğitim Sonrası**: 20-100 arası beklenir

#### b) Loss
- **Tanım**: Cross-entropy loss
- **İyi Değer**: 8-12 arası (daha düşük = daha iyi)
- **Başlangıç**: ~11-12
- **Eğitim Sonrası**: 8-10 arası beklenir

#### c) Token-Level Accuracy
- **Tanım**: Doğru tahmin edilen token yüzdesi
- **İyi Değer**: %30-50 arası (daha yüksek = daha iyi)
- **Başlangıç**: ~0.04% (random)
- **Eğitim Sonrası**: %20-40 arası beklenir

### 5. Evaluation Script Kullanımı

```bash
# Temel evaluation
python3 mm_rec/scripts/evaluate.py \
    --checkpoint ./checkpoints/checkpoint_step_500.pt \
    --test_data ./data/chat_data_real.jsonl \
    --max_samples 100 \
    --max_length 512

# Generation örnekleri ile
python3 mm_rec/scripts/evaluate.py \
    --checkpoint ./checkpoints/checkpoint_step_500.pt \
    --test_data ./data/chat_data_real.jsonl \
    --max_samples 50 \
    --generate_samples \
    --num_samples 5
```

### 6. Beklenen Sonuçlar

#### Başarılı Eğitim İşaretleri:

✅ **Loss Azalması:**
```
Step 1:  11.59
Step 10: 11.20
Step 50: 10.50
Step 100: 9.80
```

✅ **Perplexity İyileşmesi:**
```
Step 1:  ~100,000
Step 10: ~80,000
Step 50: ~40,000
Step 100: ~20,000
```

✅ **Accuracy Artışı:**
```
Step 1:  ~0.04%
Step 10: ~0.5%
Step 50: ~5%
Step 100: ~15%
```

#### Dikkat Edilmesi Gerekenler:

⚠️ **Loss Artıyorsa:**
- Learning rate çok yüksek olabilir
- Gradient explosion olabilir
- Checkpoint'ten devam ederken LR'ı düşürün

⚠️ **Loss Değişmiyorsa:**
- Learning rate çok düşük olabilir
- Model donmuş olabilir
- Gradient flow kontrol edin

⚠️ **NaN/Inf Loss:**
- Numerical instability
- Log-space hesaplamaları kontrol edin
- Gradient clipping aktif olmalı

### 7. Test Verisi Bölme

**Önerilen Split:**
- **Train**: %80 (1120 konuşma)
- **Validation**: %10 (140 konuşma)
- **Test**: %10 (140 konuşma)

**Split Script:**
```python
# mm_rec/scripts/split_data.py
import json
from pathlib import Path
import random

data_file = Path('data/chat_data_real.jsonl')
conversations = []

with open(data_file, 'r') as f:
    for line in f:
        if line.strip():
            conversations.append(json.loads(line))

random.shuffle(conversations)

train_split = int(len(conversations) * 0.8)
val_split = int(len(conversations) * 0.9)

train_data = conversations[:train_split]
val_data = conversations[train_split:val_split]
test_data = conversations[val_split:]

# Save splits
for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
    with open(f'data/{name}_data.jsonl', 'w') as f:
        for conv in data:
            f.write(json.dumps(conv) + '\n')
```

### 8. Evaluation Sonrası Analiz

**Metrikler Dosyası:**
```json
{
  "avg_loss": 9.85,
  "avg_perplexity": 18950.23,
  "avg_accuracy": 0.1245,
  "total_tokens": 125000,
  "num_valid": 140,
  "total_conversations": 140
}
```

**Karşılaştırma:**
- **Baseline (Random)**: Loss ~11.5, PPL ~100K, Accuracy ~0.04%
- **Eğitim Sonrası**: Loss ~9-10, PPL ~20K-50K, Accuracy ~10-20%
- **İyi Model**: Loss <9, PPL <10K, Accuracy >30%

### 9. Sonraki Adımlar

1. **Evaluation Çalıştır:**
   ```bash
   python3 mm_rec/scripts/evaluate.py \
       --checkpoint ./checkpoints/checkpoint_step_500.pt \
       --test_data ./data/chat_data_real.jsonl
   ```

2. **Metrikleri İncele:**
   - Loss trend'i kontrol et
   - Perplexity değerlerini karşılaştır
   - Accuracy artışını gözlemle

3. **Generation Örnekleri:**
   ```bash
   python3 mm_rec/scripts/evaluate.py \
       --checkpoint ./checkpoints/checkpoint_step_500.pt \
       --generate_samples --num_samples 10
   ```

4. **Model İyileştirme:**
   - Loss yüksekse: Daha uzun eğitim
   - Accuracy düşükse: Learning rate ayarı
   - Overfitting varsa: Dropout artır

### 10. Önemli Notlar

- **CPU Eğitimi Yavaş**: Her adım ~60-70 saniye
- **GPU Önerilir**: 10-100x hızlanma beklenir
- **Checkpoint Kaydet**: Her 100 adımda bir kaydet
- **Metrikleri İzle**: Loss trend'i önemli
- **Test Verisi**: Gerçek veri kullan (synthetic değil)

---

## Hızlı Başlangıç

```bash
# 1. Eğitim tamamlandıktan sonra
cd /home/onur/workspace/mm-rec

# 2. Son checkpoint'i bul
ls -lt checkpoints/ | head -5

# 3. Evaluation çalıştır
python3 mm_rec/scripts/evaluate.py \
    --checkpoint ./checkpoints/checkpoint_step_500.pt \
    --test_data ./data/chat_data_real.jsonl \
    --max_samples 100 \
    --generate_samples

# 4. Sonuçları incele
cat evaluation_results.json
```

