# MM-Rec Model Training Methodology

## Problem: Mevcut Eğitim Yaklaşımı

### Şu Anki Durum
- **Veri**: Sadece 1400 konuşma (çok az!)
- **Format**: Chat format (SFT - Supervised Fine-Tuning)
- **Aşama**: Fine-tuning (pre-training yok)
- **Sonuç**: Model İngilizce'yi öğrenemiyor, sadece format öğreniyor

### Sorun
1. **Pre-training eksik**: Model sıfırdan eğitilmiyor, sadece fine-tuning yapılıyor
2. **Veri yetersiz**: 1400 konuşma çok az (LLaMA 1T+ token kullanıyor)
3. **Yanlış aşama**: Model önce İngilizce'yi öğrenmeli, sonra chat format'ı

---

## Doğru LLM Eğitim Metodolojisi

### Aşama 1: Pre-training (Temel Öğrenme)

**Amaç**: Model'e İngilizce dilini, genel bilgiyi, dünya bilgisini öğretmek

**Veri Kaynakları**:
1. **CommonCrawl**: Web crawl verisi (trilyonlarca token)
2. **Wikipedia**: Yapılandırılmış bilgi (milyarlarca token)
3. **Books**: Kitap verisi (milyarlarca token)
4. **Code**: Kod verisi (GitHub, Stack Overflow)
5. **Academic Papers**: Akademik makaleler

**Örnek Veri Miktarları**:
- **LLaMA 7B**: ~1T token (1 trilyon)
- **GPT-3**: ~300B token
- **Mistral 7B**: ~2T token

**Format**: 
- **Raw text**: Sadece metin, chat format değil
- **Next token prediction**: Bir sonraki token'ı tahmin et
- **Self-supervised**: Label yok, sadece metin

**Örnek**:
```
Input: "The capital of France is"
Target: "Paris"
```

### Aşama 2: Fine-tuning (Özel Görevler)

**Amaç**: Pre-trained model'i özel görevlere adapte etmek

**Türler**:
1. **Instruction Tuning**: Talimat takip etme
2. **Chat Fine-tuning**: Sohbet formatı
3. **Domain-specific**: Belirli alan (tıp, hukuk, kod)

**Veri**: 
- Daha az veri (milyonlarca token yeterli)
- Yüksek kaliteli, özenle seçilmiş

---

## MM-Rec İçin Önerilen Eğitim Pipeline

### Phase 1: Pre-training (İngilizce Öğrenme)

**Hedef**: Model'e İngilizce dilini öğretmek

**Veri Kaynakları**:
1. **WikiText-103**: Wikipedia verisi (103M token)
2. **OpenWebText**: Reddit linklerinden toplanan veri
3. **BookCorpus**: Kitap verisi
4. **C4 (Colossal Clean Crawled Corpus)**: Temizlenmiş web verisi

**Minimum Veri Miktarı**:
- **Küçük model (100M)**: 1-10B token
- **Orta model (1B)**: 10-100B token
- **Büyük model (7B)**: 100B-1T token

**Eğitim**:
```python
# Pre-training script
python mm_rec/scripts/pretrain.py \
    --data_dir ./data/pretrain \
    --model_name mmrec_100m \
    --max_steps 100000 \
    --batch_size 4 \
    --seq_len 2048 \
    --learning_rate 3e-4
```

### Phase 2: Instruction Tuning

**Hedef**: Model'e talimat takip etmeyi öğretmek

**Veri**:
- Alpaca dataset
- ShareGPT
- Self-instruct

### Phase 3: Chat Fine-tuning (SFT)

**Hedef**: Chat format'ını öğretmek

**Veri**: 
- Mevcut chat_data_real.jsonl (yeterli)
- Daha fazla çeşitlilik eklenebilir

---

## Hızlı Başlangıç: Pre-training için Veri İndirme

### 1. WikiText-103
```bash
# Hugging Face datasets
python -c "
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
# Save to text files
"
```

### 2. OpenWebText
```bash
# Download from Hugging Face
python -c "
from datasets import load_dataset
dataset = load_dataset('openwebtext', split='train')
"
```

### 3. C4 Dataset
```bash
# Colossal Clean Crawled Corpus
python -c "
from datasets import load_dataset
dataset = load_dataset('c4', 'en', split='train', streaming=True)
"
```

---

## Önerilen Eğitim Stratejisi

### Kısa Vadeli (Hızlı Test)
1. **WikiText-103** ile pre-training (1-2 gün)
2. Sonra chat fine-tuning (mevcut veri)

### Orta Vadeli (İyi Sonuç)
1. **WikiText + OpenWebText** ile pre-training (1 hafta)
2. Instruction tuning
3. Chat fine-tuning

### Uzun Vadeli (En İyi Sonuç)
1. **C4 + Wikipedia + Books** ile pre-training (aylar)
2. Instruction tuning
3. RLHF (Reinforcement Learning from Human Feedback)
4. Chat fine-tuning

---

## Implementation Plan

### 1. Pre-training Script
- Raw text data loader
- Next token prediction loss
- Long sequence support (32K+)
- Checkpointing

### 2. Data Pipeline
- Multiple data sources
- Tokenization
- Shuffling
- Streaming support

### 3. Training Configuration
- Learning rate schedule
- Warmup
- Gradient accumulation
- Mixed precision

---

## Kaynaklar

- **LLaMA Paper**: "LLaMA: Open and Efficient Foundation Language Models"
- **GPT-3 Paper**: "Language Models are Few-Shot Learners"
- **Mistral Paper**: "Mistral 7B"
- **Pile Dataset**: Pre-training dataset collection

---

## Sonuç

**Şu anki sorun**: Model pre-training olmadan fine-tuning yapılıyor
**Çözüm**: Önce pre-training (İngilizce öğrenme), sonra fine-tuning
**Minimum gereksinim**: WikiText-103 ile başla, sonra daha büyük veri setleri

