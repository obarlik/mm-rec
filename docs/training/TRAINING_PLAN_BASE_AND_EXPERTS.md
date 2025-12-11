# MM-Rec Base Model ve Expert Eğitim Planı

## 📋 GENEL BAKIŞ

Bu plan, temel bir pretrained model oluşturup, bunu kullanarak domain-specific uzmanları (experts) eğitmeyi hedefler.

### Mimari
- **Base Model**: Genel amaçlı MM-Rec model (256 channel, 8-16 layer)
- **Expert Models**: Base modelden türetilen domain-specific modeller
  - **Text Expert**: Genel metin verileri için
  - **Code Expert**: Kod verileri için
- **Fusion Layer**: İki expert'i birleştiren katman (256+256 → 512)

---

## 🎯 STAGE 1: BASE MODEL PRE-TRAINING

### Amaç
Genel dil bilgisini öğrenen temel bir MM-Rec modeli oluşturmak.

### Model Konfigürasyonu
```python
BaseMMRecModel:
  - vocab_size: 32000
  - model_dim: 256
  - num_layers: 12  # Base için yeterli
  - num_heads: 8
  - ffn_dim: 1024
  - num_memories: 1
  - mem_dim: 256
  - max_seq_len: 32768
  - Total params: ~50-60M
```

### Veri Kaynakları
1. **WikiText-103** (genel metin)
2. **OpenWebText** (web crawl, genel metin)
3. **Books** (Project Gutenberg, genel metin)
4. **Code** (GitHub, genel kod)

**Strateji**: Tüm veri kaynaklarını karıştır (mixed domain)

### Eğitim Parametreleri
```yaml
Base Pre-training:
  - max_steps: 100000
  - batch_size: 8
  - seq_len: 2048
  - gradient_accumulation_steps: 8
  - learning_rate: 3e-4
  - weight_decay: 0.1
  - warmup_steps: 5000
  - checkpoint_interval: 10000
  - optimizer: AdamW
  - scheduler: CosineAnnealingLR
  - mixed_precision: True (CPU AMP)
  - gradient_checkpointing: True
```

### Çıktı
- **Checkpoint**: `checkpoints_base/base_model_step_100000.pt`
- **Model State**: Tüm ağırlıklar (embedding, blocks, norm, lm_head)
- **Metadata**: Training loss, perplexity, step count

---

## 🎯 STAGE 2: EXPERT SPECIALIZATION

### Amaç
Base modelden türetilen uzmanları domain-specific verilerle fine-tune etmek.

### Strateji: Knowledge Transfer

#### 2.1 Text Expert Fine-tuning

**Base Model'den Transfer:**
```python
# Base model ağırlıklarını yükle
base_state = torch.load('checkpoints_base/base_model_step_100000.pt')

# Text Expert'i oluştur
text_expert = ExpertModule(
    model_dim=256,
    num_layers=12,  # Base ile aynı
    ...
)

# Base model ağırlıklarını kopyala
text_expert.blocks.load_state_dict(base_state['blocks'])
text_expert.norm.load_state_dict(base_state['norm'])
```

**Veri Kaynakları:**
- WikiText-103 (text only)
- OpenWebText (text only)
- Books (text only)
- **NOT**: Code verisi kullanma

**Eğitim Parametreleri:**
```yaml
Text Expert Fine-tuning:
  - max_steps: 50000
  - batch_size: 8
  - seq_len: 2048
  - learning_rate: 1e-4  # Daha düşük (fine-tuning)
  - weight_decay: 0.01
  - warmup_steps: 2000
  - checkpoint_interval: 10000
  - freeze_embedding: False  # Embedding'i de fine-tune et
  - freeze_first_n_layers: 0  # Tüm katmanları fine-tune et
```

#### 2.2 Code Expert Fine-tuning

**Base Model'den Transfer:**
```python
# Base model ağırlıklarını yükle
base_state = torch.load('checkpoints_base/base_model_step_100000.pt')

# Code Expert'i oluştur
code_expert = ExpertModule(
    model_dim=256,
    num_layers=12,  # Base ile aynı
    ...
)

# Base model ağırlıklarını kopyala
code_expert.blocks.load_state_dict(base_state['blocks'])
code_expert.norm.load_state_dict(base_state['norm'])
```

**Veri Kaynakları:**
- GitHub Code (Python, JavaScript, etc.)
- Stack Overflow (code snippets)
- **NOT**: Text verisi kullanma

**Eğitim Parametreleri:**
```yaml
Code Expert Fine-tuning:
  - max_steps: 50000
  - batch_size: 8
  - seq_len: 2048
  - learning_rate: 1e-4  # Daha düşük (fine-tuning)
  - weight_decay: 0.01
  - warmup_steps: 2000
  - checkpoint_interval: 10000
  - freeze_embedding: False
  - freeze_first_n_layers: 0
```

---

## 🎯 STAGE 3: FUSION LAYER TRAINING

### Amaç
İki expert'i birleştiren Fusion Layer'ı eğitmek.

### Strateji: Joint Training

**Model:**
```python
# Her iki expert'i yükle
text_expert = ExpertModule(...)
text_expert.load_state_dict(torch.load('checkpoints_text/text_expert_step_50000.pt'))

code_expert = ExpertModule(...)
code_expert.load_state_dict(torch.load('checkpoints_code/code_expert_step_50000.pt'))

# Fusion Layer'ı oluştur
fusion = FusionLayer(expert_dim=256, fused_dim=512)

# MMRec100M modelini oluştur
model = MMRec100M(
    vocab_size=32000,
    expert_dim=256,
    num_layers=12,
    ...
)

# Expert'leri yükle
model.text_expert = text_expert
model.code_expert = code_expert
model.fusion = fusion

# Expert'leri freeze et, sadece fusion'ı eğit
for param in model.text_expert.parameters():
    param.requires_grad = False
for param in model.code_expert.parameters():
    param.requires_grad = False
```

**Veri Kaynakları:**
- **Mixed**: Text ve Code verilerini karıştır
- Her batch'te hem text hem code örnekleri olsun

**Eğitim Parametreleri:**
```yaml
Fusion Training:
  - max_steps: 20000
  - batch_size: 8
  - seq_len: 2048
  - learning_rate: 5e-4  # Fusion için biraz daha yüksek
  - weight_decay: 0.01
  - warmup_steps: 1000
  - checkpoint_interval: 5000
  - experts_frozen: True  # Expert'ler frozen
```

---

## 📊 TRAINING PIPELINE

### 1. Base Model Pre-training Script
```bash
python3 mm_rec/scripts/pretrain_base.py \
    --data_dir ./data/pretrain/mixed \
    --max_steps 100000 \
    --batch_size 8 \
    --seq_len 2048 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-4 \
    --checkpoint_dir ./checkpoints_base \
    --use_amp \
    --use_gradient_checkpointing
```

### 2. Text Expert Fine-tuning Script
```bash
python3 mm_rec/scripts/finetune_expert.py \
    --base_checkpoint ./checkpoints_base/base_model_step_100000.pt \
    --expert_type text \
    --data_dir ./data/pretrain/text \
    --max_steps 50000 \
    --batch_size 8 \
    --seq_len 2048 \
    --learning_rate 1e-4 \
    --checkpoint_dir ./checkpoints_text
```

### 3. Code Expert Fine-tuning Script
```bash
python3 mm_rec/scripts/finetune_expert.py \
    --base_checkpoint ./checkpoints_base/base_model_step_100000.pt \
    --expert_type code \
    --data_dir ./data/pretrain/code \
    --max_steps 50000 \
    --batch_size 8 \
    --seq_len 2048 \
    --learning_rate 1e-4 \
    --checkpoint_dir ./checkpoints_code
```

### 4. Fusion Layer Training Script
```bash
python3 mm_rec/scripts/train_fusion.py \
    --text_expert_checkpoint ./checkpoints_text/text_expert_step_50000.pt \
    --code_expert_checkpoint ./checkpoints_code/code_expert_step_50000.pt \
    --data_dir ./data/pretrain/mixed \
    --max_steps 20000 \
    --batch_size 8 \
    --seq_len 2048 \
    --learning_rate 5e-4 \
    --checkpoint_dir ./checkpoints_fusion
```

---

## 🔄 KNOWLEDGE TRANSFER MECHANISM

### Base → Expert Transfer

**Yöntem 1: Full Copy**
```python
# Tüm ağırlıkları kopyala
expert.blocks.load_state_dict(base_model.blocks.state_dict())
expert.norm.load_state_dict(base_model.norm.state_dict())
```

**Yöntem 2: Partial Freeze (Optional)**
```python
# İlk N katmanı freeze et
for i in range(freeze_first_n_layers):
    for param in expert.blocks[i].parameters():
        param.requires_grad = False
```

**Yöntem 3: Learning Rate Scaling**
```python
# İlk katmanlar için daha düşük LR
optimizer = optim.AdamW([
    {'params': expert.blocks[:freeze_first_n_layers].parameters(), 'lr': 1e-5},
    {'params': expert.blocks[freeze_first_n_layers:].parameters(), 'lr': 1e-4},
])
```

---

## 📈 BEKLENEN SONUÇLAR

### Base Model
- **Perplexity**: ~15-20 (genel metin)
- **Loss**: ~2.5-3.0
- **Domain Coverage**: Genel (text + code)

### Text Expert
- **Perplexity**: ~10-15 (text only)
- **Loss**: ~2.0-2.5
- **Domain**: Text-optimized

### Code Expert
- **Perplexity**: ~12-18 (code only)
- **Loss**: ~2.2-2.8
- **Domain**: Code-optimized

### Fusion Model
- **Perplexity**: ~8-12 (mixed)
- **Loss**: ~1.8-2.2
- **Domain**: Text + Code (best of both)

---

## 🎯 ÖNEMLİ NOTLAR

1. **Base Model Kalitesi**: Base model ne kadar iyi olursa, expert'ler de o kadar iyi olur
2. **Veri Kalitesi**: Domain-specific verilerin kalitesi kritik
3. **Learning Rate**: Fine-tuning için daha düşük LR kullan
4. **Freezing Strategy**: İlk katmanları freeze etmek overfitting'i önleyebilir
5. **Fusion Training**: Expert'leri freeze edip sadece fusion'ı eğit

---

## 📁 CHECKPOINT YAPISI

```
checkpoints/
├── base/
│   ├── base_model_step_10000.pt
│   ├── base_model_step_20000.pt
│   └── base_model_step_100000.pt  # Final
├── text/
│   ├── text_expert_step_10000.pt
│   └── text_expert_step_50000.pt  # Final
├── code/
│   ├── code_expert_step_10000.pt
│   └── code_expert_step_50000.pt  # Final
└── fusion/
    ├── fusion_model_step_5000.pt
    └── fusion_model_step_20000.pt  # Final
```

---

## 🚀 HIZLI BAŞLANGIÇ

1. **Base Model Eğitimi** (100K steps, ~2-3 hafta CPU'da)
2. **Text Expert Fine-tuning** (50K steps, ~1 hafta)
3. **Code Expert Fine-tuning** (50K steps, ~1 hafta)
4. **Fusion Training** (20K steps, ~3-4 gün)

**Toplam Süre**: ~1 ay (CPU'da, optimizasyonlarla)

