# MM-Rec Progressive Training Stratejisi

**Tarih**: 2025-01-27  
**Amaç**: En küçük temel modelden başlayarak büyük modellere evrilme ve uzmanlık alanlarıyla fine-tuning

---

## 🎯 Strateji Özeti

### 1. En Küçük Temel Model (Başlangıç)

**Tiny Base Model**:
- Parametre: **230K** (0.23M)
- Bellek (FP16): **0.44 MB**
- Konfigürasyon:
  - Vocab: 5,000
  - Model Dim: 128
  - Layers: 4
  - Heads: 4
  - Max Seq Len: 1,024

**Neden Bu Model?**:
- ✅ Çok hızlı eğitilir (test ve doğrulama için)
- ✅ Minimum kaynak gerektirir
- ✅ Architecture'ı doğrulamak için ideal
- ✅ Progressive training'in temelini oluşturur

### 2. Progressive Training Sequence

```
Tiny (0.23M) 
  ↓ (Weight Transfer + Training)
Mini (2M)
  ↓ (Weight Transfer + Training)
Small (10M)
  ↓ (Weight Transfer + Training)
Base (52M)
  ↓ (Weight Transfer + Training)
Medium (200M+)
  ↓ (Weight Transfer + Training)
Large (500M+)
  ↓ (Weight Transfer + Training)
7B (7.38B)
```

### 3. Her Stage'de Yapılacaklar

1. **Weight Transfer**: Önceki modelden weight'leri transfer et
2. **Training**: Yeni model boyutunda eğitim
3. **Checkpointing**: Model checkpoint kaydet
4. **Fine-tuning (Opsiyonel)**: Uzmanlık alanlarıyla fine-tune et

---

## 📋 Model Konfigürasyonları

### Tiny Base (Başlangıç)
```python
vocab_size=5000
model_dim=128
num_layers=4
num_heads=4
max_seq_len=1024
use_hem=True
use_dpg=False
use_uboo=False
```

### Mini Base
```python
vocab_size=10000
model_dim=256
num_layers=6
num_heads=4
max_seq_len=2048
use_hem=True
use_dpg=False
use_uboo=False
```

### Small Base
```python
vocab_size=20000
model_dim=512
num_layers=8
num_heads=8
max_seq_len=4096
use_hem=True
use_dpg=False
use_uboo=False
```

### Base Base
```python
vocab_size=32000
model_dim=1024
num_layers=12
num_heads=8
max_seq_len=8192
use_hem=True
use_dpg=False
use_uboo=False
```

### Medium Base
```python
vocab_size=32000
model_dim=2048
num_layers=16
num_heads=16
max_seq_len=16384
use_hem=True
use_dpg=True  # DPG aktif
use_uboo=False
```

### Large Base
```python
vocab_size=32000
model_dim=3072
num_layers=20
num_heads=24
max_seq_len=32768
use_hem=True
use_dpg=True
use_uboo=True  # UBÖO aktif
```

### 7B (Hedef)
```python
vocab_size=32000
model_dim=4096
num_layers=24
num_heads=32
max_seq_len=32768
use_hem=True
use_dpg=True
use_uboo=True
```

---

## 🚀 Kullanım

### 1. En Küçük Temel Modeli Eğitme

```bash
# Tiny model eğitimi
python mm_rec/scripts/train_base_model.py \
    --config tiny \
    --output-dir checkpoints \
    --epochs 10 \
    --batch-size 4 \
    --seq-len 512 \
    --lr 3e-4
```

### 2. Progressive Training (Tiny -> 7B)

```bash
# Progressive training (tüm sequence)
python mm_rec/scripts/train_base_model.py \
    --progressive \
    --start-config tiny \
    --end-config 7b \
    --epochs-per-stage 5 \
    --output-dir checkpoints
```

### 3. Belirli Stage'den Devam

```bash
# Mini model eğitimi (Tiny'dan sonra)
python mm_rec/scripts/train_base_model.py \
    --config mini \
    --output-dir checkpoints \
    --epochs 10 \
    --resume-from checkpoints/tiny/final_checkpoint.pt
```

### 4. Expert Fine-tuning

```bash
# Pretrained model'i uzmanlık alanıyla fine-tune et
python mm_rec/scripts/finetune_expert.py \
    --checkpoint checkpoints/base/final_checkpoint.pt \
    --expert-name medical \
    --output-dir experts \
    --epochs 5 \
    --lr 1e-5
```

---

## 🔄 Weight Transfer Stratejisi

### Akıllı Weight Transfer

1. **Embedding**: Ortak vocab kısmı transfer edilir
2. **Blocks**: Mevcut layer'lar transfer edilir, yeni layer'lar random init
3. **Layer Norm**: Aynı dimension'lar transfer edilir
4. **Output Head**: Ortak vocab kısmı transfer edilir

### Örnek: Tiny -> Mini Transfer

```python
from mm_rec.utils.model_upscaling import upscale_model
from mm_rec.configs.base_model_configs import TINY_BASE_CONFIG, MINI_BASE_CONFIG

# Tiny model yükle
tiny_model = load_checkpoint("checkpoints/tiny/final_checkpoint.pt")

# Mini model'e upscale et
mini_model = upscale_model(
    tiny_model,
    MINI_BASE_CONFIG,
    device=device
)

# Mini model'i eğit
train_model(mini_model, ...)
```

---

## 📊 Fine-tuning Stratejisi

### Expert Model Oluşturma

1. **Base Model Yükle**: Pretrained base model'i yükle
2. **Freeze Layers**: İlk N layer'ı freeze et (knowledge preservation)
3. **Train Last Layers**: Son M layer'ı train et (task adaptation)
4. **Low Learning Rate**: Fine-tuning için düşük LR (1e-5)

### Örnek Expert Alanları

- **Medical**: Tıbbi metinler
- **Code**: Kod üretimi
- **Math**: Matematik problemleri
- **Legal**: Hukuki metinler
- **Finance**: Finansal analiz

---

## ✅ Avantajlar

1. **Hızlı İterasyon**: Küçük modelle hızlı test
2. **Kaynak Verimliliği**: Progressive training kaynak tasarrufu sağlar
3. **Knowledge Transfer**: Küçük modelden büyük modele bilgi transferi
4. **Esneklik**: Her stage'de fine-tuning yapılabilir
5. **Scalability**: 7B'ye kadar ölçeklenebilir

---

## 📝 Sonraki Adımlar

1. ✅ En küçük temel model (Tiny) eğitimi
2. ⏳ Weight transfer mekanizması testi
3. ⏳ Progressive training pipeline
4. ⏳ Expert fine-tuning altyapısı
5. ⏳ Gerçek dataset entegrasyonu

---

**Hazırlayan**: MM-Rec Training Team  
**Tarih**: 2025-01-27
