# MM-Rec 7B Model - Parametre Sayısı Analizi

**Tarih**: 2025-12-08  
**Model Konfigürasyonu**: 7B (7.38B parameters)

---

## 📊 Toplam Parametre Sayısı

**7,380,553,728 parametre** (≈ **7.38 B**)

### Model Boyutu
- **BF16**: 13.75 GB
- **FP32**: 27.49 GB

---

## 📋 Detaylı Parametre Dağılımı

### 1. Embedding Layer
- **Parametreler**: 131,072,000
- **Hesaplama**: `vocab_size × model_dim = 32,000 × 4,096`
- **Not**: LM Head ile tied (aynı ağırlıklar paylaşılıyor)

### 2. LM Head
- **Parametreler**: 131,072,000 (Embedding ile tied - tekrar sayılmıyor)
- **Hesaplama**: `model_dim × vocab_size = 4,096 × 32,000`

### 3. MMRecBlock (Her Blok - 24 Katman)

#### 3.1 QKVZ Projeksiyonları
- **W_q**: 16,781,312 (4,096 × 4,096 + bias)
- **W_k**: 16,781,312 (4,096 × 4,096 + bias)
- **W_v**: 16,781,312 (4,096 × 4,096 + bias)
- **W_z**: 16,781,312 (4,096 × 4,096 + bias)
- **W_g**: 16,781,312 (4,096 × 4,096 + bias)
- **Subtotal**: 83,906,560

#### 3.2 MDI (Memory Decay/Integration)
- **Parametreler**: 50,345,984
- **Bileşenler**:
  - W_g: 4,096 × 2 × 4,096 = 33,554,432 (gating)
  - W_gamma: 4,096 × 1,024 + 1,024 × 4,096 = 8,388,608 (decay)
  - W_context: 4,096 × 1,024 + 1,024 × 4,096 = 8,388,608 (context modulation)

#### 3.3 MultiMemoryAttention
- **Parametreler**: 33,562,624
- **Bileşenler**:
  - W_q: 4,096 × 4,096 = 16,781,312
  - W_o: 4,096 × 4,096 = 16,781,312

#### 3.4 FFN (Feed-Forward Network)
- **Parametreler**: 134,238,208
- **Hesaplama**: 
  - Up projection: `model_dim × ffn_dim = 4,096 × 16,384 = 67,108,864`
  - Down projection: `ffn_dim × model_dim = 16,384 × 4,096 = 67,108,864`
  - Bias: 16,384 + 4,096 = 20,480

#### 3.5 Normalization
- **Norm1**: 4,096 (RMSNorm weight)
- **Norm2**: 4,096 (RMSNorm weight)
- **Subtotal**: 8,192

#### 3.6 Block Toplamı
- **Her Blok**: 302,061,568 parametre
- **24 Blok**: 7,249,477,632 parametre

### 4. Final Normalization
- **Parametreler**: 4,096 (RMSNorm weight)

---

## 📈 Karşılaştırma

| Model | Parametre Sayısı | Fark |
|-------|------------------|------|
| LLaMA 7B | ~7.0B | - |
| MM-Rec 7B | 7.38B | +0.38B (+5.4%) |

**Not**: MM-Rec'in fazladan parametreleri:
- MDI modülü (decay ve gating mekanizması)
- MultiMemoryAttention (O(M) complexity attention)
- Ekstra gating projeksiyonları (W_g)

---

## 🧮 Parametre Hesaplama Formülleri

### Embedding
```
params = vocab_size × model_dim
      = 32,000 × 4,096
      = 131,072,000
```

### QKVZ Projeksiyonları (her biri)
```
params = model_dim × model_dim + model_dim (bias)
      = 4,096 × 4,096 + 4,096
      = 16,781,312
```

### FFN
```
params = (model_dim × ffn_dim) + (ffn_dim × model_dim) + biases
      = (4,096 × 16,384) + (16,384 × 4,096) + (16,384 + 4,096)
      = 67,108,864 + 67,108,864 + 20,480
      = 134,238,208
```

### Toplam (Embedding hariç)
```
total = (block_params × num_layers) + final_norm
      = (302,061,568 × 24) + 4,096
      = 7,249,477,632 + 4,096
      = 7,249,481,728
```

### Toplam (Embedding dahil)
```
total = embedding + blocks + final_norm
      = 131,072,000 + 7,249,481,728
      = 7,380,553,728
```

---

## 💾 Bellek Gereksinimleri

### Model Ağırlıkları
- **BF16**: 13.75 GB
- **FP32**: 27.49 GB

### Eğitim (Gradient + Optimizer States)
- **AdamW Optimizer**: ~2x model size (FP32)
  - Momentum: 27.49 GB
  - Variance: 27.49 GB
  - **Toplam**: ~55 GB (FP32)
- **Gradient**: 27.49 GB (FP32)
- **Activations**: Sequence length'e bağlı (chunking ile azaltılabilir)

### Toplam Eğitim Belleği (Tahmini)
- **FP32 Training**: ~82 GB (model + gradients + optimizer)
- **Mixed Precision (BF16)**: ~55 GB (model BF16, optimizer FP32)

---

## 📝 Notlar

1. **Tied Weights**: Embedding ve LM Head aynı ağırlıkları paylaşıyor (weight tying). Bu, parametre sayısını azaltır ve model performansını artırır.

2. **MDI Parametreleri**: MM-Rec'in benzersiz özelliği olan Memory Decay/Integration modülü, ekstra parametreler ekler (~50M per block).

3. **Attention Parametreleri**: MultiMemoryAttention, standart Transformer attention'dan daha az parametre kullanır çünkü Key ve Value projeksiyonları HDS memory'den gelir (O(M) complexity).

4. **FFN Boyutu**: FFN dimension, model_dim'den 4x daha büyük (16,384 vs 4,096), bu standart LLM pratiğidir.

---

## 🔍 Doğrulama

Parametre sayısı `model.get_num_params()` metodu ile doğrulanmıştır:

```python
from mm_rec.model import MMRecModel

model = MMRecModel(
    vocab_size=32000,
    model_dim=4096,
    num_layers=24,
    num_heads=32,
    ffn_dim=16384
)

total_params = model.get_num_params()
print(f"Total: {total_params:,} ({total_params / 1e9:.2f}B)")
# Output: Total: 7,380,553,728 (7.38B)
```

---

**Son Güncelleme**: 2025-12-08

