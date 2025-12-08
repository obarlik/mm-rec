# MM-Rec Model Weight Conversion Guide

**Tarih**: 2025-12-08  
**Amaç**: Mevcut LLM ağırlıklarını MM-Rec mimarisine dönüştürme

---

## 🎯 Genel Bakış

MM-Rec model converter, mevcut Transformer tabanlı LLM ağırlıklarını (LLaMA, GPT, vb.) MM-Rec mimarisine dönüştürmenizi sağlar. Bu işlem **modeli çalıştırmadan**, sadece ağırlıkları analiz ederek yapılır.

### Desteklenen Özellikler

✅ **Uyumlu Bileşenler** (Doğrudan Transfer):
- Embedding layer
- LM Head (output projection)
- FFN (Feed-Forward Network)
- Layer Normalization
- Bazı attention projeksiyonları (Q, O)

⚠️ **Yeni Bileşenler** (Rastgele Başlatma):
- MDI (Memory Decay/Integration) - MM-Rec'e özgü
- HDS (Hierarchical Data Structure) - MM-Rec'e özgü
- Associative Scan - MM-Rec'e özgü
- MultiMemoryAttention (tam olarak farklı)
- Gating projeksiyonları (W_g, W_z)

---

## 📋 Kullanım

### 1. Komut Satırı Kullanımı

```bash
python -m mm_rec.scripts.convert_weights \
    --source llama-7b.pt \
    --output mmrec-7b-converted.pt \
    --vocab_size 32000 \
    --model_dim 4096 \
    --num_layers 24 \
    --num_heads 32
```

### 2. Python API Kullanımı

```python
from mm_rec.model import MMRecModel
from mm_rec.utils.model_converter import convert_model_weights

# MM-Rec model oluştur
model = MMRecModel(
    vocab_size=32000,
    model_dim=4096,
    num_layers=24,
    num_heads=32,
    ffn_dim=16384
)

# Ağırlıkları dönüştür
converted_weights, report = convert_model_weights(
    source_checkpoint_path='llama-7b.pt',
    target_model=model,
    output_path='mmrec-7b-converted.pt',
    strict=False,  # Missing keys için hata verme
    initialize_new=True  # Yeni bileşenleri başlat
)

# Model'e yükle
model.load_state_dict(converted_weights, strict=False)
```

---

## 🔍 Dönüşüm Süreci

### Adım 1: Kaynak Model Analizi

Converter, kaynak modelin state_dict'ini analiz eder:
- Model tipi (LLaMA, GPT, vb.)
- Vocab size
- Model dimension
- Layer sayısı
- Attention head sayısı
- FFN dimension

### Adım 2: Uyumlu Ağırlık Eşleştirme

Converter, kaynak ve hedef modeller arasında uyumlu ağırlıkları bulur:

| Kaynak (LLaMA) | Hedef (MM-Rec) | Uyumluluk |
|----------------|----------------|-----------|
| `embed_tokens.weight` | `embedding.weight` | ✅ 100% |
| `lm_head.weight` | `lm_head.weight` | ✅ 100% |
| `layers.{i}.norm.weight` | `blocks.{i}.norm1.weight` | ✅ 100% |
| `layers.{i}.mlp.up_proj.weight` | `blocks.{i}.ffn.0.weight` | ✅ 100% |
| `layers.{i}.mlp.down_proj.weight` | `blocks.{i}.ffn.3.weight` | ✅ 100% |
| `layers.{i}.attention.q_proj.weight` | `blocks.{i}.multi_mem_attention.W_q.weight` | ✅ 100% |
| `layers.{i}.attention.o_proj.weight` | `blocks.{i}.multi_mem_attention.W_o.weight` | ✅ 100% |

### Adım 3: Yeni Bileşenlerin Başlatılması

MM-Rec'e özgü bileşenler rastgele başlatılır:
- **MDI**: Xavier uniform initialization
- **HDS**: Memory banks sıfırdan başlatılır
- **Associative Scan**: Parametresiz (kernel-based)
- **W_g, W_z**: Benzer ağırlıklardan initialize edilebilir

### Adım 4: Dönüşüm Raporu

Converter, detaylı bir rapor oluşturur:
- Dönüştürülen ağırlık sayısı
- Eksik ağırlıklar
- Shape uyumsuzlukları
- Uyumluluk skorları

---

## 📊 Örnek Dönüşüm Raporu

```json
{
  "total_keys": 245,
  "converted_keys": 180,
  "missing_keys": 65,
  "new_keys": [
    "blocks.0.mdi.W_g.weight",
    "blocks.0.mdi.W_gamma.0.weight",
    "blocks.0.multi_mem_attention.W_q.weight"
  ],
  "source_analysis": {
    "model_type": "llama",
    "vocab_size": 32000,
    "model_dim": 4096,
    "num_layers": 24,
    "num_heads": 32,
    "ffn_dim": 11008
  },
  "compatibility_scores": {
    "embedding.weight": 1.0,
    "blocks.0.ffn.0.weight": 1.0,
    "blocks.0.multi_mem_attention.W_q.weight": 0.8
  }
}
```

---

## ⚠️ Önemli Notlar

### 1. Partial Loading

MM-Rec, Transformer'dan farklı bir mimariye sahip olduğu için **tam uyumluluk beklenmez**. Genellikle:
- **~70-80%** ağırlık transfer edilebilir
- **~20-30%** yeni bileşenler rastgele başlatılır

### 2. Fine-tuning Gerekliliği

Dönüştürülen model, **mutlaka fine-tuning** gerektirir çünkü:
- Yeni bileşenler (MDI, HDS) rastgele başlatılmıştır
- Attention mekanizması farklıdır (MultiMemoryAttention)
- Core formula (h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}) yeni bir yapıdır

### 3. Transfer Learning Stratejisi

Önerilen yaklaşım:
1. **Ağırlıkları transfer et** (uyumlu olanlar)
2. **Kısa bir fine-tuning** yap (1-5 epoch)
3. **Yeni bileşenleri öğrenmesine izin ver**

### 4. Desteklenen Formatlar

- `.pt` (PyTorch checkpoint)
- `.pth` (PyTorch checkpoint)
- `.safetensors` (SafeTensors format, `safetensors` library gerekli)

---

## 🔧 Gelişmiş Kullanım

### Özel Key Mapping

Eğer kaynak modeliniz farklı bir key yapısına sahipse, `ModelWeightConverter` sınıfını extend edebilirsiniz:

```python
from mm_rec.utils.model_converter import ModelWeightConverter

class CustomConverter(ModelWeightConverter):
    def _keys_match(self, source_pattern, source_key, target_key):
        # Özel eşleştirme mantığı
        # ...
        return super()._keys_match(source_pattern, source_key, target_key)
```

### Shape Transformation

Bazı ağırlıklar shape dönüşümü gerektirebilir (transpose, reshape):

```python
# Converter otomatik olarak transpose yapar
# Eğer source: [4096, 4096] ve target: [4096, 4096] ise
# Otomatik olarak .t() uygulanır
```

---

## 📈 Beklenen Sonuçlar

### Başarılı Dönüşüm İşaretleri

✅ **70%+ ağırlık transferi**: İyi bir başlangıç noktası  
✅ **Embedding ve FFN transferi**: Temel özellikler korunur  
✅ **Shape uyumluluğu**: Çoğu ağırlık uyumlu

### Dikkat Edilmesi Gerekenler

⚠️ **<50% transfer**: Model yapısı çok farklı olabilir  
⚠️ **Çok fazla shape mismatch**: Model boyutları uyumsuz  
⚠️ **Yeni bileşenler çok fazla**: Fine-tuning daha uzun sürebilir

---

## 🚀 Sonraki Adımlar

1. **Dönüştürülen modeli yükle**:
   ```python
   model.load_state_dict(torch.load('mmrec-7b-converted.pt'), strict=False)
   ```

2. **Fine-tuning başlat**:
   ```bash
   python -m mm_rec.scripts.train \
       --checkpoint mmrec-7b-converted.pt \
       --num_steps 10000
   ```

3. **Performansı değerlendir**:
   - Transfer edilen ağırlıklar sayesinde daha hızlı convergence
   - Yeni bileşenler öğrenilene kadar performans düşük olabilir

---

## 📝 Örnek Senaryolar

### Senaryo 1: LLaMA 7B → MM-Rec 7B

```bash
python -m mm_rec.scripts.convert_weights \
    --source llama-7b/consolidated.00.pth \
    --output mmrec-7b-from-llama.pt \
    --vocab_size 32000 \
    --model_dim 4096 \
    --num_layers 24 \
    --num_heads 32
```

**Beklenen Sonuç**: ~75% ağırlık transferi

### Senaryo 2: GPT-2 → MM-Rec

```bash
python -m mm_rec.scripts.convert_weights \
    --source gpt2.pt \
    --output mmrec-from-gpt2.pt \
    --vocab_size 50257 \
    --model_dim 768 \
    --num_layers 12 \
    --num_heads 12
```

**Beklenen Sonuç**: ~60% ağırlık transferi (daha küçük model)

---

## ❓ Sık Sorulan Sorular

**S: Tüm ağırlıklar transfer edilebilir mi?**  
C: Hayır. MM-Rec'in benzersiz bileşenleri (MDI, HDS) yeni başlatılır.

**S: Transfer edilen model hemen çalışır mı?**  
C: Evet, ama performans düşük olabilir. Fine-tuning önerilir.

**S: Hangi modeller desteklenir?**  
C: LLaMA, GPT-2, GPT-Neo gibi Transformer tabanlı modeller. Key pattern'leri otomatik tespit edilir.

**S: Model boyutları farklıysa ne olur?**  
C: Uyumlu olanlar transfer edilir, uyumsuz olanlar rastgele başlatılır.

---

**Son Güncelleme**: 2025-12-08

