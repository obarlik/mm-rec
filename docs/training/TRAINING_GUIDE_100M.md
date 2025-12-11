# MM-Rec 100M Modüler Model Eğitim Rehberi

**Versiyon**: 1.0  
**Tarih**: 2025-12-08  
**Hedef**: 100M parametreli modüler modelin kademeli eğitimi

---

## 📋 İçindekiler

1. [Hızlı Başlangıç](#1-hızlı-başlangıç)
2. [Eğitim Stratejisi](#2-eğitim-stratejisi)
3. [Veri Hazırlama](#3-veri-hazırlama)
4. [Stage-by-Stage Eğitim](#4-stage-by-stage-eğitim)
5. [Monitoring ve Debugging](#5-monitoring-ve-debugging)
6. [Best Practices](#6-best-practices)
7. [Sorun Giderme](#7-sorun-giderme)

---

## 1. Hızlı Başlangıç

### 1.1 Minimum Gereksinimler

**Donanım**:
- GPU: NVIDIA GPU (24GB+ VRAM önerilir)
- CPU: 16+ cores
- RAM: 64GB+
- Disk: 500GB+ (checkpoints ve data için)

**Yazılım**:
- CUDA 11.8+
- PyTorch 2.0+
- Triton 2.0+
- Python 3.10+

### 1.2 Kurulum

```bash
# 1. Repository'yi klonla
git clone <repo-url>
cd mm-rec

# 2. Virtual environment oluştur
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# 3. Bağımlılıkları yükle
pip install -e .

# 4. GPU kontrolü
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 1.3 İlk Eğitim (Synthetic Data)

```bash
# Stage 1: Local consistency (hızlı test)
python3 -m mm_rec.scripts.train_modular \
    --stage stage1 \
    --batch_size 2 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_interval 50 \
    --vocab_size 32000 \
    --expert_dim 256 \
    --num_layers 16 \
    --num_heads 8 \
    --ffn_dim 3072
```

**Beklenen Çıktı**:
```
Stage: Local Consistency
Description: MDI gates learn local consistency
============================================================

Stage stage1: 100%|████████| 5000/5000 [15:30<00:00, loss=2.3456, avg_loss=2.4567, lr=3.00e-04]
💾 Checkpoint saved: ./checkpoints/checkpoint_stage1_step_5000.pt
✅ Stage stage1 training completed!
```

---

## 2. Eğitim Stratejisi

### 2.1 Kademeli Yaklaşım (3 Stage)

**Stage 1: Local Consistency (Lokal Tutarlılık)**
- **Amaç**: MDI kapılarının kısa sekanslarla lokal tutarlılık kazanması
- **Sequence Length**: 512 tokens
- **Eğitim**: Her iki expert birlikte
- **Fusion**: Kullanılmaz
- **Learning Rate**: 3e-4
- **Steps**: 5,000
- **Süre**: ~2-4 saat

**Stage 2: Global Specialization (Global Uzmanlaşma)**
- **Amaç**: Uzmanların domain-specific uzun sekanslarda çalışması
- **Sequence Length**: 8,192 tokens
- **Eğitim**: Uzmanlar ayrı ayrı (Text → Code)
- **Fusion**: Kullanılmaz
- **Learning Rate**: 1e-4
- **Steps**: 10,000
- **Süre**: ~8-12 saat

**Stage 3: Fusion Training (Füzyon Eğitimi)**
- **Amaç**: Fusion layer'ın uzmanları birleştirmeyi öğrenmesi
- **Sequence Length**: 4,096 tokens
- **Eğitim**: Her iki expert + fusion layer
- **Fusion**: Kullanılır
- **Learning Rate**: 5e-5
- **Steps**: 5,000
- **Süre**: ~2-4 saat

### 2.2 Neden Kademeli?

1. **Stage 1**: MDI kapıları kısa sekanslarda stabil öğrenir
2. **Stage 2**: Uzmanlar uzun sekanslarda domain-specific bilgi öğrenir
3. **Stage 3**: Fusion layer uzmanları birleştirmeyi öğrenir

**Alternatif**: Tüm stage'leri tek seferde çalıştır:
```bash
python3 -m mm_rec.scripts.train_modular --stage all
```

---

## 3. Veri Hazırlama

### 3.1 Veri Formatı

**Text Expert için**:
- Natural language text
- Books, articles, web text
- Format: Tokenized sequences (token IDs)

**Code Expert için**:
- Source code (Python, JavaScript, etc.)
- Code comments and documentation
- Format: Tokenized sequences (token IDs)

### 3.2 Tokenization

```python
from transformers import AutoTokenizer

# Text tokenizer
text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
text_tokenizer.pad_token = text_tokenizer.eos_token

# Code tokenizer (veya code-specific tokenizer)
code_tokenizer = AutoTokenizer.from_pretrained("gpt2")
code_tokenizer.pad_token = code_tokenizer.eos_token

# Tokenize text
text_tokens = text_tokenizer.encode(text, return_tensors="pt")

# Tokenize code
code_tokens = code_tokenizer.encode(code, return_tensors="pt")
```

### 3.3 DataLoader Entegrasyonu

**Örnek DataLoader**:

```python
from torch.utils.data import Dataset, DataLoader

class TextCodeDataset(Dataset):
    def __init__(self, text_data, code_data, seq_len=512):
        self.text_data = text_data  # List of tokenized sequences
        self.code_data = code_data
        self.seq_len = seq_len
    
    def __len__(self):
        return min(len(self.text_data), len(self.code_data))
    
    def __getitem__(self, idx):
        # Get text and code sequences
        text_seq = self.text_data[idx][:self.seq_len]
        code_seq = self.code_data[idx][:self.seq_len]
        
        # Pad if necessary
        if len(text_seq) < self.seq_len:
            text_seq = torch.cat([text_seq, torch.zeros(self.seq_len - len(text_seq), dtype=torch.long)])
        if len(code_seq) < self.seq_len:
            code_seq = torch.cat([code_seq, torch.zeros(self.seq_len - len(code_seq), dtype=torch.long)])
        
        return {
            'text': text_seq,
            'code': code_seq
        }

# Create DataLoader
dataset = TextCodeDataset(text_data, code_data, seq_len=512)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### 3.4 train_modular.py'ye Entegrasyon

Şu anda `train_modular.py` synthetic data kullanıyor. Gerçek data için:

```python
# train_modular.py içinde değiştir:
# Synthetic data yerine:
for batch in dataloader:
    if config["train_both_experts"]:
        # Use both text and code
        input_ids = batch['text']  # or batch['code']
    else:
        # Stage 2: Alternate between text and code
        if step % 2 == 0:
            input_ids = batch['text']
        else:
            input_ids = batch['code']
```

---

## 4. Stage-by-Stage Eğitim

### 4.1 Stage 1: Local Consistency

**Komut**:
```bash
python3 -m mm_rec.scripts.train_modular \
    --stage stage1 \
    --batch_size 4 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_interval 100 \
    --vocab_size 32000
```

**Beklenen Davranış**:
- Loss başlangıçta yüksek (~8-10)
- İlk 1000 step'te hızlı düşüş
- 5000 step'te loss ~2-3 aralığında
- MDI kapıları stabil değerler öğrenir

**Monitoring**:
```python
# Loss tracking
- Initial loss: ~8-10
- After 1000 steps: ~4-5
- After 5000 steps: ~2-3

# Memory norms (should be stable)
- Short-term memory norm: ~0.5-2.0
- Long-term memory norm: ~0.1-1.0
```

**Checkpoint**:
- Her 100 step'te kaydedilir
- Format: `checkpoint_stage1_step_{step}.pt`
- İçerik: model, optimizer, scheduler, step, loss

### 4.2 Stage 2: Global Specialization

**Komut**:
```bash
python3 -m mm_rec.scripts.train_modular \
    --stage stage2 \
    --batch_size 2 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_interval 200 \
    --resume_from ./checkpoints/checkpoint_stage1_step_5000.pt
```

**Önemli Notlar**:
- Stage 1 checkpoint'inden resume edilir
- Batch size düşürülür (uzun sekanslar için)
- Text ve Code expert'ler ayrı ayrı eğitilir
- Her expert kendi domain'inde uzmanlaşır

**Beklenen Davranış**:
- Loss başlangıçta Stage 1'den devam eder (~2-3)
- Text expert text data'da, code expert code data'da loss düşer
- 10,000 step'te loss ~1.5-2.5 aralığında
- Her expert'in M matrisi domain-specific bilgi içerir

**Monitoring**:
```python
# Expert-specific loss tracking
- Text expert loss: ~1.5-2.0 (text data'da)
- Code expert loss: ~1.5-2.0 (code data'da)

# Memory specialization
- Text expert M: Text patterns
- Code expert M: Code patterns
```

### 4.3 Stage 3: Fusion Training

**Komut**:
```bash
python3 -m mm_rec.scripts.train_modular \
    --stage stage3 \
    --batch_size 4 \
    --checkpoint_dir ./checkpoints \
    --checkpoint_interval 100 \
    --resume_from ./checkpoints/checkpoint_stage2_step_10000.pt
```

**Önemli Notlar**:
- Stage 2 checkpoint'inden resume edilir
- Fusion layer aktif edilir
- Her iki expert birlikte eğitilir
- Fusion layer uzmanları birleştirmeyi öğrenir

**Beklenen Davranış**:
- Loss başlangıçta Stage 2'den devam eder (~1.5-2.5)
- Fusion layer öğrenirken loss düşer
- 5,000 step'te loss ~1.0-2.0 aralığında
- Fusion layer her iki expert'ten bilgi alır

**Monitoring**:
```python
# Fusion loss tracking
- Initial loss: ~1.5-2.5
- After 5000 steps: ~1.0-2.0

# Fusion layer weights
- Should learn to combine experts effectively
```

---

## 5. Monitoring ve Debugging

### 5.1 Monitoring Hooks Kullanımı

```python
from mm_rec.utils.monitoring import create_monitoring_hooks

# Create hooks
hooks = create_monitoring_hooks(model, checkpoint_dir="./checkpoints")

# In training loop
for step in range(num_steps):
    # Forward pass
    logits, memory_states = model(input_ids, return_memory=True)
    
    # Check numerical stability
    stable = hooks["stability_monitor"].check_outputs({"logits": logits}, step)
    if not stable:
        # Recovery protocol
        success, checkpoint = hooks["recovery"].recover(model, optimizer)
        if not success:
            raise RuntimeError("Recovery failed!")
    
    # Track memory norms
    norms = hooks["memory_hook"](memory_states, step)
    
    # Log norms (every 100 steps)
    if step % 100 == 0:
        print(f"Step {step}: Memory norms: {norms}")
```

### 5.2 WandB Entegrasyonu

```python
import wandb

# Initialize
wandb.init(project="mmrec-100m", name=f"stage{stage}")

# In training loop
wandb.log({
    "loss": loss.item(),
    "learning_rate": scheduler.get_last_lr()[0],
    "memory_norms": norms,
    "step": step
})
```

### 5.3 TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./logs")

# In training loop
writer.add_scalar("Loss/train", loss.item(), step)
writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], step)
writer.add_histogram("MemoryNorms", torch.tensor(list(norms.values())), step)
```

### 5.4 Debugging İpuçları

**Loss NaN/Inf**:
```python
# Check for numerical issues
if torch.isnan(loss) or torch.isinf(loss):
    print("⚠️ Numerical error detected!")
    # Use recovery protocol
    hooks["recovery"].recover(model, optimizer)
```

**Memory Norm Anomalies**:
```python
# Check for sudden norm changes
for key, norm in norms.items():
    if norm > 100 or norm < 0.001:
        print(f"⚠️ Anomalous norm in {key}: {norm}")
```

**Gradient Explosion**:
```python
# Check gradient norms
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if total_norm > 10:
    print(f"⚠️ Large gradient norm: {total_norm}")
```

---

## 6. Best Practices

### 6.1 Hyperparameter Ayarları

**Stage 1**:
```python
learning_rate = 3e-4
warmup_steps = 500
weight_decay = 0.1
gradient_clip_norm = 1.0
batch_size = 4
```

**Stage 2**:
```python
learning_rate = 1e-4  # Reduced
warmup_steps = 1000
weight_decay = 0.1
gradient_clip_norm = 1.0
batch_size = 2  # Reduced for long sequences
```

**Stage 3**:
```python
learning_rate = 5e-5  # Further reduced
warmup_steps = 500
weight_decay = 0.1
gradient_clip_norm = 1.0
batch_size = 4
```

### 6.2 Checkpointing Stratejisi

**Sık Checkpointing**:
- Her 100 step (Stage 1, 3)
- Her 200 step (Stage 2 - uzun sekanslar)

**Checkpoint İçeriği**:
```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'step': step,
    'loss': loss.item(),
    'stage': stage,
    'config': config
}
```

**Resume**:
```bash
--resume_from ./checkpoints/checkpoint_stage1_step_5000.pt
```

### 6.3 Memory Management

**Gradient Checkpointing**:
```python
# Enable for memory efficiency
model.use_gradient_checkpointing = True
```

**Chunking**:
```python
# Automatic for sequences > 32K
logits = model(input_ids, chunk_size=8192)
```

**Mixed Precision**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(input_ids)
    loss = criterion(logits, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 6.4 Data Pipeline Optimizasyonu

**Prefetching**:
```python
dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=2
)
```

**Caching**:
```python
# Cache tokenized data
import pickle

# Save
with open("tokenized_data.pkl", "wb") as f:
    pickle.dump(tokenized_data, f)

# Load
with open("tokenized_data.pkl", "rb") as f:
    tokenized_data = pickle.load(f)
```

---

## 7. Sorun Giderme

### 7.1 OOM (Out of Memory)

**Sorun**: GPU bellek yetersiz

**Çözümler**:
```bash
# 1. Batch size'ı düşür
--batch_size 1

# 2. Gradient checkpointing aktif et
model.use_gradient_checkpointing = True

# 3. Chunking kullan (uzun sekanslar için)
--chunk_size 4096

# 4. Sequence length'i düşür (Stage 2 için)
# Stage 2'de seq_len=4096 ile başla, sonra 8192'ye çık
```

### 7.2 Loss NaN/Inf

**Sorun**: Sayısal kararsızlık

**Çözümler**:
```bash
# 1. Recovery protocol kullan
./mlops/recovery_protocol.sh

# 2. Learning rate'i düşür
# Recovery otomatik olarak LR'yi 0.5x yapar

# 3. Gradient clipping'i artır
--gradient_clip_norm 0.5

# 4. Hybrid precision kullan (otomatik)
# associative_scan_hybrid.py kullanılır
```

### 7.3 Yavaş Eğitim

**Sorun**: Eğitim çok yavaş

**Çözümler**:
```bash
# 1. Kernel fusion aktif et (default: True)
model.use_kernel_fusion = True

# 2. Mixed precision kullan
# BF16 training (otomatik)

# 3. DataLoader optimizasyonu
num_workers=4, pin_memory=True

# 4. GPU utilization kontrolü
nvidia-smi  # %100 kullanım olmalı
```

### 7.4 Convergence Sorunları

**Sorun**: Loss düşmüyor

**Çözümler**:
```bash
# 1. Learning rate'i kontrol et
# Stage 1: 3e-4, Stage 2: 1e-4, Stage 3: 5e-5

# 2. Warmup steps'i artır
--warmup_steps 1000

# 3. Batch size'ı artır (mümkünse)
--batch_size 8

# 4. Gradient accumulation kullan
# Effective batch size = batch_size * gradient_accumulation_steps
```

---

## 8. Örnek Eğitim Senaryoları

### 8.1 Senaryo 1: Hızlı Test (Synthetic Data)

```bash
# Tüm stage'leri hızlı test et
python3 -m mm_rec.scripts.train_modular \
    --stage all \
    --batch_size 2 \
    --checkpoint_dir ./checkpoints_test \
    --checkpoint_interval 50 \
    --vocab_size 1000  # Küçük vocab (hızlı test)
```

**Süre**: ~1-2 saat (tüm stage'ler)

### 8.2 Senaryo 2: Production Training

```bash
# Stage 1
python3 -m mm_rec.scripts.train_modular \
    --stage stage1 \
    --batch_size 4 \
    --checkpoint_dir ./checkpoints_prod \
    --checkpoint_interval 100

# Stage 2 (resume from Stage 1)
python3 -m mm_rec.scripts.train_modular \
    --stage stage2 \
    --batch_size 2 \
    --checkpoint_dir ./checkpoints_prod \
    --checkpoint_interval 200 \
    --resume_from ./checkpoints_prod/checkpoint_stage1_step_5000.pt

# Stage 3 (resume from Stage 2)
python3 -m mm_rec.scripts.train_modular \
    --stage stage3 \
    --batch_size 4 \
    --checkpoint_dir ./checkpoints_prod \
    --checkpoint_interval 100 \
    --resume_from ./checkpoints_prod/checkpoint_stage2_step_10000.pt
```

**Süre**: ~12-20 saat (tüm stage'ler)

### 8.3 Senaryo 3: Distributed Training (Multi-GPU)

```bash
# PyTorch DDP kullan
torchrun --nproc_per_node=4 \
    -m mm_rec.scripts.train_modular \
    --stage all \
    --batch_size 8  # Per GPU
```

**Not**: Distributed training için `train_modular.py`'yi DDP ile entegre etmek gerekir.

---

## 9. Eğitim Sonrası

### 9.1 Model Değerlendirme

```python
# Load final model
model.load_state_dict(torch.load("checkpoint_stage3_step_5000.pt")['model_state_dict'])

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_loss = evaluate(model, test_dataloader)
    print(f"Test loss: {test_loss:.4f}")
```

### 9.2 Inference

```python
# Single expert (text)
logits_text = model(input_ids, expert_type="text")

# Single expert (code)
logits_code = model(input_ids, expert_type="code")

# Both experts with fusion
logits_fused = model(input_ids, expert_type=None)  # Automatic fusion
```

### 9.3 Model Export

```python
# Export for deployment
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'vocab_size': 32000,
        'expert_dim': 256,
        'num_layers': 16,
        'num_heads': 8,
        'ffn_dim': 3072
    }
}, "mmrec_100m_final.pt")
```

---

## 10. Özet Komutlar

### Tüm Stage'leri Çalıştır
```bash
python3 -m mm_rec.scripts.train_modular --stage all
```

### Tek Stage
```bash
python3 -m mm_rec.scripts.train_modular --stage stage1
```

### Resume
```bash
python3 -m mm_rec.scripts.train_modular \
    --stage stage2 \
    --resume_from ./checkpoints/checkpoint_stage1_step_5000.pt
```

### Monitoring ile
```python
# Python script içinde
from mm_rec.utils.monitoring import create_monitoring_hooks
hooks = create_monitoring_hooks(model)
# Training loop'ta hooks kullan
```

---

**Son Güncelleme**: 2025-12-08  
**Hazırlayan**: MM-Rec Development Team

