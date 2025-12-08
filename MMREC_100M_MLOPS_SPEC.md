# MM-Rec 100M Modüler Eğitim ve MLOps Tasarım Dokümantasyonu

**Versiyon**: 1.0  
**Tarih**: 2025-12-08  
**Hedef**: 100M parametreli, 256 kanallı, Modüler MM-Rec modeli ve MLOps hattı

---

## 📋 İçindekiler

1. [Model Mimarisi](#1-model-mimarisi)
2. [Hibrit Hassasiyet Implementasyonu](#2-hibrit-hassasiyet-implementasyonu)
3. [Session-Based Memory Management](#3-session-based-memory-management)
4. [Modüler Eğitim Stratejisi](#4-modüler-eğitim-stratejisi)
5. [Fusion Layer](#5-fusion-layer)
6. [MLOps Orkestrasyonu](#6-mlops-orkestrasyonu)
7. [Monitoring ve Recovery](#7-monitoring-ve-recovery)
8. [Doğrulama ve Testler](#8-doğrulama-ve-testler)

---

## 1. Model Mimarisi

### 1.1 MM-Rec 100M Model Yapısı

**Konfigürasyon**:
- **Parametre Sayısı**: ~100M
- **Expert Dimension**: 256 (her expert)
- **Expert Sayısı**: 2 (Text Expert + Code Expert)
- **Fusion Dimension**: 512 (256 + 256)
- **Layer Sayısı**: 16 per expert
- **FFN Dimension**: 3072 per expert
- **Vocab Size**: 32,000

**Mimari**:
```
Input [batch, seq_len]
  ↓
Embedding [batch, seq_len, 256]
  ↓
┌─────────────────┬─────────────────┐
│  Text Expert    │  Code Expert    │
│  (256 channels) │  (256 channels) │
│  16 layers      │  16 layers      │
└─────────────────┴─────────────────┘
  ↓                    ↓
┌──────────────────────────────────┐
│      Fusion Layer (512)          │
└──────────────────────────────────┘
  ↓
Final Norm + LM Head
  ↓
Output [batch, seq_len, vocab_size]
```

### 1.2 Expert Modülleri

**Text Expert**:
- 16 MMRecBlock layers
- 256 model dimension
- 3072 FFN dimension
- 8 attention heads
- Kendi memory state'leri (M_text, h_t_text)

**Code Expert**:
- 16 MMRecBlock layers
- 256 model dimension
- 3072 FFN dimension
- 8 attention heads
- Kendi memory state'leri (M_code, h_t_code)

### 1.3 Dosya Yapısı

```
mm_rec/
├── models/
│   └── mmrec_100m.py          # MMRec100M model
├── core/
│   ├── associative_scan_hybrid.py  # BF16 + FP64 hybrid
│   └── session_memory.py      # Session-based memory management
├── scripts/
│   └── train_modular.py       # Modüler eğitim script'i
└── utils/
    └── monitoring.py           # Monitoring hooks
```

---

## 2. Hibrit Hassasiyet Implementasyonu

### 2.1 Hassasiyet Stratejisi

**BF16 (Bfloat16)**:
- Model weights
- Activations
- Standard operations

**FP64 (Double Precision)**:
- **Sadece**: Log-space cumulative sum (prefix sum) in associative scan
- Kritik sayısal stabilite için

### 2.2 Implementasyon

**Dosya**: `mm_rec/core/associative_scan_hybrid.py`

```python
class HybridPrecisionAssociativeScan:
    @staticmethod
    def forward(gamma: torch.Tensor) -> torch.Tensor:
        # 1. Convert to FP32 for log operations
        gamma_fp32 = gamma.to(torch.float32)
        
        # 2. Convert to log-space (FP32)
        log_gamma = torch.log(gamma_fp32 + 1e-8)
        log_gamma = torch.clamp(log_gamma, -50.0, 0.0)
        
        # 3. CRITICAL: Convert to FP64 for accumulation
        log_gamma_fp64 = log_gamma.to(torch.float64)
        
        # 4. Cumulative sum in FP64 (double precision)
        log_cumsum_fp64 = torch.cumsum(log_gamma_fp64, dim=2)
        
        # 5. Convert back to FP32 for exp
        log_cumsum_fp32 = log_cumsum_fp64.to(torch.float32)
        
        # 6. Stable exponential conversion
        max_log = torch.max(log_cumsum_fp32, dim=2, keepdim=True)[0]
        stable_log = log_cumsum_fp32 - max_log
        cumulative_product = torch.exp(stable_log) * torch.exp(max_log)
        
        # 7. Convert back to BF16
        return cumulative_product.to(gamma.dtype)
```

**Kullanım**:
```python
from mm_rec.core.associative_scan_hybrid import associative_scan_exponential_hybrid

# BF16 input
gamma = torch.rand(2, 8, 1024, 64, dtype=torch.bfloat16, device='cuda')

# Hybrid precision scan (BF16 → FP64 log accumulation → BF16)
result = associative_scan_exponential_hybrid(gamma)
```

---

## 3. Session-Based Memory Management

### 3.1 Session Memory Manager

**Dosya**: `mm_rec/core/session_memory.py`

**Özellikler**:
- Session ID bazlı memory state yönetimi
- Serialize/load memory states (M ve h_t)
- File-based veya database-based storage
- I/O optimizasyonu

**Kullanım**:
```python
from mm_rec.core.session_memory import SessionMemoryManager

# Initialize manager
manager = SessionMemoryManager(base_dir="./memory_sessions")

# Serialize memory states
session_id = "user_123_session_1"
memory_states = {
    "text": [text_expert_states],
    "code": [code_expert_states]
}
save_path = manager.serialize_state(session_id, memory_states)

# Load memory states
loaded_states = manager.load_state(session_id, device=device)
```

### 3.2 Memory State Structure

**Short-term Memory (h_t)**:
- Per-timestep hidden states
- Shape: [batch, seq_len, model_dim]
- Updated sequentially

**Long-term Memory (M)**:
- Persistent memory matrix
- Shape: [batch, num_memories, M, mem_dim] where M=1024
- Updated incrementally

**Serialization Format**:
```
memory_sessions/
└── {session_id}/
    ├── text/
    │   ├── layer_0.pt
    │   ├── layer_1.pt
    │   └── ...
    ├── code/
    │   ├── layer_0.pt
    │   ├── layer_1.pt
    │   └── ...
    └── metadata.json
```

---

## 4. Modüler Eğitim Stratejisi

### 4.1 Kademeli Eğitim Planı

**Stage 1: Local Consistency (Lokal Tutarlılık)**
- **Amaç**: MDI kapılarının kısa sekanslarla lokal tutarlılık kazanması
- **Sequence Length**: 512 tokens
- **Eğitim**: Her iki expert birlikte
- **Fusion**: Kullanılmaz
- **Learning Rate**: 3e-4
- **Steps**: 5,000

**Stage 2: Global Specialization (Global Uzmanlaşma)**
- **Amaç**: Uzmanların domain-specific uzun sekanslarda çalışması
- **Sequence Length**: 8,192 tokens
- **Eğitim**: Uzmanlar ayrı ayrı (Text → Code)
- **Fusion**: Kullanılmaz
- **Learning Rate**: 1e-4
- **Steps**: 10,000

**Stage 3: Fusion Training (Füzyon Eğitimi)**
- **Amaç**: Fusion layer'ın uzmanları birleştirmeyi öğrenmesi
- **Sequence Length**: 4,096 tokens
- **Eğitim**: Her iki expert + fusion layer
- **Fusion**: Kullanılır
- **Learning Rate**: 5e-5
- **Steps**: 5,000

### 4.2 Eğitim Script'i

**Dosya**: `mm_rec/scripts/train_modular.py`

**Kullanım**:
```bash
# Stage 1 only
python -m mm_rec.scripts.train_modular --stage stage1

# Stage 2 only
python -m mm_rec.scripts.train_modular --stage stage2

# Stage 3 only
python -m mm_rec.scripts.train_modular --stage stage3

# All stages
python -m mm_rec.scripts.train_modular --stage all
```

**Checkpointing**:
- Her stage için ayrı checkpoint'ler
- Format: `checkpoint_{stage}_step_{step}.pt`
- Resume: `--resume_from checkpoint_path`

---

## 5. Fusion Layer

### 5.1 Fusion Mekanizması

**Amaç**: İki 256-kanallı expert'i 512 kanallı birleşik hafıza alanına dönüştürme

**Yöntemler**:
1. **Concatenate**: Basit birleştirme (256 + 256 = 512)
2. **Weighted**: Öğrenilebilir ağırlıklı birleştirme

**Implementasyon**:
```python
class FusionLayer(nn.Module):
    def forward(self, text_output, code_output, text_memory=None, code_memory=None):
        # Concatenate expert outputs
        combined = torch.cat([text_output, code_output], dim=-1)  # [batch, seq_len, 512]
        
        # Apply fusion projection
        fused = self.fusion_proj(combined)  # [batch, seq_len, 512]
        
        return fused
```

**Inference Kullanımı**:
```python
# Eğitilmiş expert'lerden çıktı al
text_output, text_states = model.text_expert(x, text_memory_states)
code_output, code_states = model.code_expert(x, code_memory_states)

# Fusion layer ile birleştir
fused = model.fusion(text_output, code_output, text_memory, code_memory)
# fused: [batch, seq_len, 512]
```

---

## 6. MLOps Orkestrasyonu

### 6.1 Dockerfile

**Dosya**: `mlops/Dockerfile`

**Özellikler**:
- CUDA 11.8 + cuDNN 8
- PyTorch with CUDA support
- Triton kernel support
- Tüm bağımlılıklar

**Build**:
```bash
docker build -t mmrec-100m:latest -f mlops/Dockerfile .
```

### 6.2 Kubernetes Deployment

**Dosya**: `mlops/kubernetes/job.yaml`

**Özellikler**:
- GPU resource requests/limits
- Persistent volume claims (checkpoints, data)
- Environment variables
- Resource limits

**Deploy**:
```bash
kubectl apply -f mlops/kubernetes/job.yaml
```

### 6.3 Slurm Script

**Dosya**: `mlops/slurm/train.sh`

**Özellikler**:
- GPU allocation
- Resource limits
- Logging

**Submit**:
```bash
sbatch mlops/slurm/train.sh
```

---

## 7. Monitoring ve Recovery

### 7.1 Monitoring Hooks

**Dosya**: `mm_rec/utils/monitoring.py`

**Özellikler**:

1. **Numerical Stability Monitor**:
   - NaN/Inf detection in outputs
   - Memory state stability checks
   - Alert threshold (3 consecutive issues)

2. **Memory Norm Tracker**:
   - Track M and h_t norms
   - Detect sudden changes (>10x ratio)
   - Log history

**Kullanım**:
```python
from mm_rec.utils.monitoring import create_monitoring_hooks

hooks = create_monitoring_hooks(model, checkpoint_dir="./checkpoints")

# In training loop
for step in range(num_steps):
    # Forward pass
    logits, memory_states = model(input_ids, return_memory=True)
    
    # Check outputs
    hooks["stability_monitor"].check_outputs({"logits": logits}, step)
    
    # Track memory norms
    hooks["memory_hook"](memory_states, step)
```

### 7.2 Recovery Protocol

**Dosya**: `mlops/recovery_protocol.sh`

**Özellikler**:
- Otomatik checkpoint bulma
- Learning rate reduction (0.5x)
- Training resume
- Maximum recovery attempts (3)

**Kullanım**:
```bash
# Manual recovery
./mlops/recovery_protocol.sh

# Automatic (from monitoring)
if numerical_error_detected:
    recovery.recover(model, optimizer, error_type="numerical_error")
```

**Recovery Steps**:
1. Find latest checkpoint
2. Load model and optimizer states
3. Reduce learning rate by 0.5x
4. Resume training
5. Log recovery attempt

---

## 8. Doğrulama ve Testler

### 8.1 Associative Scan Validation

**Dosya**: `mm_rec/tests/test_associative_scan_validation.py`

**Testler**:
- Short sequence (128 tokens)
- Medium sequence (1024 tokens)
- Long sequence (8192 tokens)
- Hybrid precision test
- Numerical stability test (extreme values)

**Çalıştırma**:
```bash
python -m pytest mm_rec/tests/test_associative_scan_validation.py -v
```

**Doğrulama**:
- Triton kernel output vs sequential Python implementation
- Tolerance: rtol=1e-3, atol=1e-4
- NaN/Inf checks

### 8.2 32K Sequence Test

**Dosya**: `mm_rec/tests/test_32k_sequence.py`

**Testler**:
- 32K forward pass
- 32K with memory states
- Chunking consistency (different chunk sizes)

**Çalıştırma**:
```bash
python -m pytest mm_rec/tests/test_32k_sequence.py -v
```

**Doğrulama**:
- Model processes 32,768 tokens without errors
- Output shape correctness
- No NaN/Inf in outputs
- Chunking produces consistent results

---

## 9. Kullanım Örnekleri

### 9.1 Model Oluşturma

```python
from mm_rec.models.mmrec_100m import MMRec100M

model = MMRec100M(
    vocab_size=32000,
    expert_dim=256,
    num_layers=16,
    num_heads=8,
    ffn_dim=3072
)

print(f"Parameters: {model.get_num_params():,}")  # ~100M
```

### 9.2 Modüler Eğitim

```bash
# Stage 1: Local consistency
python -m mm_rec.scripts.train_modular \
    --stage stage1 \
    --batch_size 4 \
    --checkpoint_dir ./checkpoints

# Stage 2: Global specialization
python -m mm_rec.scripts.train_modular \
    --stage stage2 \
    --batch_size 2 \
    --checkpoint_dir ./checkpoints

# Stage 3: Fusion training
python -m mm_rec.scripts.train_modular \
    --stage stage3 \
    --batch_size 4 \
    --checkpoint_dir ./checkpoints
```

### 9.3 Session Memory Management

```python
from mm_rec.core.session_memory import SessionMemoryManager

manager = SessionMemoryManager(base_dir="./memory_sessions")

# Save state
session_id = "user_123"
memory_states = {
    "text": text_expert_states,
    "code": code_expert_states
}
manager.serialize_state(session_id, memory_states)

# Load state
loaded_states = manager.load_state(session_id, device=device)
```

### 9.4 Inference with Fusion

```python
# Load trained model
model.load_state_dict(torch.load("checkpoint_stage3_final.pt"))

# Inference
input_ids = torch.randint(0, 32000, (1, 4096), device=device)

# Use both experts and fuse
logits = model(input_ids, expert_type=None)  # Automatic fusion

# Or use single expert
text_logits = model(input_ids, expert_type="text")
code_logits = model(input_ids, expert_type="code")
```

---

## 10. Performans Beklentileri

### 10.1 Model Boyutu

- **Parameters**: ~100M
- **BF16 Size**: ~200 MB
- **FP32 Size**: ~400 MB
- **Training Memory**: ~2-4 GB (with gradient checkpointing)

### 10.2 Eğitim Süresi (Tahmini)

- **Stage 1**: ~2-4 hours (5K steps, seq_len=512)
- **Stage 2**: ~8-12 hours (10K steps, seq_len=8192)
- **Stage 3**: ~2-4 hours (5K steps, seq_len=4096)
- **Total**: ~12-20 hours (single GPU)

### 10.3 Inference

- **32K sequence**: ~2-4 seconds (with chunking)
- **Memory usage**: ~2-3 GB
- **Throughput**: ~8K-16K tokens/second

---

## 11. Kritik Notlar

### 11.1 Mutlaka Yapılması Gerekenler

1. **FP64 Log Accumulation**: Associative scan'de mutlaka kullan
2. **Session Memory**: Her session için ayrı memory state yönet
3. **Staged Training**: Stage 1 → 2 → 3 sırasını takip et
4. **Monitoring**: NaN/Inf detection'ı aktif tut
5. **Checkpointing**: Her stage'de düzenli checkpoint al

### 11.2 Yapılmaması Gerekenler

1. **Direct FP64**: Tüm model'i FP64 yapma (sadece log accumulation)
2. **Skip Stages**: Stage'leri atlama
3. **Ignore Monitoring**: Monitoring hook'larını kapatma
4. **No Recovery**: Recovery protocol'ü devre dışı bırakma

---

## 12. Sonuç

MM-Rec 100M modüler model ve MLOps hattı, aşağıdaki özellikleri sağlar:

✅ **100M parametreli modüler model** (2 experts, 256 channels each)  
✅ **Hibrit hassasiyet** (BF16 + FP64 log accumulation)  
✅ **Session-based memory management**  
✅ **Kademeli modüler eğitim** (3 stage)  
✅ **Fusion layer** (512 channels from 2x256)  
✅ **MLOps orkestrasyonu** (Docker, Kubernetes, Slurm)  
✅ **Monitoring ve recovery** (NaN/Inf detection, automatic recovery)  
✅ **Doğrulama testleri** (kernel validation, 32K sequence)

**Dokümantasyon Versiyonu**: 1.0  
**Son Güncelleme**: 2025-12-08

