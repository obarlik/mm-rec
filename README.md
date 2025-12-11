# MM-Rec Architecture: Implementation Documentation

Bu repository, MM-Rec (Multi-Memory Recurrence) mimarisinin PyTorch/JAX framework'ünde büyük ölçekli LLM eğitimi (7B+ parametre) için implementasyon gereksinimlerini içerir.

## Dokümantasyon Yapısı

### 1. [ENGINEERING_OUTPUTS.md](./docs/analysis/ENGINEERING_OUTPUTS.md)
**Ana Çıktı Dokümanı** - Tüm kritik mühendislik çıktılarının detaylı checklist'i. Her implementasyon adımı için spesifik dosya isimleri, class'lar ve metodlar belirtilmiştir.

**Kullanım**: Implementasyon sırasında referans olarak kullanılacak ana doküman.

### 2. [CORE_FORMULA_SPEC.md](./docs/architecture/CORE_FORMULA_SPEC.md)
**Çekirdek Formül Spesifikasyonu** - MM-Rec'in temel recurrence formülü (`h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}`) ve Log-Sum-Exp implementasyonu.

**Kullanım**: Core recurrence formula implementasyonu için kritik rehber.

### 3. [TECHNICAL_REQUIREMENTS.md](./docs/architecture/TECHNICAL_REQUIREMENTS.md)
**Teknik Gereksinimler** - Yüksek seviye teknik gereksinimler, tensor layout'ları, CUDA kernel spesifikasyonları, distributed training gereksinimleri.

**Kullanım**: Mimari kararlar ve teknik tasarım için referans.

### 4. [IMPLEMENTATION_SPEC.md](./docs/architecture/IMPLEMENTATION_SPEC.md)
**Implementasyon Spesifikasyonu** - Kritik algoritmik bileşenlerin (Associative Scan, HDS, MDI) detaylı matematiksel tanımları ve implementasyon detayları.

**Kullanım**: Algoritma implementasyonu için detaylı rehber.

### 5. [CODE_STRUCTURE.md](./docs/architecture/CODE_STRUCTURE.md)
**Kod Yapısı ve API Tasarımı** - Proje dizin yapısı, tüm API'lerin tam kod örnekleri, kullanım örnekleri.

**Kullanım**: Kod yazmaya başlamak için doğrudan kullanılabilir template'ler.

### 6. [KERNEL_AND_TRAINING_REQUIREMENTS.md](./docs/architecture/KERNEL_AND_TRAINING_REQUIREMENTS.md)
**Kernel Geliştirme ve Büyük Ölçekli Eğitim Gereksinimleri** - Özel CUDA/Triton kernel gereksinimleri, BF16/FP16 stabilizasyon önlemleri, HDS long-term memory yönetimi.

**Kullanım**: Kernel geliştirme ve distributed training setup için doğrudan uygulanabilir checklist.

### Ortam Kurulumu (venv + bağımlılıklar)

```bash
# 1) Sanal ortam
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 2) PyTorch (GPU varsa uygun teker seç)
pip install torch --index-url https://download.pytorch.org/whl/cu118  # GPU
# veya sadece: pip install torch                                     # CPU

# 3) Opsiyonel Triton (GPU hızlandırma)
pip install triton

# 4) Proje bağımlılıkları
pip install -r requirements.txt
pip install -r requirements_cpu.txt  # veri indirme/izleme için

# 5) C++ eklentisi (BLAS otomatik tespit; MKL/OpenBLAS varsa hızlanır)
cd mm_rec/cpp
python setup.py build_ext --inplace
cd ../..

# 6) Hızlı doğrulama
python quick_test.py
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
from mm_rec.core import adaptive_learning
print("✓ mm_rec import OK")
PY
```

MKL/OpenBLAS kurulumu ve ortam değişkenleri için: `docs/install/MKL_INSTALLATION_GUIDE.md`.

## Kritik Bileşenler

### Core Recurrence Formula (Çekirdek Formül)
- **Matematiksel Tanım**: `h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}`
- **Dosya**: `mm_rec/blocks/mm_rec_block.py`
- **Özellikler**: Gated recurrence, exponential decay, parallel computation
- **CRITICAL**: Log-Sum-Exp kullanımı zorunlu (numerical stability için)

### Associative Scan (Exponential Product)
- **Dosya**: `mm_rec/core/associative_scan.py`
- **CUDA Kernel**: `mm_rec/cuda/scan_kernel.cu`
- **Özellikler**: O(log n) derinlik, parallel tree-based scan, **üstel çarpım** (not sum)
- **CRITICAL**: Log-Sum-Exp pattern ile implement edilmeli

### HDS (Hierarchical Data Structure)
- **Dosya**: `mm_rec/core/hds.py`
- **Özellikler**: İkili hafıza sistemi (h_t ve M), multi-level hierarchy, O(M) access cost
- **Dual Memory**: Short-term (h_t) ve Long-term (M) memory

### MDI (Memory Decay/Integration)
- **Dosya**: `mm_rec/core/mdi.py`
- **Özellikler**: Learnable decay coefficients (γ), context-dependent modulation, gated integration

### MM-Rec Block
- **Dosya**: `mm_rec/blocks/mm_rec_block.py`
- **Özellikler**: Complete block with core formula, memory state management, residual connections

## Hızlı Başlangıç

### Implementasyon Sırası

1. **Hafta 1-3**: Core Components
   - Associative Scan (CPU fallback → CUDA)
   - HDS hierarchy construction
   - MDI decay mechanism

2. **Hafta 4**: Block Integration
   - Complete MM-Rec block
   - Memory state management
   - End-to-end forward/backward

3. **Hafta 5-6**: Optimization
   - CUDA kernel optimization
   - Kernel fusion
   - Mixed precision

4. **Hafta 7**: Distributed Training
   - FSDP integration
   - Sequence parallelism

5. **Hafta 8**: Testing & Validation
   - Comprehensive tests
   - Performance benchmarks

### İlk Adımlar

1. `ENGINEERING_OUTPUTS.md` dosyasını okuyun - tüm çıktıların listesi
2. `CODE_STRUCTURE.md` dosyasından API tasarımını inceleyin
3. `IMPLEMENTATION_SPEC.md` dosyasından algoritma detaylarını öğrenin
4. İlk component'i (Associative Scan) implement edin

## Gereksinimler

### Software
- PyTorch 2.0+ (CUDA 11.8+)
- JAX 0.4+ (optional)
- CUDA Toolkit 11.8+
- cuDNN 8.6+
- NCCL (distributed training)

### Hardware
- NVIDIA GPUs (A100/H100 recommended)
- 80GB+ GPU memory per device (7B model için)
- High-bandwidth interconnects

## 7B Model Konfigürasyonu (REQUIRED SPECS)

```python
{
    "hidden_dim": 4096,          # D_hidden = 4096 (REQUIRED)
    "num_layers": 24,             # L_layer = 24 (REQUIRED)
    "num_memories": 8,
    "mem_dim": 512,
    "memory_size_M": 1024,        # Long-term memory size (M << seq_len)
    "num_heads": 32,
    "ffn_dim": 11008,
    "num_hds_levels": 3,
    "chunk_size": 128,
    "max_seq_len": 32768,         # N_sequence ≥ 32768 (32K+) (REQUIRED)
    "decay_init": 0.99,           # Initial γ value
    "use_log_sum_exp": True,      # CRITICAL: Use Log-Sum-Exp for stability
    "log_clamp_min": -50.0,       # Prevent underflow
    "log_clamp_max": 0.0          # Prevent overflow
}
```

## Başarı Kriterleri

- ✅ Tüm unit testler geçer
- ✅ Gradient correctness doğrulanır
- ✅ >50% teorik peak throughput
- ✅ <80GB GPU memory (7B, seq_len=2048)
- ✅ >80% scaling efficiency (8 GPUs)
- ✅ Training stability (NaN/Inf yok)

## Notlar

- Tüm implementasyonlar production-ready olmalı
- Kod standartları: PEP 8 (Python), Google C++ Style (CUDA)
- Her public API için docstring gerekli
- Git ile proper version control

## İletişim

Teknik sorular için `ENGINEERING_OUTPUTS.md` dosyasındaki spesifik checklist'i referans alın.

