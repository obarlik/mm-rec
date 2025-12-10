# 🚀 CPU için C++ Kütüphanesi - Maksimum Optimizasyon Planı

**Tarih**: 2025-01-27  
**Hedef**: CPU'da maksimum performans için native C++ kütüphanesi

---

## 🎯 Stratejik Hedef

### Amaç
MM-Rec'in CPU'da maksimum performans için kritik operasyonları C++ ile optimize etmek.

### Neden C++?
1. **SIMD Optimizasyonları**: AVX2, AVX-512 ile paralel işlem
2. **Memory Layout Kontrolü**: Cache-friendly erişim desenleri
3. **Thread Paralelizasyonu**: OpenMP ile multi-core kullanımı
4. **Compiler Optimizasyonları**: -O3, -march=native ile maksimum hız
5. **Zero Overhead**: Python overhead'i yok

---

## 📊 Kritik Operasyonlar Analizi

### 1. Associative Scan (Exponential Product) - EN KRİTİK
**Mevcut Durum**: Python/Triton (CPU fallback var ama yavaş)
**CPU'da Sorun**: 
- Log-Sum-Exp hesaplamaları Python'da yavaş
- Exponential operations CPU'da pahalı
- Sequential implementation blocking

**C++ Optimizasyon Potansiyeli**: ⭐⭐⭐⭐⭐ (5/5)
- SIMD ile paralel log/exp hesaplamaları
- OpenMP ile multi-threaded scan
- Cache-optimized memory access
- **Beklenen Hızlanma**: 10-20x

### 2. Core Recurrence Formula - KRİTİK
**Formül**: `h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}`
**Mevcut Durum**: PyTorch operations (F.linear, matmul)
**CPU'da Sorun**:
- Multiple matrix multiplications
- Element-wise operations
- Sigmoid computations

**C++ Optimizasyon Potansiyeli**: ⭐⭐⭐⭐ (4/5)
- Fused kernel: tüm operasyonlar tek kernel'de
- SIMD ile element-wise operations
- BLAS (MKL/OpenBLAS) entegrasyonu
- **Beklenen Hızlanma**: 5-10x

### 3. Memory Decay/Integration (MDI) - ÖNEMLİ
**Operasyon**: Gated integration, decay computation
**Mevcut Durum**: PyTorch operations
**CPU'da Sorun**:
- Multiple small operations
- Memory access patterns

**C++ Optimizasyon Potansiyeli**: ⭐⭐⭐ (3/5)
- Fused operations
- Cache-friendly access
- **Beklenen Hızlanma**: 3-5x

### 4. Matrix Multiplications - ÖNEMLİ
**Operasyon**: QKVZ projections, attention
**Mevcut Durum**: PyTorch (MKL backend)
**CPU'da Sorun**:
- PyTorch overhead
- Multiple small matmuls

**C++ Optimizasyon Potansiyeli**: ⭐⭐⭐ (3/5)
- Direct MKL/OpenBLAS calls
- Batch matmul optimizasyonu
- **Beklenen Hızlanma**: 2-3x

### 5. Attention Computations - ORTA
**Operasyon**: Multi-head attention
**Mevcut Durum**: PyTorch
**CPU'da Sorun**:
- Softmax operations
- Matrix multiplications

**C++ Optimizasyon Potansiyeli**: ⭐⭐⭐ (3/5)
- SIMD softmax
- Fused attention kernel
- **Beklenen Hızlanma**: 3-5x

---

## 🏗️ C++ Kütüphanesi Mimarisi

### Kütüphane Yapısı
```
mm_rec/cpp/
├── src/
│   ├── core/
│   │   ├── associative_scan_cpu.cpp      # ✅ Mevcut (geliştirilmeli)
│   │   ├── associative_scan_cpu.h
│   │   ├── log_sum_exp_simd.cpp          # 🆕 SIMD optimizasyonları
│   │   ├── log_sum_exp_simd.h
│   │   └── parallel_scan_openmp.cpp      # 🆕 OpenMP paralelizasyonu
│   ├── blocks/
│   │   ├── mm_rec_block_cpp.cpp          # ✅ Mevcut (geliştirilmeli)
│   │   ├── core_recurrence_fused.cpp     # 🆕 Fused kernel
│   │   ├── core_recurrence_fused.h
│   │   └── mdi_cpu.cpp                   # 🆕 MDI optimizasyonu
│   ├── attention/
│   │   ├── attention_cpu.cpp             # 🆕 CPU attention
│   │   ├── attention_cpu.h
│   │   └── softmax_simd.cpp              # 🆕 SIMD softmax
│   ├── utils/
│   │   ├── simd_utils.cpp                # 🆕 SIMD helper functions
│   │   ├── simd_utils.h
│   │   ├── memory_utils.cpp              # 🆕 Memory alignment
│   │   └── thread_pool.cpp                # 🆕 Thread pool
│   └── bindings/
│       ├── python_bindings.cpp            # PyTorch C++ extension
│       └── python_bindings.h
├── include/
│   └── mm_rec_cpp.h                      # Public API
├── CMakeLists.txt                        # CMake build system
├── setup.py                              # Python build
└── tests/
    ├── test_associative_scan.cpp
    ├── test_core_recurrence.cpp
    └── benchmark.cpp
```

---

## 🔧 Kritik C++ Implementasyonları

### 1. Associative Scan (Exponential Product) - EN ÖNCELİKLİ

#### SIMD Optimized Log-Sum-Exp
```cpp
// mm_rec/cpp/src/core/log_sum_exp_simd.cpp

#include <immintrin.h>  // AVX2, AVX-512
#include <cmath>

// AVX-512 optimized log-sum-exp
void log_sum_exp_avx512(
    const float* a,      // Input array 1
    const float* b,      // Input array 2
    float* output,       // Output array
    size_t n             // Array size
) {
    const size_t simd_width = 16;  // AVX-512: 16 floats
    size_t i = 0;
    
    for (; i + simd_width <= n; i += simd_width) {
        // Load 16 floats
        __m512 va = _mm512_load_ps(&a[i]);
        __m512 vb = _mm512_load_ps(&b[i]);
        
        // max(a, b)
        __m512 vmax = _mm512_max_ps(va, vb);
        
        // abs(a - b)
        __m512 vdiff = _mm512_sub_ps(va, vb);
        vdiff = _mm512_abs_ps(vdiff);
        
        // Clamp diff to [0, 20]
        __m512 vclamp = _mm512_min_ps(vdiff, _mm512_set1_ps(20.0f));
        
        // exp(-diff)
        __m512 vexp = _mm512_exp_ps(_mm512_mul_ps(vclamp, _mm512_set1_ps(-1.0f)));
        
        // log1p(exp(-diff))
        __m512 vlog1p = _mm512_log1p_ps(vexp);
        
        // max + log1p
        __m512 vresult = _mm512_add_ps(vmax, vlog1p);
        
        // Store
        _mm512_store_ps(&output[i], vresult);
    }
    
    // Scalar tail
    for (; i < n; i++) {
        float max_val = std::max(a[i], b[i]);
        float diff = std::abs(a[i] - b[i]);
        diff = std::min(diff, 20.0f);
        output[i] = max_val + std::log1p(std::exp(-diff));
    }
}
```

#### OpenMP Parallel Scan
```cpp
// mm_rec/cpp/src/core/parallel_scan_openmp.cpp

#include <omp.h>
#include <algorithm>

void associative_scan_exponential_parallel(
    const float* gamma,      // Input: decay coefficients [batch, heads, seq_len, dim]
    float* output,          // Output: cumulative products
    int batch_size,
    int num_heads,
    int seq_len,
    int dim
) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            // Log-space conversion
            float* log_gamma = new float[seq_len * dim];
            for (int t = 0; t < seq_len; t++) {
                for (int d = 0; d < dim; d++) {
                    int idx = t * dim + d;
                    float val = gamma[b * num_heads * seq_len * dim + 
                                     h * seq_len * dim + idx];
                    log_gamma[idx] = std::clamp(
                        std::log(val + 1e-8f), -50.0f, 0.0f
                    );
                }
            }
            
            // Parallel prefix sum (Blelloch scan)
            // Up-sweep phase
            for (int stride = 1; stride < seq_len; stride *= 2) {
                #pragma omp parallel for
                for (int t = stride; t < seq_len; t += 2 * stride) {
                    int left = t - stride;
                    int right = t;
                    for (int d = 0; d < dim; d++) {
                        log_gamma[right * dim + d] = 
                            log_sum_exp_simd(
                                log_gamma[left * dim + d],
                                log_gamma[right * dim + d]
                            );
                    }
                }
            }
            
            // Down-sweep phase
            for (int stride = seq_len / 2; stride > 0; stride /= 2) {
                #pragma omp parallel for
                for (int t = stride; t < seq_len; t += 2 * stride) {
                    int left = t - stride;
                    int right = t;
                    for (int d = 0; d < dim; d++) {
                        log_gamma[right * dim + d] = 
                            log_sum_exp_simd(
                                log_gamma[left * dim + d],
                                log_gamma[right * dim + d]
                            );
                    }
                }
            }
            
            // Convert back to linear space
            for (int t = 0; t < seq_len; t++) {
                for (int d = 0; d < dim; d++) {
                    int idx = t * dim + d;
                    output[b * num_heads * seq_len * dim + 
                           h * seq_len * dim + idx] = 
                        std::exp(log_gamma[idx]);
                }
            }
            
            delete[] log_gamma;
        }
    }
}
```

### 2. Core Recurrence Formula - Fused Kernel

```cpp
// mm_rec/cpp/src/blocks/core_recurrence_fused.cpp

#include <immintrin.h>
#include <mkl.h>  // Intel MKL for matrix ops

void core_recurrence_fused(
    const float* z_t,           // Input: z_t [batch, seq_len, hidden_dim]
    const float* h_prev,        // Previous state [batch, seq_len, hidden_dim]
    const float* W_g,           // Gating weights [hidden_dim, hidden_dim]
    const float* gamma,         // Decay coefficients [batch, seq_len, hidden_dim]
    float* h_t,                 // Output: new state [batch, seq_len, hidden_dim]
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    // Fused kernel: h_t = z_t ⊙ σ(W_g h_{t-1}) + γ ⊙ h_{t-1}
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            // 1. Compute gating: g = W_g @ h_prev
            float* gate = new float[hidden_dim];
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                hidden_dim, hidden_dim,
                1.0f, W_g, hidden_dim,
                &h_prev[b * seq_len * hidden_dim + t * hidden_dim], 1,
                0.0f, gate, 1
            );
            
            // 2. Apply sigmoid: σ(g)
            // SIMD sigmoid
            for (int d = 0; d < hidden_dim; d += 16) {
                __m512 vg = _mm512_load_ps(&gate[d]);
                __m512 vneg = _mm512_mul_ps(vg, _mm512_set1_ps(-1.0f));
                __m512 vexp = _mm512_exp_ps(vneg);
                __m512 vone = _mm512_set1_ps(1.0f);
                __m512 vsigmoid = _mm512_div_ps(vone, _mm512_add_ps(vone, vexp));
                _mm512_store_ps(&gate[d], vsigmoid);
            }
            
            // 3. Element-wise operations: z_t ⊙ σ(g) + γ ⊙ h_prev
            int base_idx = b * seq_len * hidden_dim + t * hidden_dim;
            for (int d = 0; d < hidden_dim; d += 16) {
                __m512 vz = _mm512_load_ps(&z_t[base_idx + d]);
                __m512 vg = _mm512_load_ps(&gate[d]);
                __m512 vh = _mm512_load_ps(&h_prev[base_idx + d]);
                __m512 vgamma = _mm512_load_ps(&gamma[base_idx + d]);
                
                // z_t ⊙ σ(g)
                __m512 vzg = _mm512_mul_ps(vz, vg);
                
                // γ ⊙ h_prev
                __m512 vgh = _mm512_mul_ps(vgamma, vh);
                
                // Sum: h_t = z_t ⊙ σ(g) + γ ⊙ h_prev
                __m512 vht = _mm512_add_ps(vzg, vgh);
                
                _mm512_store_ps(&h_t[base_idx + d], vht);
            }
            
            delete[] gate;
        }
    }
}
```

### 3. MDI Optimized

```cpp
// mm_rec/cpp/src/blocks/mdi_cpu.cpp

void mdi_update_fused(
    const float* h_new,         // New memory state
    const float* h_old,         // Old memory state
    const float* gamma,         // Decay coefficients
    const float* gate,          // Integration gate
    float* h_updated,           // Output
    int batch_size,
    int seq_len,
    int model_dim
) {
    // Fused: h_updated = gate ⊙ h_new + (1 - gate) ⊙ h_old + γ ⊙ h_old
    // Optimized with SIMD
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int base_idx = b * seq_len * model_dim + t * model_dim;
            
            for (int d = 0; d < model_dim; d += 16) {
                __m512 vnew = _mm512_load_ps(&h_new[base_idx + d]);
                __m512 vold = _mm512_load_ps(&h_old[base_idx + d]);
                __m512 vgate = _mm512_load_ps(&gate[base_idx + d]);
                __m512 vgamma = _mm512_load_ps(&gamma[base_idx + d]);
                
                // gate ⊙ h_new
                __m512 vgnew = _mm512_mul_ps(vgate, vnew);
                
                // (1 - gate) ⊙ h_old
                __m512 vone = _mm512_set1_ps(1.0f);
                __m512 voneg = _mm512_sub_ps(vone, vgate);
                __m512 voneg_old = _mm512_mul_ps(voneg, vold);
                
                // γ ⊙ h_old
                __m512 vg_old = _mm512_mul_ps(vgamma, vold);
                
                // Sum
                __m512 vresult = _mm512_add_ps(vgnew, voneg_old);
                vresult = _mm512_add_ps(vresult, vg_old);
                
                _mm512_store_ps(&h_updated[base_idx + d], vresult);
            }
        }
    }
}
```

---

## 🛠️ Build System ve Entegrasyon

### CMakeLists.txt
```cmake
cmake_minimum_required(VERSION 3.18)
project(mm_rec_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mavx512f")

# Find dependencies
find_package(OpenMP REQUIRED)
find_package(MKL REQUIRED)  # or OpenBLAS

# SIMD detection
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
    add_definitions(-DUSE_AVX512)
endif()

# Sources
set(SOURCES
    src/core/associative_scan_cpu.cpp
    src/core/log_sum_exp_simd.cpp
    src/core/parallel_scan_openmp.cpp
    src/blocks/core_recurrence_fused.cpp
    src/blocks/mdi_cpu.cpp
    src/attention/attention_cpu.cpp
    src/utils/simd_utils.cpp
    src/bindings/python_bindings.cpp
)

# Library
add_library(mm_rec_cpp SHARED ${SOURCES})
target_link_libraries(mm_rec_cpp 
    OpenMP::OpenMP_CXX
    ${MKL_LIBRARIES}
)
target_include_directories(mm_rec_cpp PUBLIC include)
```

### PyTorch C++ Extension
```cpp
// mm_rec/cpp/src/bindings/python_bindings.cpp

#include <torch/extension.h>
#include "associative_scan_cpu.h"
#include "core_recurrence_fused.h"

// Associative Scan binding
torch::Tensor associative_scan_exponential_cpp(torch::Tensor gamma) {
    // Convert to C++ types
    auto gamma_contiguous = gamma.contiguous();
    float* gamma_ptr = gamma_contiguous.data_ptr<float>();
    
    // Allocate output
    auto output = torch::empty_like(gamma);
    float* output_ptr = output.data_ptr<float>();
    
    // Call C++ function
    int batch_size = gamma.size(0);
    int num_heads = gamma.size(1);
    int seq_len = gamma.size(2);
    int dim = gamma.size(3);
    
    associative_scan_exponential_parallel(
        gamma_ptr, output_ptr,
        batch_size, num_heads, seq_len, dim
    );
    
    return output;
}

// Core Recurrence binding
torch::Tensor core_recurrence_fused_cpp(
    torch::Tensor z_t,
    torch::Tensor h_prev,
    torch::Tensor W_g,
    torch::Tensor gamma
) {
    // ... similar implementation
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("associative_scan_exponential", &associative_scan_exponential_cpp,
          "Associative scan exponential product (C++)");
    m.def("core_recurrence_fused", &core_recurrence_fused_cpp,
          "Core recurrence formula fused kernel (C++)");
}
```

---

## 📈 Beklenen Performans İyileştirmeleri

### Associative Scan
- **Mevcut**: ~1000ms (Python fallback)
- **C++ Optimized**: ~50-100ms
- **Hızlanma**: 10-20x

### Core Recurrence
- **Mevcut**: ~200ms (PyTorch)
- **C++ Fused**: ~20-40ms
- **Hızlanma**: 5-10x

### Overall Training
- **Mevcut**: ~82s/step
- **C++ Optimized**: ~10-20s/step
- **Hızlanma**: 4-8x

---

## 🎯 Uygulama Öncelikleri

### Faz 1: En Kritik (Hemen)
1. ✅ **Associative Scan C++** (SIMD + OpenMP)
   - Log-Sum-Exp SIMD
   - Parallel scan
   - **Beklenen**: 10-20x hızlanma

### Faz 2: Yüksek Öncelik
2. ✅ **Core Recurrence Fused Kernel**
   - Fused operations
   - SIMD element-wise
   - **Beklenen**: 5-10x hızlanma

### Faz 3: Orta Öncelik
3. ✅ **MDI Optimized**
   - Fused MDI operations
   - **Beklenen**: 3-5x hızlanma

### Faz 4: İyileştirmeler
4. ✅ **Attention CPU**
5. ✅ **Memory utilities**
6. ✅ **Thread pool**

---

## 🔧 Teknik Detaylar

### SIMD Seviyeleri
- **AVX2**: 8 floats paralel (256-bit)
- **AVX-512**: 16 floats paralel (512-bit) - en hızlı
- **Fallback**: Scalar (SIMD yoksa)

### Thread Stratejisi
- **OpenMP**: Parallel for loops
- **Thread Pool**: Custom pool for fine-grained control
- **Affinity**: CPU core binding

### Memory Optimizasyonları
- **Alignment**: 64-byte alignment (cache line)
- **Prefetching**: __builtin_prefetch
- **Layout**: Row-major, cache-friendly

---

## 📝 Sonraki Adımlar

### Hemen Yapılacaklar
1. ✅ Associative Scan C++ implementasyonu (SIMD + OpenMP)
2. ✅ Core Recurrence fused kernel
3. ✅ PyTorch C++ extension setup
4. ✅ Benchmark ve test

### Kısa Vadede
1. MDI optimizasyonu
2. Attention CPU kernel
3. Memory utilities
4. Comprehensive testing

### Uzun Vadede
1. Auto-tuning (optimal thread count, SIMD level)
2. Profile-guided optimization
3. JIT compilation (LLVM)
4. Distributed CPU training

---

## 🎉 Sonuç

**CPU için C++ kütüphanesi kritik!**

- ✅ **10-20x hızlanma** mümkün (Associative Scan)
- ✅ **5-10x hızlanma** mümkün (Core Recurrence)
- ✅ **Toplam 4-8x** training hızlanması
- ✅ **Maksimum CPU kullanımı** (SIMD + OpenMP)

**Öncelik**: Associative Scan C++ implementasyonu → En büyük kazanç
