# 🚀 CPU için C++ Kütüphanesi - Maksimum Optimizasyon Planı

**Tarih**: 2025-01-27  
**Hedef**: CPU'da maksimum performans için native C++ kütüphanesi  
**Durum**: Mevcut C++ kodu var ama optimize edilmeli

---

## 📊 Mevcut Durum Analizi

### ✅ Mevcut C++ Dosyaları
1. **`associative_scan_cpu.cpp`** ✅
   - AVX optimizasyonları var (kısmi)
   - OpenMP paralelizasyonu var
   - **Sorun**: exp() fonksiyonu scalar (SIMD değil)
   - **Sorun**: Blelloch scan tam implement edilmemiş (sequential)

2. **`mm_rec_block_cpp.cpp`** ✅
   - Basit sequential loop var
   - **Sorun**: Fused kernel yok
   - **Sorun**: SIMD optimizasyonları yok
   - **Sorun**: OpenMP kullanılmıyor

### ⚠️ Eksikler
1. **Vectorized exp/log fonksiyonları** (SIMD)
2. **Tam Blelloch parallel scan** (şu an sequential)
3. **Core recurrence fused kernel**
4. **MDI optimized kernel**
5. **Attention CPU kernel**
6. **Memory alignment optimizasyonları**

---

## 🎯 Kritik Optimizasyonlar

### 1. Associative Scan - EN KRİTİK ⭐⭐⭐⭐⭐

#### Mevcut Sorunlar
```cpp
// Mevcut: exp() scalar - SIMD değil
for (int i = 0; i < 8; ++i) {
    result_arr[i] = max_arr[i] + std::log1p(std::exp(-clamped_arr[i]));
}
```

#### İyileştirme: Vectorized Exp/Log
```cpp
// AVX-512 vectorized exp approximation
__m512 vectorized_exp_avx512(__m512 x) {
    // Fast exp approximation using polynomial
    // vexp ≈ 1 + x + x²/2 + x³/6 + ...
    // Optimized for [-20, 0] range
}
```

#### İyileştirme: Tam Blelloch Scan
```cpp
// Mevcut: Sequential scan (O(n))
// İyileştirme: Parallel Blelloch scan (O(log n) depth)
void blelloch_scan_parallel(
    float* input,
    float* output,
    int n,
    int num_threads
) {
    // Up-sweep phase (parallel reduction tree)
    // Down-sweep phase (parallel prefix propagation)
}
```

**Beklenen Hızlanma**: 10-20x

---

### 2. Core Recurrence Formula - KRİTİK ⭐⭐⭐⭐

#### Mevcut Durum
- PyTorch operations (F.linear, matmul, sigmoid)
- Multiple small operations
- Python overhead

#### C++ Fused Kernel
```cpp
void core_recurrence_fused_avx512(
    const float* z_t,        // [batch, seq_len, hidden_dim]
    const float* h_prev,     // [batch, seq_len, hidden_dim]
    const float* W_g,        // [hidden_dim, hidden_dim]
    const float* gamma,       // [batch, seq_len, hidden_dim]
    float* h_t,              // Output
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    // Fused: h_t = z_t ⊙ σ(W_g @ h_prev) + γ ⊙ h_prev
    // All operations in single kernel:
    // 1. Matrix-vector multiply (MKL)
    // 2. Vectorized sigmoid (SIMD)
    // 3. Element-wise operations (SIMD)
}
```

**Beklenen Hızlanma**: 5-10x

---

### 3. MDI Optimized - ÖNEMLİ ⭐⭐⭐

#### Fused MDI Kernel
```cpp
void mdi_update_fused_simd(
    const float* h_new,
    const float* h_old,
    const float* gamma,
    const float* gate,
    float* h_updated,
    int n
) {
    // SIMD: h_updated = gate ⊙ h_new + (1-gate) ⊙ h_old + γ ⊙ h_old
    // AVX-512: 16 floats at once
}
```

**Beklenen Hızlanma**: 3-5x

---

## 🏗️ Yeni C++ Kütüphanesi Mimarisi

### Klasör Yapısı
```
mm_rec/cpp/
├── src/
│   ├── core/
│   │   ├── associative_scan_cpu.cpp          # ✅ Mevcut (geliştirilmeli)
│   │   ├── associative_scan_cpu.h
│   │   ├── log_sum_exp_simd.cpp             # 🆕 Vectorized LSE
│   │   ├── log_sum_exp_simd.h
│   │   ├── exp_log_simd.cpp                 # 🆕 Vectorized exp/log
│   │   ├── exp_log_simd.h
│   │   ├── blelloch_scan_parallel.cpp        # 🆕 True parallel scan
│   │   └── blelloch_scan_parallel.h
│   ├── blocks/
│   │   ├── mm_rec_block_cpp.cpp              # ✅ Mevcut (geliştirilmeli)
│   │   ├── core_recurrence_fused.cpp         # 🆕 Fused kernel
│   │   ├── core_recurrence_fused.h
│   │   ├── mdi_cpu_optimized.cpp             # 🆕 MDI SIMD
│   │   └── mdi_cpu_optimized.h
│   ├── attention/
│   │   ├── attention_cpu.cpp                 # 🆕 CPU attention
│   │   ├── attention_cpu.h
│   │   └── softmax_simd.cpp                 # 🆕 SIMD softmax
│   ├── utils/
│   │   ├── simd_utils.cpp                    # 🆕 SIMD helpers
│   │   ├── simd_utils.h
│   │   ├── memory_utils.cpp                  # 🆕 Alignment, prefetch
│   │   └── thread_pool.cpp                  # 🆕 Custom thread pool
│   └── bindings/
│       ├── python_bindings.cpp               # PyTorch extension
│       └── python_bindings.h
├── include/
│   └── mm_rec_cpp.h                         # Public API
├── CMakeLists.txt                            # CMake build
├── setup.py                                  # ✅ Mevcut (güncellenmeli)
└── tests/
    ├── test_associative_scan.cpp
    ├── test_core_recurrence.cpp
    └── benchmark.cpp
```

---

## 🔧 Kritik C++ Implementasyonları

### 1. Vectorized Exp/Log (SIMD)

```cpp
// mm_rec/cpp/src/core/exp_log_simd.cpp

#include <immintrin.h>
#include <cmath>

// Fast exp approximation for AVX-512
// Optimized for range [-20, 0] (Log-Sum-Exp use case)
__m512 vectorized_exp_avx512(__m512 x) {
    // Clamp to [-20, 0]
    __m512 x_clamped = _mm512_max_ps(x, _mm512_set1_ps(-20.0f));
    x_clamped = _mm512_min_ps(x_clamped, _mm512_set1_ps(0.0f));
    
    // Fast polynomial approximation: exp(x) ≈ 1 + x + x²/2 + x³/6
    // For better accuracy: use Remez polynomial or lookup table
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 x2 = _mm512_mul_ps(x_clamped, x_clamped);
    __m512 x3 = _mm512_mul_ps(x2, x_clamped);
    
    __m512 result = _mm512_fmadd_ps(
        x_clamped, _mm512_set1_ps(1.0f),
        one
    );
    result = _mm512_fmadd_ps(
        x2, _mm512_set1_ps(0.5f),
        result
    );
    result = _mm512_fmadd_ps(
        x3, _mm512_set1_ps(1.0f/6.0f),
        result
    );
    
    return result;
}

// Vectorized log1p for Log-Sum-Exp
__m512 vectorized_log1p_avx512(__m512 x) {
    // Fast log1p approximation
    // log1p(x) ≈ x - x²/2 + x³/3 for small x
    // For larger x, use standard log(1+x)
    __m512 small = _mm512_cmp_ps_mask(x, _mm512_set1_ps(0.1f), _CMP_LT_OQ);
    
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 x3 = _mm512_mul_ps(x2, x);
    
    __m512 approx = _mm512_sub_ps(x, _mm512_mul_ps(x2, _mm512_set1_ps(0.5f)));
    approx = _mm512_add_ps(approx, _mm512_mul_ps(x3, _mm512_set1_ps(1.0f/3.0f)));
    
    // For larger values, use standard log
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 standard = vectorized_log_avx512(_mm512_add_ps(one, x));
    
    return _mm512_mask_blend_ps(small, standard, approx);
}
```

### 2. Tam Blelloch Parallel Scan

```cpp
// mm_rec/cpp/src/core/blelloch_scan_parallel.cpp

#include <omp.h>
#include <immintrin.h>

void blelloch_scan_parallel_avx512(
    float* input,      // Input array [n]
    float* output,     // Output array [n]
    int n,             // Array size
    int num_threads    // Number of threads
) {
    omp_set_num_threads(num_threads);
    
    // Step 1: Up-sweep Phase (Reduction Tree)
    // Build reduction tree: O(log n) depth
    for (int stride = 1; stride < n; stride *= 2) {
        #pragma omp parallel for
        for (int i = stride; i < n; i += 2 * stride) {
            int left = i - stride;
            int right = i;
            
            // Vectorized log-sum-exp
            #ifdef __AVX512F__
            for (int j = 0; j < n - 15; j += 16) {
                __m512 vleft = _mm512_load_ps(&input[left * n + j]);
                __m512 vright = _mm512_load_ps(&input[right * n + j]);
                __m512 vresult = vectorized_log_sum_exp_avx512(vleft, vright);
                _mm512_store_ps(&output[right * n + j], vresult);
            }
            #endif
        }
    }
    
    // Step 2: Down-sweep Phase (Prefix Propagation)
    // Propagate prefixes: O(log n) depth
    output[n-1] = 0.0f;  // Set last element to identity
    
    for (int stride = n / 2; stride > 0; stride /= 2) {
        #pragma omp parallel for
        for (int i = stride; i < n; i += 2 * stride) {
            int left = i - stride;
            int right = i;
            
            // Vectorized log-sum-exp
            #ifdef __AVX512F__
            for (int j = 0; j < n - 15; j += 16) {
                __m512 vleft = _mm512_load_ps(&output[left * n + j]);
                __m512 vright = _mm512_load_ps(&output[right * n + j]);
                __m512 vresult = vectorized_log_sum_exp_avx512(vleft, vright);
                _mm512_store_ps(&output[right * n + j], vresult);
            }
            #endif
        }
    }
}
```

### 3. Core Recurrence Fused Kernel

```cpp
// mm_rec/cpp/src/blocks/core_recurrence_fused.cpp

#include <immintrin.h>
#include <mkl.h>  // Intel MKL

void core_recurrence_fused_avx512(
    const float* z_t,        // [batch, seq_len, hidden_dim]
    const float* h_prev,      // [batch, seq_len, hidden_dim]
    const float* W_g,         // [hidden_dim, hidden_dim]
    const float* gamma,       // [batch, seq_len, hidden_dim]
    float* h_t,               // Output [batch, seq_len, hidden_dim]
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    // Fused kernel: h_t = z_t ⊙ σ(W_g @ h_prev) + γ ⊙ h_prev
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            int base_idx = b * seq_len * hidden_dim + t * hidden_dim;
            
            // 1. Matrix-vector multiply: g = W_g @ h_prev
            float* gate = new float[hidden_dim];
            cblas_sgemv(
                CblasRowMajor, CblasNoTrans,
                hidden_dim, hidden_dim,
                1.0f, W_g, hidden_dim,
                &h_prev[base_idx], 1,
                0.0f, gate, 1
            );
            
            // 2. Vectorized sigmoid: σ(g)
            for (int d = 0; d < hidden_dim - 15; d += 16) {
                __m512 vg = _mm512_load_ps(&gate[d]);
                __m512 vneg = _mm512_mul_ps(vg, _mm512_set1_ps(-1.0f));
                __m512 vexp = vectorized_exp_avx512(vneg);
                __m512 vone = _mm512_set1_ps(1.0f);
                __m512 vsigmoid = _mm512_div_ps(vone, _mm512_add_ps(vone, vexp));
                _mm512_store_ps(&gate[d], vsigmoid);
            }
            
            // 3. Fused element-wise: z_t ⊙ σ(g) + γ ⊙ h_prev
            for (int d = 0; d < hidden_dim - 15; d += 16) {
                __m512 vz = _mm512_load_ps(&z_t[base_idx + d]);
                __m512 vg = _mm512_load_ps(&gate[d]);
                __m512 vh = _mm512_load_ps(&h_prev[base_idx + d]);
                __m512 vgamma = _mm512_load_ps(&gamma[base_idx + d]);
                
                // z_t ⊙ σ(g)
                __m512 vzg = _mm512_mul_ps(vz, vg);
                
                // γ ⊙ h_prev
                __m512 vgh = _mm512_mul_ps(vgamma, vh);
                
                // Sum
                __m512 vht = _mm512_add_ps(vzg, vgh);
                
                _mm512_store_ps(&h_t[base_idx + d], vht);
            }
            
            delete[] gate;
        }
    }
}
```

---

## 🛠️ Build System İyileştirmeleri

### setup.py Güncellemeleri

```python
# Modern CPU optimizations
cxx_args = [
    '-O3',                    # Maximum optimization
    '-march=native',          # Auto-detect CPU features
    '-mtune=native',          # Tune for native CPU
    '-mavx2',                 # AVX2 (8 floats)
    '-mavx512f',              # AVX-512 (16 floats) - if available
    '-mavx512cd',             # AVX-512 conflict detection
    '-mfma',                  # Fused Multiply-Add
    '-fopenmp',               # OpenMP
    '-funroll-loops',         # Loop unrolling
    '-ffast-math',            # Fast math (careful with numerical stability)
    '-fno-math-errno',        # Don't set errno
    '-flto',                  # Link-time optimization
]
```

### CMakeLists.txt (Alternatif)

```cmake
# Detect CPU features
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)

if(COMPILER_SUPPORTS_AVX512)
    add_definitions(-DUSE_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512cd")
elseif(COMPILER_SUPPORTS_AVX2)
    add_definitions(-DUSE_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(mm_rec_cpp OpenMP::OpenMP_CXX)

# MKL or OpenBLAS
find_package(MKL QUIET)
if(MKL_FOUND)
    target_link_libraries(mm_rec_cpp ${MKL_LIBRARIES})
else()
    find_package(OpenBLAS QUIET)
    if(OpenBLAS_FOUND)
        target_link_libraries(mm_rec_cpp ${OpenBLAS_LIBRARIES})
    endif()
endif()
```

---

## 📈 Beklenen Performans İyileştirmeleri

### Associative Scan
- **Mevcut**: Sequential, scalar exp() → ~1000ms
- **Optimized**: Parallel Blelloch, SIMD exp() → ~50-100ms
- **Hızlanma**: **10-20x** ⭐

### Core Recurrence
- **Mevcut**: PyTorch operations → ~200ms
- **Optimized**: Fused kernel, SIMD → ~20-40ms
- **Hızlanma**: **5-10x** ⭐

### MDI
- **Mevcut**: PyTorch → ~50ms
- **Optimized**: SIMD → ~10-20ms
- **Hızlanma**: **3-5x**

### Overall Training
- **Mevcut**: ~82s/step
- **Optimized**: ~10-15s/step
- **Hızlanma**: **5-8x** ⭐⭐⭐

---

## 🎯 Uygulama Öncelikleri

### Faz 1: En Kritik (Hemen) ⭐⭐⭐⭐⭐
1. ✅ **Vectorized Exp/Log** (SIMD)
   - AVX-512 exp approximation
   - Vectorized log1p
   - **Beklenen**: 5-10x hızlanma (Associative Scan için)

2. ✅ **Tam Blelloch Parallel Scan**
   - Up-sweep + Down-sweep
   - OpenMP paralelizasyonu
   - **Beklenen**: 5-10x hızlanma

**Toplam Associative Scan**: 10-20x hızlanma

### Faz 2: Yüksek Öncelik ⭐⭐⭐⭐
3. ✅ **Core Recurrence Fused Kernel**
   - Fused operations
   - SIMD sigmoid
   - MKL matmul
   - **Beklenen**: 5-10x hızlanma

### Faz 3: Orta Öncelik ⭐⭐⭐
4. ✅ **MDI Optimized**
5. ✅ **Attention CPU**
6. ✅ **Memory utilities**

---

## 🔧 Teknik Detaylar

### SIMD Seviyeleri (Auto-detect)
```cpp
// Runtime CPU feature detection
bool has_avx512() {
    return __builtin_cpu_supports("avx512f");
}

bool has_avx2() {
    return __builtin_cpu_supports("avx2");
}

// Use best available SIMD
if (has_avx512()) {
    // Use AVX-512 (16 floats)
} else if (has_avx2()) {
    // Use AVX2 (8 floats)
} else {
    // Scalar fallback
}
```

### Thread Stratejisi
```cpp
// Optimal thread count
int optimal_threads() {
    int cores = std::thread::hardware_concurrency();
    // Use 75% of cores (leave some for OS)
    return std::max(1, (cores * 3) / 4);
}

omp_set_num_threads(optimal_threads());
```

### Memory Alignment
```cpp
// 64-byte alignment (cache line)
alignas(64) float data[hidden_dim];

// Prefetch next cache line
__builtin_prefetch(&data[i + 64], 0, 3);
```

---

## 📝 Uygulama Adımları

### Adım 1: Vectorized Exp/Log (1-2 gün)
1. AVX-512 exp approximation implementasyonu
2. Vectorized log1p
3. Test ve benchmark

### Adım 2: Blelloch Scan (2-3 gün)
1. Up-sweep phase
2. Down-sweep phase
3. OpenMP paralelizasyonu
4. Test ve doğrulama

### Adım 3: Core Recurrence (2-3 gün)
1. Fused kernel implementasyonu
2. SIMD sigmoid
3. MKL entegrasyonu
4. Test

### Adım 4: Entegrasyon (1 gün)
1. PyTorch C++ extension
2. Python bindings
3. Test ve benchmark

---

## 🎉 Sonuç

**CPU için C++ kütüphanesi kritik ve mümkün!**

### Mevcut Durum
- ✅ C++ kodu var ama optimize edilmeli
- ⚠️ Sequential scan, scalar exp()
- ⚠️ Fused kernel yok

### İyileştirme Potansiyeli
- ✅ **10-20x hızlanma** (Associative Scan)
- ✅ **5-10x hızlanma** (Core Recurrence)
- ✅ **Toplam 5-8x** training hızlanması

### Öncelik
1. **Vectorized Exp/Log** (en kritik)
2. **Blelloch Parallel Scan**
3. **Core Recurrence Fused**

**Sonuç**: CPU'da maksimum optimizasyon için C++ kütüphanesi **zorunlu** ve **mümkün**!

---

**Sonraki Adım**: Vectorized Exp/Log implementasyonu ile başla!
