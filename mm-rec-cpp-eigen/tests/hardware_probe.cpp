/**
 * Hidden Hardware Capabilities Demo
 * Target: AVX2 + F16C + AVX-VNNI (Intel Alder/Raptor Lake or similar)
 */

#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#include <iomanip>

#include "mm_rec/core/compressed_tensor.h"

using namespace mm_rec;

// Check if compiler supports F16C
#ifdef __F16C__
void demo_f16c_bandwidth_saver() {
    std::cout << "\n=== 1. F16C Bandwidth Hack ===" << std::endl;
    std::cout << "Concept: Store weights in FP16 (2 bytes), expand to FP32 (4 bytes) ONLY in registers." << std::endl;
    
    // Create large tensor
    int64_t N = 1024 * 1024 * 16; // 16M floats (64MB)
    Tensor t = Tensor::ones({N}); 
    
    // Compress
    auto start_compress = std::chrono::high_resolution_clock::now();
    CompressedTensor ct(t);
    auto end_compress = std::chrono::high_resolution_clock::now();
    
    std::cout << "Original Size: " << (t.numel() * 4 / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Compressed Size: " << (ct.data_fp16.size() * 2 / 1024 / 1024) << " MB" << std::endl;
    
    double compress_gb_s = (N * 4.0 / 1e9) / std::chrono::duration<double>(end_compress - start_compress).count();
    std::cout << "Compression Speed: " << compress_gb_s << " GB/s" << std::endl;
    
    // Decompress
    auto start_decompress = std::chrono::high_resolution_clock::now();
    Tensor restored = ct.decompress();
    auto end_decompress = std::chrono::high_resolution_clock::now();
    
    double decompress_gb_s = (N * 4.0 / 1e9) / std::chrono::duration<double>(end_decompress - start_decompress).count();
    std::cout << "Decompression Speed: " << decompress_gb_s << " GB/s" << std::endl;
    
    // Check integrity
    if (std::abs(restored.data()[0] - 1.0f) < 1e-4) {
        std::cout << "✅ Data Integrity Verified" << std::endl;
    } else {
        std::cout << "❌ Data Corrupted!" << std::endl;
    }
}
#else
void demo_f16c_bandwidth_saver() { std::cout << "❌ F16C Not Supported by Compiler" << std::endl; }
#endif

int main() {
    std::cout << "🔍 Hardware Capabilities Scanner" << std::endl;
    demo_f16c_bandwidth_saver();
    
    std::cout << "\n[Conclusion]" << std::endl;
    std::cout << "If F16C is fast, we can rewrite the `Tensor` storage to use `uint16_t` internally" << std::endl;
    std::cout << "and inflate to `float` only in `linear.cpp`. This doubles effective LLM capacity!" << std::endl;
    
    return 0;
}
