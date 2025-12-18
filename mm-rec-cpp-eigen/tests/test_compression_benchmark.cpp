/**
 * Benchmark: CompressedTensor (F16C) vs Standard Tensor (FP32)
 * 
 * Objective: Measure Memory Savings and Read Latency / Bandwidth.
 * Hypothesis: 
 *   - Memory: 50% reduction.
 *   - Latency: Higher (due to decompression overhead), but acceptable given bandwidth potential.
 */

#include "mm_rec/core/compressed_tensor.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace mm_rec;

double benchmark_standard_gather(const Tensor& t, const std::vector<int>& indices, int hidden_dim) {
    auto start = std::chrono::high_resolution_clock::now();
    
    volatile float sum = 0; // Prevent optimization
    const float* data = t.data();
    for(int idx : indices) {
        const float* row = data + idx * hidden_dim;
        for(int j=0; j<hidden_dim; ++j) {
            sum = sum + row[j];
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

double benchmark_compressed_gather(const CompressedTensor& t, const Tensor& t_indices) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // gather() returns a new Tensor (decompressed)
    Tensor out = t.gather(t_indices);
    
    // Touch data to enforce execution
    volatile float sum = 0;
    const float* data = out.data();
    for(int i=0; i<out.numel(); ++i) sum = sum + data[i];
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

int main() {
    std::cout << "=== Benchmark: Memory & Bandwidth (F16C) ===\n" << std::endl;
    
    int64_t vocab = 100000; // 100k embeddings
    int64_t hidden = 256;
    
    std::cout << "Creating Embedding Table [" << vocab << " x " << hidden << "]" << std::endl;
    size_t fp32_bytes = vocab * hidden * 4;
    size_t fp16_bytes = vocab * hidden * 2;
    
    std::cout << "Standard FP32 Memory: " << fp32_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "F16C Compressed Memory: " << fp16_bytes / (1024*1024) << " MB" << std::endl;
    std::cout << "Savings: " << (fp32_bytes - fp16_bytes) / (1024*1024) << " MB (50%)\n" << std::endl;

    Tensor weights = Tensor::randn({vocab, hidden});
    
    std::cout << "Compressing..." << std::endl;
    CompressedTensor c_weights(weights);
    
    // Benchmark Access
    int64_t batch_size = 128;
    int64_t seq_len = 64;
    int64_t num_lookups = batch_size * seq_len;
    
    std::cout << "Simulating Lookup Batch: " << num_lookups << " indices..." << std::endl;
    
    std::vector<int> indices_vec(num_lookups);
    std::vector<float> indices_float(num_lookups);
    for(int i=0; i<num_lookups; ++i) {
        int idx = rand() % vocab;
        indices_vec[i] = idx;
        indices_float[i] = (float)idx;
    }
    // Correct: Pass the vector, not .data()
    Tensor t_indices = Tensor::from_data(indices_float, {num_lookups});

    // Run Standard
    double time_std = benchmark_standard_gather(weights, indices_vec, hidden);
    std::cout << "Standard FP32 Read Time: " << time_std * 1000 << " ms" << std::endl;
    
    // Run Compressed
    double time_comp = benchmark_compressed_gather(c_weights, t_indices);
    std::cout << "Compressed F16C Read + Decompress Time: " << time_comp * 1000 << " ms" << std::endl;
    
    double slowdown = time_comp / time_std;
    std::cout << "\nAnalysis:" << std::endl;
    std::cout << "Latency Cost: " << slowdown << "x slower" << std::endl;
    std::cout << "Memory Benefit: 2.0x smaller" << std::endl;
    
    if (slowdown < 2.0) {
        std::cout << "✅ Result: EXCELLENT tradeoff for memory-constrained systems." << std::endl;
    } else {
        std::cout << "⚠️  Result: High decompression overhead. Use only if RAM is critical." << std::endl;
    }
    
    return 0;
}
