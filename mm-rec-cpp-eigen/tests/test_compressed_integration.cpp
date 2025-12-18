/**
 * Test: F16C Library Integration
 * 
 * Verifies that the new "CompressedTensor" library works correctly
 * and can be used for Embedding lookups (Gather).
 */

#include "mm_rec/core/compressed_tensor.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace mm_rec;

void test_embedding_gather() {
    std::cout << "=== Test: F16C Embedding Gather ===" << std::endl;
    
    #ifndef __F16C__
    std::cout << "⚠️  Skipping test: F16C not supported by compiler/CPU" << std::endl;
    return;
    #endif
    
    // Create "Embedding Matrix" [Vocab=10, Hidden=16]
    int64_t vocab = 10;
    int64_t hidden = 16;
    Tensor weight = Tensor::ones({vocab, hidden});
    
    // Fill with identifiable data: Row i -> contains value i.0, i.1, ...
    for(int i=0; i<vocab; ++i) {
        for(int j=0; j<hidden; ++j) {
            weight.data()[i * hidden + j] = (float)i + (float)j * 0.1f;
        }
    }
    
    // Compress it!
    std::cout << "Compressing matrix..." << std::endl;
    CompressedTensor c_weight(weight);
    
    // Indices to look up: [1, 5, 9]
    std::cout << "Gathering indices [1, 5, 9]..." << std::endl;
    Tensor indices = Tensor::from_data({1.0f, 5.0f, 9.0f}, {3});
    
    Tensor gathered = c_weight.gather(indices);
    
    // Verify shape
    assert(gathered.size(0) == 3);
    assert(gathered.size(1) == hidden);
    
    // Verify content
    // Row 0 (Index 1) should match original Row 1
    float* g_ptr = gathered.data();
    for(int j=0; j<hidden; ++j) {
        float expected = 1.0f + (float)j * 0.1f;
        float actual = g_ptr[0 * hidden + j];
        // FP16 precision is low. 9.1 -> 9.10156. Err ~ 0.0016.
        if (std::abs(actual - expected) > 1e-2) { 
            std::cerr << "Mismatch at [0, " << j << "]: " << actual << " vs " << expected << std::endl;
            exit(1);
        }
    }
    std::cout << "✅ Row 1 verified" << std::endl;
    
    // Row 2 (Index 9)
    for(int j=0; j<hidden; ++j) {
        float expected = 9.0f + (float)j * 0.1f;
        float actual = g_ptr[2 * hidden + j];
        if (std::abs(actual - expected) > 1e-2) {
            std::cerr << "Mismatch at [2, " << j << "]: " << actual << " vs " << expected << std::endl;
            exit(1);
        }
    }
    std::cout << "✅ Row 9 verified" << std::endl;
    
    std::cout << "🎉 CompressedTensor Library works for Embeddings!" << std::endl;
}

int main() {
    test_embedding_gather();
    return 0;
}
