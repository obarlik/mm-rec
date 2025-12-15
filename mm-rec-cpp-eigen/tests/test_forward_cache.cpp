/**
 * Test: Forward Cache Population
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/training/forward_cache.h"
#include <iostream>
#include <cassert>

using namespace mm_rec;

void test_forward_cache() {
    std::cout << "=== Test: Forward Cache Population ===" << std::endl;
    
    // Config: vocab=10, hidden=8, mem=4, ffn=8, layers=2
    MMRecModelConfig config{10, 8, 4, 8, 2};
    MMRecModel model(config);
    
    // Input: batch=2, seq=3
    Tensor input = Tensor::zeros({2, 3});
    // Set some dummy token ids
    input.data()[0] = 1.0f;
    input.data()[1] = 2.0f;
    
    // Create cache
    ForwardCache cache;
    
    // Forward with cache
    std::cout << "Running forward pass..." << std::endl;
    Tensor output = model.forward(input, &cache);
    
    // Verify cache structure
    std::cout << "Verifying cache..." << std::endl;
    
    // 1. Embedding
    assert(cache.embedded.ndim() == 3);
    assert(cache.embedded.size(0) == 2);
    assert(cache.embedded.size(1) == 3);
    assert(cache.embedded.size(2) == 8); // hidden
    std::cout << "✅ Embedding cache correct" << std::endl;
    
    // 2. Block caches
    assert(cache.block_caches.size() == 2);
    
    for (int i = 0; i < 2; ++i) {
        auto& bc = cache.block_caches[i];
        
        // GRU gates: [batch, seq, mem_dim]
        // mem_dim = 4
        assert(bc.update_gate.ndim() == 3);
        assert(bc.update_gate.size(0) == 2);
        assert(bc.update_gate.size(1) == 3);
        assert(bc.update_gate.size(2) == 4);
        
        // FFN output: [batch, seq, hidden]
        assert(bc.ffn_output.ndim() == 3);
        assert(bc.ffn_output.size(2) == 8);
        
        // Check non-zero (activations should involve values)
        // At least some values should be non-zero after sigmoid
        // Sigmoid(0) = 0.5.
        // So update_gate should not be all zeros.
        bool has_nonzero = false;
        for (int k = 0; k < bc.update_gate.numel(); ++k) {
            if (bc.update_gate.data()[k] != 0.0f) {
                has_nonzero = true;
                break;
            }
        }
        assert(has_nonzero);
    }
    std::cout << "✅ Block caches correct" << std::endl;
    
    // 3. All logits
    assert(cache.all_logits.ndim() == 4);
    assert(cache.all_logits.size(0) == 2); // num_layers
    assert(cache.all_logits.size(1) == 2); // batch
    assert(cache.all_logits.size(2) == 3); // seq
    assert(cache.all_logits.size(3) == 10); // vocab
    std::cout << "✅ All logits cache correct" << std::endl;
    
    std::cout << "=== ALL CACHE TESTS PASSED ===" << std::endl;
}

int main() {
    test_forward_cache();
    return 0;
}
