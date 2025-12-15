/**
 * Test: Full MM-Rec Model
 */

#include "mm_rec/model/mm_rec_model.h"
#include <iostream>
#include <cassert>

using namespace mm_rec;

int main() {
    std::cout << "=== MM-Rec Full Model Test ===" << std::endl;
    
    // Small config for testing
    MMRecModelConfig config{
        /* vocab_size */ 1000,
        /* hidden_dim */ 64,
        /* mem_dim */ 64,
        /* ffn_dim */ 128,
        /* num_layers */ 3
    };
    
    std::cout << "Creating model with:" << std::endl;
    std::cout << "  - Layers: " << config.num_layers << std::endl;
    std::cout << "  - Hidden: " << config.hidden_dim << std::endl;
    std::cout << "  - Vocab: " << config.vocab_size << std::endl;
    
    MMRecModel model(config);
    
    // Test forward pass
    int64_t batch = 2;
    int64_t seq = 8;
    
    Tensor input_ids = Tensor::zeros({batch, seq});
    
    std::cout << "[test_model] After zeros, input_ids.ndim()=" << input_ids.ndim() << std::endl;
    std::cout << "[test_model] input_ids.numel()=" << input_ids.numel() << std::endl;
    
    // Fill with random token IDs
    for (int64_t i = 0; i < batch * seq; ++i) {
        input_ids.data()[i] = static_cast<float>(i % config.vocab_size);
    }
    
    std::cout << "\nRunning forward pass..." << std::endl;
    std::cout << "  Input: [" << batch << ", " << seq << "]" << std::endl;
    
    Tensor logits = model.forward(input_ids);
    
    std::cout << "\n✅ Forward pass complete!" << std::endl;
    std::cout << "  Output shape: [" << logits.size(0) << ", " 
              << logits.size(1) << ", " << logits.size(2) << ", "
              << logits.size(3) << "]" << std::endl;
    std::cout << "  Expected: [" << config.num_layers << ", " 
              << batch << ", " << seq << ", " << config.vocab_size << "]" << std::endl;
    
    // Verify shape
    assert(logits.size(0) == config.num_layers);
    assert(logits.size(1) == batch);
    assert(logits.size(2) == seq);
    assert(logits.size(3) == config.vocab_size);
    
    std::cout << "\n✅ ALL TESTS PASSED!" << std::endl;
    std::cout << "=== Full MM-Rec Model Working! ===" << std::endl;
    
    return 0;
}
