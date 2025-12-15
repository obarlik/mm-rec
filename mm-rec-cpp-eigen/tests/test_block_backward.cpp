/**
 * Test: MMRecBlock Backward Pass
 * 
 * Verifies that backward pass runs and computes valid gradients.
 */

#include "mm_rec/model/mm_rec_block.h"
#include "mm_rec/training/forward_cache.h"
#include "mm_rec/training/gradients.h"
#include <iostream>
#include <cassert>
// #include <cmath> // Missing from original thought, potentially needed

using namespace mm_rec;

void test_block_backward_shapes() {
    std::cout << "=== Test: Block Backward Shapes & Flow ===" << std::endl;
    
    // Config
    int64_t hidden = 8;
    int64_t mem = 4;
    int64_t ffn = 8;
    int64_t vocab = 5;
    MMRecBlockConfig config{hidden, mem, ffn, vocab};
    MMRecBlock block(config);
    
    // Input
    int64_t batch = 2;
    int64_t seq = 3;
    Tensor x = Tensor::randn({batch, seq, hidden}, 0, 1.0);
    Tensor memory = Tensor::zeros({batch, mem});
    
    // Forward
    BlockCache cache;
    auto result = block.forward(x, memory, &cache);
    Tensor output = std::get<0>(result);
    // output: [batch, seq, hidden]
    
    // Backward inputs (Mock gradients)
    Tensor d_output = Tensor::randn({batch, seq, hidden}, 0, 1.0);
    Tensor d_memory_next = Tensor::zeros({batch, mem});
    Tensor d_logits = Tensor::randn({batch, seq, vocab}, 0, 1.0);
    
    // Gradients container
    BlockGradients grads;
    grads.init(hidden, mem, ffn, vocab, 4); // 4 experts by default in config
    
    // Run Backward
    std::cout << "Running backward..." << std::endl;
    auto [dx, dmemory] = block.backward(d_output, d_memory_next, d_logits, cache, grads);
    
    // Verify Shapes
    std::cout << "Verifying shapes..." << std::endl;
    assert(dx.ndim() == 3);
    assert(dx.size(0) == batch);
    assert(dx.size(1) == seq);
    assert(dx.size(2) == hidden);
    std::cout << "✅ dx shape correct" << std::endl;
    
    assert(dmemory.ndim() == 2);
    assert(dmemory.size(0) == batch);
    assert(dmemory.size(1) == mem);
    std::cout << "✅ dmemory shape correct" << std::endl;
    
    // Verify Gradient Population (Non-zero check)
    std::cout << "Verifying gradient accumulation..." << std::endl;
    
    bool has_gru_grad = false;
    // Check W_z grad (part of GRU)
    for(int i=0; i<grads.gru_grads.dW_u.numel(); ++i) {
        if (grads.gru_grads.dW_u.data()[i] != 0.0f) has_gru_grad = true;
    }
    assert(has_gru_grad);
    std::cout << "✅ GRU gradients populated" << std::endl;
    
    bool has_moe_grad = false;
    // Check Gate Gradient
    for(int i=0; i<grads.moe_grads.d_gate.numel(); ++i) {
        if (grads.moe_grads.d_gate.data()[i] != 0.0f) has_moe_grad = true;
    }
    // Check Expert Gradient (at least one expert should be active)
    bool has_expert_grad = false;
    for(const auto& t : grads.moe_grads.d_expert_up_weights) {
        for(int i=0; i<t.numel(); ++i) {
            if (t.data()[i] != 0.0f) has_expert_grad = true;
        }
    }
    
    assert(has_moe_grad);
    assert(has_expert_grad);
    std::cout << "✅ MoE gradients (Gate & Experts) populated" << std::endl;
    
    bool has_out_grad = false;
    for(int i=0; i<grads.output_proj_grads.dW.numel(); ++i) {
        if (grads.output_proj_grads.dW.data()[i] != 0.0f) has_out_grad = true;
    }
    assert(has_out_grad);
    std::cout << "✅ Output Projection gradients populated" << std::endl;
    
    std::cout << "=== BLOCK BACKWARD PASSED ===" << std::endl;
}

int main() {
    test_block_backward_shapes();
    return 0;
}
