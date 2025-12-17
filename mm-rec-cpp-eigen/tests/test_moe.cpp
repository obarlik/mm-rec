/**
 * Test: MoE Layer
 * 
 * Verifies Sparse Forward and Dynamic Routing
 */

#include "mm_rec/model/moe.h"
#include "mm_rec/training/optimizer.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace mm_rec;

void test_moe_forward() {
    std::cout << "=== Test: MoE Forward ===" << std::endl;
    
    MoEConfig config;
    config.hidden_dim = 4;
    config.ffn_dim = 8;
    config.num_experts = 4;
    config.top_k = 2; // Select top 2 out of 4
    
    MoELayer moe(config);
    
    // Input: [batch=2, seq=1, hidden=4]
    Tensor x = Tensor::randn({2, 1, 4});
    
    MoECache cache;
    Tensor output = moe.forward(x, &cache);
    
    // Verify shapes
    assert(output.size(0) == 2);
    assert(output.size(1) == 1);
    assert(output.size(2) == 4); // hidden dim preserved
    
    // Verify cache population
    assert(cache.router_logits.size(1) == 4); // experts
    assert(cache.routing_weights.size(1) == 2); // top_k
    assert(cache.selected_indices.size(1) == 2); // top_k
    
    // Check indices range
    float* indices = cache.selected_indices.data();
    for(int i=0; i<4; ++i) { // 2 batch * 2 top_k
        assert(indices[i] >= 0 && indices[i] < 4);
    }
    
    // Check Aux Loss
    std::cout << "Aux Loss: " << cache.aux_loss << std::endl;
    assert(cache.aux_loss > 0.0f); // Should be positive
    
    std::cout << "✅ Forward shapes and cache correct" << std::endl;
}

void test_moe_backward() {
    std::cout << "=== Test: MoE Backward ===" << std::endl;
    
    MoEConfig config;
    config.hidden_dim = 4;
    config.ffn_dim = 8;
    config.num_experts = 4;
    config.top_k = 1; // Simplify to Top-1 for gradient check
    
    MoELayer moe(config);
    
    Tensor x = Tensor::randn({1, 1, 4});
    MoECache cache;
    Tensor output = moe.forward(x, &cache);
    
    Tensor d_output = Tensor::ones({1, 1, 4});
    MoEGradients grads;
    grads.init(config);
    
    Tensor dx = moe.backward(d_output, x, cache, grads);
    
    // Verify gradient population
    assert(dx.sizes() == x.sizes());
    
    // Check Gate Gradients
    // Should be non-zero (unless extremely unlucky with initialization/activation)
    bool has_gate_grad = false;
    for(int i=0; i<grads.d_gate.numel(); ++i) {
        if(std::abs(grads.d_gate.data()[i]) > 1e-6) has_gate_grad = true;
    }
    std::cout << "Gate Gradient Present: " << (has_gate_grad ? "Yes" : "No") << std::endl;
    assert(has_gate_grad);
    
    // Check Expert Gradients
    // Since we ran 1 token with Top-1, EXACTLY ONE expert should have gradients.
    int active_experts = 0;
    for(int e=0; e<config.num_experts; ++e) {
        float sum = 0;
        for(int k=0; k<grads.d_expert_up_weights[e].numel(); ++k) {
            sum += std::abs(grads.d_expert_up_weights[e].data()[k]);
        }
        if(sum > 1e-6) active_experts++;
    }
    
    std::cout << "Active Experts (Should be 1): " << active_experts << std::endl;
    assert(active_experts == 1);
    
    std::cout << "✅ Backward routing correct (Sparsity verified)" << std::endl;
}

int main() {
    test_moe_forward();
    test_moe_backward();
    return 0;
}
