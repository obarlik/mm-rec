/**
 * Test: MMRecModel Backward Pass
 * 
 * Verifies end-to-end gradient flow through the full model.
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/training/forward_cache.h"
#include "mm_rec/training/gradients.h"
#include <iostream>
#include <cassert>

using namespace mm_rec;

void test_model_backward() {
    std::cout << "=== Test: Model Backward End-to-End ===" << std::endl;
    
    // Config
    int64_t vocab = 10;
    int64_t hidden = 8;
    int64_t mem = 4;
    int64_t ffn = 8;
    int64_t num_layers = 2;
    MMRecModelConfig config{vocab, hidden, mem, ffn, num_layers};
    MMRecModel model(config);
    
    // Input & Targets
    int64_t batch = 2;
    int64_t seq = 3;
    Tensor input = Tensor::zeros({batch, seq});
    Tensor targets = Tensor::zeros({batch, seq}); // indices
    
    // Set some ids
    input.data()[0] = 1.0f; input.data()[1] = 2.0f;
    targets.data()[0] = 2.0f; targets.data()[1] = 3.0f;
    
    // 1. Forward
    std::cout << "Running forward..." << std::endl;
    ForwardCache cache;
    model.forward(input, &cache);
    
    // 2. Backward
    std::cout << "Running backward..." << std::endl;
    ModelGradients grads = model.backward(targets, cache);
    
    // 3. Verify Gradients
    std::cout << "Verifying gradients..." << std::endl;
    
    // Embedding Gradients
    assert(grads.embedding_grads.ndim() == 2);
    assert(grads.embedding_grads.size(0) == vocab);
    assert(grads.embedding_grads.size(1) == hidden);
    
    bool has_emb_grad = false;
    for(int i=0; i<grads.embedding_grads.numel(); ++i) {
        if(grads.embedding_grads.data()[i] != 0.0f) has_emb_grad = true;
    }
    assert(has_emb_grad);
    std::cout << "✅ Embedding gradients populated" << std::endl;
    
    // Block Gradients
    assert(grads.block_grads.size() == num_layers);
    
    for(int l=0; l<num_layers; ++l) {
        auto& bg = grads.block_grads[l];
        bool has_gru = false;
        bool has_ffn = false;
        bool has_out = false;
        
        for(int k=0; k<bg.gru_grads.dW_u.numel(); ++k) if(bg.gru_grads.dW_u.data()[k]!=0) has_gru=true;
        for(int k=0; k<bg.ffn_up_grads.dW.numel(); ++k) if(bg.ffn_up_grads.dW.data()[k]!=0) has_ffn=true;
        for(int k=0; k<bg.output_proj_grads.dW.numel(); ++k) if(bg.output_proj_grads.dW.data()[k]!=0) has_out=true;
        
        assert(has_gru);
        assert(has_ffn);
        assert(has_out);
        std::cout << "✅ Layer " << l << " gradients populated (GRU, FFN, Output)" << std::endl;
    }
    
    std::cout << "=== MODEL BACKWARD PASSED ===" << std::endl;
}

int main() {
    test_model_backward();
    return 0;
}
