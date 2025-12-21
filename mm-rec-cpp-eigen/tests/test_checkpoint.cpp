/**
 * Test: Checkpoint Management
 */

#include "mm_rec/business/checkpoint.h"
#include "mm_rec/model/mm_rec_model.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace mm_rec;

int main() {
    std::cout << "=== Checkpoint Management Test ===" << std::endl;
    
    // Create model
    MMRecModelConfig config{
        /* vocab_size */ 1000,
        /* hidden_dim */ 64,
        /* mem_dim */ 64,
        /* ffn_dim */ 128,
        /* num_layers */ 3
    };
    
    MMRecModel model(config);
    std::cout << "Model created" << std::endl;
    
    // Modify embedding weights to make them unique
    auto& embedding = model.get_embedding_weights();
    for (int64_t i = 0; i < embedding.numel(); ++i) {
        embedding.data()[i] = static_cast<float>(i) * 0.001f;
    }
    std::cout << "Embedding weights initialized" << std::endl;
    
    // Save checkpoint
    std::string ckpt_file = "/tmp/test_checkpoint.mmrc";
    CheckpointMetadata save_metadata;
    save_metadata.epoch = 42;
    save_metadata.loss = 1.234f;
    save_metadata.learning_rate = 0.0001f;
    
    std::cout << "\nSaving checkpoint..." << std::endl;
    std::cout << "  Epoch: " << save_metadata.epoch << std::endl;
    std::cout << "  Loss: " << save_metadata.loss << std::endl;
    
    try {
        CheckpointManager::save_checkpoint(ckpt_file, model, save_metadata);
        std::cout << "✅ Checkpoint saved" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Save failed: " << e.what() << std::endl;
        return 1;
    }
    
    // Create new model and load checkpoint
    MMRecModel model2(config);
    CheckpointMetadata load_metadata;
    
    std::cout << "\nLoading checkpoint..." << std::endl;
    
    try {
        CheckpointManager::load_checkpoint(ckpt_file, model2, load_metadata);
        std::cout << "✅ Checkpoint loaded" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Load failed: " << e.what() << std::endl;
        return 1;
    }
    
    // Verify metadata
    std::cout << "\nVerifying metadata..." << std::endl;
    std::cout << "  Loaded epoch: " << load_metadata.epoch << std::endl;
    std::cout << "  Loaded loss: " << load_metadata.loss << std::endl;
    
    assert(load_metadata.epoch == save_metadata.epoch);
    assert(std::abs(load_metadata.loss - save_metadata.loss) < 1e-6f);
    assert(std::abs(load_metadata.learning_rate - save_metadata.learning_rate) < 1e-6f);
    std::cout << "✅ Metadata matches" << std::endl;
    
    // Verify embedding weights
    std::cout << "\nVerifying embedding weights..." << std::endl;
    const auto& orig_emb = model.get_embedding_weights();
    const auto& loaded_emb = model2.get_embedding_weights();
    
    assert(orig_emb.numel() == loaded_emb.numel());
    
    for (int64_t i = 0; i < orig_emb.numel(); ++i) {
        float diff = std::abs(orig_emb.data()[i] - loaded_emb.data()[i]);
        assert(diff < 1e-6f);
    }
    std::cout << "✅ Embedding weights match (" << orig_emb.numel() << " params)" << std::endl;
    
    // Cleanup
    std::remove(ckpt_file.c_str());
    
    std::cout << "\n=== ALL CHECKPOINT TESTS PASSED ===" << std::endl;
    
    return 0;
}
