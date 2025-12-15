/**
 * Test: Real Training Loop (End-to-End)
 * 
 * Verifies that the complete training loop (Forward -> Backward -> Update)
 * actually reduces loss on a toy problem.
 */

#include "mm_rec/training/trainer.h"
#include "mm_rec/model/mm_rec_model.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace mm_rec;

void test_overfitting() {
    std::cout << "=== Test: Overfitting (Convergence Check) ===" << std::endl;
    
    // 1. Config
    TrainingConfig train_config;
    train_config.learning_rate = 0.05f; // High LR for fast convergence
    train_config.num_epochs = 1;
    train_config.batch_size = 2;
    train_config.warmup_steps = 0;
    
    MMRecModelConfig model_config;
    model_config.vocab_size = 10;
    model_config.hidden_dim = 8;
    model_config.mem_dim = 8;
    model_config.ffn_dim = 16;
    model_config.num_layers = 2; // Deep model
    
    // 2. Init Model & Trainer
    MMRecModel model(model_config);
    Trainer trainer(model, train_config);
    
    // 3. Create Dummy Batch (Sequence prediction task)
    // Input:  [1, 2, 3] -> Target: [2, 3, 4]
    int64_t batch_size = 2;
    int64_t seq_len = 5;
    
    Tensor input = Tensor::zeros({batch_size, seq_len});
    Tensor targets = Tensor::zeros({batch_size, seq_len});
    
    // Pattern: Next token is (curr + 1)
    for(int b=0; b<batch_size; ++b) {
        for(int s=0; s<seq_len; ++s) {
            float val = (float)((s + 1) % 5); // 1, 2, 3, 4, 0
            input.data()[b*seq_len + s] = val;
            targets.data()[b*seq_len + s] = (float)((s + 2) % 5); // Next val
        }
    }
    
    TrainingBatch batch{input, targets};
    
    // 4. Loop
    float initial_loss = trainer.train_step(batch);
    std::cout << "Initial Loss: " << initial_loss << std::endl;
    
    float final_loss = 0;
    for (int i = 0; i < 50; ++i) {
        // We reuse the SAME batch to force overfitting
        float loss = trainer.train_step(batch);
        final_loss = loss;
        if (i % 10 == 0) {
            std::cout << "Step " << i << " Loss: " << loss << std::endl;
        }
    }
    
    std::cout << "Final Loss: " << final_loss << std::endl;
    
    // 5. Verification
    assert(final_loss < initial_loss);
    // Arbitrary threshold for success (should be very low on simple task)
    assert(final_loss < 1.0f); 
    
    std::cout << "✅ Loss decreased significantly (" << initial_loss << " -> " << final_loss << ")" << std::endl;
    std::cout << "=== REAL TRAINING PASSED ===" << std::endl;
}

int main() {
    test_overfitting();
    return 0;
}
