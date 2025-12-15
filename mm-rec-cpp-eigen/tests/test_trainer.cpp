/**
 * Test: Trainer
 */

#include "mm_rec/training/trainer.h"
#include "mm_rec/model/mm_rec_model.h"
#include <iostream>
#include <cassert>

using namespace mm_rec;

int main() {
    std::cout << "=== Trainer Test ===" << std::endl;
    
    // Create model
    MMRecModelConfig model_config{
        /* vocab_size */ 100,
        /* hidden_dim */ 32,
        /* mem_dim */ 32,
        /* ffn_dim */ 64,
        /* num_layers */ 2
    };
    
    MMRecModel model(model_config);
    std::cout << "Model created" << std::endl;
    
    // Create training config
    TrainingConfig train_config;
    train_config.learning_rate = 1e-3f;
    train_config.warmup_steps = 10;
    train_config.total_steps = 100;
    
    // Create trainer
    Trainer trainer(model, train_config);
    std::cout << "Trainer created" << std::endl;
    
    // Create dummy batch
    TrainingBatch batch;
    batch.input_ids = Tensor::zeros({4, 8});  // batch=4, seq=8
    batch.targets = Tensor::zeros({4, 8});
    
    // Fill with dummy data
    for (int i = 0; i < 32; ++i) {
        batch.input_ids.data()[i] = static_cast<float>(i % 100);
        batch.targets.data()[i] = static_cast<float>((i + 1) % 100);
    }
    
    std::cout << "\n=== Training Steps ===" << std::endl;
    
    // Run a few training steps
    for (int step = 0; step < 5; ++step) {
        float loss = trainer.train_step(batch);
        float lr = trainer.get_current_lr();
        
        std::cout << "Step " << trainer.get_step() 
                  << ": loss=" << loss 
                  << ", lr=" << lr << std::endl;
        
        assert(loss > 0);  // Loss should be positive
        assert(lr > 0);    // LR should be positive
    }
    
    std::cout << "\n=== Validation Steps ===" << std::endl;
    
    // Run validation steps
    float total_val_loss = 0;
    int num_val_steps = 3;
    
    for (int i = 0; i < num_val_steps; ++i) {
        float val_loss = trainer.validate_step(batch);
        std::cout << "Val step " << i << ": loss=" << val_loss << std::endl;
        total_val_loss += val_loss;
    }
    
    float avg_val_loss = total_val_loss / num_val_steps;
    std::cout << "Average val loss: " << avg_val_loss << std::endl;
    
    // Test LR schedule progression
    std::cout << "\n=== LR Schedule ===" << std::endl;
    std::cout << "Step 0 (warmup): " << trainer.get_current_lr() << std::endl;
    
    for (int i = 0; i < 10; ++i) {
        trainer.increment_step();
    }
    std::cout << "Step 15 (post-warmup): " << trainer.get_current_lr() << std::endl;
    
    std::cout << "\n=== ALL TRAINER TESTS PASSED ===" << std::endl;
    
    return 0;
}
