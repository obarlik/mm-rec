#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/training/trainer.h"
#include "mm_rec/training/optimizer.h"
#include "mm_rec/core/tensor.h"
#include <iostream>
#include <cstring> // memcpy
#include <iomanip>

using namespace mm_rec;

int main() {
    std::cout << "=== Debug: Overfitting Single Batch ===" << std::endl;
    
    // 1. Config
    MMRecModelConfig config;
    config.vocab_size = 16;
    config.hidden_dim = 64;
    config.mem_dim = 64; // Fix: Must match hidden_dim for MMRecBlock architecture
    config.num_layers = 1; // Simplify to 1 layer
    config.num_experts = 1; // Simplify to 1 expert (dense)
    config.top_k = 1;
    
    // 2. Model
    MMRecModel model(config);
    // model.to("cpu"); // Removed

    
    // 3. Optimizer (SGD)
    // We need to construct optimizer manually or use Trainer helper?
    // Trainer creates it. Let's use Trainer for convenience/consistency.
    TrainingConfig train_config;
    train_config.learning_rate = 0.1f; // Aggressive LR for overfitting
    train_config.optimizer_type = "sgd";
    train_config.batch_size = 1;
    
    Trainer trainer(model, train_config);
    
    // 4. Data
    // 4 -> 5, 5 -> 6, 6 -> 7, 7 -> 4
    TrainingBatch batch;
    batch.input_ids = Tensor::zeros({1, 4});
    batch.targets = Tensor::zeros({1, 4});
    
    float input_data[] = {4, 5, 6, 7};
    float target_data[] = {5, 6, 7, 4};
    
    std::memcpy(batch.input_ids.data(), input_data, 4 * sizeof(float));
    std::memcpy(batch.targets.data(), target_data, 4 * sizeof(float));
    batch.loss_mask = Tensor::ones({1, 4}); // All valid
    
    // 5. Loop
    std::cout << "Initial Loss check..." << std::endl;
    
    for(int i=0; i<500; ++i) {
        float loss = trainer.train_step(batch);
        
        if (i % 50 == 0) {
            std::cout << "Step " << i << " Loss: " << loss << std::endl;
        }
        
        if (loss < 0.01f) {
            std::cout << "✅ Converged at step " << i << " Loss: " << loss << std::endl;
            return 0; // Success
        }
    }
    
    std::cout << "❌ Failed to converge." << std::endl;
    return 1;
}
