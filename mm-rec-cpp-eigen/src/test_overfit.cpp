/**
 * Overfitting Test
 * Train on SAME batch repeatedly - loss MUST go to near zero
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/training/trainer.h"
#include "mm_rec/config/model_config.h"
#include <iostream>

using namespace mm_rec;

int main() {
    std::cout << "=== Overfitting Test ===" << std::endl;
    
    // Tiny model
    MMRecModelConfig config{256, 64, 64, 128, 2, 2, 1};
    MMRecModel model(config);
    
    TrainingConfig train_config;
    train_config.learning_rate = 0.01f;  // High LR
    train_config.batch_size = 2;
    
    Trainer trainer(model, train_config);
    
    // Create TINY batch (2 samples, 8 tokens each)
    Tensor input = Tensor::zeros({2, 8});
    Tensor target = Tensor::zeros({2, 8});
    
    // Pattern: 0,1,2,3,4 repeats
    for (int b = 0; b < 2; ++b) {
        for (int s = 0; s < 8; ++s) {
            input.data()[b * 8 + s] = (float)(s % 5);
            target.data()[b * 8 + s] = (float)((s + 1) % 5);
        }
    }
    
    TrainingBatch batch{input, target};
    
    // Train on SAME batch 50 times
    std::cout << "\nTraining on same batch (should overfit to ~0 loss):\n" << std::endl;
    
    float initial_loss = trainer.train_step(batch);
    std::cout << "Initial: " << initial_loss << std::endl;
    
    for (int i = 0; i < 50; ++i) {
        float loss = trainer.train_step(batch);
        if (i % 10 == 0) {
            std::cout << "Step " << i << ": " << loss << std::endl;
        }
    }
    
    float final_loss = trainer.train_step(batch);
    std::cout << "Final: " << final_loss << std::endl;
    
    // Verification
    if (final_loss < initial_loss * 0.5f) {
        std::cout << "\n✅ PASS: Loss decreased significantly!" << std::endl;
        std::cout << "   " << initial_loss << " → " << final_loss << std::endl;
    } else {
        std::cout << "\n❌ FAIL: Loss barely changed!" << std::endl;
        std::cout << "   " << initial_loss << " → " << final_loss << std::endl;
        std::cout << "   GRADIENTS NOT WORKING!" << std::endl;
        return 1;
    }
    
    return 0;
}
