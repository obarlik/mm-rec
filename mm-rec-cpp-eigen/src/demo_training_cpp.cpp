/**
 * Full C++ Training Demo
 * 
 * Integrates:
 * 1. C++ Data Pipeline (Loader -> Dataset)
 * 2. C++ Model (MM-Rec + MoE)
 * 3. C++ Trainer (SGD)
 * 
 * This is the final proof that we don't need Python.
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/training/trainer.h"
#include "mm_rec/data/data_loader.h"
#include "mm_rec/data/tokenizer.h"
#include "mm_rec/data/tokenizer.h"
#include "mm_rec/core/vulkan_backend.h"
#include "mm_rec/core/auto_tuner.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>

using namespace mm_rec;
namespace fs = std::filesystem;

// Helper to create a dummy dataset file
void generate_data(const std::string& path, int64_t num_tokens) {
    std::cout << "Generating " << num_tokens << " tokens of dummy data..." << std::endl;
    std::ofstream file(path, std::ios::binary);
    
    int32_t magic = 0x4D4D5245;
    int32_t version = 1;
    int64_t count = num_tokens;
    
    file.write(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<char*>(&version), sizeof(version));
    file.write(reinterpret_cast<char*>(&count), sizeof(count));
    
    // Pattern: 0 1 2 0 1 2 ...
    for (int64_t i = 0; i < num_tokens; ++i) {
        int32_t token = i % 100; // Vocab size 100
        file.write(reinterpret_cast<char*>(&token), sizeof(token));
    }
    file.close();
}

int main() {
    std::cout << "🚀 Starting Pure C++ Training Engine" << std::endl;
    
    // 0. Initialize GPU Backend (if available)
    if (VulkanBackend::get().init()) {
        std::cout << "✅ Vulkan Backend Initialized Successfully" << std::endl;
    } else {
        std::cout << "⚠️ Vulkan Backend Initialization Failed (Falling back to CPU)" << std::endl;
    }
    
    // 0.5 Auto-Tune Hardware
    // This will find the optimal Shader and CPU/GPU ratio for the current machine
    std::cout << "\n🛡️  Auto-Tuning Hardware..." << std::endl;
    mm_rec::TuningResult tuning = mm_rec::AutoTuner::tune_system(4096, true);
    std::cout << "✅ Hardware Optimized: " << tuning.peak_gflops << " GFLOPS (Ratio: " << tuning.best_cpu_ratio << ")" << std::endl;
    
    // 1. Setup Data
    // 1. Setup Data
    std::string data_path = "train_data.bin";
    int64_t total_tokens = 500000; // 500K tokens for longer run
    generate_data(data_path, total_tokens);
    
    // 2. Setup Pipeline
    auto dataset = std::make_shared<Dataset>(data_path);
    int64_t batch_size = 512; // Threshold for GPU
    int64_t seq_len = 64;
    // Note: 256 * 64 = 16,384 tokens per batch.
    DataLoader loader(dataset, batch_size, seq_len, /*shuffle=*/false, /*workers=*/4);
    
    std::cout << "📊 Pipeline Ready. Batches: " << loader.total_batches() << std::endl;
    
    // 3. Setup Model
    MMRecModelConfig config;
    config.vocab_size = 100; // Matches data generation
    config.hidden_dim = 128;
    config.mem_dim = 128; // Must match hidden_dim if MoE processes memory
    config.ffn_dim = 256;
    config.num_layers = 4;
    config.num_experts = 4; // MoE enabled
    config.top_k = 2;
    
    MMRecModel model(config);
    std::cout << "🧠 Model Initialized (MoE enabled)" << std::endl;
    
    // 4. Setup Trainer
    TrainingConfig train_config;
    train_config.learning_rate = 0.01f;
    train_config.num_epochs = 1;
    train_config.batch_size = batch_size;
    train_config.validate_every = 100;
    
    Trainer trainer(model, train_config);
    
    // 5. Training Loop
    auto start_time = std::chrono::high_resolution_clock::now();
    int64_t steps = 0;
    
    TrainingBatch batch;
    float running_loss = 0.0f;
    
    std::cout << "🏁 Training Started..." << std::endl;
    
    while(loader.next(batch)) {
        float loss = trainer.train_step(batch);
        running_loss += loss;
        steps++;
        
        if (steps % 1 == 0) { // Print EVERY step
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            float tps = (steps * batch_size * seq_len) / (duration / 1000.0f);
            
            std::cout << "\rStep " << steps 
                      << " | Loss: " << (running_loss / 1.0f) 
                      << " | Speed: " << (int)tps << " tokens/sec" << std::flush;
            running_loss = 0.0f;
        }
    }
    std::cout << "\n✅ Training Complete!" << std::endl;
    
    // Cleanup
    fs::remove(data_path);
    return 0;
}
