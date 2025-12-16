/**
 * Train Kernel-Alpha
 * 
 * Full training on Code Alpaca instruction dataset
 * Using Trainer class for clean forward/backward/update
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/training/trainer.h"
#include "mm_rec/config/model_config.h"
#include "mm_rec/utils/checkpoint.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <iomanip>

using namespace mm_rec;

// Load binary data with masks
struct MaskedDataset {
    std::vector<int32_t> tokens;
    std::vector<int32_t> masks;
    
    static MaskedDataset load(const std::string& path) {
        MaskedDataset ds;
        std::ifstream file(path, std::ios::binary);
        
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open: " + path);
        }
        
        // Read header
        int32_t magic, version;
        int64_t count;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        
        std::cout << "Loading " << count << " tokens (version " << version << ")..." << std::endl;
        
        // Read tokens
        ds.tokens.resize(count);
        for (int64_t i = 0; i < count; ++i) {
            file.read(reinterpret_cast<char*>(&ds.tokens[i]), sizeof(int32_t));
        }
        
        // Read masks (if version 2)
        if (version >= 2) {
            ds.masks.resize(count);
            for (int64_t i = 0; i < count; ++i) {
                file.read(reinterpret_cast<char*>(&ds.masks[i]), sizeof(int32_t));
            }
            int masked_tokens = 0;
            for (auto m : ds.masks) if (m == 1) masked_tokens++;
            std::cout << "  Train tokens: " << masked_tokens << " (" 
                      << (100.0f * masked_tokens / count) << "%)" << std::endl;
        } else {
            ds.masks.resize(count, 1); // Train on all
        }
        
        return ds;
    }
};

int main(int argc, char** argv) {
    std::cout << "=== Kernel-Alpha Training ===\n" << std::endl;
    
    std::string config_path = "config.txt";
    std::string data_path = "../data/instructions_train.bin";
    
    if (argc > 1) config_path = argv[1];
    if (argc > 2) data_path = argv[2];
    
    // Load config
    ModelConfig config = ModelConfig::from_file(config_path);
    config.print();
    
    // Load data
    MaskedDataset dataset = MaskedDataset::load(data_path);
    
    // Create model
    MMRecModelConfig model_config{
        config.vocab_size,
        config.hidden_dim,
        config.mem_dim,
        config.ffn_dim,
        config.num_layers,
        config.num_experts,
        config.top_k
    };
    
    MMRecModel model(model_config);
    std::cout << "✅ Model initialized\n" << std::endl;
    
    // Create Trainer (handles forward + backward + update!)
    TrainingConfig train_config;
    train_config.learning_rate = config.learning_rate;
    train_config.batch_size = config.batch_size;
    
    Trainer trainer(model, train_config);
    std::cout << "✅ Trainer initialized (lr=" << config.learning_rate << ")\n" << std::endl;
    
    // Training loop
    int64_t batch_size = config.batch_size;
    int64_t seq_len = config.max_seq_len;
    int64_t total_tokens = dataset.tokens.size();
    int64_t num_batches = total_tokens / (batch_size * seq_len);
    
    std::cout << "🏁 Starting training..." << std::endl;
    std::cout << "Batches: " << num_batches << " | Batch size: " << batch_size 
              << " | Seq len: " << seq_len << "\n" << std::endl;
    
    auto training_start = std::chrono::high_resolution_clock::now();
    float running_loss = 0.0f;
    int step = 0;
    
    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        auto step_start = std::chrono::high_resolution_clock::now();
        
        // Prepare batch with STRIDED SAMPLING for independence
        Tensor input_ids = Tensor::zeros({batch_size, seq_len});
        Tensor targets = Tensor::zeros({batch_size, seq_len});
        Tensor loss_mask = Tensor::zeros({batch_size, seq_len});
        
        // Stride ensures sequences are from different parts of dataset
        int64_t stride = num_batches * seq_len;
        
        for (int64_t b = 0; b < batch_size; ++b) {
            // Each sequence starts far apart in the dataset
            int64_t base_offset = b * stride + batch_idx * seq_len;
            
            for (int64_t s = 0; s < seq_len; ++s) {
                int64_t idx = base_offset + s;
                if (idx < total_tokens - 1) {
                    // Use real token IDs (no clipping!)
                    int32_t token = dataset.tokens[idx];
                    int32_t target_token = dataset.tokens[idx + 1];
                    
                    // Bounds check
                    if (token >= config.vocab_size || target_token >= config.vocab_size) {
                        continue;  // Skip out-of-vocab tokens
                    }
                    
                    input_ids.data()[b * seq_len + s] = (float)token;
                    targets.data()[b * seq_len + s] = (float)target_token;
                    // Use mask from preprocessed data
                    loss_mask.data()[b * seq_len + s] = (float)dataset.masks[idx + 1];
                }
            }
        }
        
        TrainingBatch batch{input_ids, targets, loss_mask};
        
        // Train step (forward + backward + update!)
        float loss = trainer.train_step(batch);
        
        running_loss += loss;
        step++;
        
        // Calculate metrics
        auto step_end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(step_end - training_start);
        auto step_time = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start);
        
        float perplexity = std::exp(loss);
        float avg_loss = running_loss / step;
        float avg_step_time = elapsed.count() / (float)step;
        float eta_seconds = avg_step_time * (num_batches - step);
        int eta_minutes = (int)(eta_seconds / 60.0f);
        
        // Print progress
        float current_lr = trainer.get_current_lr();
        std::cout << "\r[Step " << step << "/" << num_batches << "] "
                  << "Loss: " << std::fixed << std::setprecision(4) << loss 
                  << " | Avg: " << avg_loss
                  << " | PPL: " << std::setprecision(1) << perplexity
                  << " | LR: " << std::setprecision(6) << current_lr
                  << " | " << step_time.count() << "ms/step"
                  << " | ETA: " << eta_minutes << "m"
                  << std::flush;
        
        // Always print newline for logging
        std::cout << std::endl;
        
        // Save checkpoint
        if (step % 500 == 0) {
            std::string ckpt = "kernel_small_step_" + std::to_string(step) + ".bin";
            CheckpointMetadata meta;
            meta.epoch = step / 500;
            meta.loss = avg_loss;
            meta.learning_rate = config.learning_rate;
            CheckpointManager::save_checkpoint(ckpt, model, meta);
            std::cout << "\n💾 Checkpoint: " << ckpt << std::endl;
            
            // Keep only last 3 checkpoints (save disk space)
            if (step > 1500) {
                int old_step = step - 2000;  // Delete checkpoint from 4 saves ago
                std::string old_ckpt = "kernel_small_step_" + std::to_string(old_step) + ".bin";
                std::remove(old_ckpt.c_str());
            }
        }
        
        // Early stop (disabled for full training)
        // if (step >= 100) break;
    }
    
    std::cout << "\n\n✅ Training complete!" << std::endl;
    CheckpointMetadata final_meta;
    final_meta.epoch = step / 500;
    final_meta.loss = running_loss / step;
    final_meta.learning_rate = config.learning_rate;
    CheckpointManager::save_checkpoint("kernel_nano_final.bin", model, final_meta);
    
    std::cout << "Final checkpoint saved. Loss: " << (running_loss / step) << std::endl;
    
    return 0;
}
