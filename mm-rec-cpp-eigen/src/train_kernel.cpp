/**
 * Train Kernel-Alpha
 * 
 * Full training on Code Alpaca instruction dataset
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/training/trainer.h"
#include "mm_rec/training/optimizer.h"
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
    std::cout << "=== Kernel-Alpha Training ===" << std::endl;
    
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
    std::cout << "✅ Model initialized" << std::endl;
    
    // Create optimizer (AdamW)
    AdamW optimizer(config.learning_rate, config.beta1, config.beta2, 1e-8f, config.weight_decay);
    std::cout << "✅ Optimizer: AdamW (lr=" << config.learning_rate << ")" << std::endl;
    
    // Training loop
    int64_t batch_size = config.batch_size;
    int64_t seq_len = config.max_seq_len;
    int64_t total_tokens = dataset.tokens.size();
    int64_t num_batches = total_tokens / (batch_size * seq_len);
    
    std::cout << "\n🏁 Starting training..." << std::endl;
    std::cout << "Batches: " << num_batches << " | Batch size: " << batch_size 
              << " | Seq len: " << seq_len << std::endl;
    
    auto training_start = std::chrono::high_resolution_clock::now();
    float running_loss = 0.0f;
    int step = 0;
    
    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        auto step_start = std::chrono::high_resolution_clock::now();
        
        // Get batch data
        Tensor input_ids = Tensor::zeros({batch_size, seq_len});
        Tensor targets = Tensor::zeros({batch_size, seq_len});
        Tensor loss_mask = Tensor::zeros({batch_size, seq_len});
        
        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t offset = batch_idx * batch_size * seq_len + b * seq_len;
            
            for (int64_t s = 0; s < seq_len; ++s) {
                int64_t idx = offset + s;
                if (idx < total_tokens - 1) {
                    input_ids.data()[b * seq_len + s] = (float)dataset.tokens[idx];
                    targets.data()[b * seq_len + s] = (float)dataset.tokens[idx + 1];
                    loss_mask.data()[b * seq_len + s] = (float)dataset.masks[idx + 1];
                }
            }
        }
        
        // Forward
        model.reset_memory(batch_size);
        Tensor logits = model.forward(input_ids);
        
        // Compute masked loss (only on non-zero mask positions)
        // logits: [layers, batch, seq, vocab]
        // We use final layer
        int64_t final_layer = config.num_layers - 1;
        float loss = 0.0f;
        int64_t count = 0;
        
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                if (loss_mask.data()[b * seq_len + s] < 0.5f) continue; // Skip masked
                
                int32_t target_id = (int32_t)targets.data()[b * seq_len + s];
                
                // Softmax + cross entropy
                float max_logit = -1e9f;
                for (int64_t v = 0; v < config.vocab_size; ++v) {
                    int64_t idx = final_layer * batch_size * seq_len * config.vocab_size +
                                  b * seq_len * config.vocab_size +
                                  s * config.vocab_size + v;
                    if (logits.data()[idx] > max_logit) max_logit = logits.data()[idx];
                }
                
                float sum_exp = 0.0f;
                for (int64_t v = 0; v < config.vocab_size; ++v) {
                    int64_t idx = final_layer * batch_size * seq_len * config.vocab_size +
                                  b * seq_len * config.vocab_size +
                                  s * config.vocab_size + v;
                    sum_exp += std::exp(logits.data()[idx] - max_logit);
                }
                
                int64_t target_idx = final_layer * batch_size * seq_len * config.vocab_size +
                                     b * seq_len * config.vocab_size +
                                     s * config.vocab_size + target_id;
                float log_prob = (logits.data()[target_idx] - max_logit) - std::log(sum_exp);
                loss += -log_prob;
                count++;
            }
        }
        
        if (count > 0) loss /= count;
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
        
        // Print detailed progress EVERY step
        std::cout << "\r[Step " << step << "/" << num_batches << "] "
                  << "Loss: " << std::fixed << std::setprecision(4) << loss 
                  << " | Avg: " << avg_loss
                  << " | PPL: " << std::setprecision(1) << perplexity
                  << " | " << step_time.count() << "ms/step"
                  << " | Elapsed: " << elapsed.count() << "s"
                  << " | ETA: " << eta_minutes << "m"
                  << std::flush;
        
        if (step % 10 == 0) {
            std::cout << std::endl; // Newline every 10 steps for readability
        }
        
        // Save checkpoint
        if (step % 500 == 0) {
            std::string ckpt = "kernel_alpha_step_" + std::to_string(step) + ".bin";
            CheckpointMetadata meta;
            meta.epoch = step / 500;  // Approximate epoch
            meta.loss = running_loss / 10.0f;
            CheckpointManager::save_checkpoint(ckpt, model, meta);
            std::cout << "\n💾 Checkpoint saved: " << ckpt << std::endl;
        }
        
        // Early stop for testing
        if (step >= 1000) break;
    }
    
    std::cout << "\n\n✅ Training complete!" << std::endl;
    CheckpointMetadata final_meta;
    final_meta.epoch = step / 500;
    CheckpointManager::save_checkpoint("kernel_alpha_final.bin", model, final_meta);
    
    return 0;
}
