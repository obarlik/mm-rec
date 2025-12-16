/**
 * Adaptive Curriculum Training
 * 
 * Implements intelligent sample selection based on difficulty:
 * 1. Discovery: Categorize all samples (easy/medium/hard)
 * 2. Iterative: Train medium, re-evaluate hard → medium
 * 3. Convergence: When no more promotions or medium exhausted
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/config/model_config.h"
#include "mm_rec/training/trainer.h"
#include "mm_rec/training/sample_tracker.h"
#include "mm_rec/utils/checkpoint.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>

using namespace mm_rec;

// Simple binary dataset structure (matching train_kernel)
struct InstructionDataset {
    std::vector<int32_t> tokens;
    std::vector<int32_t> masks;  // int32_t not int8_t!
    int32_t version = 2;
    int64_t token_count = 0;
    
    bool load_binary(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) {
            std::cerr << "Failed to open: " << path << std::endl;
            return false;
        }
        
        // Read header (magic, version, count)
        int32_t magic;
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(int32_t));
        ifs.read(reinterpret_cast<char*>(&version), sizeof(int32_t));
        
        int64_t count;
        ifs.read(reinterpret_cast<char*>(&count), sizeof(int64_t));
        
        std::cout << "Loading " << count << " tokens (version " << version << ")..." << std::endl;
        
        // Read tokens
        tokens.resize(count);
        for (int64_t i = 0; i < count; ++i) {
            ifs.read(reinterpret_cast<char*>(&tokens[i]), sizeof(int32_t));
        }
        
        // Read masks (version 2+)
        if (version >= 2) {
            masks.resize(count);
            for (int64_t i = 0; i < count; ++i) {
                ifs.read(reinterpret_cast<char*>(&masks[i]), sizeof(int32_t));
            }
            
            int masked_count = 0;
            for (auto m : masks) if (m == 1) masked_count++;
            token_count = masked_count;
            
            std::cout << "  Train tokens: " << token_count << " (" 
                      << std::fixed << std::setprecision(3)
                      << (100.0f * token_count / count) << "%)" << std::endl;
        } else {
            masks.resize(count, 1);
            token_count = count;
        }
        
        return true;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <config_file> <data_file>" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string data_path = argv[2];
    
    std::cout << "=== Adaptive Curriculum Training ===" << std::endl;
    std::cout << std::endl;
    
    // Load config (same as train_kernel)
    MMRecModelConfig config;
    std::ifstream cfg_file(config_path);
    if (!cfg_file) {
        std::cerr << "Failed to open config: " << config_path << std::endl;
        return 1;
    }
    
    std::string line;
    float learning_rate = 0.01f;
    int batch_size = 8;
    int max_seq_len = 128;
    float uboo_weight = 0.5f;
    
    while (std::getline(cfg_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        auto eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);
        
        // Model config
        if (key == "vocab_size") config.vocab_size = std::stoi(value);
        else if (key == "hidden_dim") config.hidden_dim = std::stoi(value);
        else if (key == "mem_dim") config.mem_dim = std::stoi(value);
        else if (key == "ffn_dim") config.ffn_dim = std::stoi(value);
        else if (key == "num_layers") config.num_layers = std::stoi(value);
        else if (key == "num_experts") config.num_experts = std::stoi(value);
        else if (key == "top_k") config.top_k = std::stoi(value);
        
        // Training params (separate variables)
        else if (key == "learning_rate") learning_rate = std::stof(value);
        else if (key == "batch_size") batch_size = std::stoi(value);
        else if (key == "max_seq_len") max_seq_len = std::stoi(value);
        else if (key == "uboo_weight") uboo_weight = std::stof(value);
    }
    
    // Read adaptive parameters
    float easy_threshold = 50.0f;
    float hard_threshold = 500.0f;
    int max_iterations = 15;
    
    cfg_file.clear();
    cfg_file.seekg(0);
    while (std::getline(cfg_file, line)) {
        if (line.find("easy_threshold") != std::string::npos) {
            sscanf(line.c_str(), "easy_threshold=%f", &easy_threshold);
        } else if (line.find("hard_threshold") != std::string::npos) {
            sscanf(line.c_str(), "hard_threshold=%f", &hard_threshold);
        }
    }
    cfg_file.close();
    
    // Print config
    std::cout << "=== MM-Rec Model Configuration ===" << std::endl;
    std::cout << "Architecture:" << std::endl;
    std::cout << "  vocab_size: " << config.vocab_size << std::endl;
    std::cout << "  hidden_dim: " << config.hidden_dim << std::endl;
    std::cout << "  mem_dim: " << config.mem_dim << std::endl;
    std::cout << "  ffn_dim: " << config.ffn_dim << std::endl;
    std::cout << "  num_layers: " << config.num_layers << std::endl;
    std::cout << "  num_experts: " << config.num_experts << std::endl;
    std::cout << "  top_k: " << config.top_k << std::endl;
    std::cout << std::endl;
    std::cout << "Training:" << std::endl;
    std::cout << "  learning_rate: " << learning_rate << std::endl;
    std::cout << "  batch_size: " << batch_size << std::endl;
    std::cout << "  max_seq_len: " << max_seq_len << std::endl;
    std::cout << std::endl;
    std::cout << "UBOO:" << std::endl;
    std::cout << "  uboo_weight: " << uboo_weight << std::endl;
    std::cout << "\nAdaptive Parameters:" << std::endl;
    std::cout << "  Easy threshold: " << easy_threshold << std::endl;
    std::cout << "  Hard threshold: " << hard_threshold << std::endl;
    std::cout << "===================================" << std::endl;
    
    // Load dataset
    InstructionDataset dataset;
    dataset.load_binary(data_path);
    
    std::cout << "Loading " << dataset.tokens.size() << " tokens (version " 
              << dataset.version << ")..." << std::endl;
    std::cout << "  Train tokens: " << dataset.token_count << " (" 
              << std::fixed << std::setprecision(3)
              << (100.0f * dataset.token_count / dataset.tokens.size()) 
              << "%)" << std::endl;
    
    // Initialize model
    MMRecModel model(config);
    std::cout << "✅ Model initialized" << std::endl;
    std::cout << std::endl;
    
    // Initialize trainer
    TrainingConfig train_config;
    train_config.learning_rate = learning_rate;
    train_config.batch_size = batch_size;
    Trainer trainer(model, train_config);
    
    std::cout << "✅ Trainer initialized (lr=" << learning_rate << ")" << std::endl;
    std::cout << std::endl;
    
    // Calculate batches
    int64_t seq_len = max_seq_len;
    int64_t total_tokens = dataset.tokens.size();
    int64_t num_batches = total_tokens / (batch_size * seq_len);
    int64_t stride = num_batches * seq_len;
    
    // ========================================
    // PHASE 1: DISCOVERY - Categorize all samples
    // ========================================
    
    std::cout << "🔍 DISCOVERY PHASE: Categorizing samples..." << std::endl;
    std::cout << "Batches: " << num_batches << " | Thresholds: " 
              << easy_threshold << " / " << hard_threshold << std::endl;
    std::cout << std::endl;
    
    SampleTracker tracker(easy_threshold, hard_threshold);
    
    auto discovery_start = std::chrono::high_resolution_clock::now();
    
    for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        // Prepare batch (strided sampling)
        Tensor input_ids = Tensor::zeros({batch_size, seq_len});
        Tensor targets = Tensor::zeros({batch_size, seq_len});
        Tensor loss_mask = Tensor::zeros({batch_size, seq_len});
        
        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t base_offset = b * stride + batch_idx * seq_len;
            
            for (int64_t s = 0; s < seq_len; ++s) {
                int64_t idx = base_offset + s;
                if (idx < total_tokens - 1) {
                    int32_t token = dataset.tokens[idx];
                    int32_t target_token = dataset.tokens[idx + 1];
                    
                    if (token >= config.vocab_size || target_token >= config.vocab_size) {
                        continue;
                    }
                    
                    input_ids.data()[b * seq_len + s] = (float)token;
                    targets.data()[b * seq_len + s] = (float)target_token;
                    loss_mask.data()[b * seq_len + s] = (float)dataset.masks[idx + 1];
                }
            }
        }
        
        TrainingBatch batch{input_ids, targets, loss_mask};
        
        // Forward only (no backward!)
        auto batch_start = std::chrono::high_resolution_clock::now();
        float loss = trainer.forward_only(batch);
        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - batch_start).count();
        
        float ppl = std::exp(loss);
        
        // Categorize
        tracker.add_sample(batch_idx, ppl);
        
        // Progress (EVERY batch for debugging, with timing!)
        DifficultyCategory cat = tracker.categorize(ppl);
        const char* cat_str = (cat == DifficultyCategory::EASY) ? "EASY" :
                              (cat == DifficultyCategory::MEDIUM) ? "MEDIUM" : "HARD";
        std::cout << "[Batch " << (batch_idx + 1) << "/" << num_batches 
                  << "] PPL: " << std::fixed << std::setprecision(1) << ppl
                  << " → " << cat_str 
                  << " (" << batch_ms << "ms)"
                  << std::endl;
        std::cout.flush();  // FORCE OUTPUT!
        
        // Stop after 10 for testing
        if (batch_idx >= 9) {
            std::cout << "\n⚠️  STOPPING AFTER 10 BATCHES FOR PERFORMANCE TEST" << std::endl;
            break;
        }
    }
    
    auto discovery_end = std::chrono::high_resolution_clock::now();
    auto discovery_time = std::chrono::duration_cast<std::chrono::seconds>(discovery_end - discovery_start);
    
    // Statistics
    int easy_count, medium_count, hard_count;
    tracker.get_statistics(easy_count, medium_count, hard_count);
    
    std::cout << std::endl;
    std::cout << "📊 DISCOVERY COMPLETE (" << discovery_time.count() << "s):" << std::endl;
    std::cout << "  Easy:   " << std::setw(5) << easy_count 
              << " samples (" << std::setw(2) << (100 * easy_count / num_batches) << "%) → Will skip" << std::endl;
    std::cout << "  Medium: " << std::setw(5) << medium_count 
              << " samples (" << std::setw(2) << (100 * medium_count / num_batches) << "%) → Training focus" << std::endl;
    std::cout << "  Hard:   " << std::setw(5) << hard_count 
              << " samples (" << std::setw(2) << (100 * hard_count / num_batches) << "%) → Deferred" << std::endl;
    std::cout << std::endl;
    
    // Save difficulty map
    tracker.save("difficulty_map.bin");
    std::cout << "Saved: difficulty_map.bin" << std::endl;
    std::cout << std::endl;
    
    // ========================================
    // PHASE 2: ITERATIVE TRAINING
    // ========================================
    
    std::cout << "📚 ITERATIVE TRAINING:" << std::endl;
    std::cout << "Training medium samples until convergence..." << std::endl;
    std::cout << std::endl;
    
    int iteration = 0;
    int total_steps = 0;
    float running_loss = 0.0f;
    
    while (iteration < max_iterations) {
        iteration++;
        
        // Get trainable batches (medium difficulty)
        auto medium_batches = tracker.get_trainable_batches();
        
        if (medium_batches.empty()) {
            std::cout << "✅ All samples learned! (medium exhausted)" << std::endl;
            break;
        }
        
        std::cout << "--- Iteration " << iteration << " ---" << std::endl;
        std::cout << "Training " << medium_batches.size() << " medium samples..." << std::endl;
        
        // Train all medium samples
        for (size_t i = 0; i < medium_batches.size(); ++i) {
            int batch_idx = medium_batches[i];
            
            // Prepare batch (same as discovery)
            Tensor input_ids = Tensor::zeros({batch_size, seq_len});
            Tensor targets = Tensor::zeros({batch_size, seq_len});
            Tensor loss_mask = Tensor::zeros({batch_size, seq_len});
            
            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t base_offset = b * stride + batch_idx * seq_len;
                
                for (int64_t s = 0; s < seq_len; ++s) {
                    int64_t idx = base_offset + s;
                    if (idx < total_tokens - 1) {
                        int32_t token = dataset.tokens[idx];
                        int32_t target_token = dataset.tokens[idx + 1];
                        
                        if (token >= config.vocab_size || target_token >= config.vocab_size) {
                            continue;
                        }
                        
                        input_ids.data()[b * seq_len + s] = (float)token;
                        targets.data()[b * seq_len + s] = (float)target_token;
                        loss_mask.data()[b * seq_len + s] = (float)dataset.masks[idx +1];
                    }
                }
            }
            
            TrainingBatch batch{input_ids, targets, loss_mask};
            
            // Train step (backward + update!)
            float loss = trainer.train_step(batch);
            running_loss += loss;
            total_steps++;
            
            // Progress
            if ((i + 1) % 100 == 0) {
                float avg_loss = running_loss / total_steps;
                float ppl = std::exp(loss);
                std::cout << "[Step " << total_steps << "] "
                          << "Loss: " << std::fixed << std::setprecision(4) << loss
                          << " | Avg: " << avg_loss
                          << " | PPL: " << std::setprecision(1) << ppl
                          << std::endl;
            }
        }
        
        std::cout << "Iteration " << iteration << " complete. Total steps: " << total_steps << std::endl;
        std::cout << std::endl;
        
        // Re-evaluate HARD samples
        auto hard_batches = tracker.get_batches_by_category(DifficultyCategory::HARD);
        
        if (hard_batches.empty()) {
            std::cout << "✅ No hard samples remaining!" << std::endl;
            break;
        }
        
        std::cout << "🔄 Re-evaluating " << hard_batches.size() << " hard samples..." << std::endl;
        
        int promoted = 0;
        for (size_t i = 0; i < hard_batches.size(); ++i) {
            // Re-evaluate every Nth sample (optimization)
            if (i % 10 != 0 && i != hard_batches.size() - 1) continue;
            
            int batch_idx = hard_batches[i];
            
            // Prepare batch
            Tensor input_ids = Tensor::zeros({batch_size, seq_len});
            Tensor targets = Tensor::zeros({batch_size, seq_len});
            Tensor loss_mask = Tensor::zeros({batch_size, seq_len});
            
            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t base_offset = b * stride + batch_idx * seq_len;
                
                for (int64_t s = 0; s < seq_len; ++s) {
                    int64_t idx = base_offset + s;
                    if (idx < total_tokens - 1) {
                        int32_t token = dataset.tokens[idx];
                        int32_t target_token = dataset.tokens[idx + 1];
                        
                        if (token >= config.vocab_size || target_token >= config.vocab_size) {
                            continue;
                        }
                        
                        input_ids.data()[b * seq_len + s] = (float)token;
                        targets.data()[b * seq_len + s] = (float)target_token;
                        loss_mask.data()[b * seq_len + s] = (float)dataset.masks[idx + 1];
                    }
                }
            }
            
            TrainingBatch batch{input_ids, targets, loss_mask};
            
            // Forward only
            float loss = trainer.forward_only(batch);
            float new_ppl = std::exp(loss);
            
            // Update category
            tracker.update_sample(batch_idx, new_ppl);
            
            // Check if promoted
            if (new_ppl < hard_threshold) {
                promoted++;
            }
        }
        
        std::cout << "Promoted to medium: " << promoted << " samples" << std::endl;
        std::cout << std::endl;
        
        // Convergence check
        if (promoted == 0) {
            std::cout << "⚠️  Model plateau: No samples promoted" << std::endl;
            std::cout << "Training converged at iteration " << iteration << std::endl;
            break;
        }
    }
    
    std::cout << std::endl;
    std::cout << "✅ TRAINING COMPLETE!" << std::endl;
    std::cout << "Total steps: " << total_steps << std::endl;
    std::cout << "Total iterations: " << iteration << std::endl;
    
    // Final statistics
    tracker.get_statistics(easy_count, medium_count, hard_count);
    std::cout << std::endl;
    std::cout << "Final distribution:" << std::endl;
    std::cout << "  Easy:   " << easy_count << std::endl;
    std::cout << "  Medium: " << medium_count << std::endl;
    std::cout << "  Hard:   " << hard_count << " (couldn't learn)" << std::endl;
    
    // Save final checkpoint
    CheckpointMetadata meta;
    meta.epoch = iteration;
    meta.loss = running_loss / total_steps;
    meta.learning_rate = learning_rate;
    
    CheckpointManager::save_checkpoint("kernel_adaptive_final.bin", model, meta);
    std::cout << "\n💾 Checkpoint saved: kernel_adaptive_final.bin" << std::endl;
    
    return 0;
}
