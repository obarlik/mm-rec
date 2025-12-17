/**
 * Adaptive Curriculum Training (Online Sampling Optimized)
 * 
 * Strategy:
 * 1. For each batch, process ONLY the first sequence (fast forward).
 * 2. Calculate PPL of that single sequence.
 * 3. If Medium: Train FULL batch.
 * 4. If Hard: Defer batch (skip for now).
 * 5. If Easy: Skip batch.
 * 
 * This eliminates the 30-hour "Discovery Phase" and starts training immediately.
 */

#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/config/model_config.h"
#include "mm_rec/training/trainer.h"
#include "mm_rec/training/sample_tracker.h"
#include "mm_rec/utils/checkpoint.h"
#include "mm_rec/core/memory_manager.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace mm_rec;

// Simple binary dataset structure
struct InstructionDataset {
    std::vector<int32_t> tokens;
    std::vector<int32_t> masks;
    int32_t version = 2;
    int64_t token_count = 0;
    
    bool load_binary(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) {
            std::cerr << "Failed to open: " << path << std::endl;
            return false;
        }
        
        // Read header
        int32_t magic;
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(int32_t));
        ifs.read(reinterpret_cast<char*>(&version), sizeof(int32_t));
        
        int64_t count;
        ifs.read(reinterpret_cast<char*>(&count), sizeof(int64_t));
        
        std::cout << "Loading " << count << " tokens (version " << version << ")..." << std::endl;
        
        // Read tokens
        tokens.resize(count);
        ifs.read(reinterpret_cast<char*>(tokens.data()), count * sizeof(int32_t));
        
        // Read masks
        if (version >= 2) {
            masks.resize(count);
            ifs.read(reinterpret_cast<char*>(masks.data()), count * sizeof(int32_t));
            
            int64_t masked = 0;
            for (auto m : masks) if (m == 1) masked++;
            token_count = masked;
        } else {
            masks.resize(count, 1);
            token_count = count;
        }
        
        return true;
    }
};

#include <sys/resource.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    // Automatically lower process priority to background (nice 19)
    // This allows the user to continue using the PC without lag.
    id_t pid = getpid();
    if (setpriority(PRIO_PROCESS, pid, 19) == 0) {
        std::cout << "ℹ️  Process priority set to lowest (nice 19) for background training." << std::endl;
    } else {
        std::cerr << "⚠️  Failed to set process priority." << std::endl;
    }

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <config_file> <data_file>" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string data_path = argv[2];
    
    std::cout << "=== Adaptive Curriculum Training (Online) ===" << std::endl;
    
    // Load config manually to get training params
    MMRecModelConfig config;
    float learning_rate = 0.01f;
    int batch_size = 8;
    int max_seq_len = 128; // Restored
    // Adaptive params (defaults)
    float easy_threshold = 50.0f;
    float hard_threshold = 500.0f;
    int max_iterations = 15;
    
    std::ifstream cfg_file(config_path);
    std::string line;
    while (std::getline(cfg_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        
        // Remove inline comments
        auto comment_pos = val.find('#');
        if (comment_pos != std::string::npos) {
            val = val.substr(0, comment_pos);
        }
        
        // Trim whitespace (simple)
        while (!val.empty() && std::isspace(val.back())) val.pop_back();
        while (!val.empty() && std::isspace(val.front())) val.erase(0, 1);

        if (key == "vocab_size") config.vocab_size = std::stoi(val);
        else if (key == "hidden_dim") config.hidden_dim = std::stoi(val);
        else if (key == "mem_dim") config.mem_dim = std::stoi(val);
        else if (key == "ffn_dim") config.ffn_dim = std::stoi(val);
        else if (key == "num_layers") config.num_layers = std::stoi(val);
        else if (key == "num_experts") config.num_experts = std::stoi(val);
        else if (key == "top_k") config.top_k = std::stoi(val);
        else if (key == "learning_rate") learning_rate = std::stof(val);
        else if (key == "batch_size") batch_size = std::stoi(val);
        else if (key == "max_seq_len") max_seq_len = std::stoi(val);
        // Adaptive params
        else if (key == "easy_threshold") easy_threshold = std::stof(val);
        else if (key == "hard_threshold") hard_threshold = std::stof(val);
        else if (key == "max_iterations") max_iterations = std::stoi(val);
        
        if (key == "uboo_weight") config.uboo_weight = std::stof(val);
    }
    
    std::cout << "Config Loaded: Hard Threshold = " << hard_threshold << std::endl;
    std::cout << "               Batch Size = " << batch_size << std::endl;
    
    // Load dataset
    InstructionDataset dataset;
    if (!dataset.load_binary(data_path)) return 1;
    
    // Initialize model
    MMRecModel model(config);
    
    // Initialize trainer
    TrainingConfig train_config;
    train_config.learning_rate = learning_rate;
    train_config.batch_size = batch_size;
    Trainer trainer(model, train_config);
    
    // Calc batches
    int64_t seq_len = max_seq_len;
    int64_t total_tokens = dataset.tokens.size();
    int64_t num_batches = total_tokens / (batch_size * seq_len);
    int64_t stride = num_batches * seq_len; 
    // Wait, stride logic in train_kernel was: 
    // stride = num_batches * seq_len. 
    // seq 0: [0..seq_len]
    // seq 1: [stride..stride+seq_len]
    // This ensures seqs in a batch are far apart (Decorrelated).
    // Correct.
    
    std::cout << "\n🚀 STARTING ONLINE ADAPTIVE TRAINING" << std::endl;
    std::cout << "Batches: " << num_batches << " | Stride: " << stride << std::endl;
    
    int64_t total_steps = 0;
    
    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        std::cout << "\n=== Epoch " << iteration << " ===" << std::endl;
        
        int64_t epoch_easy = 0;
        int64_t epoch_medium = 0;
        int64_t epoch_hard = 0;
        float epoch_loss = 0.0f;
        int epoch_steps = 0;
        auto epoch_start_time = std::chrono::steady_clock::now();
        
        for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // 1. PROBE (Fast Check)
            // Use FIRST sequence of the batch
            Tensor probe_input = Tensor::zeros({1, seq_len});
            Tensor probe_target = Tensor::zeros({1, seq_len});
            Tensor probe_mask = Tensor::zeros({1, seq_len});
            
            // Offset for sequence 0 of this batch
            // In train_kernel: batch.input_ids[b, s] = tokens[b*stride + batch_idx*seq_len + s]
            // So for b=0: tokens[batch_idx*seq_len + s]
            int64_t base_offset = batch_idx * seq_len;
            
            bool valid = true;
            for (int64_t s = 0; s < seq_len; ++s) {
                int64_t idx = base_offset + s;
                if (idx >= total_tokens - 1) { valid = false; break; }
                probe_input.data()[s] = (float)dataset.tokens[idx];
                probe_target.data()[s] = (float)dataset.tokens[idx+1];
                if (dataset.masks.size() > 0)
                    probe_mask.data()[s] = (float)dataset.masks[idx+1];
                else
                    probe_mask.data()[s] = 1.0f; // Default to 1 if no masks
            }
            if (!valid) continue;
            
            // Forward probe
            TrainingBatch probe_batch{probe_input, probe_target, probe_mask};
            float probe_loss = trainer.forward_only(probe_batch);
            float ppl = std::exp(probe_loss);
            
            // Log progress every 10 batches to show scanning is active
            if (batch_idx % 10 == 0) {
                 auto now = std::chrono::steady_clock::now();
                 double elapsed_sec = std::chrono::duration<double>(now - epoch_start_time).count();
                 double batches_per_sec = (batch_idx + 1) / elapsed_sec;
                 double remaining_batches = num_batches - batch_idx;
                 double eta_sec = remaining_batches / batches_per_sec;
                 
                 // Format ETA
                 int eta_m = (int)(eta_sec / 60);
                 int eta_s = (int)(eta_sec) % 60;
                 
                 // Memory usage
                 size_t mem_bytes = MemoryManager::instance().get_total_memory();
                 double mem_mb = mem_bytes / (1024.0 * 1024.0);
                 
                 std::cout << "[Ep " << iteration << "] Batch " << batch_idx << "/" << num_batches
                           << " | E:" << epoch_easy << " M:" << epoch_medium << " H:" << epoch_hard
                           << " | PPL: " << std::fixed << std::setprecision(1) << ppl
                           << " | " << std::setprecision(2) << batches_per_sec << " it/s"
                           << " | ETA: " << eta_m << "m " << eta_s << "s"
                           << " | Mem: " << (int)mem_mb << "MB"
                           << std::endl << std::flush;
            }
            
            // Decision
            if (ppl < easy_threshold) {
                epoch_easy++; // SKIP
                // CRITICAL: Reset memory arena
                MemoryManager::instance().reset_arena();
                continue;
            }
            if (ppl > hard_threshold) {
                epoch_hard++; // DEFER (Skip for now)
                // CRITICAL: Reset memory arena
                MemoryManager::instance().reset_arena();
                continue;
            }
            
            // MEDIUM -> TRAIN
            epoch_medium++;
            
            // Construct FULL batch
            Tensor input_ids = Tensor::zeros({batch_size, seq_len});
            Tensor targets = Tensor::zeros({batch_size, seq_len});
            Tensor loss_mask = Tensor::zeros({batch_size, seq_len});
            
            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t offset = b * stride + batch_idx * seq_len;
                for (int64_t s = 0; s < seq_len; ++s) {
                    if (offset + s < total_tokens - 1) {
                        input_ids.data()[b*seq_len + s] = (float)dataset.tokens[offset+s];
                        targets.data()[b*seq_len + s] = (float)dataset.tokens[offset+s+1];
                        if (dataset.masks.size() > 0)
                            loss_mask.data()[b*seq_len + s] = (float)dataset.masks[offset+s+1];
                    }
                }
            }
            
            TrainingBatch train_batch{input_ids, targets, loss_mask};
            float loss = trainer.train_step(train_batch);
            
            // Check for NaN
            if (std::isnan(loss) || std::isinf(loss)) {
                std::cerr << "⚠️  NAN LOSS DETECTED at Batch " << batch_idx << "!" << std::endl << std::flush;
                // Optional: Reduce LR or skip? For now just warn.
                continue; 
            }
            
            epoch_loss += loss;
            epoch_steps++;
            total_steps++;
            
            // Always print training loss for the first few steps, then periodically
            if (epoch_steps < 20 || epoch_steps % 10 == 0) {
                 std::cout << " -> TRAIN Step " << total_steps 
                           << " | Loss: " << std::fixed << std::setprecision(4) << loss
                           << std::endl << std::flush;
            }
            
            // CRITICAL: Reset Arena (Free memory) at end of batch
            MemoryManager::instance().reset_arena();
        }
        
        std::cout << std::endl;
        std::cout << "Epoch Stats: E=" << epoch_easy << " M=" << epoch_medium << " H=" << epoch_hard << std::endl;
        
        // Save epoch checkpoint
        CheckpointMetadata meta;
        meta.epoch = iteration;
        meta.loss = (epoch_steps > 0) ? epoch_loss / epoch_steps : 0.0f;
        meta.learning_rate = learning_rate;
        CheckpointManager::save_checkpoint("kernel_adaptive_" + std::to_string(iteration) + ".bin", model, meta);
    }
    
    // Final
    CheckpointMetadata final_meta;
    final_meta.epoch = max_iterations;
    final_meta.learning_rate = learning_rate;
    CheckpointManager::save_checkpoint("kernel_adaptive_final.bin", model, final_meta);
    std::cout << "\n✅ DONE" << std::endl;
    
    return 0;
}
