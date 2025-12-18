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
#include "mm_rec/utils/checkpoint.h"
#include "mm_rec/core/memory_manager.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/core/vulkan_backend.h" // [NEW] GPS Support

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

#include "commands.h"

int cmd_train(int argc, char* argv[]) {
    // Automatically lower process priority to background (nice 19)
    // This allows the user to continue using the PC without lag.
    id_t pid = getpid();
    if (setpriority(PRIO_PROCESS, pid, 19) == 0) {
        std::cout << "ℹ️  Process priority set to lowest (nice 19) for background training." << std::endl;
    } else {
        std::cerr << "⚠️  Failed to set process priority." << std::endl;
    }

    if (argc < 3) {
        std::cerr << "Usage: mm_rec train <config_file> <data_file>" << std::endl;
        return 1;
    }


    std::string config_path = argv[1];
    std::string data_path = argv[2];
    
    // --- 1. Setup Logging (Tee to train.log) ---
    std::ofstream log_file("train.log", std::ios::app); // Append mode
    TeeBuf tee_cout_buf(std::cout.rdbuf(), log_file.rdbuf());
    TeeBuf tee_cerr_buf(std::cerr.rdbuf(), log_file.rdbuf());
    
    // Save old buffers to restore later (RAII would be better but this is simple CLI)
    std::streambuf* old_cout = std::cout.rdbuf(&tee_cout_buf);
    std::streambuf* old_cerr = std::cerr.rdbuf(&tee_cerr_buf);
    
    std::cout << "=== Adaptive Curriculum Training (Online) ===" << std::endl;

    // 0. Initialize GPU (Hybrid)
    if (VulkanBackend::get().init()) {
         std::cout << "🚀 GPU Backend Enabled. (Full Power Mode)" << std::endl;
    } else {
         std::cout << "⚠️ GPU Backend Failed. Using CPU." << std::endl;
    }
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
        else if (key == "max_memory_mb") {
             size_t mb = std::stoul(val);
             MemoryManager::set_global_memory_limit(mb * 1024 * 1024);
        }
    }
    
    std::cout << "Config Loaded: Hard Threshold = " << hard_threshold << std::endl;
    std::cout << "               Batch Size = " << batch_size << std::endl;
    std::cout << "               Num Layers = " << config.num_layers << std::endl;
    std::cout << "               Num Experts = " << config.num_experts << std::endl;
    
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
    
    // CRITICAL: Mark model weights as persistent so they aren't wiped by reset_arena()
    MemoryManager::instance().mark_persistent();
    
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
    
    // --- 2. Auto-Resume Logic ---
    int start_epoch = 1;
    float best_loss = 1e9;
    std::string latest_ckpt_path = "checkpoint_latest.bin";
    std::string best_ckpt_path = "checkpoint_best.bin";
    
    // Try to load latest
    try {
        if (std::ifstream(latest_ckpt_path).good()) {
            std::cout << "🔄 Found latest checkpoint. Resuming..." << std::endl;
            CheckpointMetadata loaded_meta;
            CheckpointManager::load_checkpoint(latest_ckpt_path, model, loaded_meta);
            start_epoch = loaded_meta.epoch + 1;
            best_loss = loaded_meta.loss; // Assuming latest was 'ok'. Better to track best separately but this is fine.
            
            // Fix: If checkoint says epoch 5, we finished 5. Start 6.
            std::cout << "✅ Resumed from Epoch " << loaded_meta.epoch 
                      << " (Loss: " << loaded_meta.loss << ")" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "⚠️  Failed to resume: " << e.what() << ". Starting from scratch." << std::endl;
    }
    
    int64_t total_steps = 0;
    
    for (int iteration = start_epoch; iteration <= max_iterations; ++iteration) {
        std::cout << "\n=== Epoch " << iteration << " ===" << std::endl;
        
        int64_t epoch_easy = 0;
        int64_t epoch_medium = 0;
        int64_t epoch_hard = 0;
        float epoch_loss = 0.0f;
        int epoch_steps = 0;
        auto epoch_start_time = std::chrono::steady_clock::now();
        
        for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // 1. PROBE (Vectorized Full Batch Check)
            // Construct FULL batch first
            Tensor input_ids = Tensor::zeros({batch_size, seq_len});
            Tensor targets = Tensor::zeros({batch_size, seq_len});
            Tensor loss_mask = Tensor::zeros({batch_size, seq_len});
            
            // Fill batch candidates
            int actual_batch = 0;
            std::vector<int64_t> original_indices;
            
            for (int64_t b = 0; b < batch_size; ++b) {
                int64_t offset = b * stride + batch_idx * seq_len;
                bool valid_seq = true;
                
                for (int64_t s = 0; s < seq_len; ++s) {
                    if (offset + s >= total_tokens - 1) { 
                        valid_seq = false; break; 
                    }
                    input_ids.data()[b*seq_len + s] = (float)dataset.tokens[offset+s];
                    targets.data()[b*seq_len + s] = (float)dataset.tokens[offset+s+1];
                    if (dataset.masks.size() > 0)
                        loss_mask.data()[b*seq_len + s] = (float)dataset.masks[offset+s+1];
                    else
                        loss_mask.data()[b*seq_len + s] = 1.0f;
                }
                
                if (valid_seq) {
                    actual_batch++;
                    original_indices.push_back(b);
                }
            }
            
            if (actual_batch == 0) continue;
            
            TrainingBatch candidate_batch{input_ids, targets, loss_mask};
            
            // Forward pass only (Vectorized Probe) -> Returns [Batch]
            Tensor probe_losses = trainer.forward_vectorized(candidate_batch);
            
            // Filter "Medium" samples
            std::vector<int64_t> medium_indices;
            for (int64_t b = 0; b < batch_size; ++b) {
                // If this slot was valid
                bool was_valid = false;
                for(auto idx : original_indices) if(idx == b) was_valid = true;
                if (!was_valid) continue;
                
                float loss = probe_losses.data()[b];
                float ppl = std::exp(loss);
                
                if (ppl < easy_threshold) {
                    epoch_easy++;
                    
                    // DIAGNOSTIC: Analyze why it is "Easy"
                    // Only verify the first few easy samples to avoid spam
                    if (epoch_easy <= 3) {
                        std::cout << "\n🔍 Analyzing Easy Sample (BatchIdx " << b << "): "
                                  << "PPL = " << ppl << std::endl;
                        
                        // Count trainable tokens in this sequence
                        int trainable_count = 0;
                        for(int s=0; s<seq_len; ++s) {
                             if (loss_mask.data()[b*seq_len + s] > 0.5f) trainable_count++;
                        }
                        
                        std::cout << "   📊 Trainable Tokens: " << trainable_count << "/" << seq_len << std::endl;
                                  
                        if (trainable_count == 0) {
                            std::cout << "   ✅ Reason: Fully Masked / Empty Target (Nothing to learn)." << std::endl;
                        } else if (std::abs(ppl - 1.0f) < 0.01f) {
                             std::cout << "   ⚠️  Reason: Has trainable tokens but Loss is 0? Suspicious!" << std::endl;
                             std::cout << "   First 5 Tokens: ";
                             for(int s=0; s<5; ++s) std::cout << (int)input_ids.data()[b*seq_len + s] << " ";
                             std::cout << "\n   First 5 Target: ";
                             for(int s=0; s<5; ++s) std::cout << (int)targets.data()[b*seq_len + s] << " ";
                             std::cout << std::endl;
                             
                             // Check for trivial match
                             bool exact_match = true;
                             for(int s=0; s<seq_len; ++s) { // Input[s] predicts Target[s]? 
                                 // Target[s] should generally be Input[s+1].
                                 // If Input[s] == Target[s], we are predicting the current token? No.
                             }
                        } else {
                            std::cout << "   ✅ Reason: Model confidently predicts this sequence." << std::endl;
                        }
                    }
                } else if (ppl > hard_threshold) {
                    epoch_hard++;
                } else {
                    epoch_medium++;
                    medium_indices.push_back(b);
                }
            }
            
            // Log progress every 10 batches
            if (batch_idx % 10 == 0) {
                 auto now = std::chrono::steady_clock::now();
                 double elapsed_sec = std::chrono::duration<double>(now - epoch_start_time).count();
                 double batches_per_sec = (batch_idx + 1) / elapsed_sec;
                 
                 // Show PPL of the FIRST sample as representative (just for log)
                 float first_ppl = std::exp(probe_losses.data()[0]);
                 
                 size_t mem_bytes = MemoryManager::instance().get_total_memory();
                 double mem_mb = mem_bytes / (1024.0 * 1024.0);
                 
                 std::cout << "[Ep " << iteration << "] Batch " << batch_idx << "/" << num_batches
                           << " | E:" << epoch_easy << " M:" << epoch_medium << " H:" << epoch_hard
                           << " | Probe PPL: " << std::fixed << std::setprecision(1) << first_ppl
                           << " | " << std::setprecision(2) << batches_per_sec << " it/s"
                           << " | Mem: " << (int)mem_mb << "MB"
                           << std::endl << std::flush;
            }
            
            // If no medium samples, skip training this batch
            if (medium_indices.empty()) {
                MemoryManager::instance().reset_arena();
                continue;
            }
            
            // Construct FILTERED batch
            int64_t filtered_size = medium_indices.size();
            Tensor f_input = Tensor::zeros({filtered_size, seq_len});
            Tensor f_target = Tensor::zeros({filtered_size, seq_len});
            Tensor f_mask = Tensor::zeros({filtered_size, seq_len});
            
            size_t copy_size = seq_len * sizeof(float);
            
            for (size_t i = 0; i < filtered_size; ++i) {
                int64_t src_b = medium_indices[i];
                // Copy from candidate batch
                std::memcpy(f_input.data() + i*seq_len,   candidate_batch.input_ids.data() + src_b*seq_len, copy_size);
                std::memcpy(f_target.data() + i*seq_len,  candidate_batch.targets.data() + src_b*seq_len, copy_size);
                std::memcpy(f_mask.data() + i*seq_len,    candidate_batch.loss_mask.data() + src_b*seq_len, copy_size);
            }
            
            TrainingBatch final_batch{f_input, f_target, f_mask};
            float loss = trainer.train_step(final_batch);
            
            // Check for NaN
            if (std::isnan(loss) || std::isinf(loss)) {
                std::cerr << "⚠️  NAN LOSS DETECTED at Batch " << batch_idx << "!" << std::endl << std::flush;
                continue; 
            }
            
            epoch_loss += loss;
            epoch_steps++;
            total_steps++;
            
            if (epoch_steps < 20 || epoch_steps % 10 == 0) {
                 std::cout << " -> TRAIN Step " << total_steps 
                           << " | BatchSize: " << filtered_size // Show dynamic batch size
                           << " | Loss: " << std::fixed << std::setprecision(4) << loss
                           << std::endl << std::flush;
            }
            
            // CRITICAL: Reset Arena (Free memory) at end of batch
            MemoryManager::instance().reset_arena();
        }
        
        std::cout << std::endl;
        std::cout << "Epoch Stats: E=" << epoch_easy << " M=" << epoch_medium << " H=" << epoch_hard << std::endl;
        
        // Save epoch checkpoint (Atomically overwrite latest)
        CheckpointMetadata meta;
        meta.epoch = iteration;
        meta.loss = (epoch_steps > 0) ? epoch_loss / epoch_steps : 0.0f;
        meta.learning_rate = learning_rate;
        
        std::cout << "💾 Saving " << latest_ckpt_path << "..." << std::endl;
        CheckpointManager::save_checkpoint(latest_ckpt_path, model, meta);
        
        // Save Best
        if (meta.loss < best_loss && meta.loss > 0.0f) {
             std::cout << "🏆 New Best Loss! (" << meta.loss << " < " << best_loss << "). Saving " << best_ckpt_path << std::endl;
             best_loss = meta.loss;
             CheckpointManager::save_checkpoint(best_ckpt_path, model, meta);
        }
        
        if (epoch_medium == 0) {
            std::cout << "\n🛑 Early Stopping: No trainable samples found in this epoch." << std::endl;
            std::cout << "   (All samples were either too Easy or too Hard)" << std::endl;
            break; 
        }
    }
    
    // Final


    CheckpointMetadata final_meta;
    final_meta.epoch = max_iterations;
    final_meta.learning_rate = learning_rate;
    CheckpointManager::save_checkpoint("kernel_adaptive_final.bin", model, final_meta);
    std::cout << "\n✅ DONE" << std::endl;
    
    // Restore streams (Just good hygiene)
    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);
    
    return 0;
}
