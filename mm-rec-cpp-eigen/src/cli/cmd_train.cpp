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
#include "mm_rec/training/gradient_utils.h"
#include "mm_rec/training/sample_tracker.h"
#include "mm_rec/utils/checkpoint.h"
#include "mm_rec/core/memory_manager.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/ui.h"            // [NEW] UI library
#include "mm_rec/core/vulkan_backend.h"
#include "mm_rec/utils/metrics.h"
#include "mm_rec/utils/system_optimizer.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace mm_rec;
using namespace mm_rec::ui;  // [NEW] Use UI namespace

// Simple binary dataset structure
struct InstructionDataset {
    std::vector<int32_t> tokens;
    std::vector<int32_t> masks;
    int32_t version = 2;
    int64_t token_count = 0;
    
    bool load_binary(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) {
            ui::error("Failed to open: " + path);
            return false;
        }
        
        // Read header
        int32_t magic;
        ifs.read(reinterpret_cast<char*>(&magic), sizeof(int32_t));
        ifs.read(reinterpret_cast<char*>(&version), sizeof(int32_t));
        
        int64_t count;
        ifs.read(reinterpret_cast<char*>(&count), sizeof(int64_t));
        
        ui::info("Loading " + std::to_string(count) + " tokens (version " + std::to_string(version) + ")...");
        
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
        LOG_INFO("Process priority set to lowest (nice 19) for background training.");
    } else {
        LOG_DEBUG("Failed to set process priority.");
    }

    if (argc < 3) {
        std::cerr << "Usage: mm_rec train <config_file> <data_file>" << std::endl;
        return 1;
    }


    std::string config_path = argv[1];
    std::string data_path = argv[2];
    
    // --- 1. Setup Logging ---
    Logger::instance().start_writer("train.log", LogLevel::DEBUG);
    
    ui::print_header("Adaptive Curriculum Training (Online)");

    // 0. Initialize GPU (Hybrid)
    if (VulkanBackend::get().init()) {
         ui::success("GPU Backend Enabled (Full Power Mode)");
    } else {
         ui::warning("GPU Backend Failed. Using CPU.");
    }
    MMRecModelConfig config;
    float learning_rate = 0.01f;
    std::string optimizer_type = "sgd";
    float weight_decay = 0.01f;
    float grad_clip_norm = 1.0f; // Default clip
    int batch_size = 8;
    int max_seq_len = 128; // Restored
    // Adaptive params (defaults)
    float easy_threshold = 50.0f;
    float hard_threshold = 500.0f;
    int max_iterations = 15;
    int warmup_steps = 100; // Default warmup
    
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
        else if (key == "optimizer_type") optimizer_type = val;
        else if (key == "weight_decay") weight_decay = std::stof(val);
        else if (key == "batch_size") batch_size = std::stoi(val);
        else if (key == "max_seq_len") max_seq_len = std::stoi(val);
        // Adaptive params
        else if (key == "easy_threshold") easy_threshold = std::stof(val);
        else if (key == "hard_threshold") hard_threshold = std::stof(val);
        else if (key == "grad_clip_norm") grad_clip_norm = std::stof(val);
        else if (key == "max_iterations") max_iterations = std::stoi(val);
        else if (key == "warmup_steps") warmup_steps = std::stoi(val);
        
        if (key == "uboo_weight") config.uboo_weight = std::stof(val);
        else if (key == "max_memory_mb") {
             size_t mb = std::stoul(val);
             MemoryManager::set_global_memory_limit(mb * 1024 * 1024);
        }
    }
    
    LOG_INFO("Config Loaded: Hard Threshold = " + std::to_string(hard_threshold));
    LOG_INFO("               Batch Size = " + std::to_string(batch_size));
    LOG_INFO("               Num Layers = " + std::to_string(config.num_layers));
    LOG_INFO("               Num Experts = " + std::to_string(config.num_experts));
    
    // Load dataset
    ui::info("Loading dataset: " + data_path);
    InstructionDataset dataset;
    if (!dataset.load_binary(data_path)) return 1;
    
    // Initialize model
    MMRecModel model(config);
    
    // Initialize trainer
    // Initialize trainer
    TrainingConfig train_config;
    train_config.learning_rate = learning_rate;
    train_config.batch_size = batch_size;
    train_config.optimizer_type = optimizer_type;
    train_config.weight_decay = weight_decay;
    train_config.warmup_steps = warmup_steps;
    train_config.grad_clip_norm = grad_clip_norm;
    train_config.easy_threshold = easy_threshold;
    train_config.hard_threshold = hard_threshold;
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
    
    ui::print_header("STARTING ONLINE ADAPTIVE TRAINING");
    LOG_INFO("Batches: " + std::to_string(num_batches) + " | Stride: " + std::to_string(stride));
    
    // --- 2. Auto-Resume Logic ---
    int start_epoch = 1;
    float best_loss = 1e9;
    std::string latest_ckpt_path = "checkpoint_latest.bin";
    std::string best_ckpt_path = "checkpoint_best.bin";
    
    // Try to load latest
    try {
        if (std::ifstream(latest_ckpt_path).good()) {
            ui::info("Found latest checkpoint. Resuming...");
            CheckpointMetadata loaded_meta;
            CheckpointManager::load_checkpoint(latest_ckpt_path, model, loaded_meta);
            start_epoch = loaded_meta.epoch + 1;
            best_loss = loaded_meta.loss; // Assuming latest was 'ok'. Better to track best separately but this is fine.
            
            // Fix: If checkoint says epoch 5, we finished 5. Start 6.
            ui::success("Resumed from Epoch " + std::to_string(loaded_meta.epoch) + 
                       " (Loss: " + std::to_string(loaded_meta.loss) + ")");
        }
    } catch (const std::exception& e) {
        ui::warning("Failed to resume: " + std::string(e.what()) + ". Starting from scratch.");
    }
    
    int64_t total_steps = 0;
    
    // SAVE INITIAL CHECKPOINT (For Epoch 1 Rollback)
    CheckpointMetadata init_meta;
    init_meta.epoch = 0; // Represents "Before Epoch 1"
    init_meta.learning_rate = learning_rate;
    CheckpointManager::save_checkpoint(latest_ckpt_path, model, init_meta);
    CheckpointManager::save_checkpoint(latest_ckpt_path, model, init_meta);
    LOG_INFO("Initial state saved (for Robustness safety).");
    
    // Metrics enabled by default (--no-metrics to disable)
    bool enable_metrics = true;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-metrics") {
            enable_metrics = false;
            break;
        }
    }
    
    if (enable_metrics) {
        // Production mode: Binary format + sampling
        MetricsSamplingConfig sampling;
        sampling.enabled = true;
        sampling.interval = 10;  // Record every 10th step
        sampling.warmup_events = 100;
        
        MetricsManager::instance().start_writer("training_metrics.bin", sampling);
        MetricsManager::instance().start_writer("training_metrics.bin", sampling);
        ui::info("Metrics enabled → training_metrics.bin");
    }
    
    for (int iteration = start_epoch; iteration <= max_iterations; ++iteration) {
        std::cout << "\n"; // Clean separation
        ui::print_header("Epoch " + std::to_string(iteration) + " / " + std::to_string(max_iterations), 40);
        
        int64_t epoch_easy = 0;
        int64_t epoch_medium = 0;
        int64_t epoch_hard = 0;
        float epoch_loss = 0.0f;
        int epoch_steps = 0;
        auto epoch_start_time = std::chrono::steady_clock::now();
        // Timer for Data Stall metric
        auto step_end_time = std::chrono::high_resolution_clock::now();
        
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
                 
                 size_t mem_bytes = MemoryManager::get_global_memory_usage();
                 double mem_mb = mem_bytes / (1024.0 * 1024.0);
                 
                 std::stringstream ss;
                 ss << "[Ep " << iteration << "] Batch " << batch_idx << "/" << num_batches
                    << " | E:" << epoch_easy << " M:" << epoch_medium << " H:" << epoch_hard
                    << " | Probe PPL: " << std::fixed << std::setprecision(1) << first_ppl
                    << " | " << std::setprecision(2) << batches_per_sec << " it/s"
                    << " | Mem: " << (int)mem_mb << "MB";
                 
                 // Using direct cout with carriage return for smooth update, LOG_INFO for history
                 std::cout << "\r" << color::CYAN << ss.str() << color::RESET << std::flush;
                 if (batch_idx % 100 == 0) LOG_INFO(ss.str().c_str());
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
            
            for (int64_t i = 0; i < filtered_size; ++i) {
                int64_t src_b = medium_indices[i];
                // Copy from candidate batch
                std::memcpy(f_input.data() + i*seq_len,   candidate_batch.input_ids.data() + src_b*seq_len, copy_size);
                std::memcpy(f_target.data() + i*seq_len,  candidate_batch.targets.data() + src_b*seq_len, copy_size);
                std::memcpy(f_mask.data() + i*seq_len,    candidate_batch.loss_mask.data() + src_b*seq_len, copy_size);
            }
            
            TrainingBatch final_batch{f_input, f_target, f_mask};
            // FLUX OPTIMIZER V2 (Holistic Adaptive Scaling)
            // Calculate difficulty score (0.0 = Easy, 1.0 = Threshold)
            // Scale update to be aggressive on confident data, cautious on hard data.
            
            // 1. Medium Batch Difficulty (Local)
            float avg_ppl = 0.0f;
            for (int64_t idx : medium_indices) {
                avg_ppl += std::exp(probe_losses.data()[idx]);
            }
            if (!medium_indices.empty()) avg_ppl /= medium_indices.size();
            
            float difficulty_ratio = std::min(1.0f, avg_ppl / hard_threshold);
            
            // 2. Hard Sample Feedback (Global Health)
            // If we are rejecting many samples as "Hard", the model is struggling.
            // We must slow down globally, even if the "Medium" (easy) samples look fine.
            int64_t local_hard_count = 0;
            // Iterate original indices to find hard ones
            for(int64_t b=0; b<batch_size; ++b) {
                bool was_valid = false;
                for(auto idx : original_indices) if(idx == b) was_valid = true;
                if(!was_valid) continue;
                
                float ppl = std::exp(probe_losses.data()[b]);
                if (ppl > hard_threshold) local_hard_count++;
            }
            
            float hard_ratio = (float)local_hard_count / (float)std::max((int64_t)1, (int64_t)original_indices.size());
            
            // Flux Formula V2: 
            // Base Aggression: 1.25x
            // Difficulty Penalty: -0.75x (If Medium batch is tricky)
            // Hardness Penalty: -2.0x (If we are skipping too much data - CRITICAL)
            // Min Scale: 0.1x (Never zero, always slow learning)
            
            float flux_scale = 1.25f - (0.75f * difficulty_ratio) - (2.0f * hard_ratio);
            flux_scale = std::max(0.1f, flux_scale);
            
            if (train_config.optimizer_type == "flux") {
                trainer.get_optimizer()->set_flux_scale(flux_scale);
            }
            
            // Check stop signal from Dashboard (BEFORE expensive step)
            if (trainer.should_stop()) {
                ui::warning("Training stopped by user request (Dashboard).");
                break;
            }

            // Calculate Data Stall (Time spent since last step ended)
            auto step_start_time = std::chrono::high_resolution_clock::now();
            double stall_ms = std::chrono::duration<double, std::milli>(step_start_time - step_end_time).count();
            // First step might be huge, but that's accurate (setup time)

            // Calculate memory usage (Cheap)
            size_t mem_bytes = MemoryManager::get_global_memory_usage();
            double mem_mb = mem_bytes / (1024.0 * 1024.0);

            // Calculate Speed (Tokens per second, persistent)
            auto step_end = std::chrono::steady_clock::now();
            double step_dur = std::chrono::duration<double>(step_end - epoch_start_time).count();
            double speed_tps = (epoch_steps * filtered_size * seq_len) / (step_dur + 1e-9);

            float loss = trainer.train_step(final_batch, (float)stall_ms, (float)speed_tps, (float)mem_mb);
            
            // Mark end of step for next stall calculation
            step_end_time = std::chrono::high_resolution_clock::now();
            
            // Check for NaN - Auto-Rollback Mechanism
            // Check for NaN - Auto-Rollback Mechanism
            if (!is_robust_finite(loss)) {
                ui::error("EXPLOSION DETECTED at Batch " + std::to_string(batch_idx) + "! Initiating Auto-Rollback...");
                
                // 1. Reload Last Good Checkpoint
                try {
                    CheckpointMetadata loaded_meta;
                    if (std::ifstream(latest_ckpt_path).good()) {
                        CheckpointManager::load_checkpoint(latest_ckpt_path, model, loaded_meta);
                        ui::info("Rolled back to safely saved stats (Epoch " + std::to_string(loaded_meta.epoch) + ")");
                    } else {
                        ui::error("No checkpoint to roll back to! Cannot recover.");
                        return 1; // Fatal if no backup
                    }
                } catch (...) {
                    ui::error("Rollback Failed.");
                    return 1;
                }
                
                // 2. Slash Learning Rate (The "Foolproof" Correction)
                learning_rate *= 0.5f; 
                train_config.learning_rate = learning_rate; 
                
                // CRITICAL: Update Scheduler and Optimizer via Trainer
                trainer.update_learning_rate(learning_rate);
                
                ui::warning("Emergency Brake: LR slashed to " + std::to_string(learning_rate));
                
                // 3. Clear Memory & Retry (Skip this toxic batch)
                MemoryManager::instance().reset_arena();
                continue; 
            }
            
            epoch_loss += loss;
            epoch_steps++;
            total_steps++;
            
            // 👇 ZERO-OVERHEAD: Record ALL steps to metrics (0.02ns/event)
            float current_lr = trainer.get_current_lr();
            METRIC_TRAINING_STEP(loss, current_lr);
            
            // Record Flux-specific metrics
            if (train_config.optimizer_type == "flux") {
                METRIC_RECORD(CUSTOM, flux_scale, 0, 0, "flux");
            }
            
            // 👇 EXPENSIVE: Console output only every 100 steps (or first 20)
            if (epoch_steps < 20 || total_steps % 100 == 0) {
                 // formatted output using stringstream
                 std::stringstream ss;
                 ss << "\r -> Step " << total_steps 
                    << " | Loss: " << std::fixed << std::setprecision(4) << loss
                    << " | LR: " << std::fixed << std::setprecision(5) << current_lr;
                           
                 if (train_config.optimizer_type == "flux") {
                     ss << " | Flux: " << std::fixed << std::setprecision(2) << flux_scale;
                 }
                 
                 ss << " | " << (int)speed_tps << " tok/s    ";
                 ss << " | Mem: " << (int)mem_mb << "MB";
                 
                 std::cout << color::GREEN << ss.str() << color::RESET << std::flush;
                 if (total_steps % 1000 == 0) LOG_INFO(ss.str().c_str());
            }
            
            // CRITICAL: Reset Arena (Free memory) at end of batch
            MemoryManager::instance().reset_arena();
        }
        
        std::cout << "\n";
        Table stats({"Category", "Count"}, 15);
        stats.add_row({"Easy", std::to_string(epoch_easy)});
        stats.add_row({"Medium", std::to_string(epoch_medium)});
        stats.add_row({"Hard", std::to_string(epoch_hard)});
        stats.finish();
        
        // Save epoch checkpoint (Atomically overwrite latest)
        CheckpointMetadata meta;
        meta.epoch = iteration;
        meta.loss = (epoch_steps > 0) ? epoch_loss / epoch_steps : 0.0f;
        meta.learning_rate = learning_rate;
        
        LOG_INFO("Saving " + latest_ckpt_path + "...");
        CheckpointManager::save_checkpoint(latest_ckpt_path, model, meta);
        
        // Save Best
        if (meta.loss < best_loss && meta.loss > 0.0f) {
             ui::success("New Best Loss! (" + std::to_string(meta.loss) + " < " + std::to_string(best_loss) + "). Saving " + best_ckpt_path);
             best_loss = meta.loss;
             CheckpointManager::save_checkpoint(best_ckpt_path, model, meta);
        }
        
        if (epoch_medium == 0) {
            ui::error("Early Stopping: No trainable samples found in this epoch.");
            LOG_INFO("All samples were either too Easy or too Hard");
            break; 
        }
    }
    
    // Final
    CheckpointMetadata final_meta;
    final_meta.epoch = max_iterations;
    final_meta.learning_rate = learning_rate;
    CheckpointManager::save_checkpoint("kernel_adaptive_final.bin", model, final_meta);
    
    // Stop metrics writer (flushes remaining events to disk)
    MetricsManager::instance().stop_writer();
    
    ui::success("DONE");
    
    Logger::instance().stop_writer();
    
    return 0;
}
