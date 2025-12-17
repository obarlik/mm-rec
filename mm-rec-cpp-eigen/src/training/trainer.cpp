/**
 * Trainer Implementation
 */

#include "mm_rec/training/trainer.h"
#include "mm_rec/training/uboo_loss.h"
#include "mm_rec/training/gradient_utils.h"
#include "mm_rec/training/gradient_accum.h"
#include "mm_rec/training/forward_cache.h"
#include "mm_rec/training/optimizer.h"
#include <iostream>
#include <cstring>  // memcpy
#include <cmath>
#include "mm_rec/core/memory_manager.h"
#include <atomic> // Used for total_loss
#include <omp.h> // Good practice

namespace mm_rec {

Trainer::Trainer(MMRecModel& model, const TrainingConfig& config)
    : model_(model),
      config_(config),
      step_(0)
{
    // Create learning rate scheduler
    scheduler_ = std::make_unique<CosineScheduler>(
        config.learning_rate,
        config.warmup_steps,
        config.total_steps,
        config.learning_rate * 0.01f  // min_lr
    );
    
    // Create optimizer
    optimizer_ = std::make_unique<SGD>(config.learning_rate);
}

    // 44-50 original
float Trainer::train_step(const TrainingBatch& batch) {
    int64_t batch_size = batch.input_ids.size(0);
    int64_t seq_len = batch.input_ids.size(1);
    
    // Main thread accumulator (Updated by merge step)
    ModelGradients accumulated_grads; 
    
    std::atomic<float> total_loss{0.0f};
    std::atomic<int> valid_samples{0};
    
    // Storage for per-thread results
    int max_threads = omp_get_max_threads();
    std::vector<ModelGradients> thread_results(max_threads);
    std::vector<bool> thread_has_data(max_threads, false);

    // Pre-calculate copy size
    size_t copy_size = seq_len * sizeof(float);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        // THREAD-LOCAL CONTEXT
        // Each thread processes a subset of the batch
        
        // 1. Initialize Thread-Local Accumulator
        ModelGradients local_grads;
        bool local_init = false;
        
        // 2. Init flag
        // MemoryManager::instance().mark_persistent(); // REMOVED: Too early!
        
        // 3. Parallel Loop
        #pragma omp for
        for (int64_t b = 0; b < batch_size; ++b) {
            // ... (setup code omitted for brevity in prompt, but kept in file) ...
            // Allocate temp tensors (TRANSIENT)
            Tensor single_input = Tensor::zeros({1, seq_len});
            Tensor single_target = Tensor::zeros({1, seq_len});
            Tensor single_mask;
            
            // Fast copy
            std::memcpy(single_input.data(), batch.input_ids.data() + b * seq_len, copy_size);
            std::memcpy(single_target.data(), batch.targets.data() + b * seq_len, copy_size);
            
            if (batch.loss_mask.numel() > 0) {
                single_mask = Tensor::zeros({1, seq_len});
                std::memcpy(single_mask.data(), batch.loss_mask.data() + b * seq_len, copy_size);
            }
            
            ForwardCache cache;
            Tensor logits = model_.forward(single_input, &cache);
            
            Tensor loss_tensor = compute_uboo_loss(logits, single_target, single_mask);
            float loss = loss_tensor.item();
            
            // Add Aux Loss for Reporting (Gradient already handled in MoE backward)
            float total_step_loss = loss + cache.total_aux_loss * 0.01f; // 0.01 matches weight in MoE
            
            // Atomic add for loss (relaxed ordering fine for stats)
            float current_total = total_loss.load(std::memory_order_relaxed);
            while (!total_loss.compare_exchange_weak(current_total, current_total + total_step_loss));
            
            ModelGradients grads = model_.backward(single_target, cache);
            clip_gradients_by_norm(grads, config_.grad_clip_norm);
            
            // Accumulate Locally
            if (!local_init) {
                // Determine tensor shapes from first sample
                // We use clone() to ensure deep copy into persistent memory
                local_grads = grads.clone(); 
                local_init = true;
                
                // CRITICAL FIX: Mark Persistence AFTER allocating local_grads
                // This ensures reset_arena() rewinds to HERE, preserving local_grads
                MemoryManager::instance().mark_persistent();
            } else {
                accumulate_gradients(local_grads, grads);
            }
            
            // Increment
            valid_samples++; // Atomic
            
            // Reset Transient Memory (Free logits, cache, grads, inputs)
            // Rewinds to the point set by mark_persistent() above
            MemoryManager::instance().reset_arena();
        } // End of For
        
        // 4. Save result for Main Thread merging
        if (local_init) {
             // Shallow copy of structure is enough, data pointers point to this thread's arena
             thread_results[tid] = local_grads;
             thread_has_data[tid] = true;
        }
        
    } // End Parallel Region 1 (Implicit Barrier)
    
    // 5. Main Thread Merge (Sequential but fast)
    // accumulated_grads will be allocated on Main Thread's arena
    bool main_init = false;
    for (int i = 0; i < max_threads; ++i) {
        if (!thread_has_data[i]) continue;
        
        if (!main_init) {
            accumulated_grads = thread_results[i].clone(); // Deep copy to Main Arena
            main_init = true;
        } else {
            // Accumulate from Thread i's Arena to Main Arena
            accumulate_gradients(accumulated_grads, thread_results[i]);
        }
    }
    
    // 6. Global Cleanup
    // Now that Main Thread has a copy, we can release all worker arenas
    #pragma omp parallel
    {
        // Wipe everything including the local_grads we just merged
        MemoryManager::instance().clear_persistent();
        MemoryManager::instance().reset_arena();
    }
    
    // Average
    if (valid_samples > 0) {
        float avg_loss = total_loss / valid_samples;
        scale_gradients(accumulated_grads, 1.0f / valid_samples);
        
        float current_lr = scheduler_->get_lr(step_);
        optimizer_->set_lr(current_lr);
        model_.update_parameters(*optimizer_, accumulated_grads);
        
        step_++;
        return avg_loss;
    }
    
    return 0.0f;
}

float Trainer::forward_only(const TrainingBatch& batch) {
    // Forward pass only for difficulty assessment (no backward!)
    int64_t batch_size = batch.input_ids.size(0);
    int64_t seq_len = batch.input_ids.size(1);
    
    float total_loss = 0.0f;
    int valid_samples = 0;
    
    // Pre-allocate buffers
    Tensor single_input = Tensor::zeros({1, seq_len});
    Tensor single_target = Tensor::zeros({1, seq_len});
    Tensor single_mask = Tensor::zeros({1, seq_len});
    size_t copy_size = seq_len * sizeof(float);
    
    // Process each sequence independently (same as train_step but no backward)
    for (int64_t b = 0; b < batch_size; ++b) {
        // Fast copy
        std::memcpy(single_input.data(), batch.input_ids.data() + b * seq_len, copy_size);
        std::memcpy(single_target.data(), batch.targets.data() + b * seq_len, copy_size);
        
        if (batch.loss_mask.numel() > 0) {
            std::memcpy(single_mask.data(), batch.loss_mask.data() + b * seq_len, copy_size);
        }
        
        // Reset memory for isolation
        model_.reset_memory(1);
        
        // Forward pass only
        ForwardCache cache;
        Tensor logits = model_.forward(single_input, &cache);
        
        // Compute loss
        Tensor loss_tensor = compute_uboo_loss(logits, single_target, single_mask);
        float loss = loss_tensor.item();
        total_loss += loss;
        valid_samples++;

        // Fix: Reset Arena to prevent accumulation
        MemoryManager::instance().reset_arena();
    }
    
    return total_loss / valid_samples;
}

float Trainer::validate_step(const TrainingBatch& batch) {
    // No gradient computation in validation
    Tensor logits = model_.forward(batch.input_ids);
    Tensor loss_tensor = compute_uboo_loss(logits, batch.targets);
    
    return loss_tensor.item();
}

float Trainer::get_current_lr() const {
    return scheduler_->get_lr(step_);
}

} // namespace mm_rec
