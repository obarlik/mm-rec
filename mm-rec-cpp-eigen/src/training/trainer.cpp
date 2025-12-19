
/**
 * Trainer Implementation
 */

#include "mm_rec/training/trainer.h"
#include "mm_rec/model/mm_rec_model.h"
#include "mm_rec/training/gradient_utils.h"
#include "mm_rec/training/forward_cache.h" // Keep this, it's used
#include "mm_rec/training/optimizer.h" // Keep this, it's used
#include "mm_rec/training/uboo_loss.h" // Keep this, it's used
#include "mm_rec/core/memory_manager.h"
#include "mm_rec/utils/metrics.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/dashboard_html.h"
#include <iostream>
#include <cstring>  // memcpy
#include <cmath>
#include <atomic> // Used for total_loss
#include <omp.h> // Good practice

namespace {
    // Robust Finite Check (Bypasses -ffast-math)
    // IEEE 754: Exponent bits (23-30) all 1s means Inf or NaN.
    // Mask: 0x7F800000
    inline bool is_robust_finite(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(float));
        return (u & 0x7F800000) != 0x7F800000;
    }
}

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
    
    if (config.optimizer_type == "flux") {
        LOG_INFO("Creating Flux Optimizer (Adaptive Complexity Scaling)");
        optimizer_ = std::make_unique<Flux>(config.learning_rate, 0.9f, 0.999f, 1e-8f, config.weight_decay);
    } else if (config.optimizer_type == "adamw") {
        LOG_INFO("Creating AdamW Optimizer (LR=" + std::to_string(config.learning_rate) + ", WD=" + std::to_string(config.weight_decay) + ")");
        optimizer_ = std::make_unique<AdamW>(config.learning_rate, 0.9f, 0.999f, 1e-8f, config.weight_decay);
    } else if (config.optimizer_type == "adam") {
        LOG_INFO("Creating Adam Optimizer (LR=" + std::to_string(config.learning_rate) + ")");
        optimizer_ = std::make_unique<Adam>(config.learning_rate);
    } else {
        LOG_INFO("Creating SGD Optimizer (LR=" + std::to_string(config.learning_rate) + ")");
        optimizer_ = std::make_unique<SGD>(config.learning_rate);
    }
    
    // Initialize Dashboard
    dashboard_server_ = std::make_unique<net::HttpServer>(8080);
    
    // Serve HTML
    dashboard_server_->register_handler("/", [](const std::string&) -> std::string {
        return net::HttpServer::build_response(200, "text/html", ui::DASHBOARD_HTML);
    });
    
    // Try ports 8080 to 8090
    int port = 8080;
    while (port < 8090) {
        dashboard_server_ = std::make_unique<net::HttpServer>(port);
        
        // Re-register handlers (since we created a new server instance)
        // Note: Ideally HttpServer should allow dynamic binding, but our simple impl needs reconstruction.
        // Let's refactor slightly to separate setup from binding?
        // Or just lazy lambda registration. Pity I have to duplicate registration code.
        // Let's just try to bind. If fail, recreate.
        
        // Refactored approach: helper method
        setup_dashboard_handlers();
        
        if (dashboard_server_->start()) {
            LOG_UI("📊 Dashboard active at http://localhost:" + std::to_string(port));
            break;
        } else {
            // Port busy, try next
            port++;
        }
    }
    
    if (port == 8090) {
        LOG_UI("⚠️  Failed to start Dashboard (all ports 8080-8089 busy).");
    }
}

void Trainer::setup_dashboard_handlers() {
    // Serve HTML
    dashboard_server_->register_handler("/", [](const std::string&) -> std::string {
        return net::HttpServer::build_response(200, "text/html", ui::DASHBOARD_HTML);
    });
    
    // API: Stop
    dashboard_server_->register_handler("/api/stop", [this](const std::string&) -> std::string {
        this->stop_requested_ = true;
        LOG_UI("🛑 Stop requested via Dashboard!");
        return net::HttpServer::build_response(200, "application/json", "{\"status\": \"stopping\"}");
    });
    
    // API: Stats
    dashboard_server_->register_handler("/api/stats", [this](const std::string&) -> std::string {
        std::lock_guard<std::mutex> lock(this->stats_mutex_);
        std::stringstream json;
        json << "{";
        json << "\"epoch\": " << (this->step_ / 1000) << ", "; // Approximate
        json << "\"step\": " << this->step_ << ", ";
        json << "\"lr\": " << this->get_current_lr() << ", ";
        json << "\"speed\": " << this->current_speed_ << ", ";
        json << "\"loss\": " << (this->loss_history_.empty() ? 0.0f : this->loss_history_.back()) << ", ";
        
        json << "\"history\": [";
        for (size_t i = 0; i < this->loss_history_.size(); ++i) {
            json << this->loss_history_[i];
            if (i < this->loss_history_.size() - 1) json << ",";
        }
        json << "], ";
        
        json << "\"avg_history\": [";
        for (size_t i = 0; i < this->avg_loss_history_.size(); ++i) {
            json << this->avg_loss_history_[i];
            if (i < this->avg_loss_history_.size() - 1) json << ",";
        }
        json << "], ";

        json << "\"grad_norm_history\": [";
        for (size_t i = 0; i < this->grad_norm_history_.size(); ++i) {
            json << this->grad_norm_history_[i];
            if (i < this->grad_norm_history_.size() - 1) json << ",";
        }
        json << "], ";

        json << "\"lr_history\": [";
        for (size_t i = 0; i < this->lr_history_.size(); ++i) {
            json << this->lr_history_[i];
            if (i < this->lr_history_.size() - 1) json << ",";
        }
        json << "]";
        
        json << "}";
        return net::HttpServer::build_response(200, "application/json", json.str());
    });
}

Trainer::~Trainer() {
    if (dashboard_server_) {
        dashboard_server_->stop();
    }
}

// Vectorized training step (Full Batch GPU Offload)
float Trainer::train_step(const TrainingBatch& batch) {
    // 1. Mark persistent memory start
    // We want gradients to survive until update_parameters
    MemoryManager::instance().mark_persistent();
    
    // 2. Forward Pass (Full Batch)
    ForwardCache cache;
    Tensor logits;
    {
        METRIC_SCOPE(FORWARD_PASS, "fwd");
        logits = model_.forward(batch.input_ids, &cache);
    }
    
    // 3. Loss Calculation (Full Batch)
    Tensor loss_tensor = compute_uboo_loss(logits, batch.targets, batch.loss_mask);
    float loss = 0.0f;
    for(int i=0; i<loss_tensor.size(0); ++i) loss += loss_tensor.data()[i];
    if(loss_tensor.size(0) > 0) loss /= loss_tensor.size(0);
    
    // Add Aux Loss from MoE
    float total_step_loss = loss + cache.total_aux_loss * 0.01f;
    
    // Safety Check: Detect NaN in Loss (Pre-Backward)
    if (!is_robust_finite(total_step_loss)) {
        LOG_UI("⚠️  Loss Explosion/NaN detected (Loss: " + std::to_string(total_step_loss) + "). Skipping Step.");
        MemoryManager::instance().clear_persistent();
        MemoryManager::instance().reset_arena();
        return 100.0f; // Return high penalty
    }

    // --- ACTIVE LEARNING: CURRICULUM FILTER ---
    // 1. Skip Easy Samples (Save Compute)
    if (total_step_loss < config_.easy_threshold) {
        // Just update stats, don't train
        // Maybe we want to train rarely on easy to prevent forgetting?
        // For now: Hard Skip.
        // LOG_INFO("Skipping Easy Sample (Loss: " + std::to_string(total_step_loss) + ")");
        MemoryManager::instance().clear_persistent();
        MemoryManager::instance().reset_arena();
        return total_step_loss;
    }
    
    // 2. Skip Hard Samples (Prevent Destabilization)
    if (total_step_loss > config_.hard_threshold) {
        LOG_UI("🛡️ Skipping Too Hard Sample (Loss: " + std::to_string(total_step_loss) + ")");
        MemoryManager::instance().clear_persistent();
        MemoryManager::instance().reset_arena();
        return total_step_loss;
    }
    // ------------------------------------------
    
    // 4. Backward Pass (Full Batch)
    ModelGradients grads;
    {
        METRIC_SCOPE(BACKWARD_PASS, "bwd");
        grads = model_.backward(batch.targets, cache);
    }
    
    // 5. Gradient Clipping
    float grad_norm = clip_gradients_by_norm(grads, config_.grad_clip_norm);
    
    // Safety Check: Detect NaN/Inf in gradients (Bitwise)
    // std::isnan often fails with -ffast-math, so we use robust check.
    if (!is_robust_finite(grad_norm) || grad_norm > 1000.0f) {
        LOG_UI("⚠️  Gradient Explosion/NaN detected (Norm: " + std::to_string(grad_norm) + "). Skipping parameter update.");
        MemoryManager::instance().clear_persistent();
        MemoryManager::instance().reset_arena();
        return total_step_loss; // Return loss but do not update
    }
    
    // 6. Update Parameters
    float current_lr = scheduler_->get_lr(step_);
    optimizer_->set_lr(current_lr);
    
    {
        METRIC_SCOPE(OPTIMIZER_STEP, "optim");
        model_.update_parameters(*optimizer_, grads);
    }
    
    step_++;
    
    // Update dashboard stats
    update_stats(total_step_loss, 0.0f, grad_norm, current_lr); // Speed is updated by caller, but we send health metrics here
    
    // 7. Cleanup
    MemoryManager::instance().clear_persistent();
    MemoryManager::instance().reset_arena();

    return total_step_loss;
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

Tensor Trainer::forward_vectorized(const TrainingBatch& batch) {
    // Forward pass only for difficulty assessment (no backward!)
    // Returns [Batch] loss tensor
    
    // Reset Memory just in case
    MemoryManager::instance().reset_arena();
    
    ForwardCache cache;
    Tensor logits = model_.forward(batch.input_ids, &cache);
    
    // Compute loss per sample
    Tensor loss_tensor = compute_uboo_loss(logits, batch.targets, batch.loss_mask);
    
    // We must return a copy because reset_arena might wipe it if not careful?
    // The caller is responsible for MemoryManager if they want to keep it?
    // Actually, cmd_train calls reset_arena after decision.
    // So returning this Tensor (allocated in Arena) is fine as long as caller uses it before reset.
    
    return loss_tensor;
}

float Trainer::validate_step(const TrainingBatch& batch) {
    // No gradient computation in validation
    Tensor logits = model_.forward(batch.input_ids);
    Tensor loss_tensor = compute_uboo_loss(logits, batch.targets);
    
    float avg_loss = 0.0f;
    for(int i=0; i<loss_tensor.size(0); ++i) avg_loss += loss_tensor.data()[i];
    if(loss_tensor.size(0) > 0) avg_loss /= loss_tensor.size(0);
    
    return avg_loss;
}

float Trainer::get_current_lr() const {
    return scheduler_->get_lr(step_);
}

void Trainer::update_stats(float loss, float speed, float grad_norm, float lr) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Raw Loss (Increased history for full graph visualization)
    if (loss_history_.size() >= 10000) loss_history_.pop_front();
    loss_history_.push_back(loss);
    
    // Grad Norm
    if (grad_norm_history_.size() >= 10000) grad_norm_history_.pop_front();
    grad_norm_history_.push_back(grad_norm);
    
    // LR
    if (lr_history_.size() >= 10000) lr_history_.pop_front();
    lr_history_.push_back(lr);
    
    // Speed
    if (speed > 0) current_speed_ = speed;
    
    // EMA Calculation & History
    if (current_ema_ == 0.0f) current_ema_ = loss;
    else current_ema_ = current_ema_ * 0.95f + loss * 0.05f;
    
    if (avg_loss_history_.size() >= 10000) avg_loss_history_.pop_front();
    avg_loss_history_.push_back(current_ema_);
}

} // namespace mm_rec
