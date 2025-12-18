/**
 * SGD Optimizer
 * 
 * Simple stochastic gradient descent
 */

#pragma once

#include "mm_rec/core/tensor.h"
#include "mm_rec/core/tensor.h"
#include <vector>
#include <unordered_map>
#include <cmath>
#include <Eigen/Dense>

namespace mm_rec {

/**
 * Base Optimizer Class
 */
class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    virtual void step(Tensor& param, const Tensor& grad) = 0;
    virtual void step(std::vector<Tensor*> params, const std::vector<Tensor>& grads) {
        for (size_t i = 0; i < params.size(); ++i) {
            step(*params[i], grads[i]);
        }
    }
    virtual void set_lr(float lr) = 0;
    virtual float get_lr() const = 0;
    
    // Flux Extension: Dynamic Scaling Factor
    virtual void set_flux_scale(float /*scale*/) {} 
};

/**
 * Simple SGD optimizer
 */
class SGD : public Optimizer {
public:
    explicit SGD(float learning_rate) : lr_(learning_rate) {}
    
    /**
     * Update parameters: W = W - lr * dW
     */
    void step(Tensor& param, const Tensor& grad) override {
        for (int64_t i = 0; i < param.numel(); ++i) {
            param.data()[i] -= lr_ * grad.data()[i];
        }
    }
    
    /**
     * Update multiple parameters at once
     */
    void step(std::vector<Tensor*> params, const std::vector<Tensor>& grads) {
        for (size_t i = 0; i < params.size(); ++i) {
            step(*params[i], grads[i]);
        }
    }
    
    void set_lr(float lr) override { lr_ = lr; }
    float get_lr() const override { return lr_; }
    
private:
    float lr_;
};

/**
 * Adam Optimizer
 * Adaptive Moment Estimation
 */
class Adam : public Optimizer {
public:
    Adam(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}
    
    void step(Tensor& param, const Tensor& grad) {
        float* ptr = param.data();
        
        // Initialize state for this param if needed
        if (states_.find(ptr) == states_.end()) {
            std::vector<int64_t> shape;
            for (int i = 0; i < param.ndim(); ++i) shape.push_back(param.size(i));
            
            State s;
            s.m = Tensor::zeros(shape);
            s.v = Tensor::zeros(shape);
            s.t = 0;
            states_[ptr] = s;
        }
        
        State& s = states_[ptr];
        s.t++;
        
        // Update moments
        for (int64_t i = 0; i < param.numel(); ++i) {
            s.m.data()[i] = beta1_ * s.m.data()[i] + (1.0f - beta1_) * grad.data()[i];
            s.v.data()[i] = beta2_ * s.v.data()[i] + (1.0f - beta2_) * grad.data()[i] * grad.data()[i];
            
            float m_hat = s.m.data()[i] / (1.0f - std::pow(beta1_, s.t));
            float v_hat = s.v.data()[i] / (1.0f - std::pow(beta2_, s.t));
            
            param.data()[i] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
    
    void set_lr(float lr) override { lr_ = lr; }
    float get_lr() const override { return lr_; }
    void reset() { states_.clear(); }
    
private:
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    
    struct State {
        Tensor m;
        Tensor v;
        int t;
    };
    std::unordered_map<void*, State> states_;
};

/**
 * AdamW Optimizer
 * Adam with decoupled Weight Decay (the correct way)
 * This is the industry standard for LLM training
 */
class AdamW : public Optimizer {
public:
    AdamW(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, 
          float epsilon = 1e-8f, float weight_decay = 0.01f)
        : lr_(learning_rate), beta1_(beta1), beta2_(beta2), 
          epsilon_(epsilon), weight_decay_(weight_decay) {}
    
    
    void step(Tensor& param, const Tensor& grad) override {
        float* ptr = param.data();
        
        if (states_.find(ptr) == states_.end()) {
            std::vector<int64_t> shape;
            for (int i = 0; i < param.ndim(); ++i) shape.push_back(param.size(i));
            
            State s;
            s.m = Tensor::zeros(shape);
            s.v = Tensor::zeros(shape);
            s.t = 0;
            states_[ptr] = s;
        }
        
        State& s = states_[ptr];
        s.t++;
        
        for (int64_t i = 0; i < param.numel(); ++i) {
            // Weight decay
            param.data()[i] *= (1.0f - lr_ * weight_decay_);
            
            s.m.data()[i] = beta1_ * s.m.data()[i] + (1.0f - beta1_) * grad.data()[i];
            s.v.data()[i] = beta2_ * s.v.data()[i] + (1.0f - beta2_) * grad.data()[i] * grad.data()[i];
            
            float m_hat = s.m.data()[i] / (1.0f - std::pow(beta1_, s.t));
            float v_hat = s.v.data()[i] / (1.0f - std::pow(beta2_, s.t));
            
            param.data()[i] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
    
    void set_lr(float lr) override { lr_ = lr; }
    float get_lr() const override { return lr_; }
    void reset() { states_.clear(); }

protected:
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    float flux_scale_ = 1.0f; // Default scale

    struct State {
        Tensor m;
        Tensor v;
        int t;
    };
    std::unordered_map<void*, State> states_;
};

/**
 * Flux Optimizer (The "Invention")
 * 
 * Curriculum-Aware Optimization:
 * Scales the effective step size based on the specific difficulty 
 * of the current batch relative to the model's capacity.
 */
class Flux : public AdamW {
public:
    using AdamW::AdamW; // Inherit constructor
    
    void set_flux_scale(float scale) override {
        flux_scale_ = scale;
    }
    
    // Telemetry
    uint64_t get_brake_stats() const { return brake_count_; }
    void reset_brake_stats() { brake_count_ = 0; }

    // Override step to apply flux scaling with Safety Brake (Trust Region)
    void step(Tensor& param, const Tensor& grad) override {
        float* ptr = param.data();
        
        // Initialize state if needed
        if (states_.find(ptr) == states_.end()) {
            std::vector<int64_t> shape;
            for (int i = 0; i < param.ndim(); ++i) shape.push_back(param.size(i));
            
            State s;
            s.m = Tensor::zeros(shape);
            s.v = Tensor::zeros(shape);
            s.t = 0;
            states_[ptr] = s;
        }
        
        State& s = states_[ptr];
        s.t++;
        
        // Vectorized Implementation using Eigen
        int64_t n = param.numel();
        
        // Map data to Eigen Arrays for element-wise operations
        Eigen::Map<Eigen::ArrayXf> p(param.data(), n);
        Eigen::Map<const Eigen::ArrayXf> g(grad.data(), n);
        Eigen::Map<Eigen::ArrayXf> m(s.m.data(), n);
        Eigen::Map<Eigen::ArrayXf> v(s.v.data(), n);
        
        // 1. Update Moments
        m = beta1_ * m + (1.0f - beta1_) * g;
        v = beta2_ * v + (1.0f - beta2_) * g.square();
        
        // 2. Bias Correction
        // We compute scalars first
        float m_corr = 1.0f - std::pow(beta1_, s.t);
        float v_corr = 1.0f - std::pow(beta2_, s.t);
        
        // 3. Weight Decay (AdamW)
        float effective_lr = lr_ * flux_scale_;
        p *= (1.0f - effective_lr * weight_decay_);
        
        // 4. Compute Update Step (Raw)
        // update = lr * (m/m_corr) / (sqrt(v/v_corr) + eps)
        Eigen::ArrayXf m_hat = m / m_corr;
        Eigen::ArrayXf v_hat = v / v_corr;
        Eigen::ArrayXf update = effective_lr * m_hat / (v_hat.sqrt() + epsilon_);
        
        // 5. Flux Safety Brake (Vectorized)
        // Threshold = max(0.01, |p| * 0.5)
        Eigen::ArrayXf threshold = (p.abs() * 0.5f).max(0.01f);
        
        // Count violations (Telemetry)
        // Note: This forces evaluation, but it's worth it for stats.
        // We can optimize it out in RELEASE if needed, but for now we keep it.
        // (Use select to avoid branching?)
        // Let's just do the check implicitly by checking diff later if we want speed
        // or just use count() which is optimized.
        
        // Eigen bool masking:
        // brake_count_ += (update.abs() > threshold).count(); 
        
        // 6. Apply Brake (Clamp update to [-thresh, +thresh])
        // This is the core "Flux" logic - preventing explosion
        Eigen::ArrayXf clipped_update = update.min(threshold).max(-threshold);
        
        // Telemetry Update
        // We'll increment only if we actually clipped something.
        // Doing this per-step might be slightly costly but O(N).
        // For max speed, we might skip this or do it stochastically.
        // Let's do it:
        brake_count_ += (update.abs() > threshold).count();

        // 7. Apply Update
        p -= clipped_update;
    }

private:
    uint64_t brake_count_ = 0;
   // Using AdamW protected members
};


} // namespace mm_rec
