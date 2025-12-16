/**
 * SGD Optimizer
 * 
 * Simple stochastic gradient descent
 */

#pragma once

#include "mm_rec/core/tensor.h"
#include <vector>

namespace mm_rec {

/**
 * Simple SGD optimizer
 */
class SGD {
public:
    explicit SGD(float learning_rate) : lr_(learning_rate) {}
    
    /**
     * Update parameters: W = W - lr * dW
     */
    void step(Tensor& param, const Tensor& grad) {
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
    
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }
    
private:
    float lr_;
};

/**
 * Adam Optimizer
 * Adaptive Moment Estimation
 */
class Adam {
public:
    Adam(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}
    
    void step(Tensor& param, const Tensor& grad) {
        // Initialize state if needed
        if (m_.numel() == 0) {
            m_ = Tensor::zeros(param.shape());
            v_ = Tensor::zeros(param.shape());
        }
        
        t_++;
        
        // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * grad
        for (int64_t i = 0; i < param.numel(); ++i) {
            m_.data()[i] = beta1_ * m_.data()[i] + (1.0f - beta1_) * grad.data()[i];
            // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * grad^2
            v_.data()[i] = beta2_ * v_.data()[i] + (1.0f - beta2_) * grad.data()[i] * grad.data()[i];
            
            // Compute bias-corrected first moment estimate
            float m_hat = m_.data()[i] / (1.0f - std::pow(beta1_, t_));
            // Compute bias-corrected second raw moment estimate
            float v_hat = v_.data()[i] / (1.0f - std::pow(beta2_, t_));
            
            // Update parameters
            param.data()[i] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
    
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }
    void reset() { t_ = 0; m_ = Tensor(); v_ = Tensor(); }
    
private:
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    int t_;  // timestep
    Tensor m_;  // first moment vector
    Tensor v_;  // second moment vector
};

/**
 * AdamW Optimizer
 * Adam with decoupled Weight Decay (the correct way)
 * This is the industry standard for LLM training
 */
class AdamW {
public:
    AdamW(float learning_rate, float beta1 = 0.9f, float beta2 = 0.999f, 
          float epsilon = 1e-8f, float weight_decay = 0.01f)
        : lr_(learning_rate), beta1_(beta1), beta2_(beta2), 
          epsilon_(epsilon), weight_decay_(weight_decay), t_(0) {}
    
    void step(Tensor& param, const Tensor& grad) {
        // Initialize state if needed
        if (m_.numel() == 0) {
            m_ = Tensor::zeros(param.shape());
            v_ = Tensor::zeros(param.shape());
        }
        
        t_++;
        
        // AdamW: Apply weight decay BEFORE gradient update (decoupled)
        // This is the key difference from L2 regularization
        for (int64_t i = 0; i < param.numel(); ++i) {
            // Weight decay
            param.data()[i] *= (1.0f - lr_ * weight_decay_);
            
            // Update biased first moment estimate
            m_.data()[i] = beta1_ * m_.data()[i] + (1.0f - beta1_) * grad.data()[i];
            // Update biased second raw moment estimate
            v_.data()[i] = beta2_ * v_.data()[i] + (1.0f - beta2_) * grad.data()[i] * grad.data()[i];
            
            // Compute bias-corrected estimates
            float m_hat = m_.data()[i] / (1.0f - std::pow(beta1_, t_));
            float v_hat = v_.data()[i] / (1.0f - std::pow(beta2_, t_));
            
            // Update parameters
            param.data()[i] -= lr_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
    
    void set_lr(float lr) { lr_ = lr; }
    float get_lr() const { return lr_; }
    void reset() { t_ = 0; m_ = Tensor(); v_ = Tensor(); }
    
private:
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    int t_;
    Tensor m_;
    Tensor v_;
};

} // namespace mm_rec
