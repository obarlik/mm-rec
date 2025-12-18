/**
 * UBOO Loss Implementation
 */

#include "mm_rec/training/uboo_loss.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace mm_rec {

Tensor cross_entropy_loss(const Tensor& logits, const Tensor& targets, const Tensor& mask) {
    // logits: [batch, seq, vocab]
    // targets: [batch, seq]
    // mask: [batch, seq] - optional, 1=compute, 0=skip
    
    int64_t batch = logits.size(0);
    int64_t seq = logits.size(1);
    int64_t vocab = logits.size(2);
    
    Tensor loss_per_sample = Tensor::zeros({batch});
    
    bool use_mask = (mask.numel() > 0);
    
    for (int64_t b = 0; b < batch; ++b) {
        float sample_total_loss = 0.0f;
        int64_t sample_tokens = 0;
        
        for (int64_t s = 0; s < seq; ++s) {
            // Skip if masked
            if (use_mask && mask.data()[b * seq + s] < 0.5f) {
                continue;
            }
            
            int64_t target_id = static_cast<int64_t>(targets.data()[b * seq + s]);
            
            // Numerical stability using Log-Sum-Exp TrickImplementation
            // log_prob = logit[target] - max_logit - log(sum(exp(logit - max_logit)))
            
            float max_logit = -1e9f;
            for (int64_t v = 0; v < vocab; ++v) {
                float logit = logits.data()[b * seq * vocab + s * vocab + v];
                max_logit = std::max(max_logit, logit);
            }
           
            float sum_exp = 0.0f;
            for (int64_t v = 0; v < vocab; ++v) {
                float logit = logits.data()[b * seq * vocab + s * vocab + v];
                sum_exp += std::exp(logit - max_logit);
            }
            float log_sum_exp = std::log(sum_exp);
            
            // Get target logit
            float target_logit = logits.data()[b * seq * vocab + s * vocab + target_id];
            
            // Log probability (stable)
            float log_prob = target_logit - max_logit - log_sum_exp;
            
            sample_total_loss += -log_prob;
            sample_tokens++;
        }
        
        // Average per sample
        if (sample_tokens > 0) {
            loss_per_sample.data()[b] = sample_total_loss / sample_tokens;
        } else {
             loss_per_sample.data()[b] = 0.0f;
        }
    }
    
    return loss_per_sample;
}

Tensor compute_uboo_loss(const Tensor& all_layer_logits, const Tensor& targets, const Tensor& mask) {
    // all_layer_logits: [num_layers, batch, seq, vocab]
    // targets: [batch, seq]
    // mask: [batch, seq] - optional
    
    int64_t num_layers = all_layer_logits.size(0);
    int64_t batch = all_layer_logits.size(1);
    int64_t seq = all_layer_logits.size(2);
    int64_t vocab = all_layer_logits.size(3);
    
    // We need to store [Batch] sized tensor for each layer
    std::vector<Tensor> layer_losses;
    
    // Compute loss for each layer
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        // Extract this layer's logits
        Tensor layer_logits = Tensor::zeros({batch, seq, vocab});
        
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t s = 0; s < seq; ++s) {
                for (int64_t v = 0; v < vocab; ++v) {
                    int64_t src_idx = layer * batch * seq * vocab +
                                     b * seq * vocab +
                                     s * vocab +
                                     v;
                    int64_t dst_idx = b * seq * vocab +
                                     s * vocab +
                                     v;
                    layer_logits.data()[dst_idx] = all_layer_logits.data()[src_idx];
                }
            }
        }
        
        // Compute loss for this layer (with mask) -> Returns [Batch]
        layer_losses.push_back(cross_entropy_loss(layer_logits, targets, mask));
    }
    
    // Final layer loss: [Batch]
    Tensor final_loss = layer_losses.back();
    
    // Auxiliary losses: [Batch]
    Tensor auxiliary_loss = Tensor::zeros({batch});
    if (num_layers > 1) {
        for (size_t i = 0; i < layer_losses.size() - 1; ++i) {
             for (int64_t b = 0; b < batch; ++b) {
                auxiliary_loss.data()[b] += layer_losses[i].data()[b];
             }
        }
        for (int64_t b = 0; b < batch; ++b) {
             auxiliary_loss.data()[b] /= (num_layers - 1);
        }
    }
    
    // UBOO weighted combination: [Batch]
    Tensor total_loss = Tensor::zeros({batch});
    for (int64_t b = 0; b < batch; ++b) {
        total_loss.data()[b] = 0.5f * final_loss.data()[b] + 0.5f * auxiliary_loss.data()[b];
    }
    
    return total_loss;
}

} // namespace mm_rec
