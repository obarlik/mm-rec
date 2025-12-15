/**
 * UBOO (Universal Basis for Output Optimization)
 * 
 * Key Innovation: Every layer predicts next token, not just final layer.
 * 
 * Benefits from production:
 * - Loss 8.0 → 0.003 (27 epochs)
 * - Improved gradient flow (all layers get direct supervision)
 * - Better data efficiency (lower layers learn output-relevant features)
 * 
 * Implementation: Each MM-Rec block has its own linear projection to vocabulary
 */

#pragma once

#include <torch/torch.h>

namespace mm_rec {

/**
 * Per-Layer Output Projection (UBOO Component)
 * 
 * Projects hidden state to vocabulary logits
 */
class UBOOOutput : public torch::nn::Module {
public:
    /**
     * Constructor
     * 
     * @param hidden_dim Dimension of hidden state from the layer
     * @param vocab_size Size of vocabulary
     */
    UBOOOutput(int64_t hidden_dim, int64_t vocab_size);

    /**
     * Forward: Project hidden state to logits
     * 
     * @param hidden Hidden state [batch, seq_len, hidden_dim]
     * @return Logits [batch, seq_len, vocab_size]
     */
    torch::Tensor forward(const torch::Tensor& hidden);

private:
    torch::nn::Linear projection{nullptr};
    int64_t hidden_dim_;
    int64_t vocab_size_;
};

} // namespace mm_rec
