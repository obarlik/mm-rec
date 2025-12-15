// Placeholder for log sum exp utilities
// Will implement when needed for numerical stability

#include <torch/torch.h>

namespace mm_rec {

// Stable log-sum-exp for numerical stability
inline torch::Tensor stable_log_sum_exp(
    const torch::Tensor& log_a,
    const torch::Tensor& log_b
) {
    auto max_val = torch::max(log_a, log_b);
    auto diff = torch::abs(log_a - log_b);
    auto diff_clamped = torch::clamp(diff, 0.0, 20.0);
    return max_val + torch::log1p(torch::exp(-diff_clamped));
}

} // namespace mm_rec
