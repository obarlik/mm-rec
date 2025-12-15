/**
 * Backward Pass Implementation
 */

#include "mm_rec/training/backward.h"
#include "mm_rec/core/tensor.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

namespace mm_rec {

// Helper: Eigen-optimized matmul
static Tensor eigen_matmul(const Tensor& A, const Tensor& B) {
    // A: [m, n], B: [n, p] → C: [m, p]
    int64_t m = A.size(0);
    int64_t n = A.size(1);
    int64_t p = B.size(1);
    
    // Map to Eigen matrices
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        eA(A.data(), m, n);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        eB(B.data(), n, p);
    
    // Compute with Eigen
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eC = eA * eB;
    
    // Copy back to Tensor
    Tensor C = Tensor::zeros({m, p});
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eC_map(C.data(), m, p);
    eC_map = eC;
    
    return C;
}

// Helper: Eigen-optimized stable softmax
static Tensor eigen_softmax(const Tensor& x) {
    // x: [batch, dim]
    int64_t batch = x.size(0);
    int64_t dim = x.size(1);
    
    Tensor y = Tensor::zeros(x.sizes());
    
    // Map to Eigen for vectorized operations
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eX(x.data(), batch, dim);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        eY(y.data(), batch, dim);
    
    for (int64_t b = 0; b < batch; ++b) {
        // Max for stability
        float max_val = eX.row(b).maxCoeff();
        
        // Exp and sum (vectorized)
        Eigen::VectorXf exp_vals = (eX.row(b).array() - max_val).exp();
        float sum = exp_vals.sum();
        
        // Normalize
        eY.row(b) = exp_vals / sum;
    }
    
    return y;
}


void linear_backward(
    const Tensor& x,
    const Tensor& W,
    const Tensor& dy,
    Tensor& dx,
    Tensor& dW,
    Tensor& db
) {
    // x: [batch, in_dim]
    // W: [out_dim, in_dim]
    // dy: [batch, out_dim]
    
    int64_t batch = x.size(0);
    int64_t in_dim = x.size(1);
    int64_t out_dim = dy.size(1);
    
    // dx = dy @ W  
    dx = eigen_matmul(dy, W);
    
    // dW = dy^T @ x
    dW = eigen_matmul(dy.transpose(), x);
    
    // db = sum(dy, axis=batch)
    // db[out_dim] = sum over batch dimension
    db = Tensor::zeros({out_dim});
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t o = 0; o < out_dim; ++o) {
            db.data()[o] += dy.data()[b * out_dim + o];
        }
    }
}

Tensor sigmoid_backward(const Tensor& y, const Tensor& dy) {
    // dx = dy * y * (1 - y)
    Tensor dx = Tensor::zeros(y.sizes());
    
    for (int64_t i = 0; i < y.numel(); ++i) {
        float y_val = y.data()[i];
        dx.data()[i] = dy.data()[i] * y_val * (1.0f - y_val);
    }
    
    return dx;
}

Tensor tanh_backward(const Tensor& y, const Tensor& dy) {
    // dx = dy * (1 - y^2)
    Tensor dx = Tensor::zeros(y.sizes());
    
    for (int64_t i = 0; i < y.numel(); ++i) {
        float y_val = y.data()[i];
        dx.data()[i] = dy.data()[i] * (1.0f - y_val * y_val);
    }
    
    return dx;
}

Tensor relu_backward(const Tensor& y, const Tensor& dy) {
    // dx = dy * (y > 0)
    Tensor dx = Tensor::zeros(y.sizes());
    
    for (int64_t i = 0; i < y.numel(); ++i) {
        float y_val = y.data()[i];
        dx.data()[i] = (y_val > 0.0f) ? dy.data()[i] : 0.0f;
    }
    
    return dx;
}

Tensor softmax_cross_entropy_backward(
    const Tensor& logits,
    const Tensor& targets
) {
    // logits: [batch, vocab]
    // targets: [batch] (class indices)
    
    int64_t batch = logits.size(0);
    int64_t vocab = logits.size(1);
    
    // Compute softmax
    Tensor y_pred = eigen_softmax(logits);
    
    // dlogits = y_pred - y_true (one-hot)
    // Manual copy instead of clone()
    Tensor dlogits = Tensor::zeros(y_pred.sizes());
    for (int64_t i = 0; i < y_pred.numel(); ++i) {
        dlogits.data()[i] = y_pred.data()[i];
    }
    
    for (int64_t b = 0; b < batch; ++b) {
        int64_t target_idx = static_cast<int64_t>(targets.data()[b]);
        dlogits.data()[b * vocab + target_idx] -= 1.0f;
    }
    
    // Average over batch
    for (int64_t i = 0; i < dlogits.numel(); ++i) {
        dlogits.data()[i] /= batch;
    }
    
    return dlogits;
}

} // namespace mm_rec
