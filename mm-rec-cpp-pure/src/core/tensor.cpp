/**
 * Tensor Implementation (Pure C++)
 */

#include "mm_rec/core/tensor.h"
#include <mkl.h>  // Intel MKL
#include <random>
#include <numeric>
#include <cstring>

namespace mm_rec {

// ============================================================================
// Constructors & Factory Methods
// ============================================================================

Tensor::Tensor(std::vector<int64_t> shape) : shape_(std::move(shape)) {
    compute_numel();
    data_.resize(numel_);
}

void Tensor::compute_numel() {
    numel_ = 1;
    for (auto s : shape_) {
        numel_ *= s;
    }
}

Tensor Tensor::zeros(std::vector<int64_t> shape) {
    Tensor t(shape);
    std::fill(t.data_.begin(), t.data_.end(), 0.0f);
    return t;
}

Tensor Tensor::ones(std::vector<int64_t> shape) {
    Tensor t(shape);
    std::fill(t.data_.begin(), t.data_.end(), 1.0f);
    return t;
}

Tensor Tensor::randn(std::vector<int64_t> shape, float mean, float std) {
    Tensor t(shape);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    
    for (auto& val : t.data_) {
        val = dist(gen);
    }
    
    return t;
}

Tensor Tensor::from_data(std::vector<float> data, std::vector<int64_t> shape) {
    Tensor t(shape);
    if (data.size() != static_cast<size_t>(t.numel_)) {
        throw std::runtime_error("Data size mismatch");
    }
    t.data_ = std::move(data);
    return t;
}

// ============================================================================
// Linear Algebra (MKL BLAS)
// ============================================================================

Tensor Tensor::matmul(const Tensor& other) const {
    // this: [m, k], other: [k, n] -> result: [m, n]
    
    if (dim() != 2 || other.dim() != 2) {
        throw std::runtime_error("matmul requires 2D tensors");
    }
    
    int64_t m = shape_[0];
    int64_t k = shape_[1];
    int64_t k2 = other.shape_[0];
    int64_t n = other.shape_[1];
    
    if (k != k2) {
        throw std::runtime_error("matmul shape mismatch");
    }
    
    Tensor result = zeros({m, n});
    
    // Use MKL BLAS
    // C = alpha * A * B + beta * C
    cblas_sgemm(
        CblasRowMajor,       // Row-major order
        CblasNoTrans,        // Don't transpose A
        CblasNoTrans,        // Don't transpose B
        m, n, k,             // Dimensions
        1.0f,                // alpha
        data(),              // A
        k,                   // lda
        other.data(),        // B
        n,                   // ldb
        0.0f,                // beta
        result.data(),       // C
        n                    // ldc
    );
    
    return result;
}

Tensor Tensor::transpose() const {
    if (dim() != 2) {
        throw std::runtime_error("transpose requires 2D tensor");
    }
    
    int64_t m = shape_[0];
    int64_t n = shape_[1];
    
    Tensor result({n, m});
    
    #pragma omp parallel for collapse(2)
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            result.data_[j * m + i] = data_[i * n + j];
        }
    }
    
    return result;
}

// ============================================================================
// Element-wise Operations (SIMD-optimized)
// ============================================================================

Tensor Tensor::operator+(const Tensor& other) const {
    check_shape_compatible(other);
    
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    check_shape_compatible(other);
    
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    check_shape_compatible(other);
    
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    
    return result;
}

// ============================================================================
// Activation Functions (Vectorized)
// ============================================================================

Tensor Tensor::sigmoid() const {
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
    }
    
    return result;
}

Tensor Tensor::tanh_activation() const {
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_[i] = std::tanh(data_[i]);
    }
    
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_[i] = std::max(0.0f, data_[i]);
    }
    
    return result;
}

// ============================================================================
// Utilities
// ============================================================================

int64_t Tensor::size(int64_t dim) const {
    if (dim < 0) dim += shape_.size();
    return shape_[dim];
}

float Tensor::item() const {
    if (numel_ != 1) {
        throw std::runtime_error("item() only for scalar tensors");
    }
    return data_[0];
}

void Tensor::fill_(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::zero_() {
    fill_(0.0f);
}

void Tensor::check_shape_compatible(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Shape mismatch");
    }
}

Tensor Tensor::sum() const {
    float result = 0.0f;
    
    #pragma omp simd reduction(+:result)
    for (int64_t i = 0; i < numel_; ++i) {
        result += data_[i];
    }
    
    return from_data({result}, {1});
}

Tensor Tensor::mean() const {
    return sum() * (1.0f / static_cast<float>(numel_));
}

} // namespace mm_rec
