/**
 * Tensor Implementation (Eigen-based)
 * 
 * ZERO runtime dependencies - header-only Eigen library
 */

#include "mm_rec/core/tensor.h"
#include <Eigen/Dense>
#include <random>
#include <iostream>

namespace mm_rec {

// ============================================================================
// Constructors & Factory Methods
// ============================================================================

Tensor::Tensor() : data_{}, shape_({}) {
}

Tensor::Tensor(std::vector<int64_t> shape) : shape_(std::move(shape)) {
    int64_t n = 1;
    for (auto s : shape_) n *= s;
    data_.resize(n);
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
    if (data.size() != static_cast<size_t>(t.numel())) {
        throw std::runtime_error("Data size mismatch");
    }
    t.data_ = std::move(data);
    return t;
}

Tensor Tensor::reshape(std::vector<int64_t> new_shape) const {
    int64_t new_numel = 1;
    for(auto s : new_shape) new_numel *= s;
    
    if(new_numel != numel() && new_numel != -1) {
         throw std::runtime_error("Reshape size mismatch");
    }
    
    // Handle -1 inference
    if(new_numel == -1) {
        int64_t known = 1;
        int unknown_idx = -1;
        for(size_t i=0; i<new_shape.size(); ++i) {
            if(new_shape[i] == -1) {
                if(unknown_idx != -1) throw std::runtime_error("Multiple -1 in reshape");
                unknown_idx = i;
            } else {
                known *= new_shape[i];
            }
        }
        
        if(numel() % known != 0) throw std::runtime_error("Invalid reshape");
        new_shape[unknown_idx] = numel() / known;
    }
    
    Tensor t = *this; // Copy data
    t.shape_ = new_shape;
    return t;
}

// ============================================================================
// Linear Algebra (Eigen-based - ZERO RUNTIME DEPENDENCY!)
// ============================================================================

Tensor Tensor::matmul(const Tensor& other) const {
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::runtime_error("matmul requires 2D tensors");
    }
    
    int64_t m = shape_[0];
    int64_t k = shape_[1];
    int64_t k2 = other.shape_[0];
    int64_t n = other.shape_[1];
    
    if (k != k2) {
        throw std::runtime_error("matmul shape mismatch");
    }
    
    // Use Eigen for optimized matmul
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        A(data(), m, k);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        B(other.data(), k, n);
    
    Tensor result = zeros({m, n});
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        C(result.data(), m, n);
    
    // Eigen automatically uses SIMD and optimizations
    C = A * B;
    
    return result;
}

Tensor Tensor::transpose() const {
    if (ndim() != 2) {
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
// Element-wise Operations (Vectorized)
// ============================================================================

Tensor Tensor::operator+(const Tensor& other) const {
    check_shape_compatible(other);
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    check_shape_compatible(other);
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    check_shape_compatible(other);
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel(); ++i) {
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
    for (int64_t i = 0; i < numel(); ++i) {
        result.data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
    }
    
    return result;
}

Tensor Tensor::tanh_activation() const {
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel(); ++i) {
        result.data_[i] = std::tanh(data_[i]);
    }
    
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel(); ++i) {
        result.data_[i] = std::max(0.0f, data_[i]);
    }
    
    return result;
}



float Tensor::item() const {
    if (numel() != 1) {
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
    for (int64_t i = 0; i < numel(); ++i) {
        result += data_[i];
    }
    
    return from_data({result}, {1});
}

Tensor Tensor::mean() const {
    return sum() * (1.0f / static_cast<float>(numel()));
}

} // namespace mm_rec
