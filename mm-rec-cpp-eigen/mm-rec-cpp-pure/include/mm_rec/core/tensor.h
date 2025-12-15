/**
 * Minimal Tensor Class (Pure C++)
 * 
 * Lightweight tensor implementation without LibTorch dependency.
 * Uses MKL/OpenBLAS for performance, SIMD for activations.
 */

#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <stdexcept>
#include <cmath>

namespace mm_rec {

class Tensor {
public:
    // Constructors
    Tensor() = default;
    Tensor(std::vector<int64_t> shape);
    
    // Factory methods
    static Tensor zeros(std::vector<int64_t> shape);
    static Tensor ones(std::vector<int64_t> shape);
    static Tensor randn(std::vector<int64_t> shape, float mean = 0.0f, float std = 1.0f);
    static Tensor from_data(std::vector<float> data, std::vector<int64_t> shape);
    
    // Shape & size
    std::vector<int64_t> sizes() const { return shape_; }
    int64_t size(int64_t dim) const;
    int64_t numel() const { return numel_; }
    int64_t dim() const { return shape_.size(); }
    
    // Data access
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    
    // Linear algebra (MKL-based)
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    
    // Element-wise operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // Element-wise multiply
    Tensor operator*(float scalar) const;
    
    // Activations (SIMD-optimized)
    Tensor sigmoid() const;
    Tensor tanh_activation() const;
    Tensor relu() const;
    
    // Utilities
    Tensor cat(const std::vector<Tensor>& tensors, int64_t dim);
    Tensor sum() const;
    Tensor mean() const;
    float item() const;  // For scalar tensors
    
    // Fill operations
    void fill_(float value);
    void zero_();
    
private:
    std::vector<float> data_;
    std::vector<int64_t> shape_;
    int64_t numel_;
    
    void compute_numel();
    void check_shape_compatible(const Tensor& other) const;
};

} // namespace mm_rec
