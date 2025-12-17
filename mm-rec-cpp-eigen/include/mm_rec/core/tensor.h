/**
 * Minimal Tensor Class (Pure C++)
 * 
 * Lightweight tensor implementation without LibTorch dependency.
 * Uses MKL/OpenBLAS for performance, SIMD for activations.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include "mm_rec/utils/error_handling.h"

namespace mm_rec {

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(std::vector<int64_t> shape);
    
    // Destructor
    ~Tensor();
    
    // Copy constructor
    Tensor(const Tensor& other);
    
    // Move constructor
    Tensor(Tensor&& other) noexcept;
    
    // Copy assignment
    Tensor& operator=(const Tensor& other);
    
    // Move assignment
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Factory methods
    static Tensor zeros(std::vector<int64_t> shape);
    static Tensor ones(std::vector<int64_t> shape);
    static Tensor randn(std::vector<int64_t> shape, float mean = 0.0f, float std = 1.0f);
    static Tensor from_data(std::vector<float> data, std::vector<int64_t> shape);
    
    // Shape & size
    std::vector<int64_t> sizes() const { return shape_; }
    int64_t numel() const { return numel_; }
    int64_t ndim() const { return shape_.size(); }
    
    // Element access (with bounds checking)
    float& operator[](int64_t idx) {
        check_bounds(idx, numel_, "Tensor::operator[]");
        return data_ptr_[idx];
    }
    const float& operator[](int64_t idx) const {
        check_bounds(idx, numel_, "Tensor::operator[]");
        return data_ptr_[idx];
    }
    
    float* data() { return data_ptr_; }
    const float* data() const { return data_ptr_; }
    
    // Shape
    int64_t size(int dim) const {
        check_bounds(dim, shape_.size(), "Tensor::size");
        return shape_[dim];
    }

    // Reshape
    Tensor reshape(std::vector<int64_t> new_shape) const;
    
    // Matrix Multiplication
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
    std::vector<int64_t> shape_;
    float* data_ptr_ = nullptr;
    int64_t numel_ = 0;
    
    void check_shape_compatible(const Tensor& other) const;
};

} // namespace mm_rec
