/**
 * Tensor Implementation (Eigen-based)
 * 
 * ZERO runtime dependencies - header-only Eigen library
 */

#include "mm_rec/core/tensor.h"
#include "mm_rec/core/memory_manager.h"
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cstring> // memcpy, memset

namespace mm_rec {

// ============================================================================
// Constructors & Factory Methods
// ============================================================================

Tensor::Tensor() : shape_({}), data_ptr_(nullptr), numel_(0) {
}

Tensor::Tensor(std::vector<int64_t> shape) : shape_(std::move(shape)) {
    numel_ = 1;
    for (auto s : shape_) numel_ *= s;
    
    if (numel_ > 0) {
        data_ptr_ = static_cast<float*>(MemoryManager::instance().allocate(numel_ * sizeof(float)));
        // Initialize to zero for safety? No, typically undefined for performance, but let's zero for safety
        // Actually, most ML frameworks leave uninitialized. Let's leave uninitialized (or zero if requested).
        // Let's zero it for now to avoid NaNs if user forgets initialization.
        // Actually, zeros() calls this then fills with 0. 
        // Let's leave uninitialized in base constructor for speed.
    }
}

Tensor::~Tensor() {
    if (data_ptr_) {
        MemoryManager::instance().deallocate(data_ptr_);
        data_ptr_ = nullptr;
    }
    numel_ = 0;
}

// Deep copy
Tensor Tensor::clone() const {
    // Uses copy constructor logic but explicit
    Tensor t(shape_);
    if (numel_ > 0) {
        std::memcpy(t.data_ptr_, data_ptr_, numel_ * sizeof(float));
    }
    return t;
}

// Copy constructor
Tensor::Tensor(const Tensor& other) : shape_(other.shape_), numel_(other.numel_) {
    if (numel_ > 0) {
        data_ptr_ = static_cast<float*>(MemoryManager::instance().allocate(numel_ * sizeof(float)));
        std::memcpy(data_ptr_, other.data_ptr_, numel_ * sizeof(float));
    }
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept 
    : shape_(std::move(other.shape_)), data_ptr_(other.data_ptr_), numel_(other.numel_) {
    other.data_ptr_ = nullptr;
    other.numel_ = 0;
}

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        // Free existing
        if (data_ptr_) {
            MemoryManager::instance().deallocate(data_ptr_);
        }
        
        shape_ = other.shape_;
        numel_ = other.numel_;
        
        if (numel_ > 0) {
            // Allocate memory
            data_ptr_ = static_cast<float*>(MemoryManager::instance().allocate(numel_ * sizeof(float)));
            std::memcpy(data_ptr_, other.data_ptr_, numel_ * sizeof(float));
        } else {
            data_ptr_ = nullptr;
        }
    }
    return *this;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Free existing
        if (data_ptr_) {
            MemoryManager::instance().deallocate(data_ptr_);
        }
        
        shape_ = std::move(other.shape_);
        data_ptr_ = other.data_ptr_;
        numel_ = other.numel_;
        
        other.data_ptr_ = nullptr;
        other.numel_ = 0;
    }
    return *this;
}

Tensor Tensor::zeros(std::vector<int64_t> shape) {
    Tensor t(shape);
    if (t.numel_ > 0) {
        std::memset(t.data_ptr_, 0, t.numel_ * sizeof(float));
    }
    return t;
}

Tensor Tensor::ones(std::vector<int64_t> shape) {
    Tensor t(shape);
    if (t.numel_ > 0) {
        // std::fill works with pointers too
        std::fill(t.data_ptr_, t.data_ptr_ + t.numel_, 1.0f);
    }
    return t;
}

Tensor Tensor::randn(std::vector<int64_t> shape, float mean, float std) {
    Tensor t(shape);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    
    for (int64_t i = 0; i < t.numel_; ++i) {
        t.data_ptr_[i] = dist(gen);
    }
    
    return t;
}

Tensor Tensor::from_data(std::vector<float> data, std::vector<int64_t> shape) {
    Tensor t(shape);
    if (static_cast<int64_t>(data.size()) != t.numel_) {
        throw std::runtime_error("Data size mismatch");
    }
    if (t.numel_ > 0) {
        std::memcpy(t.data_ptr_, data.data(), t.numel_ * sizeof(float));
    }
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
        A(data_ptr_, m, k);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        B(other.data_ptr_, k, n);
    
    Tensor result = zeros({m, n});
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        C(result.data_ptr_, m, n);
    
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
            result.data_ptr_[j * m + i] = data_ptr_[i * n + j];
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
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_ptr_[i] = data_ptr_[i] + other.data_ptr_[i];
    }
    
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) check_shape_compatible(other);
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_ptr_[i] = data_ptr_[i] - other.data_ptr_[i];
    }
    
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    check_shape_compatible(other);
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_ptr_[i] = data_ptr_[i] * other.data_ptr_[i];
    }
    
    return result;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_ptr_[i] = data_ptr_[i] * scalar;
    }
    
    return result;
}

// ============================================================================
// Activation Functions (Vectorized)
// ============================================================================
// Activations
Tensor Tensor::sigmoid() const {
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel_; ++i) {
        result.data_ptr_[i] = 1.0f / (1.0f + std::exp(-data_ptr_[i]));
    }
    
    return result;
}

Tensor Tensor::tanh_activation() const {
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel(); ++i) {
        result.data_ptr_[i] = std::tanh(data_ptr_[i]);
    }
    
    return result;
}

Tensor Tensor::relu() const {
    Tensor result(shape_);
    
    #pragma omp simd
    for (int64_t i = 0; i < numel(); ++i) {
        result.data_ptr_[i] = std::max(0.0f, data_ptr_[i]);
    }
    
    return result;
}



float Tensor::item() const {
    if (numel() != 1) {
        throw std::runtime_error("item() only for scalar tensors");
    }
    return data_ptr_[0];
}

// Fill operations
void Tensor::fill_(float value) {
    if (numel_ > 0) {
        std::fill(data_ptr_, data_ptr_ + numel_, value);
    }
}

void Tensor::zero_() {
    if (numel_ > 0) {
        std::memset(data_ptr_, 0, numel_ * sizeof(float));
    }
}

// Private helpers
void Tensor::check_shape_compatible(const Tensor& other) const {
    if (shape_ != other.shape_) {
        // Format error message
        std::string s1 = "[";
        for(auto d : shape_) s1 += std::to_string(d) + ",";
        s1 += "]";
        
        std::string s2 = "[";
        for(auto d : other.shape_) s2 += std::to_string(d) + ",";
        s2 += "]";
        
        throw std::runtime_error("Shape mismatch: " + s1 + " vs " + s2);
    }
}

Tensor Tensor::sum() const {
    float result = 0.0f;
    
    #pragma omp simd reduction(+:result)
    for (int64_t i = 0; i < numel(); ++i) {
        result += data_ptr_[i];
    }
    
    return from_data({result}, {1});
}

Tensor Tensor::mean() const {
    return sum() * (1.0f / static_cast<float>(numel()));
}

} // namespace mm_rec
