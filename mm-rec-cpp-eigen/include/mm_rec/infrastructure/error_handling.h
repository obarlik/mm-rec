/**
 * Error Handling Utilities
 * 
 * Bounds checking, NaN detection, numerical stability
 */

#pragma once

#include <stdexcept>
#include <string>
#include <cmath>

namespace mm_rec {

/**
 * Check if value is NaN or Inf
 */
inline bool is_invalid_float(float value) {
    return std::isnan(value) || std::isinf(value);
}

/**
 * Check tensor for NaN/Inf values
 */
inline void check_tensor_validity(const float* data, int64_t size, const std::string& name) {
    for (int64_t i = 0; i < size; ++i) {
        if (is_invalid_float(data[i])) {
            throw std::runtime_error(
                "Invalid value (NaN/Inf) detected in tensor: " + name +
                " at index " + std::to_string(i)
            );
        }
    }
}

/**
 * Check bounds for tensor access
 */
inline void check_bounds(int64_t index, int64_t size, const std::string& name) {
    if (index < 0 || index >= size) {
        throw std::out_of_range(
            "Index out of bounds in " + name + ": " +
            std::to_string(index) + " (size: " + std::to_string(size) + ")"
        );
    }
}

/**
 * Check shape compatibility for operations
 */
inline void check_shape_compatible(
    int64_t dim1, int64_t dim2,
    const std::string& op_name
) {
    if (dim1 != dim2) {
        throw std::runtime_error(
            "Shape mismatch in " + op_name + ": " +
            std::to_string(dim1) + " vs " + std::to_string(dim2)
        );
    }
}

} // namespace mm_rec
