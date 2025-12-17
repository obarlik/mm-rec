#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <algorithm>

namespace mm_rec {

/**
 * @brief FixedSizePool manages allocations of a single specific size (e.g., 64 bytes).
 * Uses a "Slab" architecture where large chunks are requested from OS,
 * and internally divided into a linked-list of free slots.
 */
class FixedSizePool {
public:
    explicit FixedSizePool(size_t block_size, size_t slab_size_bytes = 64 * 1024)
        : block_size_(std::max(block_size, sizeof(void*))), // Ensure regular pointer fits
          slab_size_(slab_size_bytes) {
        // Alignment
        while (block_size_ % 8 != 0) block_size_++;
    }

    ~FixedSizePool() {
        // Unique_ptrs in slabs_ will auto-clean
    }

    // No Copy
    FixedSizePool(const FixedSizePool&) = delete;
    FixedSizePool& operator=(const FixedSizePool&) = delete;

    void* allocate() {
        // Removed Mutex: Because this class is now owned by a thread_local PoolAllocator!
        
        // 1. Try Free List
        if (free_list_head_) {
            void* ptr = free_list_head_;
            free_list_head_ = *reinterpret_cast<void**>(free_list_head_);
            return ptr;
        }

        // 2. Try Current Slab
        if (!current_slab_ptr_ || current_slab_offset_ + block_size_ > slab_size_) {
            allocate_new_slab();
        }

        void* ptr = current_slab_ptr_ + current_slab_offset_;
        current_slab_offset_ += block_size_;
        return ptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        // Removed Mutex
        
        // Push to Free List head (LIFO)
        *reinterpret_cast<void**>(ptr) = free_list_head_;
        free_list_head_ = ptr;
    }

    size_t block_size() const { return block_size_; }

private:
    void allocate_new_slab() {
        auto new_slab = std::make_unique<char[]>(slab_size_);
        current_slab_ptr_ = new_slab.get();
        current_slab_offset_ = 0;
        slabs_.push_back(std::move(new_slab));
    }

    size_t block_size_;
    size_t slab_size_;
    
    void* free_list_head_ = nullptr;

    char* current_slab_ptr_ = nullptr;
    size_t current_slab_offset_ = 0;

    std::vector<std::unique_ptr<char[]>> slabs_;
    
    // std::mutex mutex_; // REMOVED
};

/**
 * @brief PoolAllocator manages multiple FixedSizePools to handle requests of varying sizes.
 * It routes specific size requests to the appropriate "Bin" (Pool).
 */
class PoolAllocator {
public:
    // Thread-Local Singleton
    static PoolAllocator& instance() {
        static thread_local PoolAllocator instance;
        return instance;
    }

    PoolAllocator() {
        // Register common bins
        pools_.emplace_back(std::make_unique<FixedSizePool>(32));
        pools_.emplace_back(std::make_unique<FixedSizePool>(64));
        pools_.emplace_back(std::make_unique<FixedSizePool>(128));
        pools_.emplace_back(std::make_unique<FixedSizePool>(256));
        pools_.emplace_back(std::make_unique<FixedSizePool>(512));
        pools_.emplace_back(std::make_unique<FixedSizePool>(1024));
    }

    void* allocate(size_t bytes) {
        // Find best fitting pool
        for (auto& pool : pools_) {
            if (bytes <= pool->block_size()) {
                return pool->allocate();
            }
        }
        return std::malloc(bytes);
    }

    void deallocate(void* ptr, size_t bytes) {
        if (!ptr) return;

        for (auto& pool : pools_) {
            if (bytes <= pool->block_size()) {
                pool->deallocate(ptr);
                return;
            }
        }
        std::free(ptr);
    }

private:
    std::vector<std::unique_ptr<FixedSizePool>> pools_;
};

} // namespace mm_rec
