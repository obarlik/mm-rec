#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <algorithm>
#include <cstdlib> // posix_memalign
#include <new> // bad_alloc

namespace mm_rec {

// Slab Constants
// #define DISABLE_POOL_ALLOCATOR // DEBUGGING
constexpr size_t SLAB_SIZE = 64 * 1024;      // 64KB
constexpr uintptr_t SLAB_MASK = ~(uintptr_t(SLAB_SIZE - 1)); // Mask to find slab start

struct SlabHeader {
    uint32_t block_size;  // Size of objects in this slab (0 = large object)
    uint32_t magic;      // Safety check (0xAABBCCDD)
    // Padding to 32 bytes to ensure alignment of first object
    char padding[24]; 
};
static_assert(sizeof(SlabHeader) == 32, "SlabHeader must be 32 bytes");

/**
 * @brief Global Registry to hold ownership of ALL slabs from ALL threads.
 * Prevents "Use-After-Free" if a thread dies but another holds a pointer to its data.
 * Memory is released only when the program terminates.
 */
class GlobalSlabRegistry {
public:
    static GlobalSlabRegistry& instance() {
        static GlobalSlabRegistry instance;
        return instance;
    }

    // Takes ownership of a slab, returns raw pointer for thread usage
    void* register_slab(void* slab_ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        SlabNode* node = static_cast<SlabNode*>(std::malloc(sizeof(SlabNode)));
        if (!node) {
            std::cerr << "OOM in GlobalRegistry" << std::endl;
            std::abort();
        }
        
        node->slab_ptr = slab_ptr;
        node->next = head_;
        head_ = node; // Push front
        
        return slab_ptr;
    }
    
    // IMMORTAL: Intentionally leak slabs to prevent use-after-free during static destruction.
    // 
    // RATIONALE:
    // - C++ static destruction order is undefined
    // - Leaked threads may still hold pointers to slab memory
    // - OS reclaims ALL process memory on exit anyway (zero cost)
    // - This prevents crashes from destroyed allocator accessed by running threads
    // 
    // SAFETY: This is the recommended practice for global allocators that may be
    // accessed during static destruction (see Google's tcmalloc, jemalloc, etc.)
    ~GlobalSlabRegistry() {
        // Intentionally empty - leak slabs for safety
    }

private:
    struct SlabNode {
        void* slab_ptr;
        SlabNode* next;
    };

    SlabNode* head_ = nullptr;
    std::mutex mutex_;
};

/**
 * @brief FixedSizePool manages allocations of a single specific size (e.g., 64 bytes).
 * Uses a "Slab" architecture where large chunks are requested from OS.
 * Slabs are owned by GlobalSlabRegistry to ensure safety across threads.
 */
class FixedSizePool {
public:
    explicit FixedSizePool(size_t block_size)
        : block_size_(block_size) {
        // Alignment: Critical for SIMD (Eigen/MKL)
        // Align to 32 bytes (AVX)
        while (block_size_ % 32 != 0) block_size_++;
    }

    ~FixedSizePool() {
        // No cleanup needed! Slabs are owned by GlobalRegistry.
        // We just walk away. Safe.
    }

    // No Copy
    FixedSizePool(const FixedSizePool&) = delete;
    FixedSizePool& operator=(const FixedSizePool&) = delete;

    void* allocate() {
        
        // 1. Try Free List
        if (free_list_head_) {
            void* ptr = free_list_head_;
            free_list_head_ = *reinterpret_cast<void**>(free_list_head_);
            return ptr;
        }

        // 2. Try Current Slab
        // Effective offset starts after header (32 bytes)
        size_t effective_offset = std::max(current_slab_offset_, sizeof(SlabHeader));
        
        if (!current_slab_ptr_ || effective_offset + block_size_ > SLAB_SIZE) {
            allocate_new_slab();
            effective_offset = sizeof(SlabHeader);
        }

        void* ptr = current_slab_ptr_ + effective_offset;
        current_slab_offset_ = effective_offset + block_size_;
        return ptr;
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
        
        // Push to Free List head (LIFO)
        *reinterpret_cast<void**>(ptr) = free_list_head_;
        free_list_head_ = ptr;
    }

    size_t block_size() const { return block_size_; }

private:
    void allocate_new_slab() {
        void* raw_mem = nullptr;
        // Allocate 64KB Aligned
        if (posix_memalign(&raw_mem, SLAB_SIZE, SLAB_SIZE) != 0) {
            std::cerr << "OOM: posix_memalign failed" << std::endl;
            std::abort();
        }
        
        // Write Header
        SlabHeader* header = static_cast<SlabHeader*>(raw_mem);
        header->block_size = block_size_;
        header->magic = 0xAABBCCDD;

        // Register
        GlobalSlabRegistry::instance().register_slab(raw_mem);

        current_slab_ptr_ = static_cast<char*>(raw_mem);
        current_slab_offset_ = sizeof(SlabHeader);
    }

    size_t block_size_;
    void* free_list_head_ = nullptr;
    char* current_slab_ptr_ = nullptr;
    size_t current_slab_offset_ = 0;
};

/**
 * @brief PoolAllocator manages multiple FixedSizePools to handle requests of varying sizes.
 * It routes specific size requests to the appropriate "Bin" (Pool).
 */
class PoolAllocator {
public:
    static PoolAllocator& instance() {
        static thread_local PoolAllocator instance;
        return instance;
    }

    PoolAllocator() {
        // Register common bins
        // MANUAL ALLOCATION (malloc) to avoid calling global new recursively!
        
        pools_[0] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(32);
        pools_[1] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(64);
        pools_[2] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(128);
        pools_[3] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(256);
        pools_[4] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(512);
        pools_[5] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(1024);
    }
    
    ~PoolAllocator() {
        // IMMORTAL: Do not free pools.
    }

    void* allocate(size_t bytes) {
#ifdef DISABLE_POOL_ALLOCATOR
        return std::malloc(bytes);
#else
        // 1. Try Pools (Small Objects)
        if (bytes <= 32) return pools_[0]->allocate();
        if (bytes <= 64) return pools_[1]->allocate();
        if (bytes <= 128) return pools_[2]->allocate();
        if (bytes <= 256) return pools_[3]->allocate();
        if (bytes <= 512) return pools_[4]->allocate();
        if (bytes <= 1024) return pools_[5]->allocate();
        
        // 2. Large Objects (> 1024) -> Multi-Page Slab
        size_t total_needed = sizeof(SlabHeader) + bytes;
        
        size_t alloc_size = (total_needed + SLAB_SIZE - 1) & SLAB_MASK;
        if (alloc_size < total_needed) alloc_size += SLAB_SIZE;
        
        void* raw_mem = nullptr;
        if (posix_memalign(&raw_mem, SLAB_SIZE, alloc_size) != 0) return nullptr;
        
        SlabHeader* header = static_cast<SlabHeader*>(raw_mem);
        header->block_size = 0; // 0 = Large Object
        header->magic = 0xAABBCCDD;
        
        return static_cast<char*>(raw_mem) + sizeof(SlabHeader);
#endif
    }

    void deallocate(void* ptr) {
        if (!ptr) return;
#ifdef DISABLE_POOL_ALLOCATOR
        std::free(ptr);
        return;
#endif
        
        // 1. Bitwise Masking to find Header
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t header_addr = addr & SLAB_MASK;
        SlabHeader* header = reinterpret_cast<SlabHeader*>(header_addr);
        
        // 2. Check Magic (Safety)
        if (header->magic != 0xAABBCCDD) {
             std::free(ptr); // Not ours
             return; 
        }
        
        // 3. Dispatch
        if (header->block_size == 0) {
            // Large Object
            std::free(reinterpret_cast<void*>(header));
        } else {
            // Small Object -> Pool
            uint32_t sz = header->block_size;
            int idx = -1;
            if (sz <= 32) idx=0;
            else if (sz <= 64) idx=1;
            else if (sz <= 128) idx=2;
            else if (sz <= 256) idx=3;
            else if (sz <= 512) idx=4;
            else idx=5; // 1024
            
            if (idx >= 0) pools_[idx]->deallocate(ptr);
        }
    }
    
    void deallocate(void* ptr, size_t) { deallocate(ptr); }

private:
    void init_pool(int index, size_t size) {
         pools_[index] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(size);
    }

    FixedSizePool* pools_[6];
};

} // namespace mm_rec
