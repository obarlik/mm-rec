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
    void* register_slab(std::unique_ptr<char[]> slab) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Manual Linked List Node allocation via malloc (avoids global new recursion)
        SlabNode* node = static_cast<SlabNode*>(std::malloc(sizeof(SlabNode)));
        if (!node) {
            // Panic or fallback? If we can't malloc a node, we are doomed.
            // Just return the slab ptr and leak the node tracking? No.
            // But slab is unique_ptr.
            // If malloc fails, we probably crash anyway.
            std::cerr << "OOM in GlobalRegistry" << std::endl;
            std::abort();
        }
        
        node->slab_ptr = slab.release(); // Take ownership
        node->next = head_;
        head_ = node; // Push front
        
        return node->slab_ptr;
    }
    
    ~GlobalSlabRegistry() {
        SlabNode* current = head_;
        while (current) {
            SlabNode* next = current->next;
            // Free the slab char array
            delete[] current->slab_ptr;
            // Free the node
            std::free(current);
            current = next;
        }
    }

private:
    struct SlabNode {
        char* slab_ptr;
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
    explicit FixedSizePool(size_t block_size, size_t slab_size_bytes = 64 * 1024)
        : block_size_(std::max(block_size, sizeof(void*))), 
          slab_size_(slab_size_bytes) {
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
        // Create Slab
        auto new_slab = std::make_unique<char[]>(slab_size_);
        
        // Transfer ownership to Global Registry (Thread-Safe)
        // We get back a raw pointer that is guaranteed to be valid until app exit.
        current_slab_ptr_ = static_cast<char*>(GlobalSlabRegistry::instance().register_slab(std::move(new_slab)));
        
        current_slab_offset_ = 0;
    }

    size_t block_size_;
    size_t slab_size_;
    
    void* free_list_head_ = nullptr;

    char* current_slab_ptr_ = nullptr;
    size_t current_slab_offset_ = 0;

    // No local ownership of slabs anymore
};

/**
 * @brief PoolAllocator manages multiple FixedSizePools to handle requests of varying sizes.
 * It routes specific size requests to the appropriate "Bin" (Pool).
 */
/**
 * @brief PoolAllocator manages multiple FixedSizePools.
 * Uses "Header-based" tracking to support unsized delete() and ensure Alignment.
 */
class PoolAllocator {
public:
    // 16-Byte Header to maintain 16-byte alignment for user data
    struct AllocHeader {
        size_t capacity; // The bin size (if pool) or full size (if malloc)
        size_t padding;  // Padding to ensure sizeof(Header) == 16
    };
    static constexpr size_t HEADER_SIZE = sizeof(AllocHeader);

    static PoolAllocator& instance() {
        static thread_local PoolAllocator instance;
        return instance;
    }

    PoolAllocator() {
        // Register common bins
        // MANUAL ALLOCATION (malloc) to avoid calling global new recursively!
        
        pools_[0] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(32 + HEADER_SIZE);
        pools_[1] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(64 + HEADER_SIZE);
        pools_[2] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(128 + HEADER_SIZE);
        pools_[3] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(256 + HEADER_SIZE);
        pools_[4] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(512 + HEADER_SIZE);
        pools_[5] = new(std::malloc(sizeof(FixedSizePool))) FixedSizePool(1024 + HEADER_SIZE);
    }
    
    ~PoolAllocator() {
        // IMMORTAL: Do not free pools.
        // Thread-local destruction order is unpredictable.
        // If we free these, a late running "delete" will crash.
        // Leaking 6 tiny structs per thread is safer.
        /*
        for(int i=0; i<6; ++i) {
            pools_[i]->~FixedSizePool();
            std::free(pools_[i]);
        }
        */
    }

    void* allocate(size_t bytes) {
        // 1. Try Pools
        // Unroll loop for speed and no iterators
        for (int i=0; i<6; ++i) {
            FixedSizePool* pool = pools_[i];
            size_t user_capacity = pool->block_size() - HEADER_SIZE;
            
            if (bytes <= user_capacity) {
                void* raw_ptr = pool->allocate();
                
                // Write Header
                AllocHeader* header = static_cast<AllocHeader*>(raw_ptr);
                header->capacity = user_capacity; 
                
                return static_cast<char*>(raw_ptr) + HEADER_SIZE;
            }
        }
        
        // 2. Fallback to Malloc
        void* raw_ptr = std::malloc(bytes + HEADER_SIZE);
        if (!raw_ptr) return nullptr;
        
        AllocHeader* header = static_cast<AllocHeader*>(raw_ptr);
        header->capacity = bytes; 
        
        return static_cast<char*>(raw_ptr) + HEADER_SIZE;
    }

    // New: Deallocate without knowing size (uses Header)
    void deallocate(void* ptr) {
        if (!ptr) return;
        
        char* raw_ptr = static_cast<char*>(ptr) - HEADER_SIZE;
        AllocHeader* header = reinterpret_cast<AllocHeader*>(raw_ptr);
        size_t capacity = header->capacity;

        for (int i=0; i<6; ++i) {
            FixedSizePool* pool = pools_[i];
            size_t user_capacity = pool->block_size() - HEADER_SIZE;
            if (capacity == user_capacity) {
                pool->deallocate(raw_ptr);
                return;
            }
        }
        
        std::free(raw_ptr);
    }
    
    void deallocate(void* ptr, size_t /*unused_size*/) {
        deallocate(ptr);
    }

private:
   FixedSizePool* pools_[6];
};

} // namespace mm_rec
