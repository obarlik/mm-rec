/**
 * Custom Memory Manager for MM-Rec
 * 
 * Implements bump allocation for extreme speed and lazy deallocation 
 * in a separate thread to prevent blocking the training loop.
 */

#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <iostream>
#include <cstring>

#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <iostream>
#include <cstring>

namespace mm_rec {

// A large chunk of memory (Arena Block)
struct MemoryBlock {
    size_t size;
    size_t used;
    uint8_t* data;
    
    MemoryBlock(size_t s) : size(s), used(0) {
        data = new uint8_t[size];
    }
    
    ~MemoryBlock() {
        delete[] data;
    }
    
    // Fast O(1) allocation
    void* allocate(size_t bytes) {
        // Align to 64 bytes for AVX/SIMD
        size_t alignment = 64;
        size_t aligned_used = (used + alignment - 1) & ~(alignment - 1);
        
        if (aligned_used + bytes > size) return nullptr;
        
        void* ptr = data + aligned_used;
        used = aligned_used + bytes;
        return ptr;
    }
    
    void reset() {
        used = 0;
    }
};

class MemoryManager {
public:
    static const size_t BLOCK_SIZE = 64 * 1024 * 1024; // 64 MB
    
    static MemoryManager& instance() {
        static MemoryManager instance;
        return instance;
    }
    
    // Fast Arena Allocation (Bump Pointer)
    void* allocate(size_t bytes) {
        if (!current_block_) {
            // Initial block
            if (!free_blocks_.empty()) {
                current_block_ = free_blocks_.back();
                free_blocks_.pop_back();
                current_block_->reset();
            } else {
                current_block_ = new MemoryBlock(BLOCK_SIZE);
            }
            blocks_.push_back(current_block_);
        }
        
        void* ptr = current_block_->allocate(bytes);
        
        // Block full? Get another one.
        if (!ptr) {
            // std::cout << "⚠️  Memory Block Full! Getting new block..." << std::endl;
            
            MemoryBlock* next_block = nullptr;
            if (!free_blocks_.empty()) {
                // Reuse from pool
                next_block = free_blocks_.back();
                free_blocks_.pop_back();
                next_block->reset();
            } else {
                // Allocate new from OS
                next_block = new MemoryBlock(BLOCK_SIZE);
            }
            
            current_block_ = next_block;
            blocks_.push_back(current_block_);
            ptr = current_block_->allocate(bytes);
        }
        
        return ptr;
    }
    
    // Free memory (Lazy)
    void deallocate(void* ptr) {
        // No-op in Arena allocator
    }
    
    void mark_persistent() {
        if (blocks_.empty()) {
            persistent_block_count_ = 0;
            persistent_used_ = 0;
            return;
        }
        persistent_block_count_ = blocks_.size();
        persistent_used_ = blocks_.back()->used;
        // std::cout << "DEBUG: Persistent Mark - Blocks: " << persistent_block_count_ << " Used: " << persistent_used_ << std::endl;
    }

    // Reset all memory (End of batch/step)
    // REUSE blocks conditions:
    // 1. Keep all "persistent" blocks (weights)
    // 2. Reset the LAST persistent block to the 'persistent_used_' offset
    // 3. Move all subsequent blocks to free_blocks_
    void reset_arena() {
        if (blocks_.empty()) return;

        // 1. Identify blocks to free (Transient blocks)
        if (blocks_.size() > persistent_block_count_) {
            // Note: If persistent_block_count_ is 1, and we have 5 blocks. 
            // We keep blocks_[0]. We move blocks_[1]..blocks_[4] to free.
            
            // Start index relies on whether persistent count is 0 (reset all) or N
            size_t start_idx = (persistent_block_count_ == 0) ? 1 : persistent_block_count_;
            // Wait, if count=1 (block 0 persistent), we start freeing at index 1.
            // If count=0 (nothing persistent), we keep block 0 (as current) and free 1..N.
            
            // Correction: Always keep at least ONE block active to avoid reallocation churn
            size_t keep_count = (persistent_block_count_ > 0) ? persistent_block_count_ : 1;
            
            for (size_t i = keep_count; i < blocks_.size(); ++i) {
                free_blocks_.push_back(blocks_[i]);
            }
            blocks_.resize(keep_count);
        }
        
        // 2. Reset the current active block
        if (!blocks_.empty()) {
            current_block_ = blocks_.back();
            if (persistent_block_count_ > 0 && blocks_.size() == persistent_block_count_) {
                // We are at the boundary block. Reset to offset.
                current_block_->used = persistent_used_;
            } else {
                // We are at a non-persistent block (or count=0). Reset to 0.
                current_block_->reset();
            }
        }
    }
    
    // Reporting
    size_t get_total_memory() const {
        size_t total = 0;
        for (const auto* b : blocks_) total += b->size;
        for (const auto* b : free_blocks_) total += b->size;
        return total;
    }
    
    // Destructor
    ~MemoryManager() {
        for (auto b : blocks_) delete b;
        for (auto b : free_blocks_) delete b;
        blocks_.clear();
        free_blocks_.clear();
    }

private:
    MemoryManager() : current_block_(nullptr), persistent_block_count_(0), persistent_used_(0) {}
    
    std::vector<MemoryBlock*> blocks_;       // Currently active blocks
    std::vector<MemoryBlock*> free_blocks_;  // Pool of free blocks for reuse
    MemoryBlock* current_block_;
    
    size_t persistent_block_count_;
    size_t persistent_used_;
};

} // namespace mm_rec
