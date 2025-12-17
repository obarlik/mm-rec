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
    
    // Reset all memory (End of batch/step)
    // REUSE blocks instead of deleting them!
    void reset_arena() {
        if (blocks_.empty()) return;

        // Move all used blocks (except maybe the first one? No, let's keep one active)
        // Actually, simple strategy: move ALL blocks to free_blocks_, then pick one for current.
        
        // Optimize: Keep the first block as current, move others to free pool
        if (blocks_.size() > 1) {
            for (size_t i = 1; i < blocks_.size(); ++i) {
                free_blocks_.push_back(blocks_[i]);
            }
            // Resize to keep only the first one
            MemoryBlock* first = blocks_[0];
            blocks_.clear();
            blocks_.push_back(first);
            current_block_ = first;
        }
        
        if (current_block_) {
            current_block_->reset();
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
    MemoryManager() : current_block_(nullptr) {}
    
    std::vector<MemoryBlock*> blocks_;       // Currently active blocks
    std::vector<MemoryBlock*> free_blocks_;  // Pool of free blocks for reuse
    MemoryBlock* current_block_;
};

} // namespace mm_rec
