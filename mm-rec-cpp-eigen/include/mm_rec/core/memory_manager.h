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
    static MemoryManager& instance() {
        static MemoryManager instance;
        return instance;
    }
    
    // Fast Arena Allocation (Bump Pointer)
    void* allocate(size_t bytes) {
        // Fast path: try current block
        // No mutex for speed in single-threaded training loop?
        // Actually training is single threaded, so this is safe.
        // If multi-threaded training, we need thread-local arenas.
        // Assuming single thread training based on code.
        
        if (!current_block_) {
            // First alloc: Create 1GB block
            current_block_ = new MemoryBlock(1024 * 1024 * 1024); // 1 GB
            blocks_.push_back(current_block_);
        }
        
        void* ptr = current_block_->allocate(bytes);
        
        // Slow path: Block full, allocate new block
        if (!ptr) {
            std::cout << "⚠️  Memory Block Full! Allocating new 1GB block..." << std::endl;
            current_block_ = new MemoryBlock(1024 * 1024 * 1024);
            blocks_.push_back(current_block_);
            ptr = current_block_->allocate(bytes);
        }
        
        return ptr;
    }
    
    // Free memory (Lazy)
    void deallocate(void* ptr) {
        // [ARENA STRATEGY]
        // Individual deallocation is intentionally a NO-OP for O(1) performance.
        // We do not manage fragmentation or free-lists inside the arena.
        // 
        // Memory is reclaimed IN BULK at the end of the batch via reset_arena().
        // This avoids costly syscalls and fragmentation management during the hot loop.
        
        if (!ptr) return;
        
        // (Optional: In debug mode we could track usage or poison memory, 
        // but for production training we just ignore it to be as fast as possible).
    }
    
    // Reset all memory (End of batch/step)
    // This is the true "Free"
    void reset_arena() {
        // Reuse first block, delete others
        if (blocks_.size() > 1) {
             // Move extra blocks to cleanup queue for background thread!
             // This is where lazy thread comes in!
             {
                 std::lock_guard<std::mutex> lock(queue_mutex_);
                 for (size_t i = 1; i < blocks_.size(); ++i) {
                     block_cleanup_queue_.push(blocks_[i]);
                 }
             }
             queue_cv_.notify_one();
             
             // Keep only first block
             MemoryBlock* first = blocks_[0];
             blocks_.clear();
             blocks_.push_back(first);
             current_block_ = first;
        }
        
        if (current_block_) {
            current_block_->reset();
        }
    }
    
    // Lazy deallocator thread loop
    void worker_loop() {
        while (running_) {
            MemoryBlock* block = nullptr;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return !block_cleanup_queue_.empty() || !running_; });
                
                if (!running_ && block_cleanup_queue_.empty()) return;
                
                if (!block_cleanup_queue_.empty()) {
                    block = block_cleanup_queue_.front();
                    block_cleanup_queue_.pop();
                }
            }
            
            if (block) {
                // Heavy deallocation happens here in background!
                // Security/Safety: Zero fill before release as requested
                std::memset(block->data, 0, block->size);
                delete block;
            }
        }
    }
    
    // Destructor
    ~MemoryManager() {
        running_ = false;
        queue_cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        
        for (auto b : blocks_) delete b;
        blocks_.clear();
    }

private:
    MemoryManager() : running_(true), current_block_(nullptr) {
        worker_thread_ = std::thread(&MemoryManager::worker_loop, this);
    }
    
    std::vector<MemoryBlock*> blocks_;
    MemoryBlock* current_block_;
    
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<MemoryBlock*> block_cleanup_queue_; // Cleanup whole blocks
    
    std::atomic<bool> running_;
    std::thread worker_thread_;
};

} // namespace mm_rec
