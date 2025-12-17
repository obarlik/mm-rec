/**
 * Custom Memory Manager for MM-Rec
 * 
 * Features:
 * 1. Thread-Local Arenas (Lock-free allocation)
 * 2. Background Cleaning (Zero-overhead memset)
 * 3. Global Block Pooling (Efficient reuse)
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

#ifdef _WIN32
#include <windows.h>
#else
#ifdef __linux__
#include <sys/mman.h>
#endif
#endif

namespace mm_rec {

// A large chunk of memory (Arena Block)
struct MemoryBlock {
    size_t size;
    size_t used;
    uint8_t* data;
    bool is_clean; // Debug flag
    
    MemoryBlock(size_t s) : size(s), used(0), is_clean(false) {
        #ifdef _WIN32
        // Try Large Pages first (Requires SeLockMemoryPrivilege)
        data = (uint8_t*)VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (!data) {
            // Fallback to standard pages if privilege missing or fragmentation high
            data = (uint8_t*)VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        }
        if (!data) throw std::bad_alloc();
        #else
        data = new uint8_t[size];
        #ifdef __linux__
        // Request Transparent Huge Pages (THP) for this large block
        madvise(data, size, MADV_HUGEPAGE);
        #endif
        #endif
    }
    
    ~MemoryBlock() {
        #ifdef _WIN32
        if (data) VirtualFree(data, 0, MEM_RELEASE);
        #else
        delete[] data;
        #endif
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
        // Note: We do NOT memset here. The background thread does it.
    }
    
    // Explicit clean method called by background thread
    void clean() {
        std::memset(data, 0, size);
        used = 0;
        is_clean = true;
    }
};

/**
 * Global Pool that manages all blocks and runs the background cleaner.
 * Thread-Safe Singleton.
 */
class GlobalBlockPool {
public:
    static GlobalBlockPool& instance() {
        static GlobalBlockPool instance;
        return instance;
    }

    MemoryBlock* acquire_clean_block(size_t size) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Try to get a clean block
        if (!clean_queue_.empty()) {
            MemoryBlock* block = clean_queue_.front();
            clean_queue_.pop();
            
            // Validate size (simplified: we assume fixed block size)
            if (block->size >= size) {
                return block;
            } else {
                delete block; // Wrong size
            }
        }
        
        // If no clean block, check dirty queue (emergency fallback logic could go here, but we prefer new)
        // For now, allocate new from OS if empty (to avoid waiting)
        return new MemoryBlock(size);
    }

    void release_dirty_block(MemoryBlock* block) {
        if (!block) return;
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            block->is_clean = false; // Mark as dirty
            dirty_queue_.push(block);
            atomic_dirty_count_.fetch_add(1, std::memory_order_release);
        }
        
        // Wake up cleaner
        cv_cleaner_.notify_one();
    }
    
    // Stats for debugging
    size_t get_clean_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return clean_queue_.size();
    }
    
    size_t get_dirty_count() {
        std::lock_guard<std::mutex> lock(mutex_);
        return dirty_queue_.size();
    }

    ~GlobalBlockPool() {
        stop_ = true;
        cv_cleaner_.notify_all();
        if (cleaner_thread_.joinable()) {
            cleaner_thread_.join();
        }
        
        // Cleanup leftover blocks
        while(!clean_queue_.empty()) { delete clean_queue_.front(); clean_queue_.pop(); }
        while(!dirty_queue_.empty()) { delete dirty_queue_.front(); dirty_queue_.pop(); }
    }

private:
    GlobalBlockPool() : stop_(false), atomic_dirty_count_(0) {
        // Start background cleaner
        cleaner_thread_ = std::thread(&GlobalBlockPool::cleaner_loop, this);
    }
    
    void cleaner_loop() {
        while (true) {
            MemoryBlock* block_to_clean = nullptr;
            
            // 1. Spin-Wait (Hybrid Polling)
            // Objective: Avoid sleep latency for bursts of work.
            // Check atomic variable (lock-free) for a short period.
            int spin_count = 0;
            const int MAX_SPIN = 10000; // ~10-50us depending on CPU
            
            while(atomic_dirty_count_.load(std::memory_order_acquire) == 0 && !stop_ && spin_count < MAX_SPIN) {
                std::this_thread::yield(); // Notify OS we are waiting, but stay scheduled if possible
                spin_count++;
            }
            
            // 2. Lock & Consume (or Sleep if failed)
            {
                std::unique_lock<std::mutex> lock(mutex_);
                
                // If queue is empty, we SLEEP (Wait for event)
                cv_cleaner_.wait(lock, [this]{ return !dirty_queue_.empty() || stop_; });
                
                if (stop_ && dirty_queue_.empty()) return;
                
                if (!dirty_queue_.empty()) {
                    block_to_clean = dirty_queue_.front();
                    dirty_queue_.pop();
                    atomic_dirty_count_.fetch_sub(1, std::memory_order_release);
                }
            }
            
            if (block_to_clean) {
                // Heavy lifting done without holding global lock!
                block_to_clean->clean();
                
                // Return to clean queue
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    clean_queue_.push(block_to_clean);
                }
            }
        }
    }

    std::mutex mutex_;
    std::condition_variable cv_cleaner_;
    std::queue<MemoryBlock*> clean_queue_;
    std::queue<MemoryBlock*> dirty_queue_;
    std::thread cleaner_thread_;
    std::atomic<bool> stop_;
    std::atomic<size_t> atomic_dirty_count_; // For lock-free spin check
};


class MemoryManager {
public:
    static const size_t BLOCK_SIZE = 64 * 1024 * 1024; // 64 MB
    
    static MemoryManager& instance() {
        static thread_local MemoryManager instance;
        return instance;
    }
    
    void* allocate(size_t bytes) {
        // 1. Try current block
        if (current_block_) {
            void* ptr = current_block_->allocate(bytes);
            if (ptr) return ptr;
        }
        
        // 2. Current block full or null. Get new one from Global Pool.
        // (No persistent local pool anymore - we use global pool)
        MemoryBlock* new_block = GlobalBlockPool::instance().acquire_clean_block(BLOCK_SIZE);
        blocks_.push_back(new_block);
        current_block_ = new_block;
        
        // 3. Alloc from new block (Should succeed since 64MB > requested usually)
        return current_block_->allocate(bytes);
    }
    
    void deallocate(void* ptr) {
        // No-op
    }
    
    void mark_persistent() {
        if (blocks_.empty()) {
            persistent_block_count_ = 0;
            return;
        }
        persistent_block_count_ = blocks_.size();
        persistent_used_ = blocks_.back()->used;
    }

    void reset_arena() {
        if (blocks_.empty()) return;

        // Release transient blocks to GLOBAL DIRTY POOL
        // Identify blocks to free
        size_t keep_count = (persistent_block_count_ > 0) ? persistent_block_count_ : 0;
        
        // Special case: If we keep current block (persistent), we just reset 'used'
        // But if we release it, we must null current_block_
        
        for (size_t i = keep_count; i < blocks_.size(); ++i) {
            GlobalBlockPool::instance().release_dirty_block(blocks_[i]);
        }
        
        blocks_.resize(keep_count);
        
        if (keep_count > 0) {
            // We kept some blocks. The last one is the active one.
            current_block_ = blocks_.back();
            current_block_->used = persistent_used_;
        } else {
            // We released everything.
            current_block_ = nullptr;
        }
    }
    
    size_t get_total_memory() const {
        size_t total = 0;
        for (const auto* b : blocks_) total += b->size;
        return total;
    }
    
    // Debug helpers
    size_t get_live_block_count() const { return blocks_.size(); }
    size_t get_free_block_count() const { return GlobalBlockPool::instance().get_clean_count(); } // Approximation

    ~MemoryManager() {
        // Thread exiting. Release all blocks to pool.
        for (auto b : blocks_) {
            GlobalBlockPool::instance().release_dirty_block(b);
        }
    }

private:
    MemoryManager() : current_block_(nullptr), persistent_block_count_(0), persistent_used_(0) {}
    
    std::vector<MemoryBlock*> blocks_;
    MemoryBlock* current_block_;
    
    size_t persistent_block_count_;
    size_t persistent_used_;
};

} // namespace mm_rec
