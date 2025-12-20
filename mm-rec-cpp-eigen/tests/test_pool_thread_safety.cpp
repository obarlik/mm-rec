#include "mm_rec/core/pool_allocator.h"
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <cassert>

using namespace mm_rec;

// Test 1: Same-thread allocation/deallocation (baseline)
void test_same_thread() {
    std::cout << "[TEST 1] Same-thread alloc/dealloc..." << std::flush;
    
    std::vector<void*> ptrs;
    auto& allocator = PoolAllocator::instance();
    
    // Allocate 1000 objects
    for (int i = 0; i < 1000; i++) {
        void* ptr = allocator.allocate(64);
        assert(ptr != nullptr);
        ptrs.push_back(ptr);
    }
    
    // Deallocate all
    for (void* ptr : ptrs) {
        allocator.deallocate(ptr, 64);
    }
    
    std::cout << " PASS\n";
}

// Test 2: Cross-thread deallocation
void test_cross_thread() {
    std::cout << "[TEST 2] Cross-thread alloc/dealloc..." << std::flush;
    
    std::vector<void*> ptrs;
    
    // Thread A: Allocate
    std::thread alloc_thread([&ptrs]() {
        auto& allocator = PoolAllocator::instance();
        for (int i = 0; i < 500; i++) {
            void* ptr = allocator.allocate(128);
            assert(ptr != nullptr);
            ptrs.push_back(ptr);
        }
    });
    
    alloc_thread.join();
    
    // Thread B: Deallocate (different thread!)
    std::thread dealloc_thread([&ptrs]() {
        auto& allocator = PoolAllocator::instance();
        for (void* ptr : ptrs) {
            allocator.deallocate(ptr, 128); // Cross-thread!
        }
    });
    
    dealloc_thread.join();
    
    std::cout << " PASS\n";
}

// Test 3: Multi-threaded stress test
void test_multithreaded_stress() {
    std::cout << "[TEST 3] Multi-threaded stress (4 threads, 10K ops)..." << std::flush;
    
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    
    for (int t = 0; t < 4; t++) {
        threads.emplace_back([&errors, t]() {
            auto& allocator = PoolAllocator::instance();
            std::vector<void*> local_ptrs;
            
            // Each thread: allocate, deallocate, mix
            for (int i = 0; i < 2500; i++) {
                size_t size = (i % 5 + 1) * 64; // 64, 128, 256, 384, 512
                void* ptr = allocator.allocate(size);
                if (!ptr) {
                    errors.fetch_add(1);
                    continue;
                }
                local_ptrs.push_back(ptr);
                
                // Deallocate half immediately
                if (i % 2 == 0 && !local_ptrs.empty()) {
                    allocator.deallocate(local_ptrs.back(), size);
                    local_ptrs.pop_back();
                }
            }
            
            // Cleanup remaining
            for (void* ptr : local_ptrs) {
                allocator.deallocate(ptr, 256); // Arbitrary size, header has real size
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    if (errors.load() == 0) {
        std::cout << " PASS\n";
    } else {
        std::cout << " FAIL (" << errors.load() << " errors)\n";
    }
}

// Test 4: Cross-thread object passing (realistic scenario)
void test_object_passing() {
    std::cout << "[TEST 4] Cross-thread object passing (HTTP-like)..." << std::flush;
    
    std::vector<void*> shared_queue;
    std::mutex queue_mutex;
    
    // Producer thread
    std::thread producer([&]() {
        auto& allocator = PoolAllocator::instance();
        for (int i = 0; i < 100; i++) {
            void* obj = allocator.allocate(256);
            assert(obj != nullptr);
            
            std::lock_guard lock(queue_mutex);
            shared_queue.push_back(obj);
        }
    });
    
    producer.join();
    
    // Consumer threads (like HTTP worker threads)
    std::vector<std::thread> consumers;
    for (int c = 0; c < 3; c++) {
        consumers.emplace_back([&]() {
            auto& allocator = PoolAllocator::instance();
            while (true) {
                void* obj = nullptr;
                {
                    std::lock_guard lock(queue_mutex);
                    if (shared_queue.empty()) break;
                    obj = shared_queue.back();
                    shared_queue.pop_back();
                }
                
                // Process and deallocate (cross-thread!)
                allocator.deallocate(obj, 256);
            }
        });
    }
    
    for (auto& t : consumers) {
        t.join();
    }
    
    std::cout << " PASS\n";
}

// Test 5: Large object cross-thread
void test_large_objects() {
    std::cout << "[TEST 5] Large object cross-thread (>1KB)..." << std::flush;
    
    std::vector<void*> large_ptrs;
    
    std::thread alloc_thread([&]() {
        auto& allocator = PoolAllocator::instance();
        for (int i = 0; i < 50; i++) {
            void* ptr = allocator.allocate(2048 + i * 100); // Large objects
            assert(ptr != nullptr);
            large_ptrs.push_back(ptr);
        }
    });
    
    alloc_thread.join();
    
    std::thread dealloc_thread([&]() {
        auto& allocator = PoolAllocator::instance();
        for (void* ptr : large_ptrs) {
            allocator.deallocate(ptr, 0); // Size hint ignored for large
        }
    });
    
    dealloc_thread.join();
    
    std::cout << " PASS\n";
}

int main() {
    std::cout << "=== PoolAllocator Thread-Safety Test Suite ===\n\n";
    
    try {
        test_same_thread();
        test_cross_thread();
        test_multithreaded_stress();
        test_object_passing();
        test_large_objects();
        
        std::cout << "\n✓ All tests PASSED!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}
