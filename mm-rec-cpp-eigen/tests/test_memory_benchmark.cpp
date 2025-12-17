#include "mm_rec/core/memory_manager.h"
#include <iostream>
#include <vector>
#include <thread>
#include <omp.h>
#include <cassert>
#include <chrono>
#include <cstring>

using namespace mm_rec;

void test_basic_allocation() {
    std::cout << "[Test] Basic Allocation... ";
    MemoryManager::instance().reset_arena();
    void* ptr = MemoryManager::instance().allocate(1024);
    assert(ptr != nullptr);
    std::cout << "OK." << std::endl;
}

void test_stress_reuse() {
    std::cout << "[Test] Memory Reuse Logic... ";
    
    MemoryManager::instance().reset_arena();
    size_t initial_bytes = MemoryManager::instance().get_total_memory();
    (void)initial_bytes; // Silence unused warning
    
    // 1. Allocate a lot
    for(int i=0; i<1000; ++i) MemoryManager::instance().allocate(1024);
    
    size_t peak_bytes = MemoryManager::instance().get_total_memory();
    assert(peak_bytes >= 1000*1024);
    
    // 2. Reset
    MemoryManager::instance().reset_arena();
    
    // 3. Alloc again (should REUSE, not grow)
    for(int i=0; i<1000; ++i) MemoryManager::instance().allocate(1024);
    
    size_t final_bytes = MemoryManager::instance().get_total_memory();
    
    if (final_bytes > peak_bytes * 1.5) {
        std::cerr << "FAILED: Memory grew significantly after reset! Leak?" << std::endl;
        std::cerr << "Peak: " << peak_bytes << " Final: " << final_bytes << std::endl;
        exit(1);
    }
    
    size_t free_blocks = MemoryManager::instance().get_free_block_count();
    (void)free_blocks; // Silence unused warning
    
    std::cout << "OK (High-water mark stable)." << std::endl;
}

void test_dirty_check() {
    std::cout << "[Test] Zero-Init Verification... ";
    
    MemoryManager::instance().reset_arena();
    
    // 1. Alloc and write 0xFF
    size_t size = 1024;
    uint8_t* ptr1 = (uint8_t*)MemoryManager::instance().allocate(size);
    memset(ptr1, 0xFF, size);
    
    // 2. Reset
    MemoryManager::instance().reset_arena();
    
    // 3. Alloc again (reuse same block)
    uint8_t* ptr2 = (uint8_t*)MemoryManager::instance().allocate(size);
    
    // 4. Check content
    if (ptr1 == ptr2 && ptr2[0] == 0xFF) {
        std::cout << "DIRTY (Performance Optimization)." << std::endl;
        std::cout << "      > Note: Memory remains dirty on reuse to avoid memset overhead." << std::endl;
    } else if (ptr2[0] == 0) {
        std::cout << "CLEAN (Zero-Initialized)." << std::endl;
    } else {
        std::cout << "UNKNOWN (mixed state)." << std::endl;
    }
}

void test_omp_concurrency() {
    std::cout << "[Test] OpenMP Concurrency (8 threads)... " << std::endl;
    
    MemoryManager::instance().reset_arena();
    
    std::atomic<size_t> total_allocs{0};

    #pragma omp parallel num_threads(8)
    {
        int tid = omp_get_thread_num();
        (void)tid; // Silence unused warning
        
        for (int i=0; i<10000; ++i) {
            void* p = MemoryManager::instance().allocate(1024);
            if (p) {
                memset(p, 1, 1024);
                total_allocs++;
            }
        }
        MemoryManager::instance().reset_arena();
    }
    
    std::cout << "  Allocated " << total_allocs << " blocks total." << std::endl;
    std::cout << "  OK (No crash)." << std::endl;
}

void test_stress_throughput() {
    std::cout << "[Test] Throughput Benchmark... ";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t iterations = 1000000; // 1M allocations
    
    #pragma omp parallel num_threads(8)
    {
        for(size_t i=0; i<iterations/8; ++i) {
             void* p = MemoryManager::instance().allocate(128);
             (void)p;
             if (i % 1000 == 0) MemoryManager::instance().reset_arena();
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << iterations << " allocs in " << elapsed.count() << "s (" 
              << (iterations / elapsed.count()) / 1e6 << " M ops/sec)" << std::endl;
}

int main() {
    std::cout << "=== Memory Manager Benchmark ===" << std::endl;
    #ifdef _OPENMP
        std::cout << "OpenMP Enabled. Max threads: " << omp_get_max_threads() << std::endl;
    #else
        std::cout << "WARNING: OpenMP NOT Enabled!" << std::endl;
    #endif

    test_basic_allocation();
    test_stress_reuse();
    test_dirty_check();
    test_omp_concurrency();
    test_stress_throughput();
    return 0;
}
