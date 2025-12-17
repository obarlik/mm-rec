#include "mm_rec/core/pool_allocator.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace mm_rec;

struct SmallObject {
    float x, y, z;
    int id;
    char padding[12]; // Total ~32 bytes
};

void benchmark_standard(int iterations) {
    std::cout << "[Standard] Allocating " << iterations << " objects..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<SmallObject*> objects;
    objects.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        objects.push_back(new SmallObject());
    }
    
    for (auto ptr : objects) {
        delete ptr;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "[Standard] Time: " << diff.count() << " seconds." << std::endl;
}

void benchmark_pool(int iterations) {
    std::cout << "[Pool] Allocating " << iterations << " objects..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<SmallObject*> objects;
    objects.reserve(iterations);
    
    size_t obj_size = sizeof(SmallObject);
    
    for (int i = 0; i < iterations; ++i) {
        void* ptr = PoolAllocator::instance().allocate(obj_size);
        objects.push_back(new(ptr) SmallObject()); // Placement new
    }
    
    for (auto ptr : objects) {
        // Manually destroy and free
        ptr->~SmallObject();
        PoolAllocator::instance().deallocate(ptr, obj_size);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "[Pool] Time: " << diff.count() << " seconds." << std::endl;
    
    std::cout << "[Pool] Speedup: " << (1.0 / diff.count()) << " ops/sec" << std::endl;
}

void benchmark_threaded_pool(int iterations, int threads) {
    std::cout << "[Pool Multi-Threaded] " << threads << " Threads, " << iterations << " ops/thread..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(threads)
    {
        std::vector<SmallObject*> objects;
        objects.reserve(iterations);
        size_t obj_size = sizeof(SmallObject);
        
        for(int i=0; i<iterations; ++i) {
            void* ptr = PoolAllocator::instance().allocate(obj_size);
            objects.push_back(new(ptr) SmallObject());
        }
        for(auto ptr : objects) {
             ptr->~SmallObject();
             PoolAllocator::instance().deallocate(ptr, obj_size);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "[Pool Multi-Threaded] Time: " << diff.count() << " seconds." << std::endl;
    std::cout << "[Pool Multi-Threaded] Total Ops: " << (long)iterations * threads << std::endl;
}

void benchmark_threaded_std(int iterations, int threads) {
    std::cout << "[Standard Multi-Threaded] " << threads << " Threads, " << iterations << " ops/thread..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(threads)
    {
        std::vector<SmallObject*> objects;
        objects.reserve(iterations);
        
        for(int i=0; i<iterations; ++i) {
            objects.push_back(new SmallObject());
        }
        for(auto ptr : objects) {
            delete ptr;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "[Standard Multi-Threaded] Time: " << diff.count() << " seconds." << std::endl;
}

int main() {
    std::cout << "=== Pool Allocator Benchmark ===" << std::endl;
    std::cout << "Object Size: " << sizeof(SmallObject) << " bytes" << std::endl;
    
    int N = 5000000;
    
    benchmark_standard(N);
    benchmark_pool(N);
    
    std::cout << "\n=== Multi-Threaded Benchmark (4 Threads) ===" << std::endl;
    benchmark_threaded_std(N, 4);
    benchmark_threaded_pool(N, 4);
    
    return 0;
}
