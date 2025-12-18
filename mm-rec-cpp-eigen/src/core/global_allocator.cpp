#include "mm_rec/core/pool_allocator.h"
#include <new>
#include <cstdlib>
#include <iostream>

/**
 * @brief Global Memory Overrides
 * This file redirects all "new" and "delete" calls in the entire program
 * to our high-performance Thread-Local PoolAllocator.
 */

void* operator new(size_t size) {
    // Route to our Pool Allocator
    // If PoolAllocator is not ready (mostly static init), fallback to malloc
    try {
        void* ptr = mm_rec::PoolAllocator::instance().allocate(size);
        if (!ptr) throw std::bad_alloc();
        return ptr;
    } catch (...) {
        // Fallback for extreme edge cases during static init
        void* ptr = std::malloc(size);
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }
}

void operator delete(void* ptr) noexcept {
    // Now we can handle unsized delete because we have Headers!
    mm_rec::PoolAllocator::instance().deallocate(ptr);
}

void operator delete(void* ptr, size_t /*size*/) noexcept {
    // We can ignore size hint and trust our header
    mm_rec::PoolAllocator::instance().deallocate(ptr);
}

// Array overrides
void* operator new[](size_t size) {
    return operator new(size);
}

void operator delete[](void* ptr) noexcept {
    mm_rec::PoolAllocator::instance().deallocate(ptr);
}

void operator delete[](void* ptr, size_t /*size*/) noexcept {
    mm_rec::PoolAllocator::instance().deallocate(ptr);
}
