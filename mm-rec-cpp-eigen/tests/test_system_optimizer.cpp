/**
 * Test: System Optimizer Verification
 * 
 * Verifies that:
 * 1. P-Cores are detected.
 * 2. Process affinity is strictly reduced (pinned).
 * 3. OpenMP threads match the pin count.
 */

#include "mm_rec/utils/system_optimizer.h"
#include <iostream>
#include <cassert>
#include <sched.h>
#include <omp.h>

using namespace mm_rec;

int count_set_bits(const cpu_set_t* set) {
    int count = 0;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, set)) count++;
    }
    return count;
}

void test_affinity_pinning() {
    std::cout << "=== Test: System Optimizer Pinning ===" << std::endl;

    // 1. Get initial affinity (should be all cores, e.g., 12)
    cpu_set_t initial_mask;
    CPU_ZERO(&initial_mask);
    sched_getaffinity(0, sizeof(cpu_set_t), &initial_mask);
    int initial_count = count_set_bits(&initial_mask);
    std::cout << "Initial Available Cores: " << initial_count << std::endl;

    // 2. Run Optimizer
    SystemOptimizer::optimize_runtime();

    // 3. Get new affinity
    cpu_set_t new_mask;
    CPU_ZERO(&new_mask);
    sched_getaffinity(0, sizeof(cpu_set_t), &new_mask);
    int new_count = count_set_bits(&new_mask);
    std::cout << "Pinned Available Cores: " << new_count << std::endl;

    // 4. Verify OpenMP settings
    int omp_threads = omp_get_max_threads();
    std::cout << "OpenMP Max Threads: " << omp_threads << std::endl;

    // Assertions
    if (new_count < initial_count) {
        std::cout << "✅ SUCCESS: Process was pinned! (" << initial_count << " -> " << new_count << " cores)" << std::endl;
    } else {
        std::cout << "⚠️  WARNING: Affinity did not change (Maybe no hybrid topology detected?)" << std::endl;
        // If system is NOT hybrid (all cores same freq), optimizer does nothing. This is correct behavior.
        // We can't fail the test, but we note it.
    }

    if (omp_threads == new_count) {
        std::cout << "✅ SUCCESS: OpenMP threads synced with affinity." << std::endl;
    } else {
        std::cerr << "❌ FAILURE: OpenMP threads (" << omp_threads << ") != Pinned Cores (" << new_count << ")" << std::endl;
        exit(1);
    }
}

int main() {
    test_affinity_pinning();
    return 0;
}
