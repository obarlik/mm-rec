// Quick binary test
#include "mm_rec/utils/metrics.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace mm_rec;

int main() {
    // Test binary format with sampling
    MetricsSamplingConfig sampling;
    sampling.enabled = true;
    sampling.interval = 10;
    sampling.warmup_events = 50;
    
    MetricsManager::instance().start_writer("test_binary.bin", sampling);
    
    // Record 1000 events
    for (int i = 0; i < 1000; ++i) {
        METRIC_TRAINING_STEP(1.234f, 0.001f);
    }
    
    // Give writer thread time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    MetricsManager::instance().stop_writer();
    
    std::cout << "Binary test complete. Check test_binary.bin" << std::endl;
    
    // Use _Exit to avoid static destruction order issues with singleton
    _Exit(0);
}
