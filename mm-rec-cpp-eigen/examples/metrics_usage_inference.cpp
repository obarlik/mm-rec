// Example: cmd_infer.cpp'de metrik toplama

#include "mm_rec/utils/metrics.h"

int cmd_infer(int argc, char* argv[]) {
    // Load model...
    MMRecModel model(config);
    
    // Start metrics (optional for inference)
    MetricsManager::instance().start_writer("inference_metrics.jsonl");
    
    while (true) {  // Request loop
        auto start = std::chrono::high_resolution_clock::now();
        
        // Generate response
        auto output = model.generate(input_ids, max_length);
        
        auto end = std::chrono::high_resolution_clock::now();
        float latency_ms = std::chrono::duration<float, std::milli>(end - start).count();
        
        // 👇 Log inference metrics
        METRIC_INFERENCE(latency_ms, output.size());
        
        // Detailed per-token metrics (if needed)
        METRIC_RECORD(CUSTOM, output.size(), latency_ms / output.size(), 0, "tok/ms");
    }
    
    MetricsManager::instance().stop_writer();
    return 0;
}
