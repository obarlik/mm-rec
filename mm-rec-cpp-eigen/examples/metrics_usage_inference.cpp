// Example: cmd_infer.cpp'de metrik toplama
#include "mm_rec/utils/metrics.h"
#include "mm_rec/infrastructure/i_metrics_exporter.h" // Added
#include "mm_rec/application/service_configurator.h" // For DI

int cmd_infer(int argc, char* argv[]) {
    // Load model...
    MMRecModel model(config);
    
    // 0. Initialize Services
    mm_rec::ServiceConfigurator::initialize();

    // Start metrics (optional for inference)
    // Needs #include "mm_rec/infrastructure/i_metrics_exporter.h"
    auto metrics_exporter = mm_rec::ServiceConfigurator::container().resolve<mm_rec::infrastructure::IMetricsExporter>();
    metrics_exporter->start("inference_metrics.jsonl");
    
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
    
    metrics_exporter->stop();
    return 0;
}
