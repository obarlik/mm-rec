// Example: cmd_train.cpp'ye eklenecek metrik kayıtları

#include "mm_rec/business/metrics.h"
#include "mm_rec/infrastructure/i_metrics_exporter.h"
#include "mm_rec/application/service_configurator.h" // For DI

int cmd_train(int argc, char* argv[]) {
    // ... existing code ...
    
    // 0. Initialize Services
    mm_rec::ServiceConfigurator::initialize();

    // 1. Setup Exporter (Background Thread) - Resolved via DI
    auto exporter = mm_rec::ServiceConfigurator::container().resolve<mm_rec::infrastructure::IMetricsExporter>();
    exporter->start("example_metrics.bin");
    
    for (int iteration = start_epoch; iteration <= max_iterations; ++iteration) {
        // ... batch loop ...
        
        for (int64_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // Record training step
            float loss = trainer.train_step(final_batch);
            
            // 👇 ZERO-OVERHEAD METRIC (just a push to ring buffer)
            METRIC_TRAINING_STEP(loss, trainer.get_current_lr());
            
            // Record flux-specific metrics
            if (train_config.optimizer_type == "flux") {
                auto* flux = dynamic_cast<Flux*>(trainer.get_optimizer());
                if (flux && batch_idx % 100 == 0) {  // Sample every 100 steps
                    METRIC_FLUX_BRAKE(flux->get_brake_stats());
                }
            }
            
            // Memory tracking (very cheap)
            if (batch_idx % 50 == 0) {
                size_t mem = MemoryManager::get_global_memory_usage();
                METRIC_RECORD(MEMORY_USAGE, mem / 1024.0f / 1024.0f, 0, 0, "");
            }
        }
    }
    
    // Stop writer (flushes remaining events)
    // Stop writer (flushes remaining events)
    exporter->stop();
    
    return 0;
}
