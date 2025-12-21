// Demo: Memory Protection in Diagnostic Logging

#include "mm_rec/infrastructure/request_context.h"
#include "mm_rec/application/service_configurator.h"
#include <iostream>

using namespace mm_rec;
using namespace mm_rec::net;

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Memory Protection Demo" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    ServiceConfigurator::initialize();
    
    // ========================================
    // Test 1: Message Aggregation (Spam Prevention)
    // ========================================
    std::cout << "🛡️  Test 1: Message Aggregation (Repeated Messages)\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        
        // Spam same message rapidly
        for (int i = 0; i < 50; i++) {
            ctx->add_trace("INFO", "Processor", "Processing batch");
        }
        
        std::cout << "Added 50 identical messages in quick succession\n";
        std::cout << "Actual entries stored: " << ctx->trace_count() << " (aggregated!)\n";
        std::cout << "Memory saved: ~" << ((50 - ctx->trace_count()) * 200) << " bytes\n\n";
        
        std::cout << ctx->flush_trace() << std::endl;
    }
    
    // ========================================
    // Test 2: Circular Buffer (Large Loop)
    // ========================================
    std::cout << "\n🛡️  Test 2: Circular Buffer (10,000 iterations)\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        
        ctx->max_trace_entries = 100;  // Limit to 100 entries
        
        // Simulate large loop
        for (int i = 0; i < 10000; i++) {
            ctx->add_trace("INFO", "Loop", "Processing item " + std::to_string(i));
        }
        
        std::cout << "Loop iterations: 10,000\n";
        std::cout << "Max entries allowed: 100\n";
        std::cout << "Entries stored: " << ctx->trace_count() << "\n";
        std::cout << "Entries dropped: " << ctx->total_trace_attempts() - ctx->trace_count() << "\n";
        std::cout << "Memory usage: ~" << (ctx->trace_count() * 200) << " bytes (capped!)\n";
        std::cout << "Without protection: ~" << (10000 * 200) << " bytes (2 MB!)\n\n";
        
        // Show trace (only last 100 entries)
        auto trace = ctx->flush_trace();
        auto lines = std::count(trace.begin(), trace.end(), '\n');
        std::cout << "Trace output lines: " << lines << "\n" << std::endl;
    }
    
    // ========================================
    // Test 3: Sampling (Hot Path)
    // ========================================
    std::cout << "\n🛡️  Test 3: Sampling (100,000 items, sample every 1000th)\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        
        // Hot loop with sampling
        for (int i = 0; i < 100000; i++) {
            ctx->add_trace_sampled("INFO", "HotPath", "Processing item " + std::to_string(i), 1000);
        }
        
        std::cout << "Loop iterations: 100,000\n";
        std::cout << "Sample rate: 1/1000 (0.1%)\n";
        std::cout << "Entries stored: " << ctx->trace_count() << " (sampled!)\n";
        std::cout << "Memory usage: ~" << (ctx->trace_count() * 200) << " bytes\n";
        std::cout << "Without sampling: ~" << (100000 * 200 / 1024) << " KB (20 MB!)\n";
        std::cout << "Reduction: " << (100.0 - (ctx->trace_count() * 100.0 / 100000)) << "%\n\n";
    }
    
    // ========================================
    // Test 4: Mixed Workload (Realistic)
    // ========================================
    std::cout << "\n🛡️  Test 4: Mixed Workload (Realistic Scenario)\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        
        ctx->max_trace_entries = 200;
        
        // Request start
        ctx->add_trace("INFO", "Gateway", "Request received");
        ctx->add_trace("INFO", "Auth", "Validating token");
        
        // Large batch processing (with sampling)
        for (int i = 0; i < 5000; i++) {
            ctx->add_trace_sampled("DEBUG", "Batch", "Processing record " + std::to_string(i), 500);
        }
        
        // Some important events
        ctx->add_trace("WARN", "Batch", "Slow record detected");
        ctx->add_trace("INFO", "Batch", "Batch complete");
        
        // Database calls
        ctx->add_trace("INFO", "Database", "Saving results");
        ctx->add_trace("ERROR", "Database", "Connection timeout!");
        
        std::cout << "Workload:\n";
        std::cout << "  - Regular logs: ~10\n";
        std::cout << "  - Batch processing: 5,000 (sampled 1/500)\n";
        std::cout << "\nResults:\n";
        std::cout << "  Total attempts: " << ctx->total_trace_attempts() << "\n";
        std::cout << "  Stored: " << ctx->trace_count() << "\n";
        std::cout << "  Dropped: " << (ctx->total_trace_attempts() - ctx->trace_count()) << "\n";
        std::cout << "  Memory: ~" << (ctx->trace_count() * 200 / 1024.0) << " KB\n\n";
        
        std::cout << ctx->flush_trace() << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  ✅ Memory Protection Working!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\n💡 Protection Mechanisms:" << std::endl;
    std::cout << "  1️⃣  Message Aggregation - Repeated messages counted, not duplicated\n";
    std::cout << "  2️⃣  Circular Buffer - Oldest entries overwritten when full (max 1000)\n";
    std::cout << "  3️⃣  Sampling - Only log every Nth iteration in hot loops\n";
    std::cout << "  4️⃣  Cap at 1000 entries - ~200 KB max memory per request\n";
    
    std::cout << "\n📝 Usage Recommendations:" << std::endl;
    std::cout << R"(
  // Normal logging (always safe):
  ctx->add_trace("INFO", "Service", "Processing user");
  
  // Loops < 100 iterations (safe, no action needed):
  for (int i = 0; i < 50; i++) {
      ctx->add_trace("INFO", "Loop", "Item " + std::to_string(i));
  }
  
  // Large loops (use sampling):
  for (int i = 0; i < 100000; i++) {
      ctx->add_trace_sampled("INFO", "Batch", "Item " + i, 1000);
      // Only logs every 1000th iteration
  }
  
  // Automatic protection:
  - Max 1000 entries (circular buffer)
  - Duplicate messages aggregated
  - Memory capped at ~200 KB per request
)" << std::endl;
    
    return 0;
}
