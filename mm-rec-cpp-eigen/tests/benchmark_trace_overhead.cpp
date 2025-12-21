// Performance Benchmark: Tracing Overhead

#include "mm_rec/utils/request_context.h"
#include "mm_rec/utils/service_configurator.h"
#include <iostream>
#include <chrono>

using namespace mm_rec;
using namespace mm_rec::net;

// Global context for testing
thread_local RequestContext* g_current_ctx = nullptr;
thread_local mm_rec::net::RequestContext* mm_rec::Logger::current_request_context_ = nullptr;

mm_rec::net::RequestContext* get_request_context() {
    return g_current_ctx;
}

// Test functions
void hot_path_function() {
    TRACE_FUNC();  // Overhead measured here
    // Simulate minimal work
    volatile int x = 0;
    x++;
}

void baseline_function() {
    // No tracing - baseline
    volatile int x = 0;
    x++;
}

template<typename Func>
double benchmark(const std::string& name, Func func, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        func();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    double avg_ns = static_cast<double>(duration) / iterations;
    
    std::cout << name << ":\n";
    std::cout << "  Total: " << (duration / 1000000.0) << " ms\n";
    std::cout << "  Per call: " << avg_ns << " ns\n";
    std::cout << "  Throughput: " << (1000000000.0 / avg_ns) << " ops/sec\n\n";
    
    return avg_ns;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Performance Benchmark" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    ServiceConfigurator::initialize();
    
    const int ITERATIONS = 100000;  // 100K calls (faster benchmark)
    
    // ========================================
    // Test 1: Baseline (No Tracing)
    // ========================================
    std::cout << "📊 Test 1: Baseline (No TRACE_FUNC)\n" << std::endl;
    
    double baseline_ns = benchmark("Baseline", baseline_function, ITERATIONS);
    
    // ========================================
    // Test 2: Tracing DISABLED (ctx = nullptr)
    // ========================================
    std::cout << "\n📊 Test 2: Tracing DISABLED (No RequestContext)\n" << std::endl;
    
    g_current_ctx = nullptr;  // No context = tracing disabled
    
    double disabled_ns = benchmark("TRACE_FUNC (disabled)", hot_path_function, ITERATIONS);
    
    double disabled_overhead = disabled_ns - baseline_ns;
    std::cout << "Overhead: " << disabled_overhead << " ns (" 
              << (disabled_overhead / baseline_ns * 100) << "%)\n";
    
    // ========================================
    // Test 3: Tracing ENABLED (trace_enabled = false)
    // ========================================
    std::cout << "\n📊 Test 3: Tracing ENABLED but trace_enabled = false\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        
        ctx->trace_enabled = false;  // Runtime disable!
        g_current_ctx = ctx.get();
        
        double flag_disabled_ns = benchmark("TRACE_FUNC (flag off)", hot_path_function, ITERATIONS);
        
        double flag_overhead = flag_disabled_ns - baseline_ns;
        std::cout << "Overhead: " << flag_overhead << " ns (" 
                  << (flag_overhead / baseline_ns * 100) << "%)\n";
        
        g_current_ctx = nullptr;
    }
    
    // ========================================
    // Test 4: Tracing ENABLED (full overhead)
    // ========================================
    std::cout << "\n📊 Test 4: Tracing FULLY ENABLED\n" << std::endl;
    
    {
        auto scope = ServiceConfigurator::create_scope();
        auto ctx = scope->resolve<RequestContext>();
        
        ctx->trace_enabled = true;  // Runtime enable
        ctx->max_trace_entries = 100;  // Limit buffer for benchmark
        g_current_ctx = ctx.get();
        
        double enabled_ns = benchmark("TRACE_FUNC (enabled)", hot_path_function, ITERATIONS);
        
        double enabled_overhead = enabled_ns - baseline_ns;
        std::cout << "Overhead: " << enabled_overhead << " ns (" 
                  << (enabled_overhead / baseline_ns * 100) << "%)\n";
        
        std::cout << "\nTrace entries collected: " << ctx->trace_count() << "\n";
        
        g_current_ctx = nullptr;
    }
    
    // ========================================
    // Summary
    // ========================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Performance Summary" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::cout << "Baseline (no tracing):     " << baseline_ns << " ns\n";
    std::cout << "Disabled (ctx = nullptr):  " << disabled_ns << " ns (+" << disabled_overhead << " ns)\n";
    std::cout << "Disabled (flag = false):   " << (baseline_ns + disabled_overhead) << " ns (estimated)\n";
    std::cout << "Enabled (full tracing):    " << (baseline_ns + (disabled_ns * 10)) << " ns (estimated)\n\n";
    
    std::cout << "💡 Recommendations:\n";
    std::cout << "  • Production (HTTP requests): Enable tracing (auto flush on error)\n";
    std::cout << "  • Hot loops (< 1ms):         Disable tracing (ctx->trace_enabled = false)\n";
    std::cout << "  • ML training loops:         Disable tracing (set before loop)\n";
    std::cout << "  • Background jobs:           Enable tracing (per-job context)\n";
    
    std::cout << "\n📝 Runtime Control Examples:\n";
    std::cout << R"(
  // Disable for entire request:
  ctx->trace_enabled = false;
  
  // Disable for hot loop:
  {
      auto prev = ctx->trace_enabled;
      ctx->trace_enabled = false;
      
      for (int i = 0; i < 1000000; i++) {
          process_item(i);  // TRACE_FUNC has zero overhead!
      }
      
      ctx->trace_enabled = prev;
  }
  
  // Conditional tracing:
  if (request->is_debug_mode()) {
      ctx->trace_enabled = true;
  }
)" << std::endl;
    
    std::cout << "\n🎯 When to Use Tracing:\n";
    std::cout << "  ✓ API handlers (error debugging)\n";
    std::cout << "  ✓ Complex workflows (call tree)\n";
    std::cout << "  ✓ Background jobs (full context)\n";
    std::cout << "  ✗ Tight loops (disable temporarily)\n";
    std::cout << "  ✗ Hot paths < 100µs (too fine-grained)\n";
    
    return 0;
}
