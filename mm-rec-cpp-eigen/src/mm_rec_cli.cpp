#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include "cli/commands.h"
#include "mm_rec/utils/system_optimizer.h"

using namespace mm_rec;

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <command> [args...]" << std::endl;
    std::cerr << "Commands:" << std::endl;
    std::cerr << "  prepare        Convert JSONL to binary training data" << std::endl;
    std::cerr << "  train          Start adaptive curriculum training" << std::endl;
    std::cerr << "  infer          Run inference with trained model" << std::endl;
    std::cerr << "  parse-metrics  Convert binary metrics to readable format" << std::endl;
}

int main(int argc, char* argv[]) {
    // 🔥 AUTO-TUNE: SystemOptimizer
    // By default, we use ALL cores for maximum throughput (1.5 GFLOPS).
    // If the user wants stability/efficiency (0.8 GFLOPS but low heat), they can enable P-Core pinning.
    // For now, we disable it by default based on benchmark results.
    // SystemOptimizer::optimize_runtime();
    
    // Check for explicit flag (simple check)
    for(int i=1; i<argc; ++i) {
        if(std::string(argv[i]) == "--p-core-only") {
            SystemOptimizer::optimize_runtime();
            break;
        }
    }

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "prepare") {
        return cmd_prepare(argc - 1, &argv[1]);
    } else if (command == "train") {
        return cmd_train(argc - 1, &argv[1]);
    } else if (command == "infer") {
        return cmd_infer(argc - 1, &argv[1]);
    } else if (command == "parse-metrics") {
        return cmd_parse_metrics(argc - 1, &argv[1]);
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        print_usage(argv[0]);
        return 1;
    }
}
