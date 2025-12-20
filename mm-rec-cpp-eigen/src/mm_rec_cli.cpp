#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/ui.h"
#include "cli/commands.h"
#include "mm_rec/utils/system_optimizer.h"
#include "mm_rec/utils/dashboard_manager.h" // [NEW]

using namespace mm_rec;
using namespace mm_rec::ui;

void print_usage(const char* prog_name) {
    ui::print_header("MM-Rec CLI Utility");
    
    std::cout << "Usage: " << prog_name << " <command> [args...]\n\n";
    
    ui::Table cmds({"Command", "Description"}, 20);
    cmds.add_row({"prepare", "Convert JSONL to binary training data"});
    cmds.add_row({"train", "Start adaptive curriculum training"});
    cmds.add_row({"infer", "Run inference with trained model"});
    cmds.add_row({"parse-metrics", "Convert binary metrics to readable format"});
    cmds.add_row({"view-trace", "Serve trace JSON for browser visualization"});
    cmds.finish();
}

int main(int argc, char* argv[]) {
    // 🔥 AUTO-TUNE: SystemOptimizer
    // By default, we use ALL cores for maximum throughput (1.5 GFLOPS).
    // If the user wants stability/efficiency (0.8 GFLOPS but low heat), they can enable P-Core pinning.
    // For now, we disable it by default based on benchmark results.
    // SystemOptimizer::optimize_runtime();

    // Start Global Dashboard (Production Mode)
    DashboardManager::instance().start(8085);
    
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

    // Dispatch commands directly for better argument handling
    if (command == "prepare") {
        return cmd_prepare_data(argc, argv);
    } else if (command == "train") {
        return cmd_train(argc, argv);
    } else if (command == "server") {
        return cmd_server(argc, argv);
    } else if (command == "infer") {
        return cmd_infer(argc, argv);
    }

    // For other commands, use the map-based dispatch (adjusting args)
    std::map<std::string, std::function<int(int, char**)>> commands;
    commands["parse-metrics"] = cmd_parse_metrics;
    commands["view-trace"] = cmd_view_trace;

    if (commands.find(command) == commands.end()) {
        ui::error("Unknown command: " + command);
        print_usage(argv[0]);
        return 1;
    }

    return commands[command](argc - 1, &argv[1]);
}
