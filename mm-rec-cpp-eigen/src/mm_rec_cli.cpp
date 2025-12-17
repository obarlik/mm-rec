#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include "cli/commands.h"

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <command> [args...]" << std::endl;
    std::cerr << "Commands:" << std::endl;
    std::cerr << "  prepare    Convert JSONL to binary training data" << std::endl;
    std::cerr << "  train      Start adaptive curriculum training" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "prepare") {
        // Shift args so argv[0] is "prepare" and argv[1] is the first arg
        // But cmd_prepare expects traditional argc/argv where argv[0] is prog name
        // We will pass &argv[1] as the new argv array, effectively making "prepare" the new argv[0]
        return cmd_prepare(argc - 1, &argv[1]);
    } else if (command == "train") {
        return cmd_train(argc - 1, &argv[1]);
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        print_usage(argv[0]);
        return 1;
    }
}
