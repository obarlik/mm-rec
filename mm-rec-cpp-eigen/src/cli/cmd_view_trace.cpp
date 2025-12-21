#include "commands.h"
#include "mm_rec/utils/ui.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/http_server.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <unistd.h>

using namespace mm_rec;
using namespace mm_rec::ui;

int cmd_view_trace(int argc, char* argv[]) {
    if (argc < 2) {
        ui::error("Usage: mm_rec view-trace <trace.json> [port]");
        return 1;
    }

    std::string trace_path = argv[1];
    int port = (argc >= 3) ? std::stoi(argv[2]) : 8080;

    // Check if file exists
    if (access(trace_path.c_str(), F_OK) == -1) {
        ui::error("Trace file not found: " + trace_path);
        return 1;
    }

    // Determine filename for URL
    std::string filename = trace_path.substr(trace_path.find_last_of("/\\") + 1);

    // Create Server
    net::HttpServer server(port);

    // Register Handler
    server.register_handler("/" + filename, [&](const net::Request&, std::shared_ptr<net::Response> res) {
        std::ifstream f(trace_path);
        if (f) {
            std::stringstream buffer;
            buffer << f.rdbuf();
            res->set_header("Content-Type", "application/json");
            res->set_header("Access-Control-Allow-Origin", "*"); // Ensure CORS for Perfetto
            res->send(buffer.str());
        } else {
            res->status(404);
            res->send("File not found");
        }
    });

    ui::print_header("Trace Viewer Server");

    if (!server.start()) {
        ui::error("Failed to start server on port " + std::to_string(port) + ". Port busy?");
        return 1;
    }
    
    std::string url = "http://localhost:" + std::to_string(port) + "/" + filename;
    std::string perfetto_link = "https://ui.perfetto.dev/#!/?url=" + url;
    
    ui::success("Serving: " + trace_path);
    std::cout << "\n✅  VIEW TRACE HERE:\n";
    std::cout << "   " << color::CYAN << color::BOLD << perfetto_link << color::RESET << "\n\n";
    
    ui::info("Press Ctrl+C to stop server...");

    // Keep main thread alive
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    return 0;
}
