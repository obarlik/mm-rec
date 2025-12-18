#include "mm_rec/data/tokenizer.h"
#include "mm_rec/utils/logger.h"
#include "mm_rec/utils/ui.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

using namespace mm_rec;
using namespace mm_rec::ui;

// Simple message structure
struct Message {
    std::string role;
    std::string content;
};

// Abstract Strategy
class DataParser {
public:
    virtual ~DataParser() = default;
    virtual std::vector<Message> parse(const std::string& line) = 0;
    
protected:
    // Helper to find value by key in a simple JSON string
    std::string extract_json_value(const std::string& line, const std::string& key) {
        std::string search_key = "\"" + key + "\"";
        size_t key_pos = line.find(search_key);
        if (key_pos == std::string::npos) return "";

        size_t val_start = line.find("\"", key_pos + search_key.length());
        if (val_start == std::string::npos) return "";
        val_start++; // skip opening quote

        // Simple parsing handling escaped quotes
        size_t val_end = val_start;
        while (val_end < line.size()) {
            if (line[val_end] == '"' && line[val_end - 1] != '\\') {
                break;
            }
            val_end++;
        }

        if (val_end >= line.size()) return ""; // Malformed or end of string
        return line.substr(val_start, val_end - val_start);
    }
};

// Strategy for Chat format ("messages": [{"role": "...", "content": "..."}])
class ChatParser : public DataParser {
public:
    std::vector<Message> parse(const std::string& line) override {
        std::vector<Message> msgs;
        size_t pos = 0;
        while (true) {
            // Find role
            size_t role_key = line.find("\"role\"", pos);
            if (role_key == std::string::npos) break;
            
            std::string role = extract_json_value(line.substr(role_key), "role");
            size_t role_val_end = line.find("\"", role_key + 6 + 1); // rough advance
            if (role_val_end != std::string::npos) {
                 // Try to find content after role
                 size_t content_key = line.find("\"content\"", role_val_end);
                 if (content_key == std::string::npos) break;
                 
                 std::string content = extract_json_value(line.substr(content_key), "content");
                 msgs.push_back({role, content});
                 
                 // Advance pos
                 size_t content_val_start = line.find("\"", content_key + 9) + 1;
                 size_t content_val_end = content_val_start + content.length(); 
                 pos = content_val_end;
            } else {
                break;
            }
        }
        return msgs;
    }
};

// Strategy for Alpaca format ("instruction": "...", "input": "...", "output": "...")
class AlpacaParser : public DataParser {
public:
    std::vector<Message> parse(const std::string& line) override {
        std::vector<Message> msgs;
        
        std::string instruction = extract_json_value(line, "instruction");
        std::string input = extract_json_value(line, "input");
        std::string output = extract_json_value(line, "output");

        if (instruction.empty() && output.empty()) return msgs; // Skip invalid lines

        // Construct User message
        std::string user_content = instruction;
        if (!input.empty()) {
            user_content += "\nInput:\n" + input;
        }
        msgs.push_back({"user", user_content});

        // Construct Assistant message
        msgs.push_back({"assistant", output});

        return msgs;
    }
};

#include "commands.h"

// ... (includes remain)

// ... (DataParser classes remain)

int cmd_prepare(int argc, char* argv[]) {
    if (argc < 3) {
        ui::error("Usage: mm_rec prepare <jsonl_input> <bin_output> [format: alpaca|chat]");
        return 1;
    }

    // Start Logger
    Logger::instance().start_writer("prepare.log", LogLevel::INFO);
    ui::print_header("Data Preparation Tool");
    
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    std::string format = (argc > 3) ? argv[3] : "alpaca"; // Default to alpaca for now
    
    std::unique_ptr<DataParser> parser;
    if (format == "chat") {
        parser = std::make_unique<ChatParser>();
    } else {
        parser = std::make_unique<AlpacaParser>();
    }

    ui::info("Processing " + in_path + " -> " + out_path + " (Format: " + format + ")");
    
    mm_rec::Tokenizer tokenizer;
    
    // Pass 1: Build Vocab (Char level)
    ui::info("Pass 1: Building Vocabulary...");
    {
        std::ifstream infile(in_path);
        std::string line;
        while (std::getline(infile, line)) {
            // Build vocab from raw line to capture all chars
            tokenizer.build_vocab(line);
        }
    }
    LOG_INFO("Vocab size: " + std::to_string(tokenizer.vocab_size()));
    
    // Pass 2: Encode
    ui::info("Pass 2: Encoding and Masking...");
    std::ifstream infile(in_path);
    std::vector<int32_t> all_tokens;
    std::vector<int32_t> all_masks;
    
    std::string line;
    int line_count = 0;
    while (std::getline(infile, line)) {
        auto msgs = parser->parse(line);
        if (msgs.empty()) continue;

        for (const auto& msg : msgs) {
            auto tokens = tokenizer.encode(msg.content);
            
            // Masking: Train only on Assistant (1), ignore User/System (0)
            int32_t mask_val = (msg.role == "assistant") ? 1 : 0;
            
            for (auto t : tokens) {
                all_tokens.push_back(t);
                all_masks.push_back(mask_val);
            }
        }
        
        // Add EOS separator
        all_tokens.push_back(3); // [EOS]
        all_masks.push_back(1);  // Train on EOS
        
        line_count++;
        if (line_count % 1000 == 0) std::cout << "Processed " << line_count << " lines..." << "\r" << std::flush;
    }
    ui::success("Total Tokens: " + std::to_string(all_tokens.size()));
    
    if (all_tokens.empty()) {
        ui::error("Error: No tokens produced! Check input file format.");
        return 1;
    }
    
    // Write Binary
    ui::info("Writing binary file...");
    std::ofstream outfile(out_path, std::ios::binary);
    
    int32_t magic = 0x4D4D5243;
    int32_t version = 2;
    int64_t total = all_tokens.size();
    
    outfile.write(reinterpret_cast<char*>(&magic), sizeof(int32_t));
    outfile.write(reinterpret_cast<char*>(&version), sizeof(int32_t));
    outfile.write(reinterpret_cast<char*>(&total), sizeof(int64_t));
    outfile.write(reinterpret_cast<char*>(all_tokens.data()), total * sizeof(int32_t));
    outfile.write(reinterpret_cast<char*>(all_masks.data()), total * sizeof(int32_t));
    
    ui::success("Successfully created " + out_path);
    
    Logger::instance().stop_writer();
    return 0;
}
