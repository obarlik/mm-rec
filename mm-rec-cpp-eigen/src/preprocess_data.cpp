#include "mm_rec/data/tokenizer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Simple JSON message parser
struct Message {
    std::string role;
    std::string content;
};

// Naive JSONL parser tailored for "messages": [...] structure
std::vector<Message> parse_jsonl_line(const std::string& line) {
    std::vector<Message> msgs;
    size_t pos = 0;
    while (true) {
        // Find role
        size_t role_key = line.find("\"role\"", pos);
        if (role_key == std::string::npos) break;
        
        size_t role_val_start = line.find("\"", role_key + 6); // skip "role"
        if (role_val_start == std::string::npos) break;
        role_val_start++; // skip "
        
        size_t role_val_end = line.find("\"", role_val_start);
        std::string role = line.substr(role_val_start, role_val_end - role_val_start);
        
        // Find content
        size_t content_key = line.find("\"content\"", role_val_end);
        if (content_key == std::string::npos) break;
        
        size_t content_val_start = line.find("\"", content_key + 9);
        if (content_val_start == std::string::npos) break;
        content_val_start++;
        
        // Naive content end finder (handles basic escaped quotes)
        size_t content_val_end = content_val_start;
        while (content_val_end < line.size()) {
            if (line[content_val_end] == '"' && line[content_val_end-1] != '\\') {
                break;
            }
            content_val_end++;
        }
        
        std::string content = line.substr(content_val_start, content_val_end - content_val_start);
        msgs.push_back({role, content});
        
        pos = content_val_end;
    }
    return msgs;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <jsonl_input> <bin_output>" << std::endl;
        return 1;
    }
    
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    
    std::cout << "Processing " << in_path << " -> " << out_path << std::endl;
    
    mm_rec::Tokenizer tokenizer;
    
    // Pass 1: Build Vocab (Char level)
    std::cout << "Pass 1: Building Vocabulary..." << std::endl;
    {
        std::ifstream infile(in_path);
        std::string line;
        while (std::getline(infile, line)) {
            tokenizer.build_vocab(line);
        }
    }
    std::cout << "Vocab size: " << tokenizer.vocab_size() << std::endl;
    
    // Pass 2: Encode
    std::cout << "Pass 2: Encoding and Masking..." << std::endl;
    std::ifstream infile(in_path);
    std::vector<int32_t> all_tokens;
    std::vector<int32_t> all_masks;
    
    std::string line;
    int line_count = 0;
    while (std::getline(infile, line)) {
        auto msgs = parse_jsonl_line(line);
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
    std::cout << "\nTotal Tokens: " << all_tokens.size() << std::endl;
    
    if (all_tokens.empty()) {
        std::cerr << "Error: No tokens produced! Check input file format." << std::endl;
        return 1;
    }
    
    // Write Binary
    std::cout << "Writing binary file..." << std::endl;
    std::ofstream outfile(out_path, std::ios::binary);
    
    int32_t magic = 0x4D4D5243;
    int32_t version = 2;
    int64_t total = all_tokens.size();
    
    outfile.write(reinterpret_cast<char*>(&magic), sizeof(int32_t));
    outfile.write(reinterpret_cast<char*>(&version), sizeof(int32_t));
    outfile.write(reinterpret_cast<char*>(&total), sizeof(int64_t));
    outfile.write(reinterpret_cast<char*>(all_tokens.data()), total * sizeof(int32_t));
    outfile.write(reinterpret_cast<char*>(all_masks.data()), total * sizeof(int32_t));
    
    std::cout << "✅ Successfully created " << out_path << std::endl;
    return 0;
}
