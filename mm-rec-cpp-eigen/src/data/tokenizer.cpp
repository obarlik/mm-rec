#include "mm_rec/data/tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace mm_rec {

Tokenizer::Tokenizer() {
    // Default special tokens
    add_token("[PAD]"); // 0
    add_token("[UNK]"); // 1
    add_token("[BOS]"); // 2
    add_token("[EOS]"); // 3
}

void Tokenizer::add_token(const std::string& token) {
    if (token_to_id_.find(token) == token_to_id_.end()) {
        int32_t id = token_to_id_.size();
        token_to_id_[token] = id;
        id_to_token_[id] = token;
    }
}

void Tokenizer::load_vocab(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        // Assume one token per line
        // Trim newline
        if (!line.empty() && line.back() == '\n') line.pop_back();
        if (!line.empty() && line.back() == '\r') line.pop_back();
        add_token(line);
    }
}

void Tokenizer::build_vocab(const std::string& text) {
    // Simple Char level builder
    for (char c : text) {
        std::string s(1, c);
        add_token(s);
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    
    // Simple Char level encoding for now (unless vocab loaded differently)
    // If we have a BPE vocab, splitting is harder.
    // For MVP/Demo: Treat as char-level if input is just raw text
    
    for (char c : text) {
        std::string s(1, c);
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(unk_id());
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::stringstream ss;
    for (int32_t id : tokens) {
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            // Don't print special tokens for clean output
            if (id > 3) {
                ss << it->second;
            }
        }
    }
    return ss.str();
}

} // namespace mm_rec
