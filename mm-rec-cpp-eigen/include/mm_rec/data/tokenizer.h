#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <string_view>

namespace mm_rec {

/**
 * Simple Tokenizer (Char or Word level)
 * 
 * For MVP, we'll implement a character-level tokenizer or a simple map-based one.
 * In production, this would load a BPE vocab file.
 */
class Tokenizer {
public:
    Tokenizer();
    
    // Load vocab from file (line by line)
    void load_vocab(const std::string& path);
    
    // Build vocab from text (simple char level)
    void build_vocab(const std::string& text);
    
    // Encode string to tokens
    std::vector<int32_t> encode(const std::string& text) const;
    
    // Decode tokens to string
    std::string decode(const std::vector<int32_t>& tokens) const;
    
    int64_t vocab_size() const { return token_to_id_.size(); }
    
    // Special tokens
    int32_t pad_id() const { return 0; }
    int32_t unk_id() const { return 1; }
    int32_t bos_id() const { return 2; }
    int32_t eos_id() const { return 3; }
    
private:
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    
    void add_token(const std::string& token);
};

} // namespace mm_rec
