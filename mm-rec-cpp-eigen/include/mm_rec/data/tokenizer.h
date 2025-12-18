#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <string_view>
#include <list>
#include <set>

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
    
    // Load vocab (vocab.json) and merges (merges.txt)
    void load_model(const std::string& vocab_path, const std::string& merges_path);
    
    // Encode string to tokens (BPE)
    std::vector<int32_t> encode(const std::string& text) const;
    
    // Decode tokens to string
    std::string decode(const std::vector<int32_t>& tokens) const;
    
    int64_t vocab_size() const { return encoder_.size(); }
    
    // Special tokens (Standard GPT-2 style often handles these differently, 
    // but we'll reserve IDs for simplicity if needed, or rely on vocab.json)
    int32_t pad_id() const { return 0; } // Assuming 0 is pad/null
    int32_t unk_id() const { return 1; } // Legacy UNK support
    int32_t bos_id() const { return encoder_.count("<|endoftext|>") ? encoder_.at("<|endoftext|>") : 0; }
    int32_t eos_id() const { return encoder_.count("<|endoftext|>") ? encoder_.at("<|endoftext|>") : 0; }
    
    // Legacy support for cmd_prepare (No-op in BPE mode)
    void build_vocab(const std::string&) {} 
    void add_token(const std::string&) {}
    
private:
    // BPE Maps
    std::unordered_map<std::string, int32_t> encoder_;
    std::unordered_map<int32_t, std::string> decoder_;
    std::unordered_map<std::string, int> bpe_ranks_; // Pair "A B" -> Rank
    mutable std::unordered_map<std::string, std::string> cache_; // Memoization for words
    
    // Helper methods
    std::vector<std::string> bpe(const std::string& token) const;
};

} // namespace mm_rec
