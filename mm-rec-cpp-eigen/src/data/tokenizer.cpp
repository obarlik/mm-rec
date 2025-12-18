#include "mm_rec/data/tokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <regex>
#include <algorithm>
#include <set>
#include <map>

// Helper for splitting
std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

namespace mm_rec {

Tokenizer::Tokenizer() {
    // Default fallback if no model loaded (Char/Byte level basic map?)
    // Actually, for BPE, usually we start empty or with bytes.
    // Let's rely on load_model to populate.
}

// Simple JSON parser for vocab.json (Map<String, Int>)
// Format: {"a": 1, "b": 2, ...}
std::unordered_map<std::string, int32_t> parse_vocab_json(const std::string& path) {
    std::unordered_map<std::string, int32_t> vocab;
    std::ifstream f(path);
    if (!f.is_open()) return vocab;
    
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    
    // VERY Basic Parser: strictly looks for "key": value pattern
    // Assumes standard keys (no escaped quotes inside keys for simplicity MVP)
    std::string key;
    bool in_key = false;
    std::string val_str;
    bool in_val = false;
    
    for (size_t i = 0; i < content.size(); ++i) {
        char c = content[i];
        if (c == '"') {
            if (in_key) {
                in_key = false; // End key
            } else if (!in_val) {
                // Could be start of key or start of string value (if value is string, but vocab values are ints)
                // In standard vocab.json, keys are strings, values are ints.
                // But check previous char to see if it's after a colon?
                // Let's assume start of key if we just finished a value or comma or open brace.
                in_key = true;
                key.clear();
            }
        } else if (in_key) {
            key += c;
        } else if (c == ':') {
            in_val = true;
            val_str.clear();
        } else if (in_val) {
            if (isdigit(c)) {
                val_str += c;
            } else if (c == ',' || c == '}') {
                if (!val_str.empty()) {
                    vocab[key] = std::stoi(val_str);
                    val_str.clear();
                    in_val = false;
                }
            }
        }
    }
    return vocab;
}

void Tokenizer::load_model(const std::string& vocab_path, const std::string& merges_path) {
    // 1. Load Vocab
    encoder_ = parse_vocab_json(vocab_path);
    for (const auto& kv : encoder_) {
        decoder_[kv.second] = kv.first;
    }
    
    // 2. Load Merges
    std::ifstream f(merges_path);
    std::string line;
    int rank = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue; // Skip comments/version
        // Merges format: "g e" -> means merge g and e
        // We store this as key="g e", value=rank
        bpe_ranks_[line] = rank++;
    }
    
    std::cout << "✅ Loaded BPE Model. Vocab: " << encoder_.size() << ", Merges: " << bpe_ranks_.size() << std::endl;
}

// Helper to get rank securely
int get_rank(const std::string& a, const std::string& b, const std::unordered_map<std::string, int>& ranks) {
    std::string key = a + " " + b; // Space delimiter as per merges format
    auto it = ranks.find(key);
    if (it != ranks.end()) return it->second;
    return 1e9 + 7; // Infinity
}

std::vector<std::string> Tokenizer::bpe(const std::string& token) const {
    if (cache_.count(token)) {
        return split(cache_[token], ' ');
    }
    
    // 1. Initialize List of chars
    std::list<std::string> word;
    for (char c : token) {
        word.push_back(std::string(1, c));
    }
    
    // If empty or single char, return immediately
    if (word.size() <= 1) return { std::begin(word), std::end(word) };
    
    // 2. Initialize Priority Queue (RB-Tree)
    // format: <rank, iterator_to_first_element_of_pair>
    // Using pair pointer (or similar ID) might be safer, but list iterators are stable.
    using ListIt = std::list<std::string>::iterator;
    // We need a custom comparator or just rely on pair's default (rank first, then iterator address).
    // Iterator comparison is not guaranteed portable for ordering, but for uniqueness in set it's fine.
    // However, std::set needs strict weak ordering.
    // Let's use a struct with explicit comparison.
    struct MergeAction {
        int rank;
        ListIt it; // points to 'left' of pair
        
        bool operator<(const MergeAction& other) const {
            if (rank != other.rank) return rank < other.rank;
            return &(*it) < &(*other.it); // Compare addresses of string content? Or use logic.
            // Pointers to elements in list are stable.
        }
    };
    
    std::set<MergeAction> pq;
    
    // 3. Initial Populate
    auto it = word.begin();
    while (it != word.end()) {
        auto next = std::next(it);
        if (next == word.end()) break;
        
        int r = get_rank(*it, *next, bpe_ranks_);
        if (r < 1e9) {
            pq.insert({r, it});
        }
        it++;
    }
    
    // 4. Merge Loop
    while (!pq.empty()) {
        // Get best merge
        MergeAction best = *pq.begin();
        pq.erase(pq.begin());
        
        // Validate? (If we manage updates correctly, it should be valid)
        ListIt left = best.it;
        ListIt right = std::next(left);
        
        // Safety check (shouldn't happen if logic is bug-free)
        if (right == word.end()) continue;
        
        // DO MERGE: left becomes "left+right"
        *left = *left + *right;
        
        // Remove 'right'
        // But first, we must remove neighbors involving 'right' from PQ
        // Neighbors of pair (L, R) were:
        // (Prev, L) and (R, Next)
        
        // 1. Remove (R, Next) if exists
        auto next = std::next(right);
        if (next != word.end()) {
            int r_old = get_rank(*right, *next, bpe_ranks_);
            if (r_old < 1e9) {
                // Must construct exact object to erase
                pq.erase({r_old, right}); 
            }
        }
        
        // 2. Remove (Prev, L) if exists
        // We need to re-insert it with NEW rank because L changed!
        if (left != word.begin()) {
            auto prev = std::prev(left);
            int r_old = get_rank(*prev, *right /* old L was not saved, tricky! */, bpe_ranks_); 
            // WAIT. "*left" has already been mutated to "new_merged_token"!
            // So we can't calculate r_old using *left now.
            // We need to remove (Prev, L) BEFORE mutating L.
        }
        
        // OK, Restart iteration logic to be safe:
        // 1. Identify Prev, L, R, Next
        // 2. Remove (Prev, L) and (R, Next) from PQ.
        // 3. Mutate L, Erase R.
        // 4. Calculate new ranks for (Prev, L) and (L, Next) and Insert.
        
        // Re-do pop
    }
    
    // Let's rewrite the loop cleaner with correct ordering
    word.clear();
    for (char c : token) word.push_back(std::string(1, c));
    if (word.size() <= 1) return { std::begin(word), std::end(word) };
    
    pq.clear();
    it = word.begin();
    while (it != word.end()) {
        auto next = std::next(it);
        if (next == word.end()) break;
        int r = get_rank(*it, *next, bpe_ranks_);
        if (r < 1e9) pq.insert({r, it});
        it++;
    }
    
    while (!pq.empty()) {
        MergeAction best = *pq.begin();
        pq.erase(pq.begin());
        
        ListIt left = best.it;
        ListIt right = std::next(left);
        
        // Check verification (symbol might have been merged already? No, we remove invalidated ones)
        // But let's be sure.
        
        // Pre-fetch neighbors
        auto prev = (left == word.begin()) ? word.end() : std::prev(left);
        auto next = std::next(right);
        
        // Remove neighbors from PQ
        if (prev != word.end()) {
            int r = get_rank(*prev, *left, bpe_ranks_);
            if (r < 1e9) pq.erase({r, prev});
        }
        if (next != word.end()) {
            int r = get_rank(*right, *next, bpe_ranks_);
            if (r < 1e9) pq.erase({r, right});
        }
        
        // Merge
        *left = *left + *right;
        word.erase(right);
        
        // Re-add neighbors with new token
        if (prev != word.end()) {
            int r = get_rank(*prev, *left, bpe_ranks_);
            if (r < 1e9) pq.insert({r, prev});
        }
        if (next != word.end()) {
            int r = get_rank(*left, *next, bpe_ranks_);
            if (r < 1e9) pq.insert({r, left});
        }
    }
    
    // Convert to vector
    std::vector<std::string> result(word.begin(), word.end());
    
    // Cache result
    std::string cache_val;
    for(size_t i=0; i<result.size(); ++i) {
        cache_val += result[i];
        if(i < result.size()-1) cache_val += " ";
    }
    cache_[token] = cache_val;
    
    return result;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> bpe_tokens;
    
    // 1. Pre-tokenize (Whitespace for MVP)
    // Full GPT-2 uses Regex. We'll stick to whitespace split for now to verify logic.
    std::stringstream ss(text);
    std::string word;
    
    while(ss >> word) {
        // Apply BPE to each word
        std::vector<std::string> subwords = bpe(word);
        
        for (const auto& sw : subwords) {
            if (encoder_.count(sw)) {
                bpe_tokens.push_back(encoder_.at(sw));
            } else {
                // Fallback: byte encoding (mapped to unknown or direct bytes if in vocab)
                // For this MVP, if subword not in vocab, we map to UNK (0 or 1)
                // Proper BPE has all bytes in vocab.
                bpe_tokens.push_back(unk_id()); 
            }
        }
    }
    return bpe_tokens;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string text;
    for (int32_t id : tokens) {
        if (decoder_.count(id)) {
            text += decoder_.at(id); // Usually BPE tokens concat directly
            // GPT-2 style replaces Space with Ġ.
            // Here we just append. Spaces handled by pre-tokenizer reconstructing?
            // Simple approach: Add space after every word? No, subwords shouldn't have spaces.
            // Space should be a token itself or part of the first token (e.g. " Hello").
            // For MVP: Just concat.
        }
    }
    return text;
}

} // namespace mm_rec
