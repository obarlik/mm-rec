#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <string>
#include "mm_rec/data/tokenizer.h"

using namespace mm_rec;

void create_dummy_vocab_files() {
    // 1. Vocab JSON
    // {"a": 4, "b": 5, "ab": 6, "c": 7}
    std::ofstream v("test_vocab.json");
    v << "{\n";
    v << "  \"[PAD]\": 0,\n";
    v << "  \"[UNK]\": 1,\n";
    v << "  \"[BOS]\": 2,\n";
    v << "  \"[EOS]\": 3,\n";
    v << "  \"a\": 4,\n";
    v << "  \"b\": 5,\n";
    v << "  \"ab\": 6,\n"; // Merged token
    v << "  \"c\": 7\n";
    v << "}\n";
    v.close();

    // 2. Merges TXT
    // Merge a and b to form ab
    std::ofstream m("test_merges.txt");
    m << "# version 1\n";
    m << "a b\n"; 
    m.close();
}

void test_basic_load_and_encode_simple() {
    std::cout << "[Test] test_basic_load_and_encode_simple..." << std::endl;
    Tokenizer t;
    t.load_model("test_vocab.json", "test_merges.txt");
    
    assert(t.vocab_size() == 8);
    
    // "c" -> ID 7
    auto ids = t.encode("c");
    assert(ids.size() == 1);
    assert(ids[0] == 7);
    
    // "a" -> ID 4
    ids = t.encode("a");
    assert(ids.size() == 1);
    assert(ids[0] == 4);

    std::cout << "PASS" << std::endl;
}

void test_bpe_merge() {
    std::cout << "[Test] test_bpe_merge..." << std::endl;
    Tokenizer t;
    t.load_model("test_vocab.json", "test_merges.txt");
    
    // "a b" -> BPE should merge to "ab" (ID 6)
    // Input text: "ab" (one word)
    // Pre-tokenize: "ab"
    // Initial splits: "a", "b"
    // BPE loop: find pair ("a", "b") -> merge -> "ab"
    // Look up "ab" -> ID 6
    
    auto ids = t.encode("ab");
    assert(ids.size() == 1);
    std::cout << "Encoded 'ab' -> " << ids[0] << " (Expected 6)" << std::endl;
    assert(ids[0] == 6);
    
    // "a b" (separate words)
    // Pre-tokenize: ["a", "b"]
    // "a" -> [4]
    // "b" -> [5]
    // Result: [4, 5]
    ids = t.encode("a b");
    assert(ids.size() == 2);
    assert(ids[0] == 4);
    assert(ids[1] == 5);
    
    std::cout << "PASS" << std::endl;
}

void test_unknown_handling() {
    std::cout << "[Test] test_unknown_handling..." << std::endl;
    Tokenizer t;
    t.load_model("test_vocab.json", "test_merges.txt");
    
    // "z" is not in vocab. Should be UNK (ID 1)
    auto ids = t.encode("z");
    assert(ids.size() == 1);
    assert(ids[0] == 1);
    
    std::cout << "PASS" << std::endl;
}

int main() {
    create_dummy_vocab_files();
    
    test_basic_load_and_encode_simple();
    test_bpe_merge();
    test_unknown_handling();
    
    std::cout << "✅ All Tokenizer Tests Passed!" << std::endl;
    // Cleanup
    remove("test_vocab.json");
    remove("test_merges.txt");
    return 0;
}
