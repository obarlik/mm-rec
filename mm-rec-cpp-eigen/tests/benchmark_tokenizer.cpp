#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include "mm_rec/data/tokenizer.h"

using namespace mm_rec;

// Generate random "words" to stress test BPE merging
std::string generate_random_text(size_t num_words, size_t max_len) {
    std::string text;
    std::string chars = "abcde"; // Limited alphabet to force heavy merging if merges exist
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_len(1, max_len);
    std::uniform_int_distribution<int> dist_char(0, chars.size() - 1);

    for (size_t i = 0; i < num_words; ++i) {
        int len = dist_len(rng);
        for (int j = 0; j < len; ++j) {
            text += chars[dist_char(rng)];
        }
        text += " ";
    }
    return text;
}

// Create a deep dummy vocab forcing many merges
void setup_stress_vocab() {
    // Vocab: letters + many recursive merges
    // a, b, c, d, e
    // ab, cd, ...
    std::ofstream v("stress_vocab.json");
    v << "{\"[PAD]\":0, \"a\":4, \"b\":5, \"c\":6, \"d\":7, \"e\":8 }\n";
    v.close();

    std::ofstream m("stress_merges.txt");
    m << "# version 1\n";
    // Create a chain of merges to force multiple passes
    // a b -> ab
    // c d -> cd
    // ab cd -> abcd
    // ...
    // This tests the "recursive" nature of BPE
    m << "a b\n";
    m << "c d\n";
    m << "a c\n"; 
    m << "b e\n";
    m.close();
}

int main() {
    setup_stress_vocab();
    
    Tokenizer t;
    t.load_model("stress_vocab.json", "stress_merges.txt");

    std::cout << "Generating text..." << std::endl;
    // 100k words, up to 20 chars each -> ~1MB text
    std::string text = generate_random_text(100000, 20); 
    std::cout << "Text size: " << text.size() / 1024 << " KB" << std::endl;

    std::cout << "Benchmarking encode()..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    auto tokens = t.encode(text);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    double tokens_per_sec = tokens.size() / elapsed.count();
    double chars_per_sec = text.size() / elapsed.count();

    std::cout << "--------------------------------" << std::endl;
    std::cout << "Time: " << elapsed.count() << "s" << std::endl;
    std::cout << "Throughput: " << (int)tokens_per_sec << " tokens/sec" << std::endl;
    std::cout << "Throughput: " << (int)(chars_per_sec / 1024.0 / 1024.0) << " MB/s" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    remove("stress_vocab.json");
    remove("stress_merges.txt");
    return 0;
}
