
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./check_vocab <training_data.bin>" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1], std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // Skip header (16 bytes)
    file.seekg(16, std::ios::beg);

    std::vector<int32_t> tokens;
    int32_t val;
    while (file.read(reinterpret_cast<char*>(&val), sizeof(int32_t))) {
        tokens.push_back(val);
    }
    
    if (tokens.empty()) {
        std::cout << "Empty dataset" << std::endl;
        return 0;
    }

    int32_t max_token = *std::max_element(tokens.begin(), tokens.end());
    int32_t min_token = *std::min_element(tokens.begin(), tokens.end());
    
    std::cout << "Total Tokens: " << tokens.size() << std::endl;
    std::cout << "Max Token ID: " << max_token << std::endl;
    std::cout << "Min Token ID: " << min_token << std::endl;
    
    if (max_token >= 16) {
        std::cout << "⚠️  CRITICAL: Max token " << max_token << " > Config Vocab 16" << std::endl;
    } else {
        std::cout << "✅ Safe: Max token " << max_token << " < Config Vocab 16" << std::endl;
    }

    return 0;
}
