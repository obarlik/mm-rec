#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

int main() {
    std::ofstream f("training_data.bin", std::ios::binary);
    
    // Header
    int32_t magic = 0xC0FFEE;
    int32_t version = 1;
    int64_t count = 1000; // 1000 tokens
    
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&count), 8);
    
    // Data (1000 random tokens)
    std::vector<int32_t> tokens(count);
    for(int i=0; i<count; ++i) tokens[i] = i % 100;
    
    f.write(reinterpret_cast<const char*>(tokens.data()), count * 4);
    f.close();
    
    std::cout << "Created valid training_data.bin" << std::endl;
    return 0;
}
