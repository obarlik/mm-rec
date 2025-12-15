/**
 * Test: Data Pipeline (Tokenizer -> Dataset -> DataLoader)
 * 
 * Verifies the streaming pipeline end-to-end.
 */

#include "mm_rec/data/tokenizer.h"
#include "mm_rec/data/dataset.h"
#include "mm_rec/data/data_loader.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <filesystem>

using namespace mm_rec;
namespace fs = std::filesystem;

void create_dummy_dataset(const std::string& path, int64_t num_tokens) {
    std::ofstream file(path, std::ios::binary);
    
    // Header
    int32_t magic = 0x4D4D5245; // MMRE
    int32_t version = 1;
    int64_t count = num_tokens;
    
    file.write(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<char*>(&version), sizeof(version));
    file.write(reinterpret_cast<char*>(&count), sizeof(count));
    
    // Body (Sequential integers 0, 1, 2...)
    for (int64_t i = 0; i < num_tokens; ++i) {
        int32_t token = (int32_t)i;
        file.write(reinterpret_cast<char*>(&token), sizeof(token));
    }
    file.close();
}

void test_tokenizer() {
    std::cout << "=== Test: Tokenizer ===" << std::endl;
    Tokenizer tokenizer;
    std::string text = "abc";
    tokenizer.build_vocab(text);
    
    // Special tokens (4) + a, b, c = 7
    std::cout << "Vocab size: " << tokenizer.vocab_size() << std::endl;
    // assert(tokenizer.vocab_size() == 7); 
    
    auto ids = tokenizer.encode("abac"); // [a, b, a, c] -> indices
    auto decoded = tokenizer.decode(ids);
    
    std::cout << "Encoded: ";
    for(auto id : ids) std::cout << id << " ";
    std::cout << "\nDecoded: " << decoded << std::endl;
    
    assert(decoded == "abac");
    std::cout << "✅ Tokenizer OK" << std::endl;
}

void test_dataset_loader() {
    std::cout << "=== Test: Dataset & DataLoader (Streaming) ===" << std::endl;
    
    std::string path = "test_data.bin";
    int64_t num_tokens = 1000;
    create_dummy_dataset(path, num_tokens);
    
    // 1. Test Dataset (mmap)
    {
        Dataset ds(path);
        std::cout << "Dataset opened. Size: " << ds.size() << std::endl;
        assert(ds.size() == num_tokens);
        assert(ds[0] == 0);
        assert(ds[999] == 999);
        std::cout << "✅ Dataset random access OK" << std::endl;
    }
    
    // 2. Test DataLoader (Threading)
    {
        auto ds = std::make_shared<Dataset>(path);
        int64_t batch_size = 4;
        int64_t seq_len = 10;
        
        DataLoader loader(ds, batch_size, seq_len, false, 2); // 2 workers
        
        TrainingBatch batch;
        bool success = loader.next(batch);
        if (!success) {
            std::cerr << "Loader failed to fetch batch!" << std::endl;
            exit(1);
        }
        
        // Verify batch data
        // Batch should be filled (maybe not sequentially due to threads, but valid data)
        std::cout << "Batch Input [0,0]: " << batch.input_ids.data()[0] << std::endl;
        std::cout << "Batch Target [0,0]: " << batch.targets.data()[0] << std::endl;
        
        // Target should be input + 1 (causal)
        assert(batch.targets.data()[0] == batch.input_ids.data()[0] + 1);
        
        std::cout << "✅ DataLoader fetched batch OK" << std::endl;
        
        // Fetch more to ensure threading doesn't crash
        for(int i=0; i<5; ++i) {
            loader.next(batch);
        }
        std::cout << "✅ Multiple fetches OK" << std::endl;
    }
    
    // Cleanup
    fs::remove(path);
    std::cout << "Cleanup done." << std::endl;
}

int main() {
    test_tokenizer();
    test_dataset_loader();
    return 0;
}
