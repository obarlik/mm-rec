#include "mm_rec/data/data_loader.h"
#include "mm_rec/data/dataset.h"
#include <iostream>
#include <memory>
#include <iomanip>

using namespace mm_rec;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./debug_dataloader <data_file>" << std::endl;
        return 1;
    }
    
    std::string data_file = argv[1];
    std::cout << "Opening dataset: " << data_file << std::endl;
    
    try {
        auto dataset = std::make_shared<Dataset>(data_file);
        
        // Batch=1, Seq=4 for easy reading
        int64_t batch_size = 1;
        int64_t seq_len = 4;
        
        DataLoader loader(dataset, batch_size, seq_len, false, 1); // no shuffle
        
        std::cout << "DataLoader created. Total tokens: " << dataset->size() << std::endl;
        
        TrainingBatch batch;
        if (loader.next(batch)) {
            std::cout << "Batch 0:" << std::endl;
            
            std::cout << "Input:  ";
            for(int i=0; i<seq_len; ++i) {
                std::cout << (int)batch.input_ids.data()[i] << " ";
            }
            std::cout << std::endl;
            
            std::cout << "Target: ";
            for(int i=0; i<seq_len; ++i) {
                std::cout << (int)batch.targets.data()[i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "Failed to fetch batch!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
