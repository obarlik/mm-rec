#pragma once

#include "mm_rec/data/i_data_loader.h" // Interface
#include "mm_rec/data/dataset.h"
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <vector>

namespace mm_rec {

/**
 * Streaming Data Loader
 * 
 * Feeds batches to the trainer from the memory-mapped dataset.
 * Supports multi-threaded prefetching.
 */
class DataLoader : public IDataLoader {
public:
    DataLoader(std::shared_ptr<Dataset> dataset, 
               int64_t batch_size, 
               int64_t seq_len,
               bool shuffle = false,
               int num_workers = 1);
               
    ~DataLoader() override;
    
    // Interface Implementation
    bool next(TrainingBatch& batch) override;
    void reset() override;
    int64_t total_batches() const override;
    
private:
    std::shared_ptr<Dataset> dataset_;
    int64_t batch_size_;
    int64_t seq_len_;
    bool shuffle_;
    
    // Cursor handling
    std::atomic<int64_t> current_batch_idx_;
    std::vector<int64_t> indices_;
    int64_t total_tokens_;
    
    // Threading
    std::queue<TrainingBatch> buffer_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    
    std::vector<std::thread> workers_;
    std::atomic<bool> stop_flag_;
    
    void worker_loop();
};

/**
 * Factory for creating DataLoader instances.
 * Registered in DI Container.
 */
class DataLoaderFactory : public IDataLoaderFactory {
public:
    std::unique_ptr<IDataLoader> create_loader(
        std::shared_ptr<Dataset> dataset,
        int64_t batch_size,
        int64_t seq_len,
        bool shuffle = false,
        int num_workers = 1
    ) override {
        return std::make_unique<DataLoader>(dataset, batch_size, seq_len, shuffle, num_workers);
    }
};

} // namespace mm_rec
