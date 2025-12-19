#pragma once

#include "mm_rec/data/dataset.h"
#include "mm_rec/training/trainer.h"
#include <memory>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <atomic>

namespace mm_rec {

/**
 * Streaming Data Loader
 * 
 * Feeds batches to the trainer from the memory-mapped dataset.
 * Supports multi-threaded prefetching.
 */
class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset> dataset, 
               int64_t batch_size, 
               int64_t seq_len,
               bool shuffle = false,
               int num_workers = 1);
               
    ~DataLoader();
    
    // Get next batch. Blocks if queue is empty.
    // Returns false if epoch is finished (and auto_reset=false), or stop is called.
    bool next(TrainingBatch& batch);
    
    // Reset cursor to start
    void reset();
    
    int64_t total_batches() const;
    
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

} // namespace mm_rec
