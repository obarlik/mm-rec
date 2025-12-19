#include "mm_rec/data/data_loader.h"
#include <iostream>
#include <algorithm>
#include <random>

namespace mm_rec {

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, 
                       int64_t batch_size, 
                       int64_t seq_len,
                       bool shuffle,
                       int num_workers)
    : dataset_(dataset), 
      batch_size_(batch_size), 
      seq_len_(seq_len), 
      shuffle_(shuffle),
      current_batch_idx_(0),
      stop_flag_(false) {
    
    total_tokens_ = dataset_->size();
    
    // Calculate total possible sequences (non-overlapping for simplicity)
    int64_t num_sequences = total_tokens_ / seq_len_;
    
    // Populate indices
    indices_.resize(num_sequences);
    for (int64_t i = 0; i < num_sequences; ++i) {
        indices_[i] = i * seq_len_;
    }
    
    reset(); // Initial shuffle
    
    // Start worker threads
    for (int i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&DataLoader::worker_loop, this);
    }
}

DataLoader::~DataLoader() {
    stop_flag_ = true;
    not_full_.notify_all();
    
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

void DataLoader::reset() {
    current_batch_idx_ = 0;
    
    if (shuffle_) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices_.begin(), indices_.end(), g);
        std::cout << "[DataLoader] Shuffled " << indices_.size() << " sequences." << std::endl;
    }
}

int64_t DataLoader::total_batches() const {
    // Total sequences / batch_size
    return indices_.size() / batch_size_;
}

bool DataLoader::next(TrainingBatch& batch) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait until buffer has data or stopped
    not_empty_.wait(lock, [this] {
        return !buffer_.empty() || stop_flag_;
    });
    
    if (stop_flag_ && buffer_.empty()) {
        return false;
    }
    
    if (buffer_.empty()) {
        // Should catch this if file ended naturally
        return false;
    }
    
    batch = std::move(buffer_.front());
    buffer_.pop();
    
    lock.unlock();
    not_full_.notify_one();
    
    return true;
}

void DataLoader::worker_loop() {
    while (!stop_flag_) {
        // 1. Reserve a Batch Index (which points to 'batch_size' sequences)
        int64_t batch_idx = current_batch_idx_.fetch_add(1);
        
        // Check bounds
        int64_t total_batches = (int64_t)indices_.size() / batch_size_;
        if (batch_idx >= total_batches) {
            // End of epoch
            // We need to wait for reset or just stop?
            // For now, let's wrap around or wait.
            // Simplified: Just stop submitting.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue; 
        }
        
        // 2. Prepare Batch Data
        Tensor input = Tensor::zeros({batch_size_, seq_len_});
        Tensor target = Tensor::zeros({batch_size_, seq_len_});
        
        // Fetch 'batch_size' sequences from the indices pool
        for (int64_t b = 0; b < batch_size_; ++b) {
            int64_t seq_idx = batch_idx * batch_size_ + b;
            if (seq_idx >= (int64_t)indices_.size()) break; // Safety
            
            int64_t start_token = indices_[seq_idx];
            
            // Safety check against file end
            // (indices were created safely, but just in case)
             if (start_token + seq_len_ + 1 >= total_tokens_) {
                 start_token = 0; // Fallback
             }

            for (int64_t s = 0; s < seq_len_; ++s) {
                // causal prediction: input=t, target=t+1
                input.data()[b * seq_len_ + s] = (float)(*dataset_)[start_token + s];
                target.data()[b * seq_len_ + s] = (float)(*dataset_)[start_token + s + 1];
            }
        }
        
        TrainingBatch batch { input, target, Tensor::zeros({0}) };
        
        // 3. Push to Buffer
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] {
            return buffer_.size() < 10 || stop_flag_; // Buffer size 10
        });
        
        if (stop_flag_) break;
        
        buffer_.push(std::move(batch));
        lock.unlock();
        not_empty_.notify_one();
    }
}

} // namespace mm_rec
