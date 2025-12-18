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
      current_idx_(0),
      stop_flag_(false) {
    
    total_tokens_ = dataset_->size();
    
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
    current_idx_ = 0;
    // Clear buffer? Or just let it be.
}

int64_t DataLoader::total_batches() const {
    // Total tokens / (batch * seq)
    // Actually we need (seq + 1) for [input, target] sliding window usually?
    // Or we do input[0..N-1] -> target[1..N].
    // Let's assume non-overlapping sequences for now for simplicity, or just contiguous.
    return total_tokens_ / (batch_size_ * seq_len_);
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
        // 1. Reserve a slice index
        // FIX: For Strided Sampling (where each row acts as an independent worker on a chunk),
        // we only advance by seq_len_ per step, not the full batch size.
        // If we advanced by (batch * seq), we would skip huge chunks of data between steps for each row.
        int64_t idx = current_idx_.fetch_add(seq_len_);
        
        // Check bounds (Per Chunk!)
        // Since we are striding, 'idx' only traverses one chunk_size.
        int64_t chunk_size = total_tokens_ / batch_size_;
        if (idx + seq_len_ + 1 > chunk_size) {
            // End of epoch (for this stride)
            current_idx_ = 0; 
            idx = current_idx_.fetch_add(seq_len_);
        }
        
        // 2. Prepare Batch Data
        Tensor input = Tensor::zeros({batch_size_, seq_len_});
        Tensor target = Tensor::zeros({batch_size_, seq_len_});
        
        // Read from dataset (Thread safe? Yes, mmap read is safe)
        // If shuffle is true, we need random indices. 
        // For streaming, "shuffle" usually means shuffling buffers, not random access seek.
        // Let's implement sequential reading for high perf first.
        
        // Input: [batch, seq]
        // We treat the dataset as one long stream, and chop it into batches.
        // Or we can say each row in batch comes from different part of file (better for diversity).
        // For simplest implementation: Continuous chunk.
        // Row 0: [idx ... idx+seq]
        // Row 1: [idx+seq ... idx+2seq] ...
        
        // Wait, "Streaming" usually implies Row 0 is at offset X, Row 1 is at offset Y.
        // If we just take one contiguous block, the batch is highly correlated.
        // Better: Stride through file.
        // stride = total_tokens / batch_size;
        // row[b] starts at b * stride + current_seq_ptr
        
        // Let's implement Strided Streaming (decorrelates batch rows).
        // chunk_size is already defined above
        
        // We use 'idx' as the 'step' index (0, 1, 2...)
        // Actually current_idx_ logic above was for continuous.
        // Let's change idx to mean "current sequence step".
        // fetch_add(seq_len_)
        
        // Re-calibrating logic:
        // idx is the offset for the *first* row.
        // We need to fetch 'seq_len' tokens.
        
        for (int64_t b = 0; b < batch_size_; ++b) {
            int64_t row_start = b * chunk_size + idx; // Strided access
            
            // Safety check
            if (row_start + seq_len_ + 1 >= total_tokens_) {
                 // Wrap around for this row
                 row_start = (row_start) % (total_tokens_ - seq_len_ - 1);
            }
            
            for (int64_t s = 0; s < seq_len_; ++s) {
                // causal prediction: input=t, target=t+1
                input.data()[b * seq_len_ + s] = (float)(*dataset_)[row_start + s];
                target.data()[b * seq_len_ + s] = (float)(*dataset_)[row_start + s + 1];
            }
        }
        
        TrainingBatch batch { input, target };
        
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
