#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace mm_rec {

/**
 * Memory Mapped Dataset
 * 
 * Reads a binary file of int32_t tokens directly from disk using mmap.
 * This allows O(1) random access to massive datasets without RAM overhead.
 * 
 * Logic:
 * - File Format: [Magic(4)][Version(4)][Count(8)][Tokens...]
 */
class Dataset {
public:
    Dataset(const std::string& path);
    ~Dataset();
    
    // Disable copy (mmap handle is unique-ish resource)
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;
    
    // Accessors
    int64_t size() const { return num_tokens_; }
    const int32_t* data() const { return data_ptr_; }
    
    // Access token at index
    int32_t operator[](int64_t index) const {
        if (index < 0 || index >= num_tokens_) {
            throw std::out_of_range("Dataset index out of bounds");
        }
        return data_ptr_[index];
    }
    
private:
    std::string path_;
    int fd_;            // File descriptor
    int64_t file_size_; // Bytes
    int64_t num_tokens_;// Number of tokens
    void* mapped_mem_;  // Raw mmap pointer
    const int32_t* data_ptr_; // Typed pointer to tokens
    
    void open_mmap();
    void close_mmap();
};

} // namespace mm_rec
