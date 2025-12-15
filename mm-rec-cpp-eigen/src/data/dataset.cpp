#include "mm_rec/data/dataset.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <iostream>

namespace mm_rec {

Dataset::Dataset(const std::string& path) : path_(path), fd_(-1), mapped_mem_(nullptr) {
    open_mmap();
}

Dataset::~Dataset() {
    close_mmap();
}

void Dataset::open_mmap() {
    // 1. Open File
    fd_ = open(path_.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("Failed to open dataset file: " + path_);
    }
    
    // 2. Get Size
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        throw std::runtime_error("Failed to stat dataset file");
    }
    file_size_ = sb.st_size;
    
    // Check header size (Magic + Version + Count = 4+4+8 = 16 bytes)
    if (file_size_ < 16) {
        close(fd_);
        throw std::runtime_error("Dataset file too small/corrupted");
    }
    
    // 3. MMAP
    mapped_mem_ = mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_mem_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("Failed to mmap dataset file");
    }
    
    // 4. Parse Header (Manual offset)
    const char* ptr = static_cast<const char*>(mapped_mem_);
    
    // Magic (int32) - Let's assume standard int check or skip for now
    // Version (int32)
    // Count (int64)
    
    // For MVP, lets assume format: [int32 magic][int32 version][int64 count][int32 tokens...]
    // You might need to handle endianness in production if moving between archs.
    
    // const int32_t* header_i32 = reinterpret_cast<const int32_t*>(ptr);
    // int32_t magic = header_i32[0];
    // int32_t version = header_i32[1];
    
    const int64_t* header_i64 = reinterpret_cast<const int64_t*>(ptr + 8);
    int64_t count = header_i64[0];
    
    num_tokens_ = count;
    
    // Verify size matches expected
    // 16 bytes header + count * 4 bytes
    int64_t expected_size = 16 + num_tokens_ * 4;
    // Note: file might be larger due to padding, but shouldn't be smaller
    if (file_size_ < expected_size) {
        std::cerr << "Warning: Dataset header claims " << num_tokens_ 
                  << " tokens, but file size is " << file_size_ << " bytes." << std::endl;
    }
    
    // Set data pointer to start of tokens
    data_ptr_ = reinterpret_cast<const int32_t*>(ptr + 16);
    
    std::cout << "[Dataset] Mapped " << (file_size_ / 1024.0 / 1024.0) << " MB. "
              << "Tokens: " << num_tokens_ << std::endl;
}

void Dataset::close_mmap() {
    if (mapped_mem_ && mapped_mem_ != MAP_FAILED) {
        munmap(mapped_mem_, file_size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
}

} // namespace mm_rec
