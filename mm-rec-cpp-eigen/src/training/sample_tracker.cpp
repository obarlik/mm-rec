#include "mm_rec/training/sample_tracker.h"
#include <fstream>
#include <iostream>

namespace mm_rec {

bool SampleTracker::save(const std::string& filepath) const {
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to open " << filepath << " for writing" << std::endl;
        return false;
    }
    
    // Write thresholds
    ofs.write(reinterpret_cast<const char*>(&easy_threshold_), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&hard_threshold_), sizeof(float));
    
    // Write sample count
    size_t count = samples_.size();
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(size_t));
    
    // Write samples
    for (const auto& sample : samples_) {
        ofs.write(reinterpret_cast<const char*>(&sample.batch_idx), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&sample.perplexity), sizeof(float));
        int cat = static_cast<int>(sample.category);
        ofs.write(reinterpret_cast<const char*>(&cat), sizeof(int));
    }
    
    return true;
}

bool SampleTracker::load(const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open " << filepath << " for reading" << std::endl;
        return false;
    }
    
    // Read thresholds
    ifs.read(reinterpret_cast<char*>(&easy_threshold_), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&hard_threshold_), sizeof(float));
    
    // Read sample count
    size_t count;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(size_t));
    
    // Read samples
    samples_.clear();
    samples_.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        SampleDifficulty sample;
        ifs.read(reinterpret_cast<char*>(&sample.batch_idx), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&sample.perplexity), sizeof(float));
        int cat;
        ifs.read(reinterpret_cast<char*>(&cat), sizeof(int));
        sample.category = static_cast<DifficultyCategory>(cat);
        samples_.push_back(sample);
    }
    
    return true;
}

} // namespace mm_rec
