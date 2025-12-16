/**
 * Sample Difficulty Tracker for Adaptive Curriculum Learning
 * 
 * Tracks training sample difficulty based on perplexity and manages
 * progressive filtering for efficient curriculum-based training.
 */

#pragma once

#include <vector>
#include <string>
#include <algorithm>

namespace mm_rec {

/**
 * Difficulty category for a training sample
 */
enum class DifficultyCategory {
    EASY,    // PPL < easy_threshold - skip in future epochs
    MEDIUM,  // easy_threshold <= PPL < hard_threshold - train actively
    HARD     // PPL >= hard_threshold - defer to later epochs
};

/**
 * Difficulty record for a single batch
 */
struct SampleDifficulty {
    int batch_idx;
    float perplexity;
    DifficultyCategory category;
    
    SampleDifficulty() : batch_idx(-1), perplexity(0.0f), category(DifficultyCategory::MEDIUM) {}
    SampleDifficulty(int idx, float ppl, DifficultyCategory cat) 
        : batch_idx(idx), perplexity(ppl), category(cat) {}
};

/**
 * Tracks and manages sample difficulties across epochs
 */
class SampleTracker {
public:
    SampleTracker(float easy_thresh = 50.0f, float hard_thresh = 500.0f)
        : easy_threshold_(easy_thresh), hard_threshold_(hard_thresh) {}
    
    /**
     * Categorize a sample based on its perplexity
     */
    DifficultyCategory categorize(float perplexity) const {
        if (perplexity < easy_threshold_) return DifficultyCategory::EASY;
        if (perplexity < hard_threshold_) return DifficultyCategory::MEDIUM;
        return DifficultyCategory::HARD;
    }
    
    /**
     * Add a sample with its difficulty
     */
    void add_sample(int batch_idx, float perplexity) {
        DifficultyCategory cat = categorize(perplexity);
        samples_.push_back(SampleDifficulty(batch_idx, perplexity, cat));
    }
    
    /**
     * Get indices of samples in a specific category
     */
    std::vector<int> get_batches_by_category(DifficultyCategory category) const {
        std::vector<int> indices;
        for (const auto& sample : samples_) {
            if (sample.category == category) {
                indices.push_back(sample.batch_idx);
            }
        }
        return indices;
    }
    
    /**
     * Get trainable batch indices (MEDIUM difficulty)
     */
    std::vector<int> get_trainable_batches() const {
        return get_batches_by_category(DifficultyCategory::MEDIUM);
    }
    
    /**
     * Get statistics about sample distribution
     */
    void get_statistics(int& easy_count, int& medium_count, int& hard_count) const {
        easy_count = medium_count = hard_count = 0;
        for (const auto& sample : samples_) {
            switch (sample.category) {
                case DifficultyCategory::EASY:   easy_count++; break;
               case DifficultyCategory::MEDIUM: medium_count++; break;
                case DifficultyCategory::HARD:   hard_count++; break;
            }
        }
    }
    
    /**
     * Update a sample's category (for re-evaluation)
     */
    void update_sample(int batch_idx, float new_perplexity) {
        for (auto& sample : samples_) {
            if (sample.batch_idx == batch_idx) {
                sample.perplexity = new_perplexity;
                sample.category = categorize(new_perplexity);
                break;
            }
        }
    }
    
    /**
     * Save difficulty map to file
     */
    bool save(const std::string& filepath) const;
    
    /**
     * Load difficulty map from file
     */
    bool load(const std::string& filepath);
    
    size_t size() const { return samples_.size(); }
    
private:
    float easy_threshold_;
    float hard_threshold_;
    std::vector<SampleDifficulty> samples_;
};

} // namespace mm_rec
