#pragma once

#include "mm_rec/training/trainer.h" // For TrainingBatch
#include "mm_rec/data/dataset.h"
#include <memory>

namespace mm_rec {

/**
 * Interface for Data Loader
 * Abstracts the stream of training batches.
 */
class IDataLoader {
public:
    virtual ~IDataLoader() = default;

    /**
     * Get next batch. Blocks if queue is empty.
     * @param batch Output batch
     * @return false if epoch finished or stopped
     */
    virtual bool next(TrainingBatch& batch) = 0;

    /**
     * Reset cursor to start (for new epoch)
     */
    virtual void reset() = 0;

    /**
     * Get total number of batches in an epoch
     */
    virtual int64_t total_batches() const = 0;
};

/**
 * Interface for Data Loader Factory
 * Needed because DataLoader requires runtime parameters (batch size, path)
 * that are not available at DI container configuration time.
 */
class IDataLoaderFactory {
public:
    virtual ~IDataLoaderFactory() = default;

    /**
     * Create a new DataLoader instance
     */
    virtual std::unique_ptr<IDataLoader> create_loader(
        std::shared_ptr<Dataset> dataset,
        int64_t batch_size,
        int64_t seq_len,
        bool shuffle = false,
        int num_workers = 1
    ) = 0;
};

} // namespace mm_rec
