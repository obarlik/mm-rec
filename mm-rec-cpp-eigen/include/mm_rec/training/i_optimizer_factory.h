#pragma once

#include "mm_rec/training/optimizer.h"
#include "mm_rec/training/optimizer.h"
#include "mm_rec/training/training_config.h" // Correct Include
#include <memory>
#include <memory>
#include <string>

namespace mm_rec {

/**
 * Interface for creating optimizers.
 * Allows decoupling Trainer from specific Optimizer implementations.
 */
class IOptimizerFactory {
public:
    virtual ~IOptimizerFactory() = default;

    virtual std::unique_ptr<Optimizer> create_optimizer(const TrainingConfig& config) = 0;
};

} // namespace mm_rec
