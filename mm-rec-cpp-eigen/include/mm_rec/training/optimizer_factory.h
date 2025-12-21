#pragma once

#include "mm_rec/training/i_optimizer_factory.h"

namespace mm_rec {

/**
 * Default implementation of IOptimizerFactory.
 * Creates SGD, Adam, AdamW, or Flux optimizers based on config.
 */
class OptimizerFactory : public IOptimizerFactory {
public:
    std::unique_ptr<Optimizer> create_optimizer(const TrainingConfig& config) override;
};

} // namespace mm_rec
