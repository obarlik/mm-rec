#include "mm_rec/training/optimizer_factory.h"
#include "mm_rec/infrastructure/logger.h"

namespace mm_rec {

std::unique_ptr<Optimizer> OptimizerFactory::create_optimizer(const TrainingConfig& config) {
    if (config.optimizer_type == "flux") {
        LOG_INFO("Creating Flux Optimizer (Adaptive Complexity Scaling)");
        return std::make_unique<Flux>(config.learning_rate, 0.9f, 0.999f, 1e-8f, config.weight_decay);
    } else if (config.optimizer_type == "adamw") {
        LOG_INFO("Creating AdamW Optimizer (LR=" + std::to_string(config.learning_rate) + ", WD=" + std::to_string(config.weight_decay) + ")");
        return std::make_unique<AdamW>(config.learning_rate, 0.9f, 0.999f, 1e-8f, config.weight_decay);
    } else if (config.optimizer_type == "adam") {
        LOG_INFO("Creating Adam Optimizer (LR=" + std::to_string(config.learning_rate) + ")");
        return std::make_unique<Adam>(config.learning_rate);
    } else {
        LOG_INFO("Creating SGD Optimizer (LR=" + std::to_string(config.learning_rate) + ")");
        return std::make_unique<SGD>(config.learning_rate);
    }
}

} // namespace mm_rec
