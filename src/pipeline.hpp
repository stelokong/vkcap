#pragma once

#include "cli.hpp"

#include <atomic>
#include <expected>
#include <string>

namespace vkcap {

struct PipelineSignals {
    std::atomic<bool> stop{false};
    std::atomic<bool> paused{false};
    std::atomic<bool> save_replay{false};
};

std::expected<void, std::string> run_kms_pipeline(const Config &cfg, PipelineSignals &signals);

}  // namespace vkcap
