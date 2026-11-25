#pragma once

#include "cli.hpp"

#include <atomic>
#include <expected>

namespace vkcap {

/// Runs the interactive portal display selection and full zero-copy
/// PipeWire DMA-BUF -> Vulkan Compute -> Vulkan Video AV1 -> libavformat recording pipeline
/// until stop_flag becomes true or duration expires.
std::expected<void, std::string> run_capture_pipeline(const Config &cfg, std::atomic<bool> &stop_flag);

}  // namespace vkcap
