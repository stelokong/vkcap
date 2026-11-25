#pragma once

#include "cli.hpp"

#include <atomic>
#include <expected>
#include <string>

namespace vkcap {

std::expected<void, std::string> run_kms_pipeline(const Config &cfg, std::atomic<bool> &stop_flag);

}  // namespace vkcap
