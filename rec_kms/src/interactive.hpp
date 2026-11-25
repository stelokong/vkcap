#pragma once

#include "cli.hpp"
#include "user_config.hpp"

#include <expected>
#include <filesystem>
#include <istream>
#include <ostream>

namespace vkcap {

// Runs the interactive setup prompt on `in`/`out`.
// If saved personal defaults exist at `defaults_path`, prompts whether to start immediately
// with those saved defaults; otherwise (or if declined), offers Quick Setup (profiles)
// or Advanced Setup (every parameter), followed by an optional prompt to save defaults.
// Returns true if confirmed and `cfg` is ready to record, false if cancelled (q / EOF).
std::expected<bool, std::string> run_interactive_setup(
    Config &cfg,
    const char *prog_name,
    std::istream &in,
    std::ostream &out,
    const std::filesystem::path &defaults_path = user_defaults_path());

}  // namespace vkcap
