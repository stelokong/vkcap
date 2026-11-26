#pragma once

#include "cli.hpp"

#include <expected>
#include <filesystem>
#include <optional>
#include <string>

namespace vkcap {

/// Returns the user defaults config path ($XDG_CONFIG_HOME/vkcap/defaults.conf
/// or $HOME/.config/vkcap/defaults.conf).
std::filesystem::path user_defaults_path();

/// Loads saved user defaults from `path` (if it exists).
/// Returns std::nullopt if the file does not exist.
/// Output path is automatically populated with a fresh timestamped filename using
/// the saved container extension so previous recordings are never overwritten.
std::expected<std::optional<Config>, std::string> load_user_defaults(
    const std::filesystem::path &path = user_defaults_path());

/// Saves `cfg` as the user's personal defaults at `path` (creates parent directories).
/// Never modifies or overwrites any built-in profile.
std::expected<void, std::string> save_user_defaults(
    const Config &cfg,
    const std::filesystem::path &path = user_defaults_path());

/// Formats a compact, readable multi-line summary of `cfg` for display in the interactive prompt.
std::string format_config_summary(const Config &cfg);

}  // namespace vkcap
