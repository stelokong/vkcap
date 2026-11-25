#pragma once

#include "cli.hpp"

#include <cstdint>
#include <expected>
#include <optional>
#include <string>

namespace vkcap {

struct PortalStreamInfo {
    uint32_t node_id = 0;
    std::optional<uint64_t> pipewire_serial;
    int32_t width = 0;
    int32_t height = 0;
    int pipewire_fd = -1;
    std::optional<std::string> restore_token;
};

class PortalSession {
public:
    PortalSession() = default;
    ~PortalSession();

    PortalSession(const PortalSession &) = delete;
    PortalSession &operator=(const PortalSession &) = delete;
    PortalSession(PortalSession &&other) noexcept;
    PortalSession &operator=(PortalSession &&other) noexcept;

    /// Interactively prompts the user via org.freedesktop.portal.ScreenCast
    /// (CreateSession -> SelectSources -> Start -> OpenPipeWireRemote).
    static std::expected<PortalSession, std::string> acquire(const Config &cfg);

    const PortalStreamInfo &stream() const noexcept { return stream_; }

    /// Transfers ownership of the PipeWire remote fd to the caller (-1 if already taken).
    int take_pipewire_fd() noexcept {
        int fd = stream_.pipewire_fd;
        stream_.pipewire_fd = -1;
        return fd;
    }

    /// Closes the portal session and PipeWire remote fd if still open.
    void close();

private:
    struct Impl;
    Impl *impl_ = nullptr;
    PortalStreamInfo stream_{};
};

std::optional<std::string> load_saved_restore_token();
bool save_restore_token(const std::string &token);

}  // namespace vkcap
