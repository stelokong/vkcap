#pragma once

#include "cli.hpp"
#include "portal.hpp"
#include "types.hpp"

#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vkcap {

struct CaptureCallbacks {
    std::function<void(const NegotiatedFormat &)> on_format_changed;
    std::function<void(void *buffer_handle)> on_buffer_removed;
    /// Called synchronously on the PipeWire loop thread with the freshest buffer.
    /// The consumer should finish reading the buffer (e.g. fast GPU compute into its own
    /// AVVkFrame) and return; PipeWireCapture immediately re-queues the pw_buffer upon
    /// return so the compositor is never starved of buffers.
    std::function<void(const CapturedFrame &)> on_frame;
    std::function<void(const std::string &err)> on_error;
};

class PipeWireCapture {
public:
    PipeWireCapture() = default;
    ~PipeWireCapture();

    PipeWireCapture(const PipeWireCapture &) = delete;
    PipeWireCapture &operator=(const PipeWireCapture &) = delete;

    static std::expected<std::unique_ptr<PipeWireCapture>, std::string> start(
        const Config &cfg,
        int pipewire_fd,
        const PortalStreamInfo &stream_info,
        std::vector<FormatModifierSet> supported_formats,
        CaptureCallbacks callbacks);

    void stop();
    bool is_streaming() const noexcept;

    static uint32_t spa_to_drm_fourcc(uint32_t spa_fmt) noexcept;
    static const char *spa_format_name(uint32_t spa_fmt) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vkcap
