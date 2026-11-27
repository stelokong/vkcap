#pragma once

#include "cli.hpp"
#include "types.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vkcap {

class KmsCapture {
public:
    KmsCapture() = default;
    ~KmsCapture();

    KmsCapture(const KmsCapture &) = delete;
    KmsCapture &operator=(const KmsCapture &) = delete;

    /// Enumerates all connected active DRM/KMS displays across /dev/dri/card*.
    /// Does NOT require CAP_SYS_ADMIN.
    static std::expected<std::vector<KmsDisplayInfo>, std::string> enumerate_displays();

    /// Opens the selected display (by connector name, index, or auto-first) for zero-copy KMS grab.
    static std::expected<std::unique_ptr<KmsCapture>, std::string> open(const Config &cfg);

    /// Grabs the currently scanned-out primary plane DMA-BUF (and optional hardware cursor plane)
    /// using minimal scoped CAP_SYS_ADMIN only for drmModeGetFB2 + drmPrimeHandleToFD.
    std::expected<KmsFrame, std::string> grab_frame();

    /// Same as grab_frame(), but if `vblank_seq > 0` matches the previous grab's CRTC vblank
    /// sequence AND primary fb_id + cursor state are identical, sets `out_unchanged = true`
    /// and skips all CAP_SYS_ADMIN / GETFB2 / PRIME ioctls (zero-overhead duplicate detection).
    std::expected<KmsFrame, std::string> grab_frame_dedup(uint64_t vblank_seq, bool &out_unchanged);

    struct VblankInfo {
        uint64_t sequence = 0;
        int64_t ns = 0;
    };

    /// Non-blocking query of the last CRTC vblank sequence + timestamp in CLOCK_MONOTONIC ns.
    /// Does NOT require CAP_SYS_ADMIN and never blocks (safe with VRR / PSR displays).
    std::optional<VblankInfo> query_vblank() const noexcept;

    /// Convenience wrapper returning only the last CRTC vblank timestamp in CLOCK_MONOTONIC ns.
    std::optional<int64_t> query_vblank_ns() const noexcept;

    /// Non-blocking query of the current CRTC mode refresh rate (for mid-recording modeset changes).
    std::optional<double> query_refresh_hz() const noexcept;

    const KmsDisplayInfo &display() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vkcap
