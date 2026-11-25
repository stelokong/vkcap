#pragma once

#include "cli.hpp"
#include "types.hpp"

#include <expected>
#include <memory>
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

    const KmsDisplayInfo &display() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vkcap
