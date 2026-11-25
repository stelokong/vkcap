#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>

namespace vkcap {

enum class CaptureMemoryType {
    DmaBuf,
    Shm,
};

struct FramePlane {
    int fd = -1;            // Owned by pw_buffer (do not close)
    uint32_t offset = 0;
    int32_t stride = 0;
    uint32_t size = 0;
    const void *data = nullptr;  // Non-null for MemPtr/MemFd if already mapped
};

struct CropRect {
    int32_t x = 0;
    int32_t y = 0;
    int32_t width = 0;
    int32_t height = 0;
    bool valid() const noexcept { return width > 0 && height > 0; }
};

struct NegotiatedFormat {
    uint32_t spa_format = 0;
    uint32_t drm_fourcc = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t drm_modifier = 0;
    bool has_modifier = false;
    uint32_t fps_num = 60;
    uint32_t fps_den = 1;
};

struct CapturedFrame {
    void *buffer_handle = nullptr;  // Opaque pw_buffer* key for Vulkan image cache
    CaptureMemoryType mem_type = CaptureMemoryType::DmaBuf;
    NegotiatedFormat format{};
    uint32_t num_planes = 0;
    std::array<FramePlane, 4> planes{};
    CropRect crop{};
    int64_t pts_ns = 0;
    uint64_t sequence = 0;
    std::function<void()> release;  // Returns buffer back to PipeWire stream
};

struct FormatModifierSet {
    uint32_t spa_format = 0;
    uint32_t drm_fourcc = 0;
    std::vector<uint64_t> modifiers;
    bool allow_modifierless = true;
};

}  // namespace vkcap
