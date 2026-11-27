#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <unistd.h>

namespace vkcap {

/// Shared mutex to serialize terminal status/log output across capture, encode, and replay-save threads.
inline std::mutex &stderr_mutex() noexcept {
    static std::mutex mtx;
    return mtx;
}

struct DmaBufPlane {
    int fd = -1;
    uint32_t offset = 0;
    uint32_t stride = 0;
    uint32_t size = 0;
};

struct CropRect {
    int32_t x = 0;
    int32_t y = 0;
    int32_t width = 0;
    int32_t height = 0;
    bool valid() const noexcept { return width > 0 && height > 0; }
};

struct KmsSurface {
    uint64_t inode_key = 0;       // Unique kernel dma_buf inode (fstat st_ino)
    uint32_t fb_id = 0;
    uint32_t drm_fourcc = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t drm_modifier = 0;
    bool has_modifier = false;
    bool owns_fds = true;         // False for borrowed cached FDs (e.g. cursor cache hit)
    uint32_t num_planes = 0;
    std::array<DmaBufPlane, 4> planes{};

    void close_fds() noexcept {
        if (!owns_fds) {
            for (uint32_t i = 0; i < num_planes; ++i) planes[i].fd = -1;
            return;
        }
        for (uint32_t i = 0; i < num_planes; ++i) {
            if (planes[i].fd >= 0) {
                int fd = planes[i].fd;
                // Avoid double-closing if multiple planes share the same fd
                for (uint32_t j = i; j < num_planes; ++j) {
                    if (planes[j].fd == fd) planes[j].fd = -1;
                }
                ::close(fd);
            }
        }
    }
};

struct KmsCursor {
    bool visible = false;
    int32_t x = 0;
    int32_t y = 0;
    int32_t width = 0;
    int32_t height = 0;
    KmsSurface surface{};
};

struct KmsFrame {
    KmsSurface primary{};
    CropRect plane_src{};  // Optional DRM primary plane SRC_X/Y/W/H sub-rectangle (e.g. unified multi-monitor FB)
    CropRect crop{};
    KmsCursor cursor{};

    KmsFrame() = default;
    ~KmsFrame() {
        primary.close_fds();
        cursor.surface.close_fds();
    }
    KmsFrame(const KmsFrame &) = delete;
    KmsFrame &operator=(const KmsFrame &) = delete;
    KmsFrame(KmsFrame &&other) noexcept
        : primary(other.primary), plane_src(other.plane_src), crop(other.crop), cursor(other.cursor) {
        for (auto &p : other.primary.planes) p.fd = -1;
        for (auto &p : other.cursor.surface.planes) p.fd = -1;
        other.primary.owns_fds = false;
        other.cursor.surface.owns_fds = false;
    }
    KmsFrame &operator=(KmsFrame &&other) noexcept {
        if (this != &other) {
            primary.close_fds();
            cursor.surface.close_fds();
            primary = other.primary;
            plane_src = other.plane_src;
            crop = other.crop;
            cursor = other.cursor;
            for (auto &p : other.primary.planes) p.fd = -1;
            for (auto &p : other.cursor.surface.planes) p.fd = -1;
            other.primary.owns_fds = false;
            other.cursor.surface.owns_fds = false;
        }
        return *this;
    }
};

struct ColorPrimaries {
    double rx = 0.0, ry = 0.0;
    double gx = 0.0, gy = 0.0;
    double bx = 0.0, by = 0.0;
    double wx = 0.0, wy = 0.0;
    bool valid = false;
};

struct KmsDisplayInfo {
    std::string card_path;
    std::string connector_name;
    uint32_t connector_id = 0;
    uint32_t crtc_id = 0;
    uint32_t crtc_index = 0;
    uint32_t primary_plane_id = 0;
    uint32_t cursor_plane_id = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    double refresh_hz = 60.0;
    ColorPrimaries edid_primaries{};
};

}  // namespace vkcap
