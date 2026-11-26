#include "kms_capture.hpp"
#include "privileges.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <format>
#include <iostream>
#include <libdrm/drm.h>
#include <libdrm/drm_fourcc.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

namespace vkcap {

namespace {

struct PlaneProps {
    uint64_t type = 0;
    uint32_t crtc_id = 0;
    uint32_t fb_id = 0;
    int32_t crtc_x = 0;
    int32_t crtc_y = 0;
    uint32_t crtc_w = 0;
    uint32_t crtc_h = 0;
    uint32_t src_x = 0;  // 16.16 fixed point
    uint32_t src_y = 0;
    uint32_t src_w = 0;
    uint32_t src_h = 0;
};

PlaneProps read_plane_props(int fd, uint32_t plane_id) {
    PlaneProps out{};
    drmModeObjectPropertiesPtr props = drmModeObjectGetProperties(fd, plane_id, DRM_MODE_OBJECT_PLANE);
    if (!props) return out;
    for (uint32_t i = 0; i < props->count_props; ++i) {
        drmModePropertyPtr p = drmModeGetProperty(fd, props->props[i]);
        if (!p) continue;
        const uint64_t v = props->prop_values[i];
        if (std::strcmp(p->name, "type") == 0) out.type = v;
        else if (std::strcmp(p->name, "CRTC_ID") == 0) out.crtc_id = static_cast<uint32_t>(v);
        else if (std::strcmp(p->name, "FB_ID") == 0) out.fb_id = static_cast<uint32_t>(v);
        else if (std::strcmp(p->name, "CRTC_X") == 0) out.crtc_x = static_cast<int32_t>(v);
        else if (std::strcmp(p->name, "CRTC_Y") == 0) out.crtc_y = static_cast<int32_t>(v);
        else if (std::strcmp(p->name, "CRTC_W") == 0) out.crtc_w = static_cast<uint32_t>(v);
        else if (std::strcmp(p->name, "CRTC_H") == 0) out.crtc_h = static_cast<uint32_t>(v);
        else if (std::strcmp(p->name, "SRC_X") == 0) out.src_x = static_cast<uint32_t>(v);
        else if (std::strcmp(p->name, "SRC_Y") == 0) out.src_y = static_cast<uint32_t>(v);
        else if (std::strcmp(p->name, "SRC_W") == 0) out.src_w = static_cast<uint32_t>(v);
        else if (std::strcmp(p->name, "SRC_H") == 0) out.src_h = static_cast<uint32_t>(v);
        drmModeFreeProperty(p);
    }
    drmModeFreeObjectProperties(props);
    return out;
}

void close_gem_handles(int drm_fd, const uint32_t handles[4]) noexcept {
    for (int i = 0; i < 4; ++i) {
        if (handles[i] == 0) continue;
        bool dup = false;
        for (int j = 0; j < i; ++j) {
            if (handles[j] == handles[i]) { dup = true; break; }
        }
        if (dup) continue;
        drm_gem_close cl{};
        cl.handle = handles[i];
        ::ioctl(drm_fd, DRM_IOCTL_GEM_CLOSE, &cl);
    }
}

// Fast zero-allocation direct-ioctl export of a DRM framebuffer to KmsSurface.
// Caller must hold ScopedCapSysAdmin for the duration of this call.
std::expected<KmsSurface, std::string> export_fb_to_surface_locked(int drm_fd, uint32_t fb_id) {
    drm_mode_fb_cmd2 fb2{};
    fb2.fb_id = fb_id;
    if (::ioctl(drm_fd, DRM_IOCTL_MODE_GETFB2, &fb2) != 0) {
        int e = errno;
        if (e == EACCES || e == EPERM) {
            return std::unexpected(
                "drmModeGetFB2 denied (CAP_SYS_ADMIN required). Run once:\n"
                "  sudo setcap cap_sys_admin=p ./build/vkcap");
        }
        return std::unexpected(std::format("DRM_IOCTL_MODE_GETFB2(fb={}) failed: {}", fb_id, std::strerror(e)));
    }

    if (fb2.handles[0] == 0) {
        return std::unexpected(
            "drmModeGetFB2 returned null GEM handle (CAP_SYS_ADMIN required). Run once:\n"
            "  sudo setcap cap_sys_admin=p ./build/vkcap");
    }

    KmsSurface surf{};
    surf.fb_id = fb_id;
    surf.drm_fourcc = fb2.pixel_format;
    surf.width = fb2.width;
    surf.height = fb2.height;
    surf.has_modifier = (fb2.flags & DRM_MODE_FB_MODIFIERS) != 0;
    surf.drm_modifier = surf.has_modifier ? fb2.modifier[0] : DRM_FORMAT_MOD_INVALID;

    uint32_t num_planes = 0;
    for (int i = 0; i < 4; ++i) {
        if (fb2.handles[i] != 0) num_planes = static_cast<uint32_t>(i + 1);
    }
    surf.num_planes = num_planes;

    int export_err = 0;
    for (uint32_t i = 0; i < num_planes; ++i) {
        int shared_fd = -1;
        for (uint32_t j = 0; j < i; ++j) {
            if (fb2.handles[j] == fb2.handles[i]) {
                shared_fd = surf.planes[j].fd;
                break;
            }
        }
        if (shared_fd >= 0) {
            surf.planes[i].fd = shared_fd;
        } else {
            drm_prime_handle prime{};
            prime.handle = fb2.handles[i];
            prime.flags = DRM_CLOEXEC;
            prime.fd = -1;
            if (::ioctl(drm_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime) != 0 || prime.fd < 0) {
                export_err = errno;
                break;
            }
            surf.planes[i].fd = prime.fd;
        }
        surf.planes[i].offset = fb2.offsets[i];
        surf.planes[i].stride = fb2.pitches[i];
        surf.planes[i].size = fb2.pitches[i] * fb2.height;
    }

    close_gem_handles(drm_fd, fb2.handles);

    if (export_err != 0) {
        surf.close_fds();
        return std::unexpected(std::format("DRM_IOCTL_PRIME_HANDLE_TO_FD(fb={}) failed: {}",
                                           fb_id, std::strerror(export_err)));
    }

    struct stat st{};
    if (::fstat(surf.planes[0].fd, &st) == 0) {
        surf.inode_key = static_cast<uint64_t>(st.st_ino);
    } else {
        surf.inode_key = static_cast<uint64_t>(fb_id) << 32;
    }

    return surf;
}

ColorPrimaries read_connector_edid_primaries(int drm_fd, uint32_t connector_id) {
    ColorPrimaries out{};
    drmModeObjectPropertiesPtr props = drmModeObjectGetProperties(drm_fd, connector_id, DRM_MODE_OBJECT_CONNECTOR);
    if (!props) return out;

    uint64_t edid_blob_id = 0;
    for (uint32_t i = 0; i < props->count_props; ++i) {
        drmModePropertyPtr p = drmModeGetProperty(drm_fd, props->props[i]);
        if (!p) continue;
        if (std::strcmp(p->name, "EDID") == 0) {
            edid_blob_id = props->prop_values[i];
            drmModeFreeProperty(p);
            break;
        }
        drmModeFreeProperty(p);
    }
    drmModeFreeObjectProperties(props);

    if (edid_blob_id == 0) return out;
    drmModePropertyBlobPtr blob = drmModeGetPropertyBlob(drm_fd, static_cast<uint32_t>(edid_blob_id));
    if (!blob) return out;

    if (blob->length >= 128 && blob->data != nullptr) {
        const auto *d = static_cast<const uint8_t *>(blob->data);
        // Verify EDID header 00 FF FF FF FF FF FF 00
        if (d[0] == 0x00 && d[1] == 0xFF && d[6] == 0xFF && d[7] == 0x00) {
            const uint8_t rg_lsb = d[25];
            const uint8_t bw_lsb = d[26];
            out.rx = static_cast<double>((static_cast<uint32_t>(d[27]) << 2) | ((rg_lsb >> 6) & 3u)) / 1024.0;
            out.ry = static_cast<double>((static_cast<uint32_t>(d[28]) << 2) | ((rg_lsb >> 4) & 3u)) / 1024.0;
            out.gx = static_cast<double>((static_cast<uint32_t>(d[29]) << 2) | ((rg_lsb >> 2) & 3u)) / 1024.0;
            out.gy = static_cast<double>((static_cast<uint32_t>(d[30]) << 2) | ((rg_lsb >> 0) & 3u)) / 1024.0;
            out.bx = static_cast<double>((static_cast<uint32_t>(d[31]) << 2) | ((bw_lsb >> 6) & 3u)) / 1024.0;
            out.by = static_cast<double>((static_cast<uint32_t>(d[32]) << 2) | ((bw_lsb >> 4) & 3u)) / 1024.0;
            out.wx = static_cast<double>((static_cast<uint32_t>(d[33]) << 2) | ((bw_lsb >> 2) & 3u)) / 1024.0;
            out.wy = static_cast<double>((static_cast<uint32_t>(d[34]) << 2) | ((bw_lsb >> 0) & 3u)) / 1024.0;
            if (out.rx > 0.4 && out.ry > 0.2 && out.gx > 0.1 && out.gy > 0.4 &&
                out.bx > 0.05 && out.by > 0.01 && out.wx > 0.2 && out.wy > 0.2) {
                out.valid = true;
            }
        }
    }
    drmModeFreePropertyBlob(blob);
    return out;
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

struct KmsCapture::Impl {
    Config cfg;
    KmsDisplayInfo display{};
    int drm_fd = -1;

    // Cached atomic property IDs for cursor plane so per-frame cursor queries are minimal
    uint32_t prop_cursor_fb = 0;
    uint32_t prop_cursor_x = 0;
    uint32_t prop_cursor_y = 0;
    uint32_t prop_cursor_w = 0;
    uint32_t prop_cursor_h = 0;

    ~Impl() {
        if (drm_fd >= 0) {
            ::close(drm_fd);
            drm_fd = -1;
        }
    }
};

KmsCapture::~KmsCapture() = default;

std::expected<std::vector<KmsDisplayInfo>, std::string> KmsCapture::enumerate_displays() {
    std::vector<std::string> card_paths;
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator("/dev/dri", ec)) {
        std::string name = entry.path().filename().string();
        if (name.starts_with("card") && name.size() > 4 && std::isdigit(static_cast<unsigned char>(name[4]))) {
            card_paths.push_back(entry.path().string());
        }
    }
    std::sort(card_paths.begin(), card_paths.end());

    std::vector<KmsDisplayInfo> displays;
    for (const auto &path : card_paths) {
        int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) continue;

        drmSetClientCap(fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
        drmSetClientCap(fd, DRM_CLIENT_CAP_ATOMIC, 1);

        drmModeResPtr res = drmModeGetResources(fd);
        if (!res) {
            ::close(fd);
            continue;
        }

        drmModePlaneResPtr pres = drmModeGetPlaneResources(fd);

        for (int ci = 0; ci < res->count_connectors; ++ci) {
            drmModeConnectorPtr conn = drmModeGetConnector(fd, res->connectors[ci]);
            if (!conn) continue;

            if (conn->connection == DRM_MODE_CONNECTED && conn->encoder_id != 0) {
                drmModeEncoderPtr enc = drmModeGetEncoder(fd, conn->encoder_id);
                if (enc && enc->crtc_id != 0) {
                    drmModeCrtcPtr crtc = drmModeGetCrtc(fd, enc->crtc_id);
                    if (crtc && crtc->mode_valid) {
                        uint32_t crtc_idx = 0;
                        for (int k = 0; k < res->count_crtcs; ++k) {
                            if (res->crtcs[k] == crtc->crtc_id) {
                                crtc_idx = static_cast<uint32_t>(k);
                                break;
                            }
                        }

                        KmsDisplayInfo info{};
                        info.card_path = path;
                        const char *tname = drmModeGetConnectorTypeName(conn->connector_type);
                        info.connector_name = std::format("{}-{}", tname ? tname : "Unknown", conn->connector_type_id);
                        info.connector_id = conn->connector_id;
                        info.crtc_id = crtc->crtc_id;
                        info.crtc_index = crtc_idx;
                        info.width = crtc->mode.hdisplay;
                        info.height = crtc->mode.vdisplay;
                        if (crtc->mode.htotal > 0 && crtc->mode.vtotal > 0) {
                            info.refresh_hz = (static_cast<double>(crtc->mode.clock) * 1000.0) /
                                              (static_cast<double>(crtc->mode.htotal) * crtc->mode.vtotal);
                        }
                        info.edid_primaries = read_connector_edid_primaries(fd, conn->connector_id);

                        if (pres) {
                            for (uint32_t pi = 0; pi < pres->count_planes; ++pi) {
                                drmModePlanePtr pl = drmModeGetPlane(fd, pres->planes[pi]);
                                if (!pl) continue;
                                if ((pl->possible_crtcs & (1u << crtc_idx)) != 0) {
                                    PlaneProps pp = read_plane_props(fd, pl->plane_id);
                                    if (pp.type == DRM_PLANE_TYPE_PRIMARY) {
                                        if (pl->crtc_id == crtc->crtc_id || info.primary_plane_id == 0) {
                                            info.primary_plane_id = pl->plane_id;
                                        }
                                    } else if (pp.type == DRM_PLANE_TYPE_CURSOR) {
                                        if (pl->crtc_id == crtc->crtc_id || info.cursor_plane_id == 0) {
                                            info.cursor_plane_id = pl->plane_id;
                                        }
                                    }
                                }
                                drmModeFreePlane(pl);
                            }
                        }
                        displays.push_back(std::move(info));
                    }
                    if (crtc) drmModeFreeCrtc(crtc);
                }
                if (enc) drmModeFreeEncoder(enc);
            }
            drmModeFreeConnector(conn);
        }

        if (pres) drmModeFreePlaneResources(pres);
        drmModeFreeResources(res);
        ::close(fd);
    }

    return displays;
}

std::expected<std::unique_ptr<KmsCapture>, std::string> KmsCapture::open(const Config &cfg) {
    auto disps_res = enumerate_displays();
    if (!disps_res) return std::unexpected(disps_res.error());
    const auto &disps = *disps_res;
    if (disps.empty()) {
        return std::unexpected("No connected active DRM/KMS displays found in /dev/dri/card*");
    }

    size_t selected = 0;
    if (!cfg.display.empty() && to_lower(cfg.display) != "auto") {
        bool found = false;
        size_t idx = 0;
        auto [p, ec] = std::from_chars(cfg.display.data(), cfg.display.data() + cfg.display.size(), idx);
        if (ec == std::errc() && p == cfg.display.data() + cfg.display.size() && idx < disps.size()) {
            selected = idx;
            found = true;
        } else {
            std::string want = to_lower(cfg.display);
            for (size_t i = 0; i < disps.size(); ++i) {
                if (to_lower(disps[i].connector_name) == want) {
                    selected = i;
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            return std::unexpected(std::format("Display '{}' not found. Run with --list-displays.", cfg.display));
        }
    }

    auto cap = std::unique_ptr<KmsCapture>(new KmsCapture());
    cap->impl_ = std::make_unique<Impl>();
    Impl *impl = cap->impl_.get();
    impl->cfg = cfg;
    impl->display = disps[selected];

    impl->drm_fd = ::open(impl->display.card_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (impl->drm_fd < 0) {
        return std::unexpected(std::format("Failed to open '{}': {}", impl->display.card_path, std::strerror(errno)));
    }
    drmSetClientCap(impl->drm_fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
    drmSetClientCap(impl->drm_fd, DRM_CLIENT_CAP_ATOMIC, 1);

    if (impl->display.cursor_plane_id != 0) {
        drmModeObjectPropertiesPtr props = drmModeObjectGetProperties(
            impl->drm_fd, impl->display.cursor_plane_id, DRM_MODE_OBJECT_PLANE);
        if (props) {
            for (uint32_t i = 0; i < props->count_props; ++i) {
                drmModePropertyPtr p = drmModeGetProperty(impl->drm_fd, props->props[i]);
                if (!p) continue;
                if (std::strcmp(p->name, "FB_ID") == 0) impl->prop_cursor_fb = props->props[i];
                else if (std::strcmp(p->name, "CRTC_X") == 0) impl->prop_cursor_x = props->props[i];
                else if (std::strcmp(p->name, "CRTC_Y") == 0) impl->prop_cursor_y = props->props[i];
                else if (std::strcmp(p->name, "CRTC_W") == 0) impl->prop_cursor_w = props->props[i];
                else if (std::strcmp(p->name, "CRTC_H") == 0) impl->prop_cursor_h = props->props[i];
                drmModeFreeProperty(p);
            }
            drmModeFreeObjectProperties(props);
        }
    }

    if (cfg.verbose) {
        std::cerr << std::format(
            "[kms] opened {} ({} {}x{}@{:.1f}Hz, crtc={}, primary_plane={}, cursor_plane={})\n",
            impl->display.card_path, impl->display.connector_name,
            impl->display.width, impl->display.height, impl->display.refresh_hz,
            impl->display.crtc_id, impl->display.primary_plane_id, impl->display.cursor_plane_id);
    }

    return cap;
}

std::expected<KmsFrame, std::string> KmsCapture::grab_frame() {
    // 1. Query primary plane FB_ID via direct stack ioctl (0 heap allocations, 1 ioctl)
    uint32_t fb_id = 0;
    if (impl_->display.primary_plane_id != 0) {
        drm_mode_get_plane pl{};
        pl.plane_id = impl_->display.primary_plane_id;
        if (::ioctl(impl_->drm_fd, DRM_IOCTL_MODE_GETPLANE, &pl) == 0) {
            fb_id = pl.fb_id;
        }
    }
    if (fb_id == 0) {
        drm_mode_crtc crtc{};
        crtc.crtc_id = impl_->display.crtc_id;
        if (::ioctl(impl_->drm_fd, DRM_IOCTL_MODE_GETCRTC, &crtc) == 0) {
            fb_id = crtc.fb_id;
        }
    }
    if (fb_id == 0) {
        return std::unexpected("No active framebuffer currently scanned out on selected display");
    }

    // 2. Query cursor plane properties via direct stack ioctl (1 ioctl instead of 2, 0 heap allocs)
    uint32_t c_fb = 0;
    int32_t cx = 0, cy = 0;
    uint32_t cw = 0, ch = 0;
    if (impl_->cfg.capture_cursor && impl_->display.cursor_plane_id != 0 && impl_->prop_cursor_fb != 0) {
        uint32_t prop_ids[32]{};
        uint64_t prop_vals[32]{};
        drm_mode_obj_get_properties req{};
        req.obj_id = impl_->display.cursor_plane_id;
        req.obj_type = DRM_MODE_OBJECT_PLANE;
        req.count_props = 32;
        req.props_ptr = reinterpret_cast<uintptr_t>(prop_ids);
        req.prop_values_ptr = reinterpret_cast<uintptr_t>(prop_vals);
        if (::ioctl(impl_->drm_fd, DRM_IOCTL_MODE_OBJ_GETPROPERTIES, &req) == 0) {
            const uint32_t n = std::min<uint32_t>(req.count_props, 32u);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t id = prop_ids[i];
                const uint64_t val = prop_vals[i];
                if (id == impl_->prop_cursor_fb) c_fb = static_cast<uint32_t>(val);
                else if (id == impl_->prop_cursor_x) cx = static_cast<int32_t>(val);
                else if (id == impl_->prop_cursor_y) cy = static_cast<int32_t>(val);
                else if (id == impl_->prop_cursor_w) cw = static_cast<uint32_t>(val);
                else if (id == impl_->prop_cursor_h) ch = static_cast<uint32_t>(val);
            }
        }
    }

    // 3. Single ScopedCapSysAdmin window for both primary + cursor FB2/PRIME exports
    //    (cuts SYS_capset syscalls from 8/frame down to 2/frame!)
    KmsFrame frame{};
    {
        ScopedCapSysAdmin cap_guard;
        auto prim_res = export_fb_to_surface_locked(impl_->drm_fd, fb_id);
        if (!prim_res) {
            return std::unexpected(prim_res.error());
        }
        frame.primary = *prim_res;

        if (c_fb != 0 && cw > 0 && ch > 0) {
            auto cur_res = export_fb_to_surface_locked(impl_->drm_fd, c_fb);
            if (cur_res) {
                frame.cursor.visible = true;
                frame.cursor.x = cx;
                frame.cursor.y = cy;
                frame.cursor.width = static_cast<int32_t>(cw);
                frame.cursor.height = static_cast<int32_t>(ch);
                frame.cursor.surface = *cur_res;
            }
        }
    }

    return frame;
}

const KmsDisplayInfo &KmsCapture::display() const noexcept {
    return impl_->display;
}

}  // namespace vkcap
