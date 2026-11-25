#include "pipewire_capture.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <format>
#include <iostream>
#include <libdrm/drm_fourcc.h>
#include <mutex>
#include <pipewire/pipewire.h>
#include <spa/buffer/meta.h>
#include <spa/param/video/format-utils.h>
#include <spa/pod/builder.h>
#include <spa/utils/result.h>
#include <unistd.h>

namespace vkcap {

uint32_t PipeWireCapture::spa_to_drm_fourcc(uint32_t spa_fmt) noexcept {
    switch (spa_fmt) {
        case SPA_VIDEO_FORMAT_BGRx:        return DRM_FORMAT_XRGB8888;
        case SPA_VIDEO_FORMAT_BGRA:        return DRM_FORMAT_ARGB8888;
        case SPA_VIDEO_FORMAT_RGBx:        return DRM_FORMAT_XBGR8888;
        case SPA_VIDEO_FORMAT_RGBA:        return DRM_FORMAT_ABGR8888;
        case SPA_VIDEO_FORMAT_xRGB_210LE:  return DRM_FORMAT_XRGB2101010;
        case SPA_VIDEO_FORMAT_ARGB_210LE:  return DRM_FORMAT_ARGB2101010;
        case SPA_VIDEO_FORMAT_xBGR_210LE:  return DRM_FORMAT_XBGR2101010;
        case SPA_VIDEO_FORMAT_ABGR_210LE:  return DRM_FORMAT_ABGR2101010;
        case SPA_VIDEO_FORMAT_RGBA_F16:    return DRM_FORMAT_ABGR16161616F;
        case SPA_VIDEO_FORMAT_NV12:        return DRM_FORMAT_NV12;
        case SPA_VIDEO_FORMAT_P010_10LE:   return DRM_FORMAT_P010;
        default:                           return DRM_FORMAT_INVALID;
    }
}

const char *PipeWireCapture::spa_format_name(uint32_t spa_fmt) noexcept {
    switch (spa_fmt) {
        case SPA_VIDEO_FORMAT_BGRx:        return "BGRx";
        case SPA_VIDEO_FORMAT_BGRA:        return "BGRA";
        case SPA_VIDEO_FORMAT_RGBx:        return "RGBx";
        case SPA_VIDEO_FORMAT_RGBA:        return "RGBA";
        case SPA_VIDEO_FORMAT_xRGB_210LE:  return "xRGB_210LE";
        case SPA_VIDEO_FORMAT_ARGB_210LE:  return "ARGB_210LE";
        case SPA_VIDEO_FORMAT_xBGR_210LE:  return "xBGR_210LE";
        case SPA_VIDEO_FORMAT_ABGR_210LE:  return "ABGR_210LE";
        case SPA_VIDEO_FORMAT_RGBA_F16:    return "RGBA_F16";
        case SPA_VIDEO_FORMAT_NV12:        return "NV12";
        case SPA_VIDEO_FORMAT_P010_10LE:   return "P010_10LE";
        default:                           return "UNKNOWN";
    }
}

namespace {

const spa_pod *build_format_pod(spa_pod_builder *b,
                                uint32_t spa_fmt,
                                const std::vector<uint64_t> *modifiers,
                                bool fixated_modifier,
                                uint64_t fixated_mod_val,
                                uint32_t target_fps) {
    spa_pod_frame f_obj{};
    spa_pod_builder_push_object(b, &f_obj, SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat);
    spa_pod_builder_add(b,
        SPA_FORMAT_mediaType,    SPA_POD_Id(SPA_MEDIA_TYPE_video),
        SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
        SPA_FORMAT_VIDEO_format, SPA_POD_Id(spa_fmt),
        0);

    spa_rectangle def_size = SPA_RECTANGLE(1920, 1080);
    spa_rectangle min_size = SPA_RECTANGLE(16, 16);
    spa_rectangle max_size = SPA_RECTANGLE(8192, 4352);
    spa_pod_builder_add(b,
        SPA_FORMAT_VIDEO_size, SPA_POD_CHOICE_RANGE_Rectangle(&def_size, &min_size, &max_size),
        0);

    // Wayland compositors (KWin/Mutter/wlroots) are damage-driven VFR: advertise
    // framerate = 0/1 and cap via maxFramerate = target_fps/1.
    spa_fraction vfr_fps = SPA_FRACTION(0, 1);
    spa_fraction def_max_fps = SPA_FRACTION(target_fps, 1);
    spa_fraction min_fps = SPA_FRACTION(1, 1);
    spa_fraction max_fps = SPA_FRACTION(1000, 1);
    spa_pod_builder_add(b,
        SPA_FORMAT_VIDEO_framerate,    SPA_POD_Fraction(&vfr_fps),
        SPA_FORMAT_VIDEO_maxFramerate, SPA_POD_CHOICE_RANGE_Fraction(&def_max_fps, &min_fps, &max_fps),
        0);

    if (fixated_modifier) {
        spa_pod_builder_prop(b, SPA_FORMAT_VIDEO_modifier, SPA_POD_PROP_FLAG_MANDATORY);
        spa_pod_builder_long(b, static_cast<int64_t>(fixated_mod_val));
    } else if (modifiers && !modifiers->empty()) {
        spa_pod_builder_prop(b, SPA_FORMAT_VIDEO_modifier,
                             SPA_POD_PROP_FLAG_MANDATORY | SPA_POD_PROP_FLAG_DONT_FIXATE);
        spa_pod_frame f_choice{};
        spa_pod_builder_push_choice(b, &f_choice, SPA_CHOICE_Enum, 0);
        // Default value followed by all supported modifiers
        spa_pod_builder_long(b, static_cast<int64_t>(modifiers->front()));
        for (uint64_t mod : *modifiers) {
            spa_pod_builder_long(b, static_cast<int64_t>(mod));
        }
        spa_pod_builder_pop(b, &f_choice);
    }

    return static_cast<const spa_pod *>(spa_pod_builder_pop(b, &f_obj));
}

}  // namespace

struct PipeWireCapture::Impl {
    Config cfg;
    std::vector<FormatModifierSet> supported_formats;
    CaptureCallbacks callbacks;

    pw_thread_loop *loop = nullptr;
    pw_context *context = nullptr;
    pw_core *core = nullptr;
    pw_stream *stream = nullptr;
    spa_hook core_listener{};
    spa_hook stream_listener{};

    NegotiatedFormat current_fmt{};
    std::atomic<bool> streaming{false};
    std::atomic<bool> stopped{false};
    uint64_t seq_counter = 0;

    static void on_core_error(void *data, uint32_t id, int seq, int res, const char *message) {
        auto *self = static_cast<Impl *>(data);
        if (id == PW_ID_CORE && res == -EPIPE) {
            self->streaming.store(false, std::memory_order_relaxed);
            if (self->callbacks.on_error) {
                self->callbacks.on_error(std::format("PipeWire core disconnected ({}): {}", seq, message ? message : ""));
            }
        }
    }

    static void on_state_changed(void *data, enum pw_stream_state old_state,
                                 enum pw_stream_state state, const char *error) {
        auto *self = static_cast<Impl *>(data);
        if (self->cfg.verbose) {
            std::cerr << std::format("[pipewire] state {} -> {}{}\n",
                                     pw_stream_state_as_string(old_state),
                                     pw_stream_state_as_string(state),
                                     error ? std::format(" ({})", error) : "");
        }
        self->streaming.store(state == PW_STREAM_STATE_STREAMING, std::memory_order_relaxed);
        if (state == PW_STREAM_STATE_ERROR && self->callbacks.on_error) {
            self->callbacks.on_error(std::format("PipeWire stream error: {}", error ? error : "unknown"));
        }
    }

    static void on_param_changed(void *data, uint32_t id, const struct spa_pod *param) {
        auto *self = static_cast<Impl *>(data);
        if (!param || id != SPA_PARAM_Format) {
            return;
        }

        spa_video_info_raw info{};
        if (spa_format_video_raw_parse(param, &info) < 0) {
            return;
        }

        uint8_t pod_buf[8192];
        spa_pod_builder b = SPA_POD_BUILDER_INIT(pod_buf, sizeof(pod_buf));
        std::vector<const spa_pod *> params;

        if ((info.flags & SPA_VIDEO_FLAG_MODIFIER_FIXATION_REQUIRED) != 0) {
            uint64_t chosen_mod = static_cast<uint64_t>(info.modifier);
            if (self->cfg.verbose) {
                std::cerr << std::format("[pipewire] fixating modifier 0x{:016x} for fmt {}\n",
                                         chosen_mod, spa_format_name(info.format));
            }
            params.push_back(build_format_pod(&b, info.format, nullptr, true, chosen_mod, self->cfg.fps));
            params.push_back(build_format_pod(&b, info.format, nullptr, false, 0, self->cfg.fps));
            pw_stream_update_params(self->stream, params.data(), static_cast<uint32_t>(params.size()));
            return;
        }

        NegotiatedFormat nf{};
        nf.spa_format = info.format;
        nf.drm_fourcc = spa_to_drm_fourcc(info.format);
        nf.width = info.size.width;
        nf.height = info.size.height;
        nf.has_modifier = (info.flags & SPA_VIDEO_FLAG_MODIFIER) != 0;
        nf.drm_modifier = nf.has_modifier ? static_cast<uint64_t>(info.modifier) : DRM_FORMAT_MOD_INVALID;
        nf.fps_num = info.framerate.num ? static_cast<uint32_t>(info.framerate.num) : self->cfg.fps;
        nf.fps_den = info.framerate.denom ? static_cast<uint32_t>(info.framerate.denom) : 1;
        self->current_fmt = nf;

        if (self->cfg.verbose) {
            std::cerr << std::format(
                "[pipewire] negotiated format={} {}x{} @ {}/{} fps modifier={}\n",
                spa_format_name(nf.spa_format), nf.width, nf.height,
                nf.fps_num, nf.fps_den,
                nf.has_modifier ? std::format("0x{:016x}", nf.drm_modifier) : "none");
        }

        if (self->callbacks.on_format_changed) {
            self->callbacks.on_format_changed(nf);
        }

        uint32_t data_types = (1u << SPA_DATA_DmaBuf);
        if (!self->cfg.dmabuf_only) {
            data_types |= (1u << SPA_DATA_MemFd) | (1u << SPA_DATA_MemPtr);
        }

        params.push_back(reinterpret_cast<const spa_pod *>(spa_pod_builder_add_object(
            &b, SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
            SPA_PARAM_BUFFERS_buffers, SPA_POD_CHOICE_RANGE_Int(8, 4, 16),
            SPA_PARAM_BUFFERS_blocks,  SPA_POD_CHOICE_RANGE_Int(1, 1, 4),
            SPA_PARAM_BUFFERS_dataType, SPA_POD_Int(data_types))));

        params.push_back(reinterpret_cast<const spa_pod *>(spa_pod_builder_add_object(
            &b, SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
            SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Header),
            SPA_PARAM_META_size, SPA_POD_Int(sizeof(spa_meta_header)))));

        params.push_back(reinterpret_cast<const spa_pod *>(spa_pod_builder_add_object(
            &b, SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
            SPA_PARAM_META_type, SPA_POD_Id(SPA_META_VideoCrop),
            SPA_PARAM_META_size, SPA_POD_Int(sizeof(spa_meta_region)))));

        pw_stream_update_params(self->stream, params.data(), static_cast<uint32_t>(params.size()));
    }

    static void on_remove_buffer(void *data, struct pw_buffer *b) {
        auto *self = static_cast<Impl *>(data);
        if (self->callbacks.on_buffer_removed) {
            self->callbacks.on_buffer_removed(b);
        }
    }

    static void on_process(void *data) {
        auto *self = static_cast<Impl *>(data);

        // Drain any burst of queued buffers and keep only the freshest valid frame so we
        // never waste GPU compute/encode cycles on stale frames under heavy desktop load.
        pw_buffer *b = nullptr;
        while (pw_buffer *next = pw_stream_dequeue_buffer(self->stream)) {
            if (b) {
                pw_stream_queue_buffer(self->stream, b);
            }
            b = next;
        }
        if (!b) {
            return;
        }

        spa_buffer *sb = b->buffer;
        if (!sb || sb->n_datas == 0 || !sb->datas[0].chunk ||
            (sb->datas[0].chunk->flags & SPA_CHUNK_FLAG_CORRUPTED) != 0 ||
            sb->datas[0].chunk->size == 0) {
            pw_stream_queue_buffer(self->stream, b);
            return;
        }

        const uint32_t dtype = sb->datas[0].type;
        if (dtype != SPA_DATA_DmaBuf && (self->cfg.dmabuf_only ||
            (dtype != SPA_DATA_MemFd && dtype != SPA_DATA_MemPtr))) {
            pw_stream_queue_buffer(self->stream, b);
            return;
        }

        CapturedFrame cf{};
        cf.buffer_handle = b;
        cf.mem_type = (dtype == SPA_DATA_DmaBuf) ? CaptureMemoryType::DmaBuf : CaptureMemoryType::Shm;
        cf.format = self->current_fmt;
        cf.num_planes = std::min<uint32_t>(sb->n_datas, 4u);
        for (uint32_t i = 0; i < cf.num_planes; ++i) {
            const spa_data &d = sb->datas[i];
            cf.planes[i].fd = static_cast<int>(d.fd);
            cf.planes[i].offset = d.chunk ? d.chunk->offset : 0;
            cf.planes[i].stride = d.chunk ? d.chunk->stride : 0;
            cf.planes[i].size = d.chunk ? d.chunk->size : d.maxsize;
            cf.planes[i].data = d.data;
        }

        if (auto *hdr = static_cast<spa_meta_header *>(
                spa_buffer_find_meta_data(sb, SPA_META_Header, sizeof(spa_meta_header)))) {
            cf.pts_ns = hdr->pts;
        }
        if (auto *crop = static_cast<spa_meta_region *>(
                spa_buffer_find_meta_data(sb, SPA_META_VideoCrop, sizeof(spa_meta_region)))) {
            if (spa_meta_region_is_valid(crop)) {
                cf.crop = {crop->region.position.x, crop->region.position.y,
                           static_cast<int32_t>(crop->region.size.width),
                           static_cast<int32_t>(crop->region.size.height)};
            }
        }
        cf.sequence = self->seq_counter++;

        if (self->callbacks.on_frame) {
            self->callbacks.on_frame(cf);
        }
        // Always return the pw_buffer immediately on the PipeWire thread so the
        // compositor NEVER runs out of buffers or stalls display presentation.
        pw_stream_queue_buffer(self->stream, b);
    }

    void teardown() {
        stopped.store(true, std::memory_order_release);
        if (loop) {
            pw_thread_loop_lock(loop);
        }
        if (stream) {
            spa_hook_remove(&stream_listener);
            pw_stream_destroy(stream);
            stream = nullptr;
        }
        if (core) {
            spa_hook_remove(&core_listener);
            pw_core_disconnect(core);
            core = nullptr;
        }
        if (context) {
            pw_context_destroy(context);
            context = nullptr;
        }
        if (loop) {
            pw_thread_loop_unlock(loop);
            pw_thread_loop_stop(loop);
            pw_thread_loop_destroy(loop);
            loop = nullptr;
        }
    }
};

PipeWireCapture::~PipeWireCapture() {
    stop();
}

void PipeWireCapture::stop() {
    if (impl_) {
        impl_->teardown();
        impl_.reset();
    }
}

bool PipeWireCapture::is_streaming() const noexcept {
    return impl_ && impl_->streaming.load(std::memory_order_relaxed);
}

std::expected<std::unique_ptr<PipeWireCapture>, std::string> PipeWireCapture::start(
    const Config &cfg,
    int pipewire_fd,
    const PortalStreamInfo &stream_info,
    std::vector<FormatModifierSet> supported_formats,
    CaptureCallbacks callbacks) {
    if (pipewire_fd < 0) {
        return std::unexpected("Invalid PipeWire fd");
    }

    static std::once_flag pw_init_once;
    std::call_once(pw_init_once, []() { pw_init(nullptr, nullptr); });

    auto cap = std::unique_ptr<PipeWireCapture>(new PipeWireCapture());
    cap->impl_ = std::make_unique<Impl>();
    Impl *impl = cap->impl_.get();
    impl->cfg = cfg;
    impl->supported_formats = std::move(supported_formats);
    impl->callbacks = std::move(callbacks);

    impl->loop = pw_thread_loop_new("vkcap-pw", nullptr);
    if (!impl->loop) {
        ::close(pipewire_fd);
        return std::unexpected("Failed to create PipeWire thread loop");
    }

    impl->context = pw_context_new(pw_thread_loop_get_loop(impl->loop), nullptr, 0);
    if (!impl->context) {
        ::close(pipewire_fd);
        return std::unexpected("Failed to create PipeWire context");
    }

    impl->core = pw_context_connect_fd(impl->context, pipewire_fd, nullptr, 0);
    if (!impl->core) {
        // pw_context_connect_fd closes pipewire_fd on error or takes ownership
        return std::unexpected("Failed to connect to PipeWire remote fd");
    }

    static const pw_core_events core_events = {
        .version = PW_VERSION_CORE_EVENTS,
        .error = Impl::on_core_error,
    };
    pw_core_add_listener(impl->core, &impl->core_listener, &core_events, impl);

    pw_properties *props = pw_properties_new(
        PW_KEY_MEDIA_TYPE, "Video",
        PW_KEY_MEDIA_CATEGORY, "Capture",
        PW_KEY_MEDIA_ROLE, "Screen",
        nullptr);

    if (stream_info.pipewire_serial.has_value()) {
        std::string serial_str = std::to_string(*stream_info.pipewire_serial);
        pw_properties_set(props, PW_KEY_TARGET_OBJECT, serial_str.c_str());
    }

    impl->stream = pw_stream_new(impl->core, "vkcap-screencast", props);
    if (!impl->stream) {
        return std::unexpected("Failed to create PipeWire stream");
    }

    static const pw_stream_events stream_events = {
        .version = PW_VERSION_STREAM_EVENTS,
        .state_changed = Impl::on_state_changed,
        .param_changed = Impl::on_param_changed,
        .remove_buffer = Impl::on_remove_buffer,
        .process = Impl::on_process,
    };
    pw_stream_add_listener(impl->stream, &impl->stream_listener, &stream_events, impl);

    std::vector<uint8_t> pod_storage(32768);
    spa_pod_builder b = SPA_POD_BUILDER_INIT(pod_storage.data(), static_cast<uint32_t>(pod_storage.size()));
    std::vector<const spa_pod *> params;
    for (const auto &fmt : impl->supported_formats) {
        if (!fmt.modifiers.empty()) {
            params.push_back(build_format_pod(&b, fmt.spa_format, &fmt.modifiers, false, 0, cfg.fps));
        }
        if (fmt.allow_modifierless) {
            params.push_back(build_format_pod(&b, fmt.spa_format, nullptr, false, 0, cfg.fps));
        }
    }

    if (params.empty()) {
        return std::unexpected("No supported Vulkan video capture formats to advertise to PipeWire");
    }

    uint32_t target_id = stream_info.pipewire_serial.has_value() ? PW_ID_ANY : stream_info.node_id;
    // Do NOT pass PW_STREAM_FLAG_MAP_BUFFERS: mapping DMA-BUFs into CPU address space
    // incurs heavy DRM/VRAM overhead and can force staging copies under GPU load.
    int res = pw_stream_connect(
        impl->stream,
        PW_DIRECTION_INPUT,
        target_id,
        PW_STREAM_FLAG_AUTOCONNECT,
        params.data(),
        static_cast<uint32_t>(params.size()));
    if (res < 0) {
        return std::unexpected(std::format("pw_stream_connect failed: {}", spa_strerror(res)));
    }

    if (pw_thread_loop_start(impl->loop) < 0) {
        return std::unexpected("Failed to start PipeWire thread loop");
    }

    return cap;
}

}  // namespace vkcap
