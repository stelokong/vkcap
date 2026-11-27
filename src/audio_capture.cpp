#include "audio_capture.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstring>
#include <format>
#include <iostream>
#include <mutex>

#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>

namespace vkcap {

namespace {

void ensure_pw_init() {
    static std::once_flag once;
    std::call_once(once, [] { pw_init(nullptr, nullptr); });
}

std::string lower_copy(std::string_view sv) {
    std::string s(sv);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

inline float soft_limit(float x) noexcept {
    if (x > 0.95f) [[unlikely]] {
        return 0.95f + 0.05f * std::tanh((x - 0.95f) * 20.0f);
    }
    if (x < -0.95f) [[unlikely]] {
        return -0.95f + 0.05f * std::tanh((x + 0.95f) * 20.0f);
    }
    return x;
}

struct EnumState {
    pw_main_loop *loop = nullptr;
    std::vector<AudioDeviceInfo> mics;
    int sync_seq = 0;
};

void on_reg_global(void *data, uint32_t id, uint32_t /*perms*/,
                   const char *type, uint32_t /*version*/, const spa_dict *props) {
    auto *st = static_cast<EnumState *>(data);
    if (!props || !type || std::strcmp(type, PW_TYPE_INTERFACE_Node) != 0) return;

    const char *mclass = spa_dict_lookup(props, PW_KEY_MEDIA_CLASS);
    if (!mclass) return;
    if (std::strcmp(mclass, "Audio/Source") != 0 &&
        std::strcmp(mclass, "Audio/Source/Virtual") != 0) {
        return;
    }

    const char *name = spa_dict_lookup(props, PW_KEY_NODE_NAME);
    const char *desc = spa_dict_lookup(props, PW_KEY_NODE_DESCRIPTION);
    if (!desc) desc = spa_dict_lookup(props, PW_KEY_NODE_NICK);
    if (!desc) desc = name;

    AudioDeviceInfo info{};
    info.id = id;
    info.name = name ? name : std::to_string(id);
    info.description = desc ? desc : info.name;
    st->mics.push_back(std::move(info));
}

void on_core_done(void *data, uint32_t id, int seq) {
    auto *st = static_cast<EnumState *>(data);
    if (id == PW_ID_CORE && seq == st->sync_seq) {
        pw_main_loop_quit(st->loop);
    }
}

struct CaptureStream {
    pw_stream *stream = nullptr;
    spa_hook listener{};
    AdaptiveAudioBuffer ring;
    std::atomic<uint32_t> channels{2};
};

void on_stream_param_changed(void *userdata, uint32_t id, const spa_pod *param) {
    if (!param || id != SPA_PARAM_Format) return;
    auto *cs = static_cast<CaptureStream *>(userdata);
    spa_audio_info_raw info{};
    if (spa_format_audio_raw_parse(param, &info) >= 0 && info.channels > 0) {
        cs->channels.store(info.channels, std::memory_order_release);
    }
}

void on_stream_process(void *userdata) {
    auto *cs = static_cast<CaptureStream *>(userdata);
    pw_buffer *b = pw_stream_dequeue_buffer(cs->stream);
    if (!b) return;

    spa_buffer *buf = b->buffer;
    if (buf->n_datas > 0 && buf->datas[0].data != nullptr && buf->datas[0].chunk != nullptr) {
        const auto *chunk = buf->datas[0].chunk;
        if ((chunk->flags & SPA_CHUNK_FLAG_CORRUPTED) == 0 && chunk->size > 0) {
            const uint32_t ch = cs->channels.load(std::memory_order_acquire);
            if (ch > 0) {
                const auto *base = static_cast<const uint8_t *>(buf->datas[0].data) + chunk->offset;
                const auto *samples = reinterpret_cast<const float *>(base);
                const size_t stride = sizeof(float) * static_cast<size_t>(ch);
                const size_t nframes = chunk->size / stride;
                if (ch == 2) {
                    cs->ring.push_frames(samples, nframes);
                } else if (ch == 1) {
                    // Mono USB/XLR/Pro-Audio fallback: duplicate mono -> interleaved stereo on stack (0 allocs)
                    float stereo[512];
                    size_t pos = 0;
                    while (pos < nframes) {
                        const size_t chunk_n = std::min<size_t>(256, nframes - pos);
                        for (size_t i = 0; i < chunk_n; ++i) {
                            const float s = samples[pos + i];
                            stereo[i * 2] = s;
                            stereo[i * 2 + 1] = s;
                        }
                        cs->ring.push_frames(stereo, chunk_n);
                        pos += chunk_n;
                    }
                } else {
                    // Multi-channel (>2ch) downmix on stack (0 allocs): standard SMPTE 5.1/7.1 fold
                    // when ch >= 6 (FL=0, FR=1, FC=2, LFE=3, RL/SL=4, RR/SR=5), else take channels 0/1.
                    float stereo[512];
                    size_t pos = 0;
                    while (pos < nframes) {
                        const size_t chunk_n = std::min<size_t>(256, nframes - pos);
                        for (size_t i = 0; i < chunk_n; ++i) {
                            const float *f = samples + (pos + i) * ch;
                            float l = f[0];
                            float r = f[1];
                            if (ch >= 6) {
                                const float c = 0.70710678f * f[2];
                                l = (l + c + 0.5f * f[4]) * 0.7f;
                                r = (r + c + 0.5f * f[5]) * 0.7f;
                            }
                            stereo[i * 2] = l;
                            stereo[i * 2 + 1] = r;
                        }
                        cs->ring.push_frames(stereo, chunk_n);
                        pos += chunk_n;
                    }
                }
            }
        }
    }
    pw_stream_queue_buffer(cs->stream, b);
}

const pw_stream_events kStreamEvents = [] {
    pw_stream_events ev{};
    ev.version = PW_VERSION_STREAM_EVENTS;
    ev.param_changed = on_stream_param_changed;
    ev.process = on_stream_process;
    return ev;
}();

}  // namespace

struct AudioCapture::Impl {
    AudioCaptureConfig cfg;
    pw_thread_loop *loop = nullptr;
    pw_context *ctx = nullptr;
    pw_core *core = nullptr;

    std::unique_ptr<CaptureStream> sys_stream;
    std::unique_ptr<CaptureStream> mic_stream;
    std::vector<float> tmp_mic;

    ~Impl() {
        if (loop) {
            pw_thread_loop_lock(loop);
            if (sys_stream && sys_stream->stream) {
                spa_hook_remove(&sys_stream->listener);
                pw_stream_destroy(sys_stream->stream);
                sys_stream->stream = nullptr;
            }
            if (mic_stream && mic_stream->stream) {
                spa_hook_remove(&mic_stream->listener);
                pw_stream_destroy(mic_stream->stream);
                mic_stream->stream = nullptr;
            }
            if (core) {
                pw_core_disconnect(core);
                core = nullptr;
            }
            if (ctx) {
                pw_context_destroy(ctx);
                ctx = nullptr;
            }
            pw_thread_loop_unlock(loop);
            pw_thread_loop_stop(loop);
            pw_thread_loop_destroy(loop);
            loop = nullptr;
        }
    }
};

AudioCapture::~AudioCapture() = default;

std::expected<std::vector<AudioDeviceInfo>, std::string> AudioCapture::enumerate_microphones() {
    ensure_pw_init();

    pw_main_loop *loop = pw_main_loop_new(nullptr);
    if (!loop) return std::unexpected("pw_main_loop_new failed");

    pw_context *ctx = pw_context_new(pw_main_loop_get_loop(loop), nullptr, 0);
    if (!ctx) {
        pw_main_loop_destroy(loop);
        return std::unexpected("pw_context_new failed");
    }

    pw_core *core = pw_context_connect(ctx, nullptr, 0);
    if (!core) {
        pw_context_destroy(ctx);
        pw_main_loop_destroy(loop);
        return std::unexpected("Could not connect to PipeWire audio daemon");
    }

    pw_registry *reg = pw_core_get_registry(core, PW_VERSION_REGISTRY, 0);
    EnumState st{};
    st.loop = loop;

    spa_hook reg_hook{};
    pw_registry_events reg_ev{};
    reg_ev.version = PW_VERSION_REGISTRY_EVENTS;
    reg_ev.global = on_reg_global;
    pw_registry_add_listener(reg, &reg_hook, &reg_ev, &st);

    spa_hook core_hook{};
    pw_core_events core_ev{};
    core_ev.version = PW_VERSION_CORE_EVENTS;
    core_ev.done = on_core_done;
    pw_core_add_listener(core, &core_hook, &core_ev, &st);

    st.sync_seq = pw_core_sync(core, PW_ID_CORE, 0);
    pw_main_loop_run(loop);

    spa_hook_remove(&core_hook);
    spa_hook_remove(&reg_hook);
    pw_proxy_destroy(reinterpret_cast<pw_proxy *>(reg));
    pw_core_disconnect(core);
    pw_context_destroy(ctx);
    pw_main_loop_destroy(loop);

    std::sort(st.mics.begin(), st.mics.end(),
              [](const AudioDeviceInfo &a, const AudioDeviceInfo &b) { return a.id < b.id; });
    return st.mics;
}

std::expected<std::unique_ptr<AudioCapture>, std::string> AudioCapture::start(const AudioCaptureConfig &cfg) {
    if (!cfg.capture_system && !cfg.capture_mic) {
        return std::unexpected("AudioCapture::start called with both system and mic disabled");
    }
    ensure_pw_init();

    // Resolve microphone target (index, node ID, node.name, or description substring)
    std::string resolved_mic_target;
    std::string resolved_mic_desc = "default";
    if (cfg.capture_mic && !cfg.mic_target.empty() && lower_copy(cfg.mic_target) != "default") {
        auto mics_res = enumerate_microphones();
        if (mics_res && !mics_res->empty()) {
            const auto &mics = *mics_res;
            bool matched = false;
            size_t idx = 0;
            auto [p, ec] = std::from_chars(cfg.mic_target.data(),
                                           cfg.mic_target.data() + cfg.mic_target.size(), idx);
            if (ec == std::errc() && p == cfg.mic_target.data() + cfg.mic_target.size()) {
                // Match by PipeWire node ID first, then by 0-based list index
                for (const auto &m : mics) {
                    if (m.id == idx) {
                        resolved_mic_target = m.name;
                        resolved_mic_desc = m.description;
                        matched = true;
                        break;
                    }
                }
                if (!matched && idx < mics.size()) {
                    resolved_mic_target = mics[idx].name;
                    resolved_mic_desc = mics[idx].description;
                    matched = true;
                }
            }
            if (!matched) {
                const std::string want = lower_copy(cfg.mic_target);
                for (const auto &m : mics) {
                    if (lower_copy(m.name) == want || lower_copy(m.description) == want) {
                        resolved_mic_target = m.name;
                        resolved_mic_desc = m.description;
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) {
                const std::string want = lower_copy(cfg.mic_target);
                for (const auto &m : mics) {
                    if (lower_copy(m.name).find(want) != std::string::npos ||
                        lower_copy(m.description).find(want) != std::string::npos) {
                        resolved_mic_target = m.name;
                        resolved_mic_desc = m.description;
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) {
                return std::unexpected(std::format(
                    "Microphone '{}' not found. Run with --list-mics to see available devices.",
                    cfg.mic_target));
            }
        } else {
            resolved_mic_target = cfg.mic_target;
            resolved_mic_desc = cfg.mic_target;
        }
    }

    auto cap = std::unique_ptr<AudioCapture>(new AudioCapture());
    cap->impl_ = std::make_unique<Impl>();
    Impl *impl = cap->impl_.get();
    impl->cfg = cfg;

    impl->loop = pw_thread_loop_new("vkcap-audio-pw", nullptr);
    if (!impl->loop) return std::unexpected("pw_thread_loop_new failed");

    impl->ctx = pw_context_new(pw_thread_loop_get_loop(impl->loop), nullptr, 0);
    if (!impl->ctx) return std::unexpected("pw_context_new failed");

    if (pw_thread_loop_start(impl->loop) < 0) {
        return std::unexpected("pw_thread_loop_start failed");
    }

    pw_thread_loop_lock(impl->loop);
    impl->core = pw_context_connect(impl->ctx, nullptr, 0);
    if (!impl->core) {
        pw_thread_loop_unlock(impl->loop);
        return std::unexpected("Could not connect to PipeWire audio daemon");
    }

    auto connect_stream = [&](const char *stream_name, pw_properties *props,
                              std::unique_ptr<CaptureStream> &out_cs) -> std::expected<void, std::string> {
        auto cs = std::make_unique<CaptureStream>();
        cs->channels.store(cfg.channels, std::memory_order_relaxed);
        cs->stream = pw_stream_new(impl->core, stream_name, props);
        if (!cs->stream) {
            return std::unexpected(std::format("pw_stream_new({}) failed", stream_name));
        }
        pw_stream_add_listener(cs->stream, &cs->listener, &kStreamEvents, cs.get());

        uint8_t pod_buf[1024];
        spa_pod_builder b = SPA_POD_BUILDER_INIT(pod_buf, sizeof(pod_buf));
        spa_audio_info_raw info{};
        info.format = SPA_AUDIO_FORMAT_F32;
        info.rate = cfg.sample_rate;
        info.channels = cfg.channels;
        info.position[0] = SPA_AUDIO_CHANNEL_FL;
        info.position[1] = SPA_AUDIO_CHANNEL_FR;

        const spa_pod *params[1];
        params[0] = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &info);

        const auto flags = static_cast<pw_stream_flags>(
            PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS);
        if (pw_stream_connect(cs->stream, PW_DIRECTION_INPUT, PW_ID_ANY, flags, params, 1) < 0) {
            spa_hook_remove(&cs->listener);
            pw_stream_destroy(cs->stream);
            cs->stream = nullptr;
            return std::unexpected(std::format("pw_stream_connect({}) failed", stream_name));
        }
        out_cs = std::move(cs);
        return {};
    };

    if (cfg.capture_system) {
        pw_properties *props = pw_properties_new(
            PW_KEY_MEDIA_TYPE, "Audio",
            PW_KEY_MEDIA_CATEGORY, "Capture",
            PW_KEY_MEDIA_ROLE, "Screen",
            PW_KEY_STREAM_CAPTURE_SINK, "true",
            PW_KEY_NODE_LATENCY, "960/48000",
            nullptr);
        auto r = connect_stream("vkcap-system", props, impl->sys_stream);
        if (!r) {
            pw_thread_loop_unlock(impl->loop);
            return std::unexpected(r.error());
        }
    }

    if (cfg.capture_mic) {
        pw_properties *props = pw_properties_new(
            PW_KEY_MEDIA_TYPE, "Audio",
            PW_KEY_MEDIA_CATEGORY, "Capture",
            PW_KEY_MEDIA_ROLE, "Communication",
            PW_KEY_NODE_LATENCY, "960/48000",
            nullptr);
        if (!resolved_mic_target.empty()) {
            pw_properties_set(props, PW_KEY_TARGET_OBJECT, resolved_mic_target.c_str());
        }
        auto r = connect_stream("vkcap-mic", props, impl->mic_stream);
        if (!r) {
            pw_thread_loop_unlock(impl->loop);
            return std::unexpected(r.error());
        }
        impl->tmp_mic.resize(2048 * 2);
    }

    pw_thread_loop_unlock(impl->loop);

    if (cfg.verbose) {
        std::cerr << std::format(
            "[audio] PipeWire capture started ({} Hz stereo F32, system={}, mic={})\n",
            cfg.sample_rate, cfg.capture_system ? "on" : "off",
            cfg.capture_mic ? resolved_mic_desc : "off");
    }
    return cap;
}

void AudioCapture::reset_sync() noexcept {
    if (impl_->sys_stream) impl_->sys_stream->ring.reset();
    if (impl_->mic_stream) impl_->mic_stream->ring.reset();
}

void AudioCapture::resume_sync() noexcept {
    if (impl_->sys_stream) impl_->sys_stream->ring.trim_to_cushion();
    if (impl_->mic_stream) impl_->mic_stream->ring.trim_to_cushion();
}

void AudioCapture::read_mixed_frames(float *dst, size_t nb_frames) noexcept {
    const size_t nsamples = nb_frames * 2;
    const float sg = impl_->cfg.system_gain;
    const float mg = impl_->cfg.mic_gain;

    if (impl_->sys_stream && impl_->mic_stream) {
        impl_->sys_stream->ring.pop_resampled(dst, nb_frames);
        if (impl_->tmp_mic.size() < nsamples) impl_->tmp_mic.resize(nsamples);
        impl_->mic_stream->ring.pop_resampled(impl_->tmp_mic.data(), nb_frames);
        for (size_t i = 0; i < nsamples; ++i) {
            dst[i] = soft_limit(dst[i] * sg + impl_->tmp_mic[i] * mg);
        }
    } else if (impl_->sys_stream) {
        impl_->sys_stream->ring.pop_resampled(dst, nb_frames);
        if (std::abs(sg - 1.0f) > 1e-4f) {
            for (size_t i = 0; i < nsamples; ++i) {
                dst[i] = soft_limit(dst[i] * sg);
            }
        }
    } else if (impl_->mic_stream) {
        impl_->mic_stream->ring.pop_resampled(dst, nb_frames);
        if (std::abs(mg - 1.0f) > 1e-4f) {
            for (size_t i = 0; i < nsamples; ++i) {
                dst[i] = soft_limit(dst[i] * mg);
            }
        }
    } else {
        std::memset(dst, 0, nsamples * sizeof(float));
    }
}

uint32_t AudioCapture::sample_rate() const noexcept { return impl_->cfg.sample_rate; }
uint32_t AudioCapture::channels() const noexcept { return impl_->cfg.channels; }

AudioSyncStats AudioCapture::stats() const noexcept {
    AudioSyncStats out{};
    if (impl_->sys_stream) {
        auto s = impl_->sys_stream->ring.stats();
        out.underruns += s.underruns;
        out.overrun_trims += s.overrun_trims;
        out.ratio_ppm = s.ratio_ppm;
    }
    if (impl_->mic_stream) {
        auto m = impl_->mic_stream->ring.stats();
        out.underruns += m.underruns;
        out.overrun_trims += m.overrun_trims;
        if (!impl_->sys_stream) out.ratio_ppm = m.ratio_ppm;
    }
    return out;
}

}  // namespace vkcap
