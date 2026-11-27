#include "pipeline.hpp"

#include "audio_capture.hpp"
#include "color_math.hpp"
#include "kms_capture.hpp"
#include "video_encoder.hpp"
#include "vulkan_compute.hpp"

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <format>
#include <iostream>
#include <mutex>
#include <poll.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern "C" {
#include <libavutil/frame.h>
}

namespace vkcap {

namespace {

int64_t monotonic_now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}

void sleep_until_monotonic_ns(int64_t target_ns) {
    timespec ts{};
    ts.tv_sec = target_ns / 1'000'000'000LL;
    ts.tv_nsec = target_ns % 1'000'000'000LL;
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR) {
    }
}

struct EncodedInput {
    AVFrame *frame = nullptr;
    int64_t pts_ticks = 0;
};

class EncodeFrameQueue {
public:
    explicit EncodeFrameQueue(size_t capacity)
        : capacity_(std::max<size_t>(1, capacity)), ring_(capacity_) {}

    void push(AVFrame *frame, int64_t pts_ticks) {
        AVFrame *evicted = nullptr;
        {
            std::lock_guard lk(mtx_);
            if (closed_) {
                av_frame_free(&frame);
                return;
            }
            if (count_ == capacity_) {
                evicted = ring_[head_].frame;
                head_ = (head_ + 1) % capacity_;
                --count_;
                ++dropped_;
            }
            ring_[tail_] = {frame, pts_ticks};
            tail_ = (tail_ + 1) % capacity_;
            ++count_;
        }
        if (evicted) av_frame_free(&evicted);
        cv_.notify_one();
    }

    bool pop(EncodedInput &out) {
        std::unique_lock lk(mtx_);
        cv_.wait(lk, [this] { return closed_ || count_ > 0; });
        if (count_ == 0) return false;
        out = ring_[head_];
        head_ = (head_ + 1) % capacity_;
        --count_;
        return true;
    }

    void close() {
        std::vector<AVFrame *> leftover;
        {
            std::lock_guard lk(mtx_);
            closed_ = true;
            leftover.reserve(count_);
            while (count_ > 0) {
                leftover.push_back(ring_[head_].frame);
                head_ = (head_ + 1) % capacity_;
                --count_;
            }
        }
        cv_.notify_all();
        for (AVFrame *f : leftover) av_frame_free(&f);
    }

    uint64_t dropped() const {
        std::lock_guard lk(mtx_);
        return dropped_;
    }

private:
    size_t capacity_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::vector<EncodedInput> ring_;
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t count_ = 0;
    bool closed_ = false;
    uint64_t dropped_ = 0;
};

struct ScopedRawKeyReader {
    bool active = false;
    termios orig{};

    explicit ScopedRawKeyReader(bool enable) noexcept {
        if (enable && ::isatty(STDIN_FILENO) != 0 && ::tcgetattr(STDIN_FILENO, &orig) == 0) {
            termios raw = orig;
            raw.c_lflag &= ~static_cast<tcflag_t>(ICANON | ECHO);
            raw.c_cc[VMIN] = 0;
            raw.c_cc[VTIME] = 0;
            if (::tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0) {
                active = true;
            }
        }
    }

    ~ScopedRawKeyReader() {
        if (active) {
            ::tcsetattr(STDIN_FILENO, TCSANOW, &orig);
        }
    }

    char poll_key() const noexcept {
        if (!active) return '\0';
        pollfd pfd{STDIN_FILENO, POLLIN, 0};
        if (::poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN) != 0) {
            char ch = '\0';
            if (::read(STDIN_FILENO, &ch, 1) == 1) return ch;
        }
        return '\0';
    }
};

}  // namespace

std::expected<void, std::string> run_kms_pipeline(const Config &cfg, PipelineSignals &signals) {
    auto &stop_flag = signals.stop;

    auto kms_res = KmsCapture::open(cfg);
    if (!kms_res) return std::unexpected(kms_res.error());
    auto kms = std::move(*kms_res);

    const auto &disp = kms->display();
    const ResolvedGamut gamut = resolve_input_gamut(cfg, disp.edid_primaries);

    // Probe the first KMS frame BEFORE initializing Vulkan so missing CAP_SYS_ADMIN fails
    // instantaneously and exact scanout dimensions are known upfront.
    auto first_res = kms->grab_frame();
    if (!first_res) return std::unexpected(first_res.error());

    const uint32_t src_w = first_res->primary.width;
    const uint32_t src_h = first_res->primary.height;

    CropRect eff_crop{};
    if (cfg.crop.valid()) {
        if (static_cast<uint32_t>(cfg.crop.x) >= src_w || static_cast<uint32_t>(cfg.crop.y) >= src_h) {
            return std::unexpected(std::format(
                "Crop origin ({},{}) is outside display scanout ({}x{})",
                cfg.crop.x, cfg.crop.y, src_w, src_h));
        }
        eff_crop.x = cfg.crop.x;
        eff_crop.y = cfg.crop.y;
        eff_crop.width = std::min<int32_t>(cfg.crop.width, static_cast<int32_t>(src_w) - eff_crop.x);
        eff_crop.height = std::min<int32_t>(cfg.crop.height, static_cast<int32_t>(src_h) - eff_crop.y);
        if (eff_crop.width < 2 || eff_crop.height < 2) {
            return std::unexpected("Clamped crop region must be at least 2x2 pixels");
        }
        first_res->crop = eff_crop;
    }

    const uint32_t active_w = eff_crop.valid() ? static_cast<uint32_t>(eff_crop.width) : src_w;
    const uint32_t active_h = eff_crop.valid() ? static_cast<uint32_t>(eff_crop.height) : src_h;
    const uint32_t enc_w = cfg.out_width ? cfg.out_width : (active_w & ~1u);
    const uint32_t enc_h = cfg.out_height ? cfg.out_height : (active_h & ~1u);

    auto vk_res = VulkanCompute::create(cfg, gamut.enabled, gamut.matrix, disp.card_path);
    if (!vk_res) return std::unexpected(vk_res.error());
    auto vk = std::move(*vk_res);

    auto cfg_res = vk->configure_output(enc_w, enc_h);
    if (!cfg_res) return std::unexpected(cfg_res.error());

    // Start PipeWire low-latency audio capture if requested
    const bool want_audio = cfg.audio && (cfg.capture_system_audio || cfg.capture_mic);
    std::unique_ptr<AudioCapture> audio;
    if (want_audio) {
        AudioCaptureConfig acfg{};
        acfg.capture_system = cfg.capture_system_audio;
        acfg.capture_mic = cfg.capture_mic;
        acfg.mic_target = cfg.mic_device;
        acfg.system_gain = static_cast<float>(cfg.system_volume_pct / 100.0);
        acfg.mic_gain = static_cast<float>(cfg.mic_volume_pct / 100.0);
        acfg.verbose = cfg.verbose;
        auto ares = AudioCapture::start(acfg);
        if (!ares) {
            if (cfg.capture_mic && !cfg.mic_device.empty()) {
                return std::unexpected(ares.error());
            }
            std::cerr << "[vkcap] Warning: audio disabled (" << ares.error() << ")\n";
        } else {
            audio = std::move(*ares);
        }
    }

    auto enc_res = VideoEncoder::open(cfg, enc_w, enc_h, vk->hw_device_ref(), vk->hw_frames_ref(),
                                      audio != nullptr);
    if (!enc_res) return std::unexpected(enc_res.error());
    auto encoder = std::move(*enc_res);
    if (!encoder->audio_enabled()) {
        audio.reset();
    }

    auto acodec_label = [&]() -> std::string {
        switch (cfg.audio_codec) {
            case AudioCodec::Opus: return std::format("Opus {}k", cfg.audio_bitrate / 1000);
            case AudioCodec::Aac:  return std::format("AAC {}k", cfg.audio_bitrate / 1000);
            case AudioCodec::Flac: return "FLAC lossless";
        }
        return "Opus";
    };

    std::string audio_desc = "off";
    if (audio) {
        if (cfg.capture_system_audio && cfg.capture_mic) {
            audio_desc = std::format("{} (system {:.0f}% + mic:{} {:.0f}%)",
                                     acodec_label(), cfg.system_volume_pct,
                                     cfg.mic_device.empty() ? "default" : cfg.mic_device,
                                     cfg.mic_volume_pct);
        } else if (cfg.capture_system_audio) {
            audio_desc = std::format("{} (system {:.0f}%)", acodec_label(), cfg.system_volume_pct);
        } else {
            audio_desc = std::format("{} (mic:{} {:.0f}%)",
                                     acodec_label(),
                                     cfg.mic_device.empty() ? "default" : cfg.mic_device,
                                     cfg.mic_volume_pct);
        }
    }

    const char *vcodec_str = (encoder->active_codec() == Codec::Av1) ? "AV1"
                           : (encoder->active_codec() == Codec::Hevc) ? "HEVC" : "H.264";
    const std::string crop_str = eff_crop.valid()
        ? std::format(", crop {}x{}+{}+{}", eff_crop.width, eff_crop.height, eff_crop.x, eff_crop.y)
        : "";
    if (encoder->replay_mode()) {
        std::cerr << std::format(
            "[vkcap] Instant Replay ({}s rolling buffer, {} {}x{}@{:.1f}Hz -> {}x{}, {} {}, CFR {}fps{}{})\n"
            "[vkcap] Color gamut: {} | Audio: {}\n"
            "[vkcap] Press 's'/Enter (or send SIGUSR2) to save replay, Ctrl+C to stop.\n",
            cfg.replay_seconds, disp.connector_name, src_w, src_h, disp.refresh_hz, enc_w, enc_h,
            vcodec_str, cfg.pixfmt == PixelFormat::P010 ? "10-bit P010" : "8-bit NV12",
            cfg.fps, crop_str, cfg.capture_cursor ? ", hw-cursor" : "",
            gamut.description, audio_desc);
    } else {
        std::cerr << std::format(
            "[vkcap] Recording {} ({} {}x{}@{:.1f}Hz -> {}x{}, {} {}, CFR {}fps{}{})\n"
            "[vkcap] Color gamut: {} | Audio: {}\n",
            cfg.output_path, disp.connector_name, src_w, src_h, disp.refresh_hz, enc_w, enc_h,
            vcodec_str, cfg.pixfmt == PixelFormat::P010 ? "10-bit P010" : "8-bit NV12",
            cfg.fps, crop_str, cfg.capture_cursor ? ", hw-cursor" : "",
            gamut.description, audio_desc);
    }

    EncodeFrameQueue encode_queue(cfg.queue_size);
    std::mutex err_mtx;
    std::string async_err;

    auto record_err = [&](const std::string &e) {
        std::lock_guard lk(err_mtx);
        if (async_err.empty()) async_err = e;
        stop_flag.store(true, std::memory_order_release);
    };

    const auto start_wall = std::chrono::steady_clock::now();
    auto last_report = start_wall;
    uint64_t last_enc_frames = 0;

    // Stage 2: Dedicated Vulkan Video Encode thread (av1_vulkan on VCN)
    std::jthread encode_thread([&](std::stop_token) {
        pthread_setname_np(pthread_self(), "vkcap-enc");
        EncodedInput item{};
        while (encode_queue.pop(item)) {
            auto enc_err = encoder->encode_frame(item.frame, item.pts_ticks);
            av_frame_free(&item.frame);
            if (!enc_err) {
                record_err(enc_err.error());
                break;
            }

            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - last_report).count();
            if (dt >= 1.0) {
                auto est = encoder->stats();
                auto cst = vk->stats();
                double inst_fps = static_cast<double>(est.frames_submitted - last_enc_frames) / dt;
                double elapsed_total = std::chrono::duration<double>(now - start_wall).count();
                double mbps = elapsed_total > 0.0
                    ? (static_cast<double>(est.bytes_written) * 8.0 / 1'000'000.0) / elapsed_total
                    : 0.0;
                double avg_comp_ms = cst.frames_processed > 0
                    ? (static_cast<double>(cst.total_compute_ns) / static_cast<double>(cst.frames_processed)) / 1e6
                    : 0.0;
                if (encoder->replay_mode()) {
                    auto rst = encoder->replay_stats();
                    std::cerr << std::format(
                        "\r[vkcap] REPLAY | {:.1f} fps | buf {:.1f}s ({:.1f} MiB) | saved {} | compute {:.2f} ms/f   ",
                        inst_fps, rst.buffered_sec,
                        static_cast<double>(rst.buffered_bytes) / (1024.0 * 1024.0),
                        rst.clips_saved, avg_comp_ms)
                              << std::flush;
                } else {
                    const int total_sec = static_cast<int>(elapsed_total);
                    const int hrs = total_sec / 3600;
                    const int mins = (total_sec % 3600) / 60;
                    const int secs = total_sec % 60;
                    const std::string time_str = (hrs > 0)
                        ? std::format("{}:{:02d}:{:02d}", hrs, mins, secs)
                        : std::format("{:02d}:{:02d}", mins, secs);
                    const double mib = static_cast<double>(est.bytes_written) / (1024.0 * 1024.0);
                    std::cerr << std::format(
                        "\r[vkcap] {} | {:.1f} fps | {} frames | {} dropped | {:.2f} Mbps ({:.1f} MiB) | compute {:.2f} ms/f   ",
                        time_str, inst_fps, est.frames_submitted, encode_queue.dropped(), mbps, mib, avg_comp_ms)
                              << std::flush;
                }
                last_enc_frames = est.frames_submitted;
                last_report = now;
            }
        }
    });

    // Stage 1: Precision clock-paced KMS grab + isolated low-priority GPU compute
    pthread_setname_np(pthread_self(), "vkcap");
    ::prctl(PR_SET_TIMERSLACK, 1, 0, 0, 0);
    const int64_t frame_interval_ns = 1'000'000'000LL / std::max<uint32_t>(1u, cfg.fps);
    int64_t base_ns = monotonic_now_ns();
    std::atomic<int64_t> pause_offset_ns{0};
    int64_t frame_idx = 0;

    // Stage 1b: Wall-clock mastered Opus audio encoder thread (locked to exact same base_ns)
    std::atomic<bool> audio_stop{false};
    std::jthread audio_thread;
    if (audio) {
        audio->reset_sync();
        const int a_frame_sz = encoder->audio_frame_size();
        const uint32_t a_rate = audio->sample_rate();
        audio_thread = std::jthread([&, base_ns, a_frame_sz, a_rate](std::stop_token) {
            pthread_setname_np(pthread_self(), "vkcap-aenc");
            ::prctl(PR_SET_TIMERSLACK, 1, 0, 0, 0);
            std::vector<float> pcm(static_cast<size_t>(a_frame_sz) * 2);
            int64_t a_idx = 0;
            while (!audio_stop.load(std::memory_order_acquire) &&
                   !stop_flag.load(std::memory_order_acquire)) {
                if (signals.paused.load(std::memory_order_acquire)) {
                    sleep_until_monotonic_ns(monotonic_now_ns() + 20'000'000LL);
                    continue;
                }
                const int64_t target_ns = base_ns + pause_offset_ns.load(std::memory_order_acquire) +
                    ((a_idx + 1) * static_cast<int64_t>(a_frame_sz) * 1'000'000'000LL) /
                    static_cast<int64_t>(a_rate);
                if (monotonic_now_ns() < target_ns) {
                    sleep_until_monotonic_ns(target_ns);
                }
                if (audio_stop.load(std::memory_order_acquire) ||
                    stop_flag.load(std::memory_order_acquire)) {
                    break;
                }
                if (signals.paused.load(std::memory_order_acquire)) {
                    continue;
                }
                audio->read_mixed_frames(pcm.data(), static_cast<size_t>(a_frame_sz));
                auto aerr = encoder->encode_audio_samples(
                    pcm.data(), a_frame_sz, a_idx * static_cast<int64_t>(a_frame_sz));
                if (!aerr) {
                    record_err(aerr.error());
                    break;
                }
                ++a_idx;
            }
        });
    }

    // Submit the already-grabbed first frame immediately
    {
        auto gpu_res = vk->process_frame(*first_res);
        if (!gpu_res) {
            record_err(gpu_res.error());
        } else {
            encode_queue.push(*gpu_res, frame_idx++);
        }
    }

    ScopedRawKeyReader key_reader(encoder->replay_mode());
    const int64_t end_ns = (cfg.duration_sec > 0.0)
        ? (base_ns + static_cast<int64_t>(cfg.duration_sec * 1e9))
        : 0;

    while (!stop_flag.load(std::memory_order_acquire)) {
        // Check non-blocking terminal keypresses in instant-replay mode (~6 Hz, zero overhead)
        if (key_reader.active && (frame_idx % 10) == 0) {
            const char ch = key_reader.poll_key();
            if (ch == 's' || ch == 'S' || ch == '\n' || ch == ' ') {
                signals.save_replay.store(true, std::memory_order_release);
            } else if (ch == 'p' || ch == 'P') {
                bool cur = signals.paused.load(std::memory_order_relaxed);
                signals.paused.store(!cur, std::memory_order_release);
            } else if (ch == 'q' || ch == 'Q') {
                stop_flag.store(true, std::memory_order_release);
                break;
            }
        }

        // Trigger zero-copy async replay save on SIGUSR2 or keypress
        if (encoder->replay_mode() && signals.save_replay.exchange(false, std::memory_order_acq_rel)) {
            auto sres = encoder->save_replay_async();
            if (!sres) {
                std::cerr << "\n[vkcap] Replay save skipped: " << sres.error() << '\n' << std::flush;
            } else {
                std::cerr << "\n[vkcap] Saving replay -> '" << *sres << "'...\n" << std::flush;
            }
        }

        // Handle SIGUSR1 pause/resume cleanly while preserving continuous A/V timestamps
        if (signals.paused.load(std::memory_order_acquire)) {
            const int64_t p_start = monotonic_now_ns();
            std::cerr << "\n[vkcap] Paused (send SIGUSR1 to resume)...\n" << std::flush;
            while (signals.paused.load(std::memory_order_acquire) &&
                   !stop_flag.load(std::memory_order_acquire)) {
                sleep_until_monotonic_ns(monotonic_now_ns() + 50'000'000LL);
            }
            if (stop_flag.load(std::memory_order_acquire)) break;
            const int64_t p_dur = monotonic_now_ns() - p_start;
            pause_offset_ns.fetch_add(p_dur, std::memory_order_release);
            if (audio) audio->reset_sync();
            std::cerr << "[vkcap] Resumed.\n" << std::flush;
            continue;
        }

        const int64_t p_off = pause_offset_ns.load(std::memory_order_relaxed);
        int64_t target_ns = base_ns + p_off + (frame_idx * 1'000'000'000LL) / static_cast<int64_t>(cfg.fps);
        int64_t now_ns = monotonic_now_ns();
        if (end_ns > 0 && now_ns >= (end_ns + p_off)) {
            break;
        }
        if (now_ns < target_ns) {
            sleep_until_monotonic_ns(target_ns);
        } else if (now_ns - target_ns > frame_interval_ns * 2) {
            base_ns = now_ns - p_off - (frame_idx * 1'000'000'000LL) / static_cast<int64_t>(cfg.fps);
        }

        if (stop_flag.load(std::memory_order_acquire)) break;

        auto frame_res = kms->grab_frame();
        if (!frame_res) {
            // Brief transient miss during mode-set / fullscreen transition: skip tick cleanly
            if (cfg.verbose) {
                std::cerr << "\n[kms] transient grab miss: " << frame_res.error() << '\n';
            }
            frame_idx++;
            continue;
        }
        if (eff_crop.valid()) {
            frame_res->crop = eff_crop;
        }

        auto gpu_res = vk->process_frame(*frame_res);
        if (!gpu_res) {
            record_err(gpu_res.error());
            break;
        }
        encode_queue.push(*gpu_res, frame_idx++);
    }

    audio_stop.store(true, std::memory_order_release);
    if (audio_thread.joinable()) audio_thread.join();
    encode_queue.close();
    encode_thread.join();

    std::cerr << "\n";
    auto fin = encoder->finish();
    if (!fin) return std::unexpected(fin.error());

    auto est = encoder->stats();
    auto cst = vk->stats();
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_wall).count();
    double avg_fps = elapsed > 0.0 ? static_cast<double>(est.frames_submitted) / elapsed : 0.0;
    double avg_comp_ms = cst.frames_processed > 0
        ? (static_cast<double>(cst.total_compute_ns) / static_cast<double>(cst.frames_processed)) / 1e6
        : 0.0;
    if (encoder->replay_mode()) {
        auto rst = encoder->replay_stats();
        std::cerr << std::format(
            "[vkcap] Instant Replay stopped ({} frames buffered, {:.1f}s, {} clip(s) saved, compute {:.2f} ms/f)\n",
            est.frames_submitted, elapsed, rst.clips_saved, avg_comp_ms);
    } else {
        std::cerr << std::format(
            "[vkcap] Saved '{}' ({} frames, {:.1f}s, {:.1f} avg fps, {} dropped, {:.2f} MiB, compute {:.2f} ms/f)\n",
            cfg.output_path, est.frames_submitted, elapsed, avg_fps, encode_queue.dropped(),
            static_cast<double>(est.bytes_written) / (1024.0 * 1024.0), avg_comp_ms);
    }

    std::lock_guard lk(err_mtx);
    if (!async_err.empty()) return std::unexpected(async_err);
    return {};
}

}  // namespace vkcap
