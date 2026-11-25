#include "pipeline.hpp"

#include "pipewire_capture.hpp"
#include "portal.hpp"
#include "video_encoder.hpp"
#include "vulkan_compute.hpp"

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <format>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <thread>

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

/// Latest-frame mailbox for precision CFR pacing.
/// Stores a reference to the newest GPU-resident AVFrame* without ever holding a PipeWire buffer.
class LatestFrameMailbox {
public:
    ~LatestFrameMailbox() { close(); }

    void store(AVFrame *frame) {
        AVFrame *old = nullptr;
        {
            std::lock_guard lk(mtx_);
            if (closed_) {
                av_frame_free(&frame);
                return;
            }
            old = latest_;
            latest_ = frame;
            generation_++;
        }
        if (old) av_frame_free(&old);
        cv_.notify_one();
    }

    bool wait_first() {
        std::unique_lock lk(mtx_);
        cv_.wait(lk, [this] { return closed_ || latest_ != nullptr; });
        return latest_ != nullptr && !closed_;
    }

    /// Clones a reference to the current latest GPU frame (keeps latest_ alive for static repeats).
    AVFrame *clone_latest(uint64_t *out_gen = nullptr) {
        std::lock_guard lk(mtx_);
        if (!latest_) return nullptr;
        if (out_gen) *out_gen = generation_;
        return av_frame_clone(latest_);
    }

    void close() {
        AVFrame *old = nullptr;
        {
            std::lock_guard lk(mtx_);
            closed_ = true;
            old = latest_;
            latest_ = nullptr;
        }
        if (old) av_frame_free(&old);
        cv_.notify_all();
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    AVFrame *latest_ = nullptr;
    uint64_t generation_ = 0;
    bool closed_ = false;
};

struct EncodedInput {
    AVFrame *frame = nullptr;
    int64_t pts_us = 0;
};

/// Bounded GPU-resident AVFrame queue for VFR mode.
class EncodeFrameQueue {
public:
    explicit EncodeFrameQueue(size_t capacity) : capacity_(std::max<size_t>(1, capacity)) {}

    bool is_full() const {
        std::lock_guard lk(mtx_);
        return queue_.size() >= capacity_;
    }

    void push(AVFrame *frame, int64_t pts_us) {
        AVFrame *evicted = nullptr;
        {
            std::lock_guard lk(mtx_);
            if (closed_) {
                av_frame_free(&frame);
                return;
            }
            if (queue_.size() >= capacity_) {
                evicted = queue_.front().frame;
                queue_.pop_front();
                dropped_++;
            }
            queue_.push_back({frame, pts_us});
        }
        if (evicted) av_frame_free(&evicted);
        cv_.notify_one();
    }

    bool pop(EncodedInput &out) {
        std::unique_lock lk(mtx_);
        cv_.wait(lk, [this] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) return false;
        out = queue_.front();
        queue_.pop_front();
        return true;
    }

    void close() {
        std::deque<EncodedInput> leftover;
        {
            std::lock_guard lk(mtx_);
            closed_ = true;
            leftover.swap(queue_);
        }
        cv_.notify_all();
        for (auto &item : leftover) av_frame_free(&item.frame);
    }

    void record_drop() {
        std::lock_guard lk(mtx_);
        dropped_++;
    }

    uint64_t dropped() const {
        std::lock_guard lk(mtx_);
        return dropped_;
    }

private:
    size_t capacity_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<EncodedInput> queue_;
    bool closed_ = false;
    uint64_t dropped_ = 0;
};

}  // namespace

std::expected<void, std::string> run_capture_pipeline(const Config &cfg, std::atomic<bool> &stop_flag) {
    auto vk_res = VulkanCompute::create(cfg);
    if (!vk_res) {
        return std::unexpected(vk_res.error());
    }
    auto vk = std::move(*vk_res);

    auto formats = vk->query_supported_capture_formats();
    if (formats.empty()) {
        return std::unexpected("Vulkan device reported no supported DMA-BUF capture formats");
    }

    std::cerr << "[vkcap] Requesting display selection via xdg-desktop-portal...\n";
    auto portal_res = PortalSession::acquire(cfg);
    if (!portal_res) {
        return std::unexpected(portal_res.error());
    }
    auto portal = std::move(*portal_res);
    PortalStreamInfo stream_info = portal.stream();
    int pw_fd = portal.take_pipewire_fd();

    LatestFrameMailbox mailbox;
    EncodeFrameQueue vfr_queue(cfg.queue_size);
    std::mutex err_mtx;
    std::string async_err;
    std::atomic<uint64_t> captured_frames{0};
    std::atomic<uint64_t> late_ticks{0};

    auto record_err = [&](const std::string &e) {
        std::lock_guard lk(err_mtx);
        if (async_err.empty()) async_err = e;
        stop_flag.store(true, std::memory_order_release);
    };

    std::mutex enc_init_mtx;
    std::unique_ptr<VideoEncoder> encoder;
    std::optional<int64_t> first_pts_ns;
    int64_t last_compute_ns = 0;
    const int64_t frame_interval_ns = 1'000'000'000LL / std::max<uint32_t>(1u, cfg.fps);
    const auto start_wall = std::chrono::steady_clock::now();
    auto last_report = start_wall;
    uint64_t last_enc_frames = 0;

    auto maybe_report_stats = [&](VideoEncoder *enc_ptr) {
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_report).count();
        if (dt < 1.0) return;
        auto est = enc_ptr->stats();
        auto cst = vk->stats();
        double inst_fps = static_cast<double>(est.frames_submitted - last_enc_frames) / dt;
        double elapsed_total = std::chrono::duration<double>(now - start_wall).count();
        double mbps = elapsed_total > 0.0
            ? (static_cast<double>(est.bytes_written) * 8.0 / 1'000'000.0) / elapsed_total
            : 0.0;
        double avg_comp_ms = cst.frames_processed > 0
            ? (static_cast<double>(cst.total_compute_ns) / static_cast<double>(cst.frames_processed)) / 1e6
            : 0.0;
        uint64_t drops = cfg.vfr ? vfr_queue.dropped() : late_ticks.load(std::memory_order_relaxed);
        std::cerr << std::format(
            "\r[vkcap] {:.1f} fps | {} frames | {} {} | {:.2f} Mbps | compute {:.2f} ms/f   ",
            inst_fps, est.frames_submitted, drops, cfg.vfr ? "dropped" : "late", mbps, avg_comp_ms)
                  << std::flush;
        last_enc_frames = est.frames_submitted;
        last_report = now;
    };

    // Stage 2: Dedicated Vulkan Video Encode thread (precision CFR pacer or VFR consumer)
    std::jthread encode_thread([&](std::stop_token) {
        pthread_setname_np(pthread_self(), "vkcap-enc");
        if (!cfg.vfr) {
            if (!mailbox.wait_first()) return;
            int64_t base_ns = monotonic_now_ns();
            int64_t frame_idx = 0;

            while (!stop_flag.load(std::memory_order_acquire)) {
                int64_t target_ns = base_ns + (frame_idx * 1'000'000'000LL) / static_cast<int64_t>(cfg.fps);
                int64_t now_ns = monotonic_now_ns();
                if (now_ns < target_ns) {
                    sleep_until_monotonic_ns(target_ns);
                } else if (now_ns - target_ns > frame_interval_ns * 2) {
                    // Resync pacer cleanly if system paused/spiked so we never burst-encode stale ticks
                    late_ticks.fetch_add(1, std::memory_order_relaxed);
                    base_ns = now_ns - (frame_idx * 1'000'000'000LL) / static_cast<int64_t>(cfg.fps);
                }

                if (stop_flag.load(std::memory_order_acquire)) break;

                VideoEncoder *enc_ptr = nullptr;
                {
                    std::lock_guard lk(enc_init_mtx);
                    enc_ptr = encoder.get();
                }
                if (!enc_ptr) continue;

                AVFrame *avf = mailbox.clone_latest();
                if (!avf) break;

                auto enc_err = enc_ptr->encode_frame(avf, frame_idx);
                av_frame_free(&avf);
                if (!enc_err) {
                    record_err(enc_err.error());
                    break;
                }
                frame_idx++;
                maybe_report_stats(enc_ptr);
            }
            return;
        }

        EncodedInput item{};
        while (vfr_queue.pop(item)) {
            VideoEncoder *enc_ptr = nullptr;
            {
                std::lock_guard lk(enc_init_mtx);
                enc_ptr = encoder.get();
            }
            if (!enc_ptr) {
                av_frame_free(&item.frame);
                continue;
            }
            auto enc_err = enc_ptr->encode_frame(item.frame, item.pts_us);
            av_frame_free(&item.frame);
            if (!enc_err) {
                record_err(enc_err.error());
                break;
            }
            maybe_report_stats(enc_ptr);
        }
    });

    // Stage 1: PipeWire thread runs fast isolated compute (~0.12ms) and immediately returns pw_buffer
    CaptureCallbacks cbs{};
    cbs.on_buffer_removed = [&](void *handle) {
        vk->evict_buffer(handle);
    };
    cbs.on_error = [&](const std::string &e) {
        record_err(e);
    };
    cbs.on_frame = [&](const CapturedFrame &cf) {
        captured_frames.fetch_add(1, std::memory_order_relaxed);
        if (stop_flag.load(std::memory_order_acquire)) {
            return;
        }

        // Capture dequeue timestamp BEFORE compute so GPU jitter never corrupts VFR timestamps
        const int64_t dequeue_ns = monotonic_now_ns();

        {
            std::lock_guard lk(enc_init_mtx);
            if (!encoder) {
                uint32_t src_w = cf.crop.valid() ? static_cast<uint32_t>(cf.crop.width) : cf.format.width;
                uint32_t src_h = cf.crop.valid() ? static_cast<uint32_t>(cf.crop.height) : cf.format.height;
                uint32_t enc_w = cfg.out_width ? cfg.out_width : (src_w & ~1u);
                uint32_t enc_h = cfg.out_height ? cfg.out_height : (src_h & ~1u);

                auto cfg_res = vk->configure_output(enc_w, enc_h);
                if (!cfg_res) {
                    record_err(cfg_res.error());
                    return;
                }
                auto enc_res = VideoEncoder::open(cfg, enc_w, enc_h, vk->hw_device_ref(), vk->hw_frames_ref());
                if (!enc_res) {
                    record_err(enc_res.error());
                    return;
                }
                encoder = std::move(*enc_res);
                std::cerr << std::format(
                    "[vkcap] Recording {} ({}x{} -> {}x{} {}, {}, {} {}fps)\n",
                    cfg.output_path, src_w, src_h, enc_w, enc_h,
                    cf.mem_type == CaptureMemoryType::DmaBuf ? "DMA-BUF zero-copy" : "SHM",
                    cfg.pixfmt == PixelFormat::P010 ? "AV1 10-bit P010" : "AV1 8-bit NV12",
                    cfg.vfr ? "VFR" : "CFR", cfg.fps);
            }
        }

        if (cfg.vfr) {
            if (vfr_queue.is_full()) {
                vfr_queue.record_drop();
                return;
            }
        } else {
            // In CFR mode, avoid wasting GPU compute on intermediate frames when a 144Hz/240Hz
            // display fires much faster than the target recording fps.
            if (last_compute_ns > 0 && (dequeue_ns - last_compute_ns) < ((frame_interval_ns * 3) / 4)) {
                return;
            }
        }

        auto gpu_res = vk->process_frame(cf);
        if (!gpu_res) {
            record_err(gpu_res.error());
            return;
        }
        last_compute_ns = dequeue_ns;

        if (!cfg.vfr) {
            mailbox.store(*gpu_res);
        } else {
            int64_t pts_ns = cf.pts_ns > 0 ? cf.pts_ns : dequeue_ns;
            if (!first_pts_ns) first_pts_ns = pts_ns;
            int64_t pts_us = std::max<int64_t>(0, (pts_ns - *first_pts_ns) / 1000);
            vfr_queue.push(*gpu_res, pts_us);
        }
    };

    auto cap_res = PipeWireCapture::start(cfg, pw_fd, stream_info, std::move(formats), std::move(cbs));
    if (!cap_res) {
        mailbox.close();
        vfr_queue.close();
        encode_thread.join();
        return std::unexpected(cap_res.error());
    }
    auto capture = std::move(*cap_res);

    while (!stop_flag.load(std::memory_order_acquire)) {
        if (cfg.duration_sec > 0.0) {
            double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_wall).count();
            if (elapsed >= cfg.duration_sec) {
                stop_flag.store(true, std::memory_order_release);
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    capture->stop();
    mailbox.close();
    vfr_queue.close();
    encode_thread.join();
    portal.close();

    std::cerr << "\n";
    if (encoder) {
        auto fin = encoder->finish();
        if (!fin) {
            return std::unexpected(fin.error());
        }
        auto est = encoder->stats();
        auto cst = vk->stats();
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_wall).count();
        double avg_fps = elapsed > 0.0 ? static_cast<double>(est.frames_submitted) / elapsed : 0.0;
        double avg_comp_ms = cst.frames_processed > 0
            ? (static_cast<double>(cst.total_compute_ns) / static_cast<double>(cst.frames_processed)) / 1e6
            : 0.0;
        uint64_t drops = cfg.vfr ? vfr_queue.dropped() : late_ticks.load(std::memory_order_relaxed);
        std::cerr << std::format(
            "[vkcap] Saved '{}' ({} frames, {:.1f}s, {:.1f} avg fps, {} {}, {:.2f} MiB, compute {:.2f} ms/f)\n",
            cfg.output_path, est.frames_submitted, elapsed, avg_fps, drops,
            cfg.vfr ? "dropped" : "late",
            static_cast<double>(est.bytes_written) / (1024.0 * 1024.0), avg_comp_ms);
    }

    std::lock_guard lk(err_mtx);
    if (!async_err.empty()) {
        return std::unexpected(async_err);
    }
    return {};
}

}  // namespace vkcap
