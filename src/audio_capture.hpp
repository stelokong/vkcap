#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace vkcap {

struct AudioDeviceInfo {
    uint32_t id = 0;
    std::string name;         // PipeWire node.name (stable target.object key)
    std::string description;  // Human-readable label (node.description / node.nick)
};

struct AudioCaptureConfig {
    bool capture_system = true;
    bool capture_mic = false;
    std::string mic_target;   // Empty or "default" = system default mic; otherwise node.name, ID, or substring
    float system_gain = 1.0f; // Linear gain multiplier (e.g. 1.0 = 100%, 2.0 = 200%)
    float mic_gain = 1.0f;    // Linear gain multiplier (e.g. 1.0 = 100%, 2.0 = 200%)
    uint32_t sample_rate = 48000;
    uint32_t channels = 2;
    bool verbose = false;
};

struct AudioSyncStats {
    uint64_t underruns = 0;
    uint64_t overrun_trims = 0;
    int32_t ratio_ppm = 0;    // Converged PI clock-skew compensation in parts-per-million
};

/// Lock-free SPSC stereo float32 ring buffer paired with a PI-controlled 4-point Catmull-Rom
/// fractional resampler. Smoothly absorbs hardware audio crystal skew (+-100+ ppm vs CLOCK_MONOTONIC)
/// with zero cumulative A/V drift and zero audible clicks/dropouts.
class AdaptiveAudioBuffer {
public:
    static constexpr size_t kCapFrames = 65536;  // ~1.36s @ 48 kHz (power of two)
    static constexpr size_t kMask = kCapFrames - 1;
    // ~42.7 ms cushion safely absorbs 256/512/960/1024/2048-frame PipeWire quanta without underruns
    static constexpr size_t kDefaultTargetCushion = 2048;

    explicit AdaptiveAudioBuffer(size_t target_cushion_frames = kDefaultTargetCushion) noexcept
        : target_cushion_(target_cushion_frames) {}

    /// Discards buffered samples and resets resampler state (call on recording start / resume).
    void reset() noexcept {
        const uint64_t w = write_pos_.load(std::memory_order_acquire);
        read_pos_.store(w, std::memory_order_release);
        primed_ = false;
        was_idle_ = true;
        frac_ = 0.0;
        err_ema_ = 0.0;
        err_ema2_ = 0.0;
        integral_ = 0.0;
        ratio_ = 1.0;
        fade_in_rem_ = 0;
        h_ = {};
        ratio_ppm_.store(0, std::memory_order_relaxed);
    }

    size_t available_frames() const noexcept {
        const uint64_t w = write_pos_.load(std::memory_order_acquire);
        const uint64_t r = read_pos_.load(std::memory_order_relaxed);
        return (w >= r) ? static_cast<size_t>(w - r) : 0;
    }

    /// Producer (PipeWire RT callback): lock-free, allocation-free push of interleaved stereo frames.
    void push_frames(const float *interleaved, size_t nframes) noexcept {
        if (nframes == 0) return;
        const uint64_t w = write_pos_.load(std::memory_order_relaxed);
        const uint64_t r = read_pos_.load(std::memory_order_acquire);
        const uint64_t used = (w >= r) ? (w - r) : 0;
        const size_t free_space = (used < kCapFrames) ? static_cast<size_t>(kCapFrames - used) : 0;
        const size_t to_write = std::min(nframes, free_space);
        if (to_write == 0) return;

        const size_t idx = static_cast<size_t>(w & kMask);
        const size_t first = std::min(to_write, kCapFrames - idx);
        std::memcpy(&buf_[idx * 2], interleaved, first * 2 * sizeof(float));
        if (to_write > first) {
            std::memcpy(&buf_[0], interleaved + first * 2, (to_write - first) * 2 * sizeof(float));
        }
        write_pos_.store(w + to_write, std::memory_order_release);
    }

    /// Consumer (vkcap-aenc): produces exactly `nb_frames` interleaved stereo float32 frames,
    /// dynamically resampling via PI feedback to hold ring fill at `target_cushion_` frames.
    void pop_resampled(float *dst, size_t nb_frames) noexcept {
        if (nb_frames == 0) return;

        uint64_t r = read_pos_.load(std::memory_order_relaxed);
        const uint64_t w = write_pos_.load(std::memory_order_acquire);
        uint64_t avail = (w >= r) ? (w - r) : 0;

        constexpr size_t kFadeLen = 64;  // ~1.33 ms click-free ramp @ 48 kHz

        // Idle / suspended PipeWire sink: output clean silence without winding up PI integrator
        if (avail == 0) {
            if (!was_idle_) {
                underruns_.fetch_add(1, std::memory_order_relaxed);
                const size_t ramp = std::min(nb_frames, kFadeLen);
                const float inv_ramp = 1.0f / static_cast<float>(ramp);
                for (size_t i = 0; i < ramp; ++i) {
                    const float g = 1.0f - static_cast<float>(i + 1) * inv_ramp;
                    dst[i * 2] = h_[1][0] * g;
                    dst[i * 2 + 1] = h_[1][1] * g;
                }
                if (ramp < nb_frames) {
                    std::memset(dst + ramp * 2, 0, (nb_frames - ramp) * 2 * sizeof(float));
                }
                was_idle_ = true;
                primed_ = false;
                frac_ = 0.0;
                err_ema_ = 0.0;
                err_ema2_ = 0.0;
                integral_ = 0.0;
                ratio_ = 1.0;
                h_ = {};
                ratio_ppm_.store(0, std::memory_order_relaxed);
            } else {
                std::memset(dst, 0, nb_frames * 2 * sizeof(float));
            }
            return;
        }

        const uint64_t desired_pre_pop = static_cast<uint64_t>(nb_frames + target_cushion_);
        const uint64_t hard_max_backlog = static_cast<uint64_t>(nb_frames + target_cushion_ * 4);
        const uint64_t min_start_frames = static_cast<uint64_t>(nb_frames + (target_cushion_ * 3) / 4);

        if (was_idle_) {
            // Wait until the jitter cushion has filled before starting playback so we never
            // enter a repeated underrun/fade-in loop when PipeWire starts or wakes from suspend.
            if (avail < min_start_frames) {
                std::memset(dst, 0, nb_frames * 2 * sizeof(float));
                return;
            }
            if (avail > desired_pre_pop + target_cushion_) {
                const uint64_t skip = avail - desired_pre_pop;
                r += skip;
                avail -= skip;
            }
            was_idle_ = false;
            primed_ = false;
            frac_ = 0.0;
            err_ema_ = 0.0;
            err_ema2_ = 0.0;
            integral_ = 0.0;
            ratio_ = 1.0;
        } else if (avail > hard_max_backlog) {
            // Extreme OS stall (>190 ms backlog): hard-trim back to target cushion with fade-in
            const uint64_t skip = avail - desired_pre_pop;
            r += skip;
            avail -= skip;
            overrun_trims_.fetch_add(1, std::memory_order_relaxed);
            primed_ = false;
            frac_ = 0.0;
            err_ema_ = 0.0;
            err_ema2_ = 0.0;
        } else {
            // Steady-state PI controller locking audio crystal to CLOCK_MONOTONIC.
            // 2-pole cascaded EMA (alpha=0.04) attenuates 1024/2048-frame PipeWire quantum
            // sawtooth by >60 dB (<10 ppm ripple) for zero audible pitch flutter.
            const double err = static_cast<double>(static_cast<int64_t>(avail) -
                                                   static_cast<int64_t>(desired_pre_pop));
            err_ema_ += 0.04 * (err - err_ema_);
            err_ema2_ += 0.04 * (err_ema_ - err_ema2_);

            constexpr double kMaxSkew = 0.005;                // +-0.5% (+-5000 ppm, inaudible)
            constexpr double kKp = 1.0 / (48000.0 * 4.0);     // smooth proportional (<10 ppm ripple @ 2048q)
            constexpr double kKi = 1.0 / (48000.0 * 1600.0);  // critically damped (zeta ~ 0.71) crystal lock
            integral_ = std::clamp(integral_ + err_ema2_ * kKi, -kMaxSkew, kMaxSkew);
            ratio_ = std::clamp(1.0 + err_ema2_ * kKp + integral_, 1.0 - kMaxSkew, 1.0 + kMaxSkew);
            ratio_ppm_.store(static_cast<int32_t>(std::llround((ratio_ - 1.0) * 1'000'000.0)),
                             std::memory_order_relaxed);
        }

        auto fetch_one = [&](std::array<float, 2> &out) noexcept -> bool {
            if (avail == 0) return false;
            const size_t idx = static_cast<size_t>(r & kMask);
            out[0] = buf_[idx * 2];
            out[1] = buf_[idx * 2 + 1];
            ++r;
            --avail;
            return true;
        };

        if (!primed_) {
            std::array<float, 2> s0{0.0f, 0.0f};
            std::array<float, 2> s1{0.0f, 0.0f};
            fetch_one(s0);
            if (!fetch_one(s1)) s1 = s0;
            h_[0] = s0;
            h_[1] = s0;
            h_[2] = s1;
            if (!fetch_one(h_[3])) h_[3] = s1;
            primed_ = true;
            frac_ = 0.0;
            fade_in_rem_ = kFadeLen;
        }

        bool starved = false;
        size_t starve_idx = nb_frames;

        for (size_t i = 0; i < nb_frames; ++i) {
            const float t = static_cast<float>(frac_);
            float l = catmull_rom(h_[0][0], h_[1][0], h_[2][0], h_[3][0], t);
            float rv = catmull_rom(h_[0][1], h_[1][1], h_[2][1], h_[3][1], t);
            if (fade_in_rem_ > 0) {
                const float g = static_cast<float>(kFadeLen - fade_in_rem_ + 1) / static_cast<float>(kFadeLen);
                l *= g;
                rv *= g;
                --fade_in_rem_;
            }
            dst[i * 2] = l;
            dst[i * 2 + 1] = rv;

            frac_ += ratio_;
            while (frac_ >= 1.0) {
                frac_ -= 1.0;
                h_[0] = h_[1];
                h_[1] = h_[2];
                h_[2] = h_[3];
                if (!fetch_one(h_[3])) {
                    h_[3] = h_[2];
                    if (i + 1 < nb_frames) {
                        starved = true;
                        starve_idx = i + 1;
                    }
                    break;
                }
            }
            if (starved) break;
        }

        read_pos_.store(r, std::memory_order_release);

        if (starved) {
            underruns_.fetch_add(1, std::memory_order_relaxed);
            const size_t rem = nb_frames - starve_idx;
            const size_t ramp = std::min(rem, kFadeLen);
            if (ramp > 0) {
                const float last_l = (starve_idx > 0) ? dst[(starve_idx - 1) * 2] : 0.0f;
                const float last_r = (starve_idx > 0) ? dst[(starve_idx - 1) * 2 + 1] : 0.0f;
                const float inv_ramp = 1.0f / static_cast<float>(ramp);
                for (size_t j = 0; j < ramp; ++j) {
                    const float g = 1.0f - static_cast<float>(j + 1) * inv_ramp;
                    dst[(starve_idx + j) * 2] = last_l * g;
                    dst[(starve_idx + j) * 2 + 1] = last_r * g;
                }
            }
            if (starve_idx + ramp < nb_frames) {
                std::memset(dst + (starve_idx + ramp) * 2, 0,
                            (nb_frames - starve_idx - ramp) * 2 * sizeof(float));
            }
            primed_ = false;
            was_idle_ = true;
            frac_ = 0.0;
        }
    }

    AudioSyncStats stats() const noexcept {
        return {
            underruns_.load(std::memory_order_relaxed),
            overrun_trims_.load(std::memory_order_relaxed),
            ratio_ppm_.load(std::memory_order_relaxed),
        };
    }

    double current_ratio() const noexcept { return ratio_; }

private:
    static inline float catmull_rom(float y0, float y1, float y2, float y3, float t) noexcept {
        const float a0 = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
        const float a1 = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
        const float a2 = -0.5f * y0 + 0.5f * y2;
        const float a3 = y1;
        return ((a0 * t + a1) * t + a2) * t + a3;
    }

    size_t target_cushion_ = kDefaultTargetCushion;
    std::array<float, kCapFrames * 2> buf_{};
    alignas(64) std::atomic<uint64_t> write_pos_{0};
    alignas(64) std::atomic<uint64_t> read_pos_{0};

    // Consumer-only state (vkcap-aenc thread)
    std::array<std::array<float, 2>, 4> h_{};
    bool primed_ = false;
    bool was_idle_ = true;
    size_t fade_in_rem_ = 0;
    double frac_ = 0.0;
    double err_ema_ = 0.0;
    double err_ema2_ = 0.0;
    double integral_ = 0.0;
    double ratio_ = 1.0;

    std::atomic<uint64_t> underruns_{0};
    std::atomic<uint64_t> overrun_trims_{0};
    std::atomic<int32_t> ratio_ppm_{0};
};

class AudioCapture {
public:
    AudioCapture() = default;
    ~AudioCapture();

    AudioCapture(const AudioCapture &) = delete;
    AudioCapture &operator=(const AudioCapture &) = delete;

    /// Enumerates PipeWire Audio/Source (microphone) nodes without requiring elevated privileges.
    static std::expected<std::vector<AudioDeviceInfo>, std::string> enumerate_microphones();

    /// Starts low-latency PipeWire capture stream(s) on a dedicated real-time thread loop.
    static std::expected<std::unique_ptr<AudioCapture>, std::string> start(const AudioCaptureConfig &cfg);

    /// Resets ring buffers and PI resamplers to synchronize sample 0 with the video start timestamp.
    void reset_sync() noexcept;

    /// Reads exactly `nb_frames` interleaved stereo float32 frames (2 * nb_frames floats)
    /// into `dst`, independently drift-resampling and mixing system audio + microphone
    /// (when enabled) with soft-clipping and click-free silence handling.
    void read_mixed_frames(float *dst, size_t nb_frames) noexcept;

    uint32_t sample_rate() const noexcept;
    uint32_t channels() const noexcept;
    AudioSyncStats stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vkcap
