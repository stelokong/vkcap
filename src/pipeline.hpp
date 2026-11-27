#pragma once

#include "cli.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <expected>
#include <string>

namespace vkcap {

struct PipelineSignals {
    std::atomic<bool> stop{false};
    std::atomic<bool> paused{false};
    std::atomic<bool> save_replay{false};
};

/// Drift-free CLOCK_MONOTONIC frame pacer with rational CRTC vblank phase-locking.
/// - Base timeline (`base_ns`) is immutable so audio and video remain strictly wall-clock locked.
/// - On stalls (>2 frames late), advances `frame_idx` rather than shifting `base_ns`.
/// - Works across arbitrary refresh rates: finds the smallest reduced rational `p/q` (q in 1..12)
///   matching `refresh_hz / fps` (e.g. 59.94/60 -> 1/1, 144/60 -> 12/5, 165/60 -> 11/4,
///   100/60 -> 5/3, 280/120 -> 7/3) with comb spacing `vblank_period / q >= 1 ms`.
///   - For integer ratios (q = 1): phase-locks capture wake-up to ~1.5 ms after vblank.
///   - For rational ratios (q > 1): centers the entire q-frame beat comb halfway between vblank
///     commits (`lead = lock_period / 2`), eliminating beat-edge coin-flipping on 144/165/280 Hz.
/// - Wraps `phase_offset_ns_` atomically across `+-lock_period/2` on fractional clock skew
///   (e.g. 59.94 Hz, 143.98 Hz) instead of clamping or slewing across vblank.
/// - Automatically ignores stale (PSR) or VRR-varying vblank timestamps and gently relaxes
///   phase back to 0 during sustained VRR.
class FramePacer {
public:
    FramePacer(int64_t base_ns, uint32_t fps, double refresh_hz) noexcept
        : base_ns_(base_ns),
          fps_(std::max<uint32_t>(1u, fps)),
          frame_interval_ns_(1'000'000'000LL / static_cast<int64_t>(fps_)) {
        configure_refresh_hz(refresh_hz);
    }

    int64_t base_ns() const noexcept { return base_ns_; }
    int64_t frame_interval_ns() const noexcept { return frame_interval_ns_; }
    int64_t phase_offset_ns() const noexcept { return phase_offset_ns_; }
    bool vblank_lock_eligible() const noexcept { return vblank_lock_eligible_; }
    uint32_t vblank_denominator() const noexcept { return vblank_den_; }
    uint64_t vblank_wraps() const noexcept { return vblank_wraps_; }

    /// Re-configures vblank phase-locking if the display changes refresh rate mid-recording
    /// (e.g. fullscreen game modeset) without mutating `base_ns_`.
    void update_refresh_hz(double new_refresh_hz) noexcept {
        if (new_refresh_hz <= 1.0 || std::abs(new_refresh_hz - refresh_hz_) <= 0.05) {
            return;
        }
        configure_refresh_hz(new_refresh_hz);
        phase_offset_ns_ = 0;
        last_vblank_ns_ = 0;
        vblank_locked_ = false;
        vrr_unstable_streak_ = 0;
    }

    int64_t next_target_ns(int64_t frame_idx, int64_t pause_offset_ns) const noexcept {
        const int64_t f = static_cast<int64_t>(fps_);
        const int64_t tick_ns = (frame_idx / f) * 1'000'000'000LL +
                                ((frame_idx % f) * 1'000'000'000LL) / f;
        return base_ns_ + pause_offset_ns + phase_offset_ns_ + tick_ns;
    }

    /// If capture fell behind by more than 2 frames, advances `frame_idx` to the current
    /// wall-clock tick without mutating `base_ns` (preserving exact A/V sync).
    bool recover_if_stalled(int64_t &frame_idx, int64_t now_ns, int64_t pause_offset_ns) const noexcept {
        const int64_t target_ns = next_target_ns(frame_idx, pause_offset_ns);
        if (now_ns - target_ns <= frame_interval_ns_ * 2) {
            return false;
        }
        const int64_t elapsed_ns = now_ns - base_ns_ - pause_offset_ns - phase_offset_ns_;
        if (elapsed_ns > 0) {
            const int64_t f = static_cast<int64_t>(fps_);
            const int64_t expected_idx = (elapsed_ns / 1'000'000'000LL) * f +
                                         ((elapsed_ns % 1'000'000'000LL) * f) / 1'000'000'000LL;
            if (expected_idx > frame_idx) {
                frame_idx = expected_idx;
                return true;
            }
        }
        return false;
    }

    /// Feeds a non-blocking CRTC vblank timestamp (CLOCK_MONOTONIC ns) to nudge capture phase
    /// safely away from vblank commits across both integer and rational refresh-rate ratios.
    void observe_vblank(int64_t vblank_ns, int64_t frame_idx,
                        int64_t now_ns, int64_t pause_offset_ns) noexcept {
        if (!vblank_lock_eligible_ || vblank_ns <= 0 || vblank_period_ns_ <= 0 || lock_period_ns_ <= 0) {
            return;
        }
        // Ignore stale timestamps (e.g. PSR / idle display) or future timestamps
        const int64_t age_ns = now_ns - vblank_ns;
        if (age_ns < 0 || age_ns > vblank_period_ns_ * 2) {
            return;
        }
        // Detect VRR / unstable refresh cadence (>5% deviation from nominal vblank period)
        if (last_vblank_ns_ > 0 && vblank_ns > last_vblank_ns_) {
            const int64_t dt = vblank_ns - last_vblank_ns_;
            const int64_t n = std::max<int64_t>(
                1, static_cast<int64_t>(std::llround(static_cast<double>(dt) / static_cast<double>(vblank_period_ns_))));
            const int64_t avg_period = dt / n;
            if (std::abs(avg_period - vblank_period_ns_) > vblank_period_ns_ / 20) {
                last_vblank_ns_ = vblank_ns;
                ++vrr_unstable_streak_;
                // On sustained VRR (3+ unstable intervals), gently relax phase_offset_ns_ back to 0
                if (vrr_unstable_streak_ >= 3 && phase_offset_ns_ != 0) {
                    constexpr int64_t kRelaxStepNs = 250'000LL;
                    if (std::abs(phase_offset_ns_) <= kRelaxStepNs) {
                        phase_offset_ns_ = 0;
                    } else {
                        phase_offset_ns_ += (phase_offset_ns_ > 0) ? -kRelaxStepNs : kRelaxStepNs;
                    }
                    vblank_locked_ = false;
                }
                return;
            }
            vrr_unstable_streak_ = 0;
        }
        last_vblank_ns_ = vblank_ns;

        const int64_t half_period = lock_period_ns_ / 2;
        const int64_t target_ns = next_target_ns(frame_idx, pause_offset_ns);
        const int64_t ideal_ns = vblank_ns + vblank_lead_ns_;
        int64_t err_ns = (ideal_ns - target_ns) % lock_period_ns_;
        if (err_ns > half_period) err_ns -= lock_period_ns_;
        if (err_ns < -half_period) err_ns += lock_period_ns_;

        if (!vblank_locked_) {
            phase_offset_ns_ = std::clamp(err_ns, -half_period, half_period);
            vblank_locked_ = true;
            return;
        }
        const int64_t max_slew = std::min<int64_t>(500'000LL, std::max<int64_t>(100'000LL, half_period / 4));
        const int64_t step = std::clamp(err_ns / 4, -max_slew, max_slew);
        int64_t next_phase = phase_offset_ns_ + step;
        // Atomic wrap on fractional refresh skew (e.g. 59.94 Hz vs 60 fps, 143.98 Hz vs 60 fps):
        // avoids clamping and multi-frame slew across vblank while keeping |phase_offset_ns_| <= half_period.
        if (next_phase > half_period) {
            next_phase -= lock_period_ns_;
            ++vblank_wraps_;
        } else if (next_phase < -half_period) {
            next_phase += lock_period_ns_;
            ++vblank_wraps_;
        }
        phase_offset_ns_ = std::clamp(next_phase, -half_period, half_period);
    }

    void on_resume() noexcept {
        last_vblank_ns_ = 0;
        vblank_locked_ = false;
        vrr_unstable_streak_ = 0;
    }

private:
    void configure_refresh_hz(double refresh_hz) noexcept {
        refresh_hz_ = refresh_hz;
        vblank_lock_eligible_ = false;
        vblank_period_ns_ = 0;
        lock_period_ns_ = 0;
        vblank_lead_ns_ = 0;
        vblank_den_ = 0;
        if (refresh_hz <= 1.0) return;

        const double ratio = refresh_hz / static_cast<double>(fps_);
        const int64_t vb_period = static_cast<int64_t>(std::llround(1e9 / refresh_hz));
        if (vb_period <= 0) return;

        // Search smallest rational denominator q in 1..12 where p/q matches refresh_hz/fps
        // within 1.5% (for q=1) or 0.5% (for q>1) and comb spacing >= 1.0 ms.
        constexpr int64_t kMinCombPeriodNs = 1'000'000LL;
        for (uint32_t q = 1; q <= 12; ++q) {
            const int64_t sub_period = vb_period / static_cast<int64_t>(q);
            if (sub_period < kMinCombPeriodNs) break;
            const double p = std::round(ratio * static_cast<double>(q));
            if (p < 1.0) continue;
            const double approx = p / static_cast<double>(q);
            const double tol = (q == 1) ? 0.015 : 0.0025;
            if (std::abs(ratio - approx) / approx <= tol) {
                vblank_lock_eligible_ = true;
                vblank_period_ns_ = vb_period;
                lock_period_ns_ = sub_period;
                vblank_den_ = q;
                if (q == 1) {
                    vblank_lead_ns_ = std::min<int64_t>(1'500'000LL, std::max<int64_t>(250'000LL, vb_period / 8));
                } else {
                    // Center beat comb maximally far from every vblank commit
                    vblank_lead_ns_ = sub_period / 2;
                }
                return;
            }
        }
    }

    int64_t base_ns_ = 0;
    uint32_t fps_ = 60;
    double refresh_hz_ = 60.0;
    int64_t frame_interval_ns_ = 16'666'666LL;
    bool vblank_lock_eligible_ = false;
    bool vblank_locked_ = false;
    int64_t vblank_period_ns_ = 0;
    int64_t lock_period_ns_ = 0;
    int64_t vblank_lead_ns_ = 0;
    uint32_t vblank_den_ = 0;
    int64_t phase_offset_ns_ = 0;
    int64_t last_vblank_ns_ = 0;
    uint32_t vrr_unstable_streak_ = 0;
    uint64_t vblank_wraps_ = 0;
};

std::expected<void, std::string> run_kms_pipeline(const Config &cfg, PipelineSignals &signals);

}  // namespace vkcap
