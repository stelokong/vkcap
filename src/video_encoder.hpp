#pragma once

#include "cli.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
}

namespace vkcap {

struct EncoderStats {
    uint64_t frames_submitted = 0;
    uint64_t packets_written = 0;
    uint64_t bytes_written = 0;
};

struct ReplayStats {
    uint64_t buffered_bytes = 0;
    double buffered_sec = 0.0;
    uint64_t clips_saved = 0;
};

class VideoEncoder {
public:
    VideoEncoder() = default;
    ~VideoEncoder();

    VideoEncoder(const VideoEncoder &) = delete;
    VideoEncoder &operator=(const VideoEncoder &) = delete;

    static std::expected<std::unique_ptr<VideoEncoder>, std::string> open(
        const Config &cfg,
        uint32_t width,
        uint32_t height,
        AVBufferRef *hw_device_ref,
        AVBufferRef *hw_frames_ref,
        bool enable_audio = false);

    std::expected<void, std::string> encode_frame(AVFrame *frame, int64_t pts_ticks);
    std::expected<void, std::string> encode_audio_samples(const float *interleaved_stereo,
                                                          int nb_samples,
                                                          int64_t pts_samples);
    std::expected<void, std::string> finish();

    bool audio_enabled() const noexcept;
    int audio_frame_size() const noexcept;
    Codec active_codec() const noexcept;
    bool replay_mode() const noexcept;
    ReplayStats replay_stats() const noexcept;
    /// Takes an O(1) zero-copy refcounted snapshot of the rolling replay buffer and muxes it
    /// to disk on a low-priority background thread without interrupting live capture/encoding.
    std::expected<std::string, std::string> save_replay_async(std::string_view out_path_override = {});
    EncoderStats stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vkcap
