#pragma once

#include "cli.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <string>

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
    EncoderStats stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vkcap
