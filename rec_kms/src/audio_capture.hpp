#pragma once

#include <cstddef>
#include <cstdint>
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

    /// Resets ring buffers to synchronize sample 0 with the video recording start timestamp.
    void reset_sync() noexcept;

    /// Reads exactly `nb_frames` interleaved stereo float32 frames (2 * nb_frames floats)
    /// into `dst`, mixing system audio + microphone (when enabled) with soft-clipping and
    /// silence-padding if a stream is temporarily idle/suspended.
    void read_mixed_frames(float *dst, size_t nb_frames) noexcept;

    uint32_t sample_rate() const noexcept;
    uint32_t channels() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vkcap
