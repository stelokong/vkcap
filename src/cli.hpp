#pragma once

#include "types.hpp"

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vkcap {

enum class Codec { Av1, Hevc, H264 };
enum class PixelFormat { Nv12, P010 };
enum class RateControl { Auto, Cqp, Cbr, Vbr };
enum class Tune { Default, Hq, Ll, Ull, Lossless };
enum class Usage { Default, Transcode, Stream, Record, Conference };
enum class Content { Default, Camera, Desktop, Rendered };
enum class ColorRange { Limited, Full };
enum class ColorSpace { Bt709, Bt2020 };
enum class InputGamut { Auto, Edid, Srgb, DisplayP3, Bt2020 };
enum class AudioCodec { Opus, Aac, Flac };

struct Config {
    std::string output_path;
    std::string display;      // Connector name (e.g. "DP-1"), index ("0"), or empty/auto
    std::string profile;      // Premade profile name (empty = default)
    bool list_displays = false;
    bool list_profiles = false;
    bool list_mics = false;
    bool interactive = false;

    uint32_t fps = 60;
    uint32_t out_width = 0;   // 0 = native capture width
    uint32_t out_height = 0;  // 0 = native capture height
    CropRect crop{};          // Optional GPU crop region (WxH+X+Y)
    double duration_sec = 0.0;
    uint32_t replay_seconds = 0;  // >0 = instant replay mode (rolling buffer in seconds)
    bool capture_cursor = true;

    // Audio capture (PipeWire low-latency, system audio ON by default, mic OFF by default)
    bool audio = true;
    bool capture_system_audio = true;
    bool capture_mic = false;
    std::string mic_device;        // Empty or "default" = system default mic
    double system_volume_pct = 100.0;  // Arbitrary percentage (e.g. 0..1000, default 100%)
    double mic_volume_pct = 100.0;     // Arbitrary percentage (e.g. 0..1000, default 100%)
    AudioCodec audio_codec = AudioCodec::Opus;
    int64_t audio_bitrate = 192'000;   // Opus/AAC bitrate in bps (default 192k)

    Codec codec = Codec::Av1;
    bool codec_explicit = false;  // True if user explicitly passed --codec
    PixelFormat pixfmt = PixelFormat::Nv12;
    RateControl rc = RateControl::Auto;
    int qp = -1;              // -1 = unset (when --qp is passed with --rc=auto, selects CQP)
    int64_t bitrate = 20'000'000;      // bps; default 20M VBR (1440p60 reference)
    int64_t max_bitrate = 32'000'000;  // bps; default 32M VBR ceiling
    int quality = -1;         // -1 = auto default (clamped to GPU maxQualityLevels)
    Tune tune = Tune::Hq;
    Usage usage = Usage::Record;
    Content content = Content::Rendered;
    int idr_interval = 0;     // 0 = auto 2-second GOP (fps * 2)
    int async_depth = 4;
    int b_depth = 1;
    uint32_t device_index = 0;
    bool device_explicit = false;  // True if user explicitly passed --device

    ColorRange color_range = ColorRange::Limited;
    ColorSpace color_space = ColorSpace::Bt709;
    InputGamut input_gamut = InputGamut::Auto;
    bool dither = true;       // Static spatial IGN dithering for 8-bit NV12
    size_t queue_size = 4;
    bool verbose = false;
    bool show_help = false;

    // Tracks explicit user CLI overrides so hardware-aware profile adaptation
    // never clobbers an explicitly requested flag or saved user defaults.
    bool fps_explicit = false;
    bool pixfmt_explicit = false;
    bool rc_explicit = false;
    bool qp_explicit = false;
    bool bitrate_explicit = false;
    bool max_bitrate_explicit = false;
    bool quality_explicit = false;
    bool from_saved_defaults = false;
};

struct CodecEncodeCaps {
    bool supported = false;
    bool supports_10bit = false;
    bool supports_vbr = true;
    bool supports_cqp = true;
    bool supports_cbr = true;
    uint32_t max_quality_levels = 2;
    uint32_t max_coded_width = 0;
    uint32_t max_coded_height = 0;
    uint64_t max_hw_bitrate = 0;
};

struct ProfileTarget {
    uint32_t width = 2560;
    uint32_t height = 1440;
    double refresh_hz = 144.0;
    CodecEncodeCaps av1{true, true, true, true, true, 2, 0, 0, 0};
    CodecEncodeCaps hevc{true, true, true, true, true, 2, 0, 0, 0};
    CodecEncodeCaps h264{true, false, true, true, true, 2, 0, 0, 0};
    std::string gpu_name;
    std::string display_label;

    bool has_any_codec() const noexcept {
        return av1.supported || hevc.supported || h264.supported;
    }
    const CodecEncodeCaps &caps_for(Codec c) const noexcept {
        switch (c) {
            case Codec::Av1:  return av1;
            case Codec::Hevc: return hevc;
            case Codec::H264: return h264;
        }
        return av1;
    }
    CodecEncodeCaps &caps_for(Codec c) noexcept {
        switch (c) {
            case Codec::Av1:  return av1;
            case Codec::Hevc: return hevc;
            case Codec::H264: return h264;
        }
        return av1;
    }
};

struct ProfileInfo {
    std::string_view name;
    std::string_view summary;
    std::string_view details;
};

struct AdaptiveProfileInfo {
    std::string name;
    std::string summary;
    std::string details;
};

std::optional<Codec> best_codec_for(const ProfileTarget &target) noexcept;
std::string no_encode_support_error(std::string_view gpu_name = {});
uint32_t recommended_standard_fps(double refresh_hz) noexcept;
uint32_t recommended_gameplay_fps(double refresh_hz) noexcept;
int64_t recommended_bitrate(uint32_t width, uint32_t height, uint32_t fps,
                            Codec codec, PixelFormat pixfmt, double base_1440p60_mbps,
                            uint64_t max_hw_bitrate = 0) noexcept;
std::pair<uint32_t, uint32_t> clamp_resolution_to_encoder(uint32_t width, uint32_t height,
                                                          uint32_t max_width, uint32_t max_height) noexcept;
std::expected<void, std::string> adapt_config_to_target(Config &cfg, const ProfileTarget &target);
std::optional<std::string> check_container_compatibility(std::string_view output_path,
                                                         Codec vcodec,
                                                         bool has_audio,
                                                         AudioCodec acodec);

std::span<const ProfileInfo> builtin_profiles() noexcept;
std::vector<AdaptiveProfileInfo> describe_profiles(const ProfileTarget &target);
std::expected<void, std::string> apply_profile(Config &cfg, std::string_view name,
                                               const ProfileTarget &target = {});
void print_profiles(const ProfileTarget &target = {});

std::expected<Config, std::string> parse_args(int argc, char **argv, const Config *base_defaults = nullptr);
std::expected<int64_t, std::string> parse_bitrate(const std::string &str);
std::expected<double, std::string> parse_volume_pct(std::string_view str);
std::expected<std::pair<uint32_t, uint32_t>, std::string> parse_resolution(const std::string &str);
std::expected<CropRect, std::string> parse_crop(std::string_view str);
constexpr int max_qp_for_codec(Codec c) noexcept { return c == Codec::Av1 ? 255 : 51; }
constexpr int default_qp_for_codec(Codec c) noexcept { return c == Codec::Av1 ? 52 : 22; }
void print_help(const char *prog_name);
std::string default_output_filename(std::string_view ext = ".mkv");
std::string expand_user_path(std::string_view path);

}  // namespace vkcap
