#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>

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
    double duration_sec = 0.0;
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
    PixelFormat pixfmt = PixelFormat::Nv12;
    RateControl rc = RateControl::Auto;
    int qp = -1;              // -1 = codec-aware default (52 for AV1, 22 for HEVC/H264)
    int64_t bitrate = 0;      // bps; 0 = unset
    int64_t max_bitrate = 0;  // bps; 0 = unset
    int quality = -1;         // -1 = RDNA 4 auto default (1)
    Tune tune = Tune::Hq;
    Usage usage = Usage::Record;
    Content content = Content::Rendered;
    int idr_interval = 0;     // 0 = auto 2-second GOP (fps * 2)
    int async_depth = 4;
    int b_depth = 1;
    uint32_t device_index = 0;

    ColorRange color_range = ColorRange::Limited;
    ColorSpace color_space = ColorSpace::Bt709;
    InputGamut input_gamut = InputGamut::Auto;
    bool dither = true;       // Static spatial IGN dithering for 8-bit NV12
    size_t queue_size = 4;
    bool verbose = false;
    bool show_help = false;
};

struct ProfileInfo {
    std::string_view name;
    std::string_view summary;
    std::string_view details;
};

std::span<const ProfileInfo> builtin_profiles() noexcept;
std::expected<void, std::string> apply_profile(Config &cfg, std::string_view name);
void print_profiles();

std::expected<Config, std::string> parse_args(int argc, char **argv);
std::expected<int64_t, std::string> parse_bitrate(const std::string &str);
std::expected<double, std::string> parse_volume_pct(std::string_view str);
std::expected<std::pair<uint32_t, uint32_t>, std::string> parse_resolution(const std::string &str);
constexpr int max_qp_for_codec(Codec c) noexcept { return c == Codec::Av1 ? 255 : 51; }
constexpr int default_qp_for_codec(Codec c) noexcept { return c == Codec::Av1 ? 52 : 22; }
void print_help(const char *prog_name);
std::string default_output_filename(std::string_view ext = ".mkv");

}  // namespace vkcap
