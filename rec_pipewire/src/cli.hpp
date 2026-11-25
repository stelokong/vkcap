#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string>

namespace vkcap {

enum class Codec { Av1, Hevc, H264 };
enum class PixelFormat { Nv12, P010 };
enum class RateControl { Auto, Cqp, Cbr, Vbr };
enum class Tune { Default, Hq, Ll, Ull, Lossless };
enum class Usage { Default, Transcode, Stream, Record, Conference };
enum class Content { Default, Camera, Desktop, Rendered };
enum class CursorMode { Embedded, Hidden, Metadata };
enum class SourceType { Monitor, Window, All };
enum class ColorRange { Limited, Full };
enum class ColorSpace { Bt709, Bt2020 };

struct Config {
    std::string output_path;
    uint32_t fps = 60;
    uint32_t out_width = 0;   // 0 = native capture width
    uint32_t out_height = 0;  // 0 = native capture height
    double duration_sec = 0.0;

    Codec codec = Codec::Av1;
    PixelFormat pixfmt = PixelFormat::Nv12;
    RateControl rc = RateControl::Auto;
    int qp = -1;              // -1 = codec-aware default (52 for AV1, 22 for HEVC/H264)
    int64_t bitrate = 0;      // bps; 0 = unset
    int64_t max_bitrate = 0;  // bps; 0 = unset
    int quality = -1;         // -1 = GPU/rate-aware auto (2 up to 1440p120/4K60, 1 above)
    Tune tune = Tune::Hq;
    Usage usage = Usage::Record;
    Content content = Content::Rendered;
    int idr_interval = 0;     // 0 = auto 2-second GOP (fps * 2)
    int async_depth = 4;
    int b_depth = 1;
    uint32_t device_index = 0;

    CursorMode cursor = CursorMode::Embedded;
    SourceType source_type = SourceType::Monitor;
    std::optional<std::string> restore_token;
    bool use_saved_restore = false;
    bool save_restore = false;

    ColorRange color_range = ColorRange::Limited;
    ColorSpace color_space = ColorSpace::Bt709;
    bool dither = true;       // Spatial IGN dithering for 8-bit NV12

    bool dmabuf_only = false;
    bool vfr = false;          // false = precision CFR pacing (default), true = VFR
    size_t queue_size = 4;
    bool verbose = false;
    bool show_help = false;
};

std::expected<Config, std::string> parse_args(int argc, char **argv);
std::expected<int64_t, std::string> parse_bitrate(const std::string &str);
std::expected<std::pair<uint32_t, uint32_t>, std::string> parse_resolution(const std::string &str);
void print_help(const char *prog_name);
std::string default_output_filename();

}  // namespace vkcap
