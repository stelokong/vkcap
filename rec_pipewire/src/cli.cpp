#include "cli.hpp"

#include <charconv>
#include <ctime>
#include <format>
#include <getopt.h>
#include <iostream>
#include <string_view>

namespace vkcap {

std::string default_output_filename() {
    std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);
    char buf[64];
    std::strftime(buf, sizeof(buf), "vkcap_%Y%m%d_%H%M%S.mkv", &tm);
    return std::string(buf);
}

std::expected<int64_t, std::string> parse_bitrate(const std::string &str) {
    if (str.empty()) {
        return std::unexpected("Empty bitrate string");
    }
    std::string_view sv(str);
    int64_t mult = 1;
    char last = sv.back();
    if (last == 'k' || last == 'K') {
        mult = 1'000;
        sv.remove_suffix(1);
    } else if (last == 'm' || last == 'M') {
        mult = 1'000'000;
        sv.remove_suffix(1);
    } else if (last == 'g' || last == 'G') {
        mult = 1'000'000'000;
        sv.remove_suffix(1);
    }
    if (sv.empty()) {
        return std::unexpected(std::format("Invalid bitrate '{}'", str));
    }
    double val = 0.0;
    auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), val);
    if (ec != std::errc() || ptr != sv.data() + sv.size() || val <= 0.0) {
        return std::unexpected(std::format("Invalid bitrate '{}'", str));
    }
    return static_cast<int64_t>(val * static_cast<double>(mult));
}

std::expected<std::pair<uint32_t, uint32_t>, std::string> parse_resolution(const std::string &str) {
    auto pos = str.find('x');
    if (pos == std::string::npos) pos = str.find('X');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= str.size()) {
        return std::unexpected(std::format("Invalid resolution '{}', expected WxH", str));
    }
    uint32_t w = 0, h = 0;
    auto [pw, ew] = std::from_chars(str.data(), str.data() + pos, w);
    auto [ph, eh] = std::from_chars(str.data() + pos + 1, str.data() + str.size(), h);
    if (ew != std::errc() || pw != str.data() + pos ||
        eh != std::errc() || ph != str.data() + str.size() ||
        w == 0 || h == 0 || (w & 1u) != 0 || (h & 1u) != 0) {
        return std::unexpected(std::format("Resolution '{}' must be positive even WxH", str));
    }
    return std::pair<uint32_t, uint32_t>{w, h};
}

void print_help(const char *prog_name) {
    std::cout <<
R"(vkcap — Zero-copy PipeWire DMA-BUF -> Vulkan Compute -> Vulkan Video (AV1) screen recorder

Usage: )" << prog_name << R"( [OPTIONS]

Output & Capture:
  -o, --output <path>         Output file (.mkv, .mp4, .webm, .ivf) [default: vkcap_<timestamp>.mkv]
  -f, --fps <n>               Target/max PipeWire stream framerate [default: 60]
  -s, --scale <WxH>           Scale output on GPU compute shader (even WxH) [default: native]
  -d, --duration <sec>        Stop recording automatically after <sec> seconds
      --cursor <mode>         Cursor mode: embedded, hidden, metadata [default: embedded]
      --source-type <type>    Portal source types: monitor, window, all [default: monitor]
      --restore-token <tok>   Reuse a specific xdg-desktop-portal restore token
      --use-saved-restore     Load restore token from XDG state directory if present
      --save-restore          Persist returned restore token to XDG state directory
      --dmabuf-only           Refuse non-DMA-BUF PipeWire memory fallbacks
      --vfr                   Use variable frame rate timestamps instead of locked CFR pacing

Vulkan Video Encoder (AV1 primary, tuned for RDNA 4 / VCN 5):
      --codec <name>          av1, hevc, h264 [default: av1]
      --pixfmt <fmt>          nv12 (8-bit + IGN dither) or p010 (10-bit) [default: nv12]
      --rc <mode>             Rate control: auto, cqp, cbr, vbr [default: auto]
      --qp <0..255>           Quantizer for CQP mode [default: 52 for AV1, 22 for HEVC/H264]
  -b, --bitrate <rate>        Target bitrate (e.g. 20M, 15000k); selects VBR if --rc=auto
      --max-bitrate <rate>    Max bitrate for VBR mode
      --quality <0..3>        Encoder quality level (higher = better/slower) [default: auto]
      --tune <mode>           default, hq, ll, ull, lossless [default: hq]
      --usage <mode>          default, transcode, stream, record, conference [default: record]
      --content <mode>        default, camera, desktop, rendered [default: rendered]
      --idr-interval <n>      Keyframe interval in frames [default: 2s (fps * 2)]
      --async-depth <1..64>   GPU encode pipeline depth [default: 4]
      --b-depth <1..8>        Max B-frame reference depth [default: 1]
      --device <index>        Vulkan physical device index [default: 0]

Color & Diagnostics:
      --color-range <r>       limited (tv) or full (pc) [default: limited]
      --color-space <cs>      bt709 or bt2020 [default: bt709]
      --no-dither             Disable spatial IGN dithering for 8-bit NV12
      --queue-size <n>        Bounded capture-to-GPU queue depth [default: 4]
  -v, --verbose               Print detailed negotiation, Vulkan, and frame telemetry
  -h, --help                  Show this help
)";
}

namespace {

enum OptId {
    OPT_CODEC = 1000,
    OPT_PIXFMT,
    OPT_RC,
    OPT_QP,
    OPT_MAX_BITRATE,
    OPT_QUALITY,
    OPT_TUNE,
    OPT_USAGE,
    OPT_CONTENT,
    OPT_IDR_INTERVAL,
    OPT_ASYNC_DEPTH,
    OPT_B_DEPTH,
    OPT_DEVICE,
    OPT_CURSOR,
    OPT_SOURCE_TYPE,
    OPT_RESTORE_TOKEN,
    OPT_USE_SAVED_RESTORE,
    OPT_SAVE_RESTORE,
    OPT_COLOR_RANGE,
    OPT_COLOR_SPACE,
    OPT_NO_DITHER,
    OPT_DMABUF_ONLY,
    OPT_VFR,
    OPT_QUEUE_SIZE,
};

template <typename T>
std::expected<T, std::string> parse_enum(std::string_view val,
                                         std::initializer_list<std::pair<std::string_view, T>> map,
                                         std::string_view opt_name) {
    for (const auto &[k, v] : map) {
        if (val == k) return v;
    }
    return std::unexpected(std::format("Invalid value '{}' for {}", val, opt_name));
}

}  // namespace

std::expected<Config, std::string> parse_args(int argc, char **argv) {
    Config cfg;
    optind = 1;

    static const option long_opts[] = {
        {"output",            required_argument, nullptr, 'o'},
        {"fps",               required_argument, nullptr, 'f'},
        {"scale",             required_argument, nullptr, 's'},
        {"duration",          required_argument, nullptr, 'd'},
        {"bitrate",           required_argument, nullptr, 'b'},
        {"verbose",           no_argument,       nullptr, 'v'},
        {"help",              no_argument,       nullptr, 'h'},
        {"codec",             required_argument, nullptr, OPT_CODEC},
        {"pixfmt",            required_argument, nullptr, OPT_PIXFMT},
        {"rc",                required_argument, nullptr, OPT_RC},
        {"qp",                required_argument, nullptr, OPT_QP},
        {"max-bitrate",       required_argument, nullptr, OPT_MAX_BITRATE},
        {"quality",           required_argument, nullptr, OPT_QUALITY},
        {"tune",              required_argument, nullptr, OPT_TUNE},
        {"usage",             required_argument, nullptr, OPT_USAGE},
        {"content",           required_argument, nullptr, OPT_CONTENT},
        {"idr-interval",      required_argument, nullptr, OPT_IDR_INTERVAL},
        {"async-depth",       required_argument, nullptr, OPT_ASYNC_DEPTH},
        {"b-depth",           required_argument, nullptr, OPT_B_DEPTH},
        {"device",            required_argument, nullptr, OPT_DEVICE},
        {"cursor",            required_argument, nullptr, OPT_CURSOR},
        {"source-type",       required_argument, nullptr, OPT_SOURCE_TYPE},
        {"restore-token",     required_argument, nullptr, OPT_RESTORE_TOKEN},
        {"use-saved-restore", no_argument,       nullptr, OPT_USE_SAVED_RESTORE},
        {"save-restore",      no_argument,       nullptr, OPT_SAVE_RESTORE},
        {"color-range",       required_argument, nullptr, OPT_COLOR_RANGE},
        {"color-space",       required_argument, nullptr, OPT_COLOR_SPACE},
        {"no-dither",         no_argument,       nullptr, OPT_NO_DITHER},
        {"dmabuf-only",       no_argument,       nullptr, OPT_DMABUF_ONLY},
        {"vfr",               no_argument,       nullptr, OPT_VFR},
        {"queue-size",        required_argument, nullptr, OPT_QUEUE_SIZE},
        {nullptr, 0, nullptr, 0},
    };

    auto parse_int = [](const char *s, int min_v, int max_v, std::string_view name)
        -> std::expected<int, std::string> {
        std::string_view sv(s);
        int v = 0;
        auto [p, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
        if (ec != std::errc() || p != sv.data() + sv.size() || v < min_v || v > max_v) {
            return std::unexpected(std::format("Invalid integer '{}' for {} ({}..{})", s, name, min_v, max_v));
        }
        return v;
    };

    int c = 0;
    while ((c = getopt_long(argc, argv, "o:f:s:d:b:vh", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'o': cfg.output_path = optarg; break;
            case 'f': {
                auto r = parse_int(optarg, 1, 1000, "--fps");
                if (!r) return std::unexpected(r.error());
                cfg.fps = static_cast<uint32_t>(*r);
                break;
            }
            case 's': {
                auto r = parse_resolution(optarg);
                if (!r) return std::unexpected(r.error());
                cfg.out_width = r->first;
                cfg.out_height = r->second;
                break;
            }
            case 'd': {
                std::string_view sv(optarg);
                double sec = 0.0;
                auto [p, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), sec);
                if (ec != std::errc() || p != sv.data() + sv.size() || sec <= 0.0) {
                    return std::unexpected(std::format("Invalid duration '{}'", optarg));
                }
                cfg.duration_sec = sec;
                break;
            }
            case 'b': {
                auto r = parse_bitrate(optarg);
                if (!r) return std::unexpected(r.error());
                cfg.bitrate = *r;
                break;
            }
            case 'v': cfg.verbose = true; break;
            case 'h': cfg.show_help = true; return cfg;
            case OPT_CODEC: {
                auto r = parse_enum<Codec>(optarg, {{"av1", Codec::Av1}, {"hevc", Codec::Hevc}, {"h264", Codec::H264}}, "--codec");
                if (!r) return std::unexpected(r.error());
                cfg.codec = *r;
                break;
            }
            case OPT_PIXFMT: {
                auto r = parse_enum<PixelFormat>(optarg, {{"nv12", PixelFormat::Nv12}, {"p010", PixelFormat::P010}}, "--pixfmt");
                if (!r) return std::unexpected(r.error());
                cfg.pixfmt = *r;
                break;
            }
            case OPT_RC: {
                auto r = parse_enum<RateControl>(optarg, {
                    {"auto", RateControl::Auto}, {"cqp", RateControl::Cqp},
                    {"cbr", RateControl::Cbr}, {"vbr", RateControl::Vbr}}, "--rc");
                if (!r) return std::unexpected(r.error());
                cfg.rc = *r;
                break;
            }
            case OPT_QP: {
                auto r = parse_int(optarg, 0, 255, "--qp");
                if (!r) return std::unexpected(r.error());
                cfg.qp = *r;
                break;
            }
            case OPT_MAX_BITRATE: {
                auto r = parse_bitrate(optarg);
                if (!r) return std::unexpected(r.error());
                cfg.max_bitrate = *r;
                break;
            }
            case OPT_QUALITY: {
                auto r = parse_int(optarg, 0, 16, "--quality");
                if (!r) return std::unexpected(r.error());
                cfg.quality = *r;
                break;
            }
            case OPT_TUNE: {
                auto r = parse_enum<Tune>(optarg, {
                    {"default", Tune::Default}, {"hq", Tune::Hq}, {"ll", Tune::Ll},
                    {"ull", Tune::Ull}, {"lossless", Tune::Lossless}}, "--tune");
                if (!r) return std::unexpected(r.error());
                cfg.tune = *r;
                break;
            }
            case OPT_USAGE: {
                auto r = parse_enum<Usage>(optarg, {
                    {"default", Usage::Default}, {"transcode", Usage::Transcode},
                    {"stream", Usage::Stream}, {"record", Usage::Record},
                    {"conference", Usage::Conference}}, "--usage");
                if (!r) return std::unexpected(r.error());
                cfg.usage = *r;
                break;
            }
            case OPT_CONTENT: {
                auto r = parse_enum<Content>(optarg, {
                    {"default", Content::Default}, {"camera", Content::Camera},
                    {"desktop", Content::Desktop}, {"rendered", Content::Rendered}}, "--content");
                if (!r) return std::unexpected(r.error());
                cfg.content = *r;
                break;
            }
            case OPT_IDR_INTERVAL: {
                auto r = parse_int(optarg, 0, 100000, "--idr-interval");
                if (!r) return std::unexpected(r.error());
                cfg.idr_interval = *r;
                break;
            }
            case OPT_ASYNC_DEPTH: {
                auto r = parse_int(optarg, 1, 64, "--async-depth");
                if (!r) return std::unexpected(r.error());
                cfg.async_depth = *r;
                break;
            }
            case OPT_B_DEPTH: {
                auto r = parse_int(optarg, 1, 8, "--b-depth");
                if (!r) return std::unexpected(r.error());
                cfg.b_depth = *r;
                break;
            }
            case OPT_DEVICE: {
                auto r = parse_int(optarg, 0, 16, "--device");
                if (!r) return std::unexpected(r.error());
                cfg.device_index = static_cast<uint32_t>(*r);
                break;
            }
            case OPT_CURSOR: {
                auto r = parse_enum<CursorMode>(optarg, {
                    {"embedded", CursorMode::Embedded}, {"hidden", CursorMode::Hidden},
                    {"metadata", CursorMode::Metadata}}, "--cursor");
                if (!r) return std::unexpected(r.error());
                cfg.cursor = *r;
                break;
            }
            case OPT_SOURCE_TYPE: {
                auto r = parse_enum<SourceType>(optarg, {
                    {"monitor", SourceType::Monitor}, {"window", SourceType::Window},
                    {"all", SourceType::All}}, "--source-type");
                if (!r) return std::unexpected(r.error());
                cfg.source_type = *r;
                break;
            }
            case OPT_RESTORE_TOKEN: cfg.restore_token = optarg; break;
            case OPT_USE_SAVED_RESTORE: cfg.use_saved_restore = true; break;
            case OPT_SAVE_RESTORE: cfg.save_restore = true; break;
            case OPT_COLOR_RANGE: {
                auto r = parse_enum<ColorRange>(optarg, {
                    {"limited", ColorRange::Limited}, {"tv", ColorRange::Limited},
                    {"full", ColorRange::Full}, {"pc", ColorRange::Full}}, "--color-range");
                if (!r) return std::unexpected(r.error());
                cfg.color_range = *r;
                break;
            }
            case OPT_COLOR_SPACE: {
                auto r = parse_enum<ColorSpace>(optarg, {
                    {"bt709", ColorSpace::Bt709}, {"bt2020", ColorSpace::Bt2020}}, "--color-space");
                if (!r) return std::unexpected(r.error());
                cfg.color_space = *r;
                break;
            }
            case OPT_NO_DITHER: cfg.dither = false; break;
            case OPT_DMABUF_ONLY: cfg.dmabuf_only = true; break;
            case OPT_VFR: cfg.vfr = true; break;
            case OPT_QUEUE_SIZE: {
                auto r = parse_int(optarg, 1, 64, "--queue-size");
                if (!r) return std::unexpected(r.error());
                cfg.queue_size = static_cast<size_t>(*r);
                break;
            }
            default:
                return std::unexpected("Unknown option. Run with --help for usage.");
        }
    }

    if (optind < argc) {
        if (cfg.output_path.empty() && optind + 1 == argc) {
            cfg.output_path = argv[optind];
        } else {
            return std::unexpected(std::format("Unexpected positional argument '{}'", argv[optind]));
        }
    }
    if (cfg.output_path.empty()) {
        cfg.output_path = default_output_filename();
    }
    return cfg;
}

}  // namespace vkcap
