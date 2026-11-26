#include "cli.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <ctime>
#include <format>
#include <getopt.h>
#include <iostream>
#include <string_view>

namespace vkcap {

namespace {

constexpr std::array<ProfileInfo, 7> kProfiles = {{
    {"default",        "Near-native quality, low overhead (RDNA 4 default)",
                       "60 fps | AV1 | 8-bit NV12 + IGN dither | CQP 52 | quality 1 | hq"},
    {"gameplay",       "High-framerate 120 fps gameplay capture",
                       "120 fps | AV1 | 8-bit NV12 + IGN dither | CQP 56 | quality 1 | hq"},
    {"gameplay-10bit", "120 fps 10-bit gameplay (zero banding, 35 Mbps VBR)",
                       "120 fps | AV1 | 10-bit P010 | VBR 35M (max 52.5M) | quality 1 | hq"},
    {"master",         "Archival / editing master (10-bit, ultra-high quality)",
                       "60 fps | AV1 | 10-bit P010 | CQP 36 | quality 1 | hq"},
    {"compact",        "Small file sizes for Discord / sharing / long sessions",
                       "60 fps | AV1 | 8-bit NV12 + IGN dither | CQP 78 | 4s GOP | hq"},
    {"desktop",        "Crisp desktop UI, text, and tutorials at low bitrate",
                       "60 fps | AV1 | 8-bit NV12 + IGN dither | CQP 62 | 4s GOP | desktop"},
    {"low-latency",    "Minimum GPU/VCN footprint for maximum game FPS",
                       "60 fps | AV1 | 8-bit NV12 + IGN dither | CQP 64 | quality 0 | ll | async 2"},
}};

std::string to_lower_copy(std::string_view sv) {
    std::string s(sv);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

std::span<const ProfileInfo> builtin_profiles() noexcept {
    return kProfiles;
}

std::expected<void, std::string> apply_profile(Config &cfg, std::string_view name) {
    const std::string key = to_lower_copy(name);
    if (key.empty() || key == "default" || key == "balanced") {
        cfg.profile = "default";
        cfg.fps = 60;
        cfg.codec = Codec::Av1;
        cfg.pixfmt = PixelFormat::Nv12;
        cfg.rc = RateControl::Auto;
        cfg.qp = -1;
        cfg.bitrate = 0;
        cfg.max_bitrate = 0;
        cfg.quality = -1;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = 0;
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        cfg.dither = true;
        return {};
    }
    if (key == "gameplay") {
        cfg.profile = "gameplay";
        cfg.fps = 120;
        cfg.codec = Codec::Av1;
        cfg.pixfmt = PixelFormat::Nv12;
        cfg.rc = RateControl::Cqp;
        cfg.qp = 56;
        cfg.bitrate = 0;
        cfg.max_bitrate = 0;
        cfg.quality = 1;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = 0;
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        cfg.dither = true;
        return {};
    }
    if (key == "gameplay-10bit" || key == "gameplay10") {
        cfg.profile = "gameplay-10bit";
        cfg.fps = 120;
        cfg.codec = Codec::Av1;
        cfg.pixfmt = PixelFormat::P010;
        cfg.rc = RateControl::Vbr;
        cfg.qp = -1;
        cfg.bitrate = 35'000'000;
        cfg.max_bitrate = 52'500'000;
        cfg.quality = 1;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = 0;
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        return {};
    }
    if (key == "master" || key == "archival") {
        cfg.profile = "master";
        cfg.fps = 60;
        cfg.codec = Codec::Av1;
        cfg.pixfmt = PixelFormat::P010;
        cfg.rc = RateControl::Cqp;
        cfg.qp = 36;
        cfg.bitrate = 0;
        cfg.max_bitrate = 0;
        cfg.quality = 1;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = 0;
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        return {};
    }
    if (key == "compact" || key == "small") {
        cfg.profile = "compact";
        cfg.fps = 60;
        cfg.codec = Codec::Av1;
        cfg.pixfmt = PixelFormat::Nv12;
        cfg.rc = RateControl::Cqp;
        cfg.qp = 78;
        cfg.bitrate = 0;
        cfg.max_bitrate = 0;
        cfg.quality = 1;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = 240;  // 4 seconds @ 60fps
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        cfg.dither = true;
        return {};
    }
    if (key == "desktop" || key == "ui") {
        cfg.profile = "desktop";
        cfg.fps = 60;
        cfg.codec = Codec::Av1;
        cfg.pixfmt = PixelFormat::Nv12;
        cfg.rc = RateControl::Cqp;
        cfg.qp = 62;
        cfg.bitrate = 0;
        cfg.max_bitrate = 0;
        cfg.quality = 1;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Desktop;
        cfg.idr_interval = 240;  // 4 seconds @ 60fps
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        cfg.dither = true;
        return {};
    }
    if (key == "low-latency" || key == "ll" || key == "light") {
        cfg.profile = "low-latency";
        cfg.fps = 60;
        cfg.codec = Codec::Av1;
        cfg.pixfmt = PixelFormat::Nv12;
        cfg.rc = RateControl::Cqp;
        cfg.qp = 64;
        cfg.bitrate = 0;
        cfg.max_bitrate = 0;
        cfg.quality = 0;
        cfg.tune = Tune::Ll;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = 0;
        cfg.async_depth = 2;
        cfg.b_depth = 1;
        cfg.dither = true;
        return {};
    }
    return std::unexpected(std::format("Unknown profile '{}'. Run with --list-profiles to see available profiles.", name));
}

void print_profiles() {
    std::cout << "Available premade profiles (use with -p <name>; explicit flags override profile values):\n\n";
    for (const auto &p : kProfiles) {
        std::cout << std::format("  {:<15} {}\n                  [{}]\n", p.name, p.summary, p.details);
    }
}

std::string default_output_filename(std::string_view ext) {
    std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);
    char buf[64];
    std::strftime(buf, sizeof(buf), "vkcap_%Y%m%d_%H%M%S", &tm);
    std::string out(buf);
    if (ext.empty()) {
        out += ".mkv";
    } else if (ext.front() == '.') {
        out += ext;
    } else {
        out += '.';
        out += ext;
    }
    return out;
}

std::expected<double, std::string> parse_volume_pct(std::string_view str) {
    while (!str.empty() && std::isspace(static_cast<unsigned char>(str.front()))) str.remove_prefix(1);
    while (!str.empty() && std::isspace(static_cast<unsigned char>(str.back()))) str.remove_suffix(1);
    if (!str.empty() && str.back() == '%') str.remove_suffix(1);
    if (str.empty()) return std::unexpected("Empty volume percentage");
    double val = 0.0;
    auto [p, ec] = std::from_chars(str.data(), str.data() + str.size(), val);
    if (ec != std::errc() || p != str.data() + str.size() || val < 0.0 || val > 1000.0) {
        return std::unexpected(std::format("Invalid volume '{}', expected 0..1000 (e.g. 100, 200%)", str));
    }
    return val;
}

std::expected<int64_t, std::string> parse_bitrate(const std::string &str) {
    if (str.empty()) return std::unexpected("Empty bitrate string");
    std::string_view sv(str);
    int64_t mult = 1;
    char last = sv.back();
    if (last == 'k' || last == 'K') { mult = 1'000; sv.remove_suffix(1); }
    else if (last == 'm' || last == 'M') { mult = 1'000'000; sv.remove_suffix(1); }
    else if (last == 'g' || last == 'G') { mult = 1'000'000'000; sv.remove_suffix(1); }
    if (sv.empty()) return std::unexpected(std::format("Invalid bitrate '{}'", str));
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
R"(vkcap — Zero-copy Linux DRM/KMS -> Vulkan Compute -> Vulkan Video (AV1) screen recorder
        Compositor-independent, least-privilege (requires only cap_sys_admin=p).

Usage: )" << prog_name << R"( [OPTIONS]
       Run with no arguments in a terminal to launch the interactive setup prompt.

Profiles & Interactive Mode:
  -p, --profile <name>        Apply a premade profile (explicit flags override profile values)
      --list-profiles         List all premade profiles and exit
  -i, --interactive           Launch the interactive guided setup prompt

Display Selection & Capture:
  -l, --list-displays         List connected DRM/KMS displays and exit (no privileges needed)
  -D, --display <name|idx>    Select display by connector (e.g. DP-1, HDMI-A-1) or index [default: auto]
  -o, --output <path>         Output file (.mkv, .mp4, .webm, .ivf) [default: vkcap_<ts>.mkv]
  -f, --fps <n>               Locked CFR capture framerate [default: 60]
  -s, --scale <WxH>           Scale output via GPU Catmull-Rom bicubic filter (even WxH)
  -d, --duration <sec>        Stop recording automatically after <sec> seconds
      --no-cursor             Disable GPU compositing of the hardware KMS cursor plane

Audio Capture (PipeWire low-latency, system audio ON by default):
      --no-sound, --no-audio  Disable all audio recording (video-only)
      --no-system-audio       Disable desktop/application audio (e.g. for mic-only)
      --mic [dev]             Enable microphone recording (optional: name, id, idx, or 'default')
      --list-mics             List available PipeWire microphones and exit (no privileges needed)
      --system-volume <pct>   System audio volume percentage (e.g. 80, 100%, 150) [default: 100%]
      --mic-volume <pct>      Microphone volume percentage (e.g. 100, 200%) [default: 100%]
      --audio-codec <name>    Audio codec: opus, aac, flac [default: opus]
      --audio-bitrate <rate>  Audio bitrate for opus/aac (e.g. 128k, 192k, 256k) [default: 192k]

Vulkan Video Encoder (AV1 primary, tuned for RDNA 4 / VCN 5):
      --codec <name>          av1, hevc, h264 [default: av1]
      --pixfmt <fmt>          nv12 (8-bit + IGN dither) or p010 (10-bit) [default: nv12]
      --rc <mode>             Rate control: auto, cqp, cbr, vbr [default: auto]
      --qp <n>                Quantizer for CQP (0..255 for AV1 [52], 0..51 for HEVC/H264 [22])
  -b, --bitrate <rate>        Target bitrate (e.g. 20M, 35M); selects VBR if --rc=auto
      --max-bitrate <rate>    Max bitrate for VBR mode
      --quality <0..3>        Encoder quality level (higher = slower) [default: auto (1)]
      --tune <mode>           default, hq, ll, ull, lossless [default: hq]
      --usage <mode>          default, transcode, stream, record, conference [default: record]
      --content <mode>        default, camera, desktop, rendered [default: rendered]
      --idr-interval <n>      Keyframe interval in frames [default: 2s (fps * 2)]
      --async-depth <1..64>   GPU encode pipeline depth [default: 4]
      --b-depth <1..8>        Max B-frame reference depth [default: 1]
      --device <index>        Vulkan physical device index [default: 0]

Color & Diagnostics:
      --input-gamut <g>       auto, edid, srgb, p3, bt2020 [default: auto (EDID on wide-gamut Wayland)]
      --color-range <r>       limited (tv) or full (pc) [default: limited]
      --color-space <cs>      bt709 or bt2020 [default: bt709]
      --no-dither             Disable static spatial IGN dithering for 8-bit NV12
      --queue-size <n>        Bounded GPU-to-encoder queue depth [default: 4]
  -v, --verbose               Print DRM/KMS plane, Vulkan, and encoder telemetry
  -h, --help                  Show this help

Least-Privilege Setup (run once after building):
  sudo setcap cap_sys_admin=p ./build/vkcap
)";
}

namespace {

enum OptId {
    OPT_NO_CURSOR = 1000,
    OPT_LIST_PROFILES,
    OPT_NO_AUDIO,
    OPT_NO_SYSTEM_AUDIO,
    OPT_MIC,
    OPT_LIST_MICS,
    OPT_SYSTEM_VOLUME,
    OPT_MIC_VOLUME,
    OPT_AUDIO_CODEC,
    OPT_AUDIO_BITRATE,
    OPT_CODEC,
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
    OPT_COLOR_RANGE,
    OPT_COLOR_SPACE,
    OPT_INPUT_GAMUT,
    OPT_NO_DITHER,
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

    // Pre-scan for -p / --profile so explicit CLI flags always override profile settings
    // regardless of argument order (e.g. `-f 120 -p gameplay` and `-p gameplay -f 120`).
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if ((arg == "-p" || arg == "--profile") && i + 1 < argc) {
            auto pr = apply_profile(cfg, argv[i + 1]);
            if (!pr) return std::unexpected(pr.error());
            ++i;
        } else if (arg.starts_with("--profile=")) {
            auto pr = apply_profile(cfg, arg.substr(10));
            if (!pr) return std::unexpected(pr.error());
        }
    }

    optind = 1;

    static const option long_opts[] = {
        {"list-displays",     no_argument,       nullptr, 'l'},
        {"list-profiles",     no_argument,       nullptr, OPT_LIST_PROFILES},
        {"profile",           required_argument, nullptr, 'p'},
        {"interactive",       no_argument,       nullptr, 'i'},
        {"display",           required_argument, nullptr, 'D'},
        {"output",            required_argument, nullptr, 'o'},
        {"fps",               required_argument, nullptr, 'f'},
        {"scale",             required_argument, nullptr, 's'},
        {"duration",          required_argument, nullptr, 'd'},
        {"bitrate",           required_argument, nullptr, 'b'},
        {"verbose",           no_argument,       nullptr, 'v'},
        {"help",              no_argument,       nullptr, 'h'},
        {"no-cursor",         no_argument,       nullptr, OPT_NO_CURSOR},
        {"no-sound",          no_argument,       nullptr, OPT_NO_AUDIO},
        {"no-audio",          no_argument,       nullptr, OPT_NO_AUDIO},
        {"no-system-audio",   no_argument,       nullptr, OPT_NO_SYSTEM_AUDIO},
        {"mic",               optional_argument, nullptr, OPT_MIC},
        {"with-mic",          no_argument,       nullptr, OPT_MIC},
        {"list-mics",         no_argument,       nullptr, OPT_LIST_MICS},
        {"system-volume",     required_argument, nullptr, OPT_SYSTEM_VOLUME},
        {"mic-volume",        required_argument, nullptr, OPT_MIC_VOLUME},
        {"audio-codec",       required_argument, nullptr, OPT_AUDIO_CODEC},
        {"audio-bitrate",     required_argument, nullptr, OPT_AUDIO_BITRATE},
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
        {"color-range",       required_argument, nullptr, OPT_COLOR_RANGE},
        {"color-space",       required_argument, nullptr, OPT_COLOR_SPACE},
        {"input-gamut",       required_argument, nullptr, OPT_INPUT_GAMUT},
        {"no-dither",         no_argument,       nullptr, OPT_NO_DITHER},
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
    while ((c = getopt_long(argc, argv, "lp:iD:o:f:s:d:b:vh", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'l': cfg.list_displays = true; break;
            case OPT_LIST_PROFILES: cfg.list_profiles = true; break;
            case 'p': break;  // Already applied in pre-scan above
            case 'i': cfg.interactive = true; break;
            case 'D': cfg.display = optarg; break;
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
            case OPT_NO_CURSOR: cfg.capture_cursor = false; break;
            case OPT_NO_AUDIO:
                cfg.audio = false;
                cfg.capture_system_audio = false;
                cfg.capture_mic = false;
                break;
            case OPT_NO_SYSTEM_AUDIO:
                cfg.capture_system_audio = false;
                break;
            case OPT_MIC:
                cfg.capture_mic = true;
                if (optarg && optarg[0] != '\0') {
                    cfg.mic_device = optarg;
                } else if (optind < argc && argv[optind] && argv[optind][0] != '-') {
                    // Support `--mic <device>` with space as well as `--mic=<device>` or bare `--mic`
                    // Only consume next token if it doesn't look like an output video filename
                    std::string_view next(argv[optind]);
                    if (!next.ends_with(".mkv") && !next.ends_with(".mp4") &&
                        !next.ends_with(".webm") && !next.ends_with(".ivf")) {
                        cfg.mic_device = argv[optind++];
                    }
                }
                break;
            case OPT_LIST_MICS:
                cfg.list_mics = true;
                break;
            case OPT_SYSTEM_VOLUME: {
                auto r = parse_volume_pct(optarg);
                if (!r) return std::unexpected(r.error());
                cfg.system_volume_pct = *r;
                break;
            }
            case OPT_MIC_VOLUME: {
                auto r = parse_volume_pct(optarg);
                if (!r) return std::unexpected(r.error());
                cfg.mic_volume_pct = *r;
                break;
            }
            case OPT_AUDIO_CODEC: {
                auto r = parse_enum<AudioCodec>(optarg, {
                    {"opus", AudioCodec::Opus}, {"aac", AudioCodec::Aac}, {"flac", AudioCodec::Flac}},
                    "--audio-codec");
                if (!r) return std::unexpected(r.error());
                cfg.audio_codec = *r;
                break;
            }
            case OPT_AUDIO_BITRATE: {
                auto r = parse_bitrate(optarg);
                if (!r) return std::unexpected(r.error());
                cfg.audio_bitrate = *r;
                break;
            }
            case OPT_CODEC: {
                auto r = parse_enum<Codec>(optarg, {{"av1", Codec::Av1}, {"hevc", Codec::Hevc}, {"h264", Codec::H264}}, "--codec");
                if (!r) return std::unexpected(r.error());
                cfg.codec = *r;
                cfg.codec_explicit = true;
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
                cfg.device_explicit = true;
                break;
            }
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
            case OPT_INPUT_GAMUT: {
                auto r = parse_enum<InputGamut>(optarg, {
                    {"auto", InputGamut::Auto}, {"edid", InputGamut::Edid},
                    {"srgb", InputGamut::Srgb}, {"bt709", InputGamut::Srgb},
                    {"p3", InputGamut::DisplayP3}, {"display-p3", InputGamut::DisplayP3},
                    {"bt2020", InputGamut::Bt2020}}, "--input-gamut");
                if (!r) return std::unexpected(r.error());
                cfg.input_gamut = *r;
                break;
            }
            case OPT_NO_DITHER: cfg.dither = false; break;
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

    if (cfg.qp >= 0) {
        const int max_qp = max_qp_for_codec(cfg.codec);
        if (cfg.qp > max_qp) {
            const char *cname = (cfg.codec == Codec::Av1) ? "av1"
                              : (cfg.codec == Codec::Hevc) ? "hevc" : "h264";
            return std::unexpected(std::format(
                "Invalid --qp {} for {} (valid range: 0..{})", cfg.qp, cname, max_qp));
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
