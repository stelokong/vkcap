#include "cli.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <format>
#include <getopt.h>
#include <iostream>
#include <string_view>

namespace vkcap {

namespace {

constexpr std::array<ProfileInfo, 7> kProfiles = {{
    {"default",        "Near-native quality, bounded bitrate (recommended default)",
                       "60 fps | AV1 | 8-bit NV12 + IGN dither | VBR 20M (max 32M) | quality 1 | hq"},
    {"gameplay",       "High-framerate gameplay capture (up to 120 fps)",
                       "120 fps | AV1 | 8-bit NV12 + IGN dither | VBR 35M (max 55M) | quality 1 | hq"},
    {"gameplay-10bit", "High-framerate 10-bit gameplay (zero banding, bounded VBR)",
                       "120 fps | AV1 | 10-bit P010 | VBR 40M (max 60M) | quality 1 | hq"},
    {"master",         "Archival / editing master (10-bit, uncapped CQP)",
                       "60 fps | AV1 | 10-bit P010 | CQP 36 | quality 1 | hq"},
    {"compact",        "Small file sizes for Discord / sharing / long sessions",
                       "60 fps | AV1 | 8-bit NV12 + IGN dither | VBR 8M (max 12M) | 4s GOP | hq"},
    {"desktop",        "Crisp desktop UI, text, and tutorials at low bitrate",
                       "60 fps | AV1 | 8-bit NV12 + IGN dither | CQP 62 | 4s GOP | desktop"},
    {"low-latency",    "Minimum GPU encoder footprint for maximum game FPS",
                       "60 fps | AV1 | 8-bit NV12 + IGN dither | VBR 18M (max 28M) | quality 0 | ll | async 2"},
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

std::optional<Codec> best_codec_for(const ProfileTarget &target) noexcept {
    if (target.av1.supported) return Codec::Av1;
    if (target.hevc.supported) return Codec::Hevc;
    if (target.h264.supported) return Codec::H264;
    return std::nullopt;
}

std::string no_encode_support_error(std::string_view gpu_name) {
    if (!gpu_name.empty()) {
        return std::format(
            "GPU '{}' does not advertise any Vulkan Video encode extensions "
            "(VK_KHR_video_encode_av1, VK_KHR_video_encode_h265, VK_KHR_video_encode_h264). "
            "Hardware encoding is unavailable on this device; check your Vulkan driver or try another GPU with --device <index>.",
            gpu_name);
    }
    return "Selected GPU does not advertise any Vulkan Video encode extensions "
           "(VK_KHR_video_encode_av1, VK_KHR_video_encode_h265, VK_KHR_video_encode_h264). "
           "Hardware encoding is unavailable on this device; check your Vulkan driver or try another GPU with --device <index>.";
}

uint32_t recommended_standard_fps(double refresh_hz) noexcept {
    if (refresh_hz <= 0.0) return 60;
    // Treat 59.94 / 59.998 / 60+ Hz panels as 60 fps standard; only step down
    // when the panel genuinely refreshes below 58 Hz (e.g. 29.97/30 Hz, 48 Hz, 50 Hz).
    if (refresh_hz >= 58.0) return 60;
    return static_cast<uint32_t>(std::clamp<long long>(std::llround(refresh_hz), 15, 60));
}

uint32_t recommended_gameplay_fps(double refresh_hz) noexcept {
    if (refresh_hz <= 0.0) return 120;
    // Account for fractional NTSC/VESA refresh rates (119.88 -> 120, 89.9 -> 90, 74.97 -> 75)
    if (refresh_hz >= 118.0) return 120;
    if (refresh_hz >= 88.0) return 90;
    if (refresh_hz >= 73.0) return 75;
    return recommended_standard_fps(refresh_hz);
}

int64_t recommended_bitrate(uint32_t width, uint32_t height, uint32_t fps,
                            Codec codec, PixelFormat /*pixfmt*/, double base_1440p60_mbps,
                            uint64_t max_hw_bitrate) noexcept {
    constexpr double kRefPixels = 2560.0 * 1440.0;
    const double pixels = std::max(320.0 * 240.0, static_cast<double>(width) * static_cast<double>(height));
    const double res_scale = std::pow(pixels / kRefPixels, 0.75);
    const double fps_scale = std::pow(static_cast<double>(std::max<uint32_t>(1u, fps)) / 60.0, 0.75);
    const double codec_scale = (codec == Codec::Av1) ? 1.0
                             : (codec == Codec::Hevc) ? 1.20 : 1.55;

    const double raw_mbps = base_1440p60_mbps * res_scale * fps_scale * codec_scale;
    int64_t bps = std::max<int64_t>(2, static_cast<int64_t>(std::llround(raw_mbps))) * 1'000'000LL;
    if (max_hw_bitrate > 0 && static_cast<uint64_t>(bps) > max_hw_bitrate) {
        bps = std::max<int64_t>(1'000'000LL, static_cast<int64_t>(max_hw_bitrate));
    }
    return bps;
}

std::pair<uint32_t, uint32_t> clamp_resolution_to_encoder(uint32_t width, uint32_t height,
                                                          uint32_t max_width, uint32_t max_height) noexcept {
    if (width == 0 || height == 0) return {width, height};
    double scale = 1.0;
    if (max_width > 0 && width > max_width) {
        scale = std::min(scale, static_cast<double>(max_width) / static_cast<double>(width));
    }
    if (max_height > 0 && height > max_height) {
        scale = std::min(scale, static_cast<double>(max_height) / static_cast<double>(height));
    }
    if (scale >= 1.0) return {width, height};
    uint32_t cw = static_cast<uint32_t>(std::floor(static_cast<double>(width) * scale)) & ~1u;
    uint32_t ch = static_cast<uint32_t>(std::floor(static_cast<double>(height) * scale)) & ~1u;
    if (max_width > 0) cw = std::min(cw, max_width & ~1u);
    if (max_height > 0) ch = std::min(ch, max_height & ~1u);
    return {std::max<uint32_t>(2u, cw), std::max<uint32_t>(2u, ch)};
}

std::optional<std::string> check_container_compatibility(std::string_view output_path,
                                                         Codec vcodec,
                                                         bool has_audio,
                                                         AudioCodec acodec) {
    auto dot = output_path.rfind('.');
    if (dot == std::string_view::npos) return std::nullopt;
    const std::string ext = to_lower_copy(output_path.substr(dot));
    if (ext == ".ivf") {
        if (vcodec != Codec::Av1) {
            return ".ivf container only supports AV1 video; use .mkv or .mp4 for HEVC/H.264.";
        }
        if (has_audio) {
            return ".ivf is a video-only container (audio will not be muxed); use .mkv or .mp4 to keep audio.";
        }
    } else if (ext == ".webm") {
        if (vcodec != Codec::Av1) {
            return ".webm container only supports AV1 (not HEVC or H.264); use .mkv or .mp4.";
        }
    } else if (ext == ".mp4") {
        if (has_audio && acodec == AudioCodec::Flac) {
            return "FLAC audio in .mp4 has limited player support; .mkv is recommended for FLAC.";
        }
    }
    return std::nullopt;
}

namespace {

int64_t round_mbps(double mbps, uint64_t max_hw_bitrate = 0) noexcept {
    int64_t bps = std::max<int64_t>(2, static_cast<int64_t>(std::llround(mbps))) * 1'000'000LL;
    if (max_hw_bitrate > 0 && static_cast<uint64_t>(bps) > max_hw_bitrate) {
        bps = std::max<int64_t>(1'000'000LL, static_cast<int64_t>(max_hw_bitrate));
    }
    return bps;
}

const char *codec_label(Codec c) noexcept {
    switch (c) {
        case Codec::Av1:  return "AV1";
        case Codec::Hevc: return "HEVC";
        case Codec::H264: return "H.264";
    }
    return "AV1";
}

void apply_vbr_or_fallback(Config &cfg, const ProfileTarget &target, const CodecEncodeCaps &cc,
                           double base_1440p60_mbps, double max_mult,
                           int fallback_av1_qp, int fallback_other_qp) noexcept {
    if (cc.supports_vbr) {
        cfg.rc = RateControl::Vbr;
        cfg.qp = -1;
        cfg.bitrate = recommended_bitrate(target.width, target.height, cfg.fps, cfg.codec,
                                          cfg.pixfmt, base_1440p60_mbps, cc.max_hw_bitrate);
        cfg.max_bitrate = round_mbps((static_cast<double>(cfg.bitrate) / 1e6) * max_mult,
                                     cc.max_hw_bitrate);
    } else if (cc.supports_cqp) {
        cfg.rc = RateControl::Cqp;
        cfg.qp = (cfg.codec == Codec::Av1) ? fallback_av1_qp : fallback_other_qp;
        cfg.bitrate = 0;
        cfg.max_bitrate = 0;
    } else {
        cfg.rc = RateControl::Cbr;
        cfg.qp = -1;
        cfg.bitrate = recommended_bitrate(target.width, target.height, cfg.fps, cfg.codec,
                                          cfg.pixfmt, base_1440p60_mbps, cc.max_hw_bitrate);
        cfg.max_bitrate = 0;
    }
}

void apply_cqp_or_fallback(Config &cfg, const ProfileTarget &target, const CodecEncodeCaps &cc,
                           int av1_qp, int other_qp, double fallback_1440p60_mbps) noexcept {
    if (cc.supports_cqp) {
        cfg.rc = RateControl::Cqp;
        cfg.qp = (cfg.codec == Codec::Av1) ? av1_qp : other_qp;
        cfg.bitrate = 0;
        cfg.max_bitrate = 0;
    } else {
        apply_vbr_or_fallback(cfg, target, cc, fallback_1440p60_mbps, 1.50, av1_qp, other_qp);
    }
}

std::expected<void, std::string> apply_profile_impl(
    Config &cfg, std::string_view name, const ProfileTarget &target,
    std::optional<Codec> forced_codec,
    std::optional<uint32_t> forced_fps,
    std::optional<PixelFormat> forced_pixfmt) {
    const std::string key = to_lower_copy(name);
    const std::optional<Codec> auto_codec = best_codec_for(target);
    if (!forced_codec && !auto_codec) {
        return std::unexpected(no_encode_support_error(target.gpu_name));
    }
    const Codec eff_c = forced_codec ? *forced_codec : *auto_codec;
    const CodecEncodeCaps &cc = target.caps_for(eff_c);
    const bool supports_10bit = cc.supports_10bit && (eff_c == Codec::Av1 || eff_c == Codec::Hevc);
    const int max_q = static_cast<int>(std::max<uint32_t>(1u, cc.max_quality_levels)) - 1;
    const int std_q = std::min(1, max_q);
    const uint32_t std_fps = forced_fps ? *forced_fps : recommended_standard_fps(target.refresh_hz);
    const uint32_t gp_fps = forced_fps ? *forced_fps : recommended_gameplay_fps(target.refresh_hz);

    if (key.empty() || key == "default" || key == "balanced") {
        cfg.profile = "default";
        cfg.fps = std_fps;
        cfg.codec = eff_c;
        cfg.pixfmt = forced_pixfmt ? *forced_pixfmt : PixelFormat::Nv12;
        apply_vbr_or_fallback(cfg, target, cc, 20.0, 1.60, 52, 22);
        cfg.quality = -1;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = 0;
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        cfg.dither = (cfg.pixfmt == PixelFormat::Nv12);
        return {};
    }
    if (key == "gameplay") {
        cfg.profile = "gameplay";
        cfg.fps = gp_fps;
        cfg.codec = eff_c;
        cfg.pixfmt = forced_pixfmt ? *forced_pixfmt : PixelFormat::Nv12;
        apply_vbr_or_fallback(cfg, target, cc, 20.8, 1.57, 56, 24);
        cfg.quality = std_q;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = 0;
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        cfg.dither = (cfg.pixfmt == PixelFormat::Nv12);
        return {};
    }
    if (key == "gameplay-10bit" || key == "gameplay10") {
        cfg.profile = "gameplay-10bit";
        cfg.fps = gp_fps;
        cfg.codec = eff_c;
        cfg.pixfmt = forced_pixfmt ? *forced_pixfmt : (supports_10bit ? PixelFormat::P010 : PixelFormat::Nv12);
        apply_vbr_or_fallback(cfg, target, cc, 23.8, 1.50, 52, 22);
        cfg.quality = std_q;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = 0;
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        cfg.dither = (cfg.pixfmt == PixelFormat::Nv12);
        return {};
    }
    if (key == "master" || key == "archival") {
        cfg.profile = "master";
        cfg.fps = std_fps;
        cfg.codec = eff_c;
        cfg.pixfmt = forced_pixfmt ? *forced_pixfmt : (supports_10bit ? PixelFormat::P010 : PixelFormat::Nv12);
        apply_cqp_or_fallback(cfg, target, cc, 36, 18, 45.0);
        cfg.quality = std_q;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = 0;
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        cfg.dither = (cfg.pixfmt == PixelFormat::Nv12);
        return {};
    }
    if (key == "compact" || key == "small") {
        cfg.profile = "compact";
        cfg.fps = std_fps;
        cfg.codec = eff_c;
        cfg.pixfmt = forced_pixfmt ? *forced_pixfmt : PixelFormat::Nv12;
        apply_vbr_or_fallback(cfg, target, cc, 8.0, 1.50, 78, 30);
        cfg.quality = std_q;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = static_cast<int>(cfg.fps * 4u);  // 4-second GOP
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        cfg.dither = (cfg.pixfmt == PixelFormat::Nv12);
        return {};
    }
    if (key == "desktop" || key == "ui") {
        cfg.profile = "desktop";
        cfg.fps = std_fps;
        cfg.codec = eff_c;
        cfg.pixfmt = forced_pixfmt ? *forced_pixfmt : PixelFormat::Nv12;
        apply_cqp_or_fallback(cfg, target, cc, 62, 26, 12.0);
        cfg.quality = std_q;
        cfg.tune = Tune::Hq;
        cfg.usage = Usage::Record;
        cfg.content = Content::Desktop;
        cfg.idr_interval = static_cast<int>(cfg.fps * 4u);  // 4-second GOP
        cfg.async_depth = 4;
        cfg.b_depth = 1;
        cfg.dither = (cfg.pixfmt == PixelFormat::Nv12);
        return {};
    }
    if (key == "low-latency" || key == "ll" || key == "light") {
        cfg.profile = "low-latency";
        cfg.fps = std_fps;
        cfg.codec = eff_c;
        cfg.pixfmt = forced_pixfmt ? *forced_pixfmt : PixelFormat::Nv12;
        apply_vbr_or_fallback(cfg, target, cc, 18.0, 1.56, 64, 26);
        cfg.quality = 0;
        cfg.tune = Tune::Ll;
        cfg.usage = Usage::Record;
        cfg.content = Content::Rendered;
        cfg.idr_interval = 0;
        cfg.async_depth = 2;
        cfg.b_depth = 1;
        cfg.dither = (cfg.pixfmt == PixelFormat::Nv12);
        return {};
    }
    return std::unexpected(std::format("Unknown profile '{}'. Run with --list-profiles to see available profiles.", name));
}

}  // namespace

std::expected<void, std::string> apply_profile(Config &cfg, std::string_view name, const ProfileTarget &target) {
    return apply_profile_impl(cfg, name, target, std::nullopt, std::nullopt, std::nullopt);
}

std::expected<void, std::string> adapt_config_to_target(Config &cfg, const ProfileTarget &target) {
    if (!target.has_any_codec()) {
        return std::unexpected(no_encode_support_error(target.gpu_name));
    }
    Config adapted{};
    auto res = apply_profile_impl(
        adapted,
        cfg.profile.empty() ? "default" : cfg.profile,
        target,
        cfg.codec_explicit ? std::optional<Codec>{cfg.codec} : std::nullopt,
        cfg.fps_explicit ? std::optional<uint32_t>{cfg.fps} : std::nullopt,
        cfg.pixfmt_explicit ? std::optional<PixelFormat>{cfg.pixfmt} : std::nullopt);
    if (!res) return res;

    if (!cfg.fps_explicit) cfg.fps = adapted.fps;
    if (!cfg.codec_explicit) cfg.codec = adapted.codec;
    if (!cfg.pixfmt_explicit) {
        cfg.pixfmt = adapted.pixfmt;
        cfg.dither = adapted.dither;
    }
    if (!cfg.quality_explicit) cfg.quality = adapted.quality;

    if (!cfg.rc_explicit && !cfg.qp_explicit && !cfg.bitrate_explicit) {
        cfg.rc = adapted.rc;
        cfg.qp = adapted.qp;
        cfg.bitrate = adapted.bitrate;
        if (!cfg.max_bitrate_explicit) cfg.max_bitrate = adapted.max_bitrate;
    } else if (!cfg.bitrate_explicit && (cfg.rc == RateControl::Vbr || cfg.rc == RateControl::Cbr)) {
        cfg.bitrate = adapted.bitrate;
        if (!cfg.max_bitrate_explicit) {
            cfg.max_bitrate = (cfg.rc == RateControl::Vbr) ? adapted.max_bitrate : 0;
        }
    } else if (!cfg.qp_explicit && cfg.rc == RateControl::Cqp && cfg.qp < 0) {
        cfg.qp = (adapted.qp >= 0) ? adapted.qp : default_qp_for_codec(cfg.codec);
    }
    return {};
}

std::vector<AdaptiveProfileInfo> describe_profiles(const ProfileTarget &target) {
    std::vector<AdaptiveProfileInfo> out;
    if (!target.has_any_codec()) return out;
    out.reserve(kProfiles.size());
    for (const auto &p : kProfiles) {
        Config c{};
        if (!apply_profile(c, p.name, target).has_value()) continue;
        std::string rc_desc = (c.rc == RateControl::Cqp)
            ? std::format("CQP {}", c.qp)
            : (c.rc == RateControl::Cbr)
                ? std::format("CBR {}M", c.bitrate / 1'000'000)
                : std::format("VBR {}M (max {}M)", c.bitrate / 1'000'000, c.max_bitrate / 1'000'000);
        const char *fmt_desc = (c.pixfmt == PixelFormat::P010) ? "10-bit P010" : "8-bit NV12 + IGN";
        const auto &cc = target.caps_for(c.codec);
        const int eff_q = (c.quality >= 0) ? c.quality
            : (static_cast<int>(std::max<uint32_t>(1u, cc.max_quality_levels)) > 1 ? 1 : 0);
        std::string extra;
        if (c.idr_interval > 0 && c.idr_interval == static_cast<int>(c.fps * 4u)) extra += " | 4s GOP";
        if (c.content == Content::Desktop) extra += " | desktop";
        else if (c.tune == Tune::Ll) extra += std::format(" | quality {} | ll | async {}", eff_q, c.async_depth);
        else extra += std::format(" | quality {} | hq", eff_q);

        out.push_back({
            std::string(p.name),
            std::string(p.summary),
            std::format("{} fps | {} | {} | {}{}", c.fps, codec_label(c.codec), fmt_desc, rc_desc, extra),
        });
    }
    return out;
}

void print_profiles(const ProfileTarget &target) {
    if (!target.has_any_codec()) {
        std::cerr << "Error: " << no_encode_support_error(target.gpu_name) << '\n';
        return;
    }
    std::cout << "Available adaptive profiles (use with -p <name>; explicit flags override):\n";
    if (!target.display_label.empty() || !target.gpu_name.empty()) {
        const Codec best_c = *best_codec_for(target);
        std::cout << std::format("  Tailored for: {}{}{}{}\n",
                                 target.display_label.empty() ? std::format("{}x{}@{:.0f}Hz", target.width, target.height, target.refresh_hz) : target.display_label,
                                 target.gpu_name.empty() ? "" : " | ",
                                 target.gpu_name,
                                 std::format(" ({})", codec_label(best_c)));
    }
    std::cout << '\n';
    for (const auto &p : describe_profiles(target)) {
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

std::string expand_user_path(std::string_view path) {
    if (path.size() >= 2 && path[0] == '~' && path[1] == '/') {
        if (const char *home = std::getenv("HOME"); home && home[0] != '\0') {
            return std::string(home) + std::string(path.substr(1));
        }
    }
    return std::string(path);
}

std::expected<std::pair<uint32_t, uint32_t>, std::string> parse_resolution(const std::string &str) {
    const std::string lo = to_lower_copy(str);
    if (lo == "720p" || lo == "hd") return std::pair<uint32_t, uint32_t>{1280, 720};
    if (lo == "1080p" || lo == "fhd") return std::pair<uint32_t, uint32_t>{1920, 1080};
    if (lo == "1440p" || lo == "qhd" || lo == "2k") return std::pair<uint32_t, uint32_t>{2560, 1440};
    if (lo == "2160p" || lo == "4k" || lo == "uhd") return std::pair<uint32_t, uint32_t>{3840, 2160};

    auto pos = str.find('x');
    if (pos == std::string::npos) pos = str.find('X');
    if (pos == std::string::npos || pos == 0 || pos + 1 >= str.size()) {
        return std::unexpected(std::format("Invalid resolution '{}', expected WxH or preset (720p, 1080p, 1440p, 4k)", str));
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

std::expected<CropRect, std::string> parse_crop(std::string_view str) {
    // Expected format: WxH+X+Y (e.g. 1920x1080+320+180)
    auto x_pos = str.find('x');
    if (x_pos == std::string_view::npos) x_pos = str.find('X');
    auto p1 = str.find('+');
    auto p2 = (p1 != std::string_view::npos) ? str.find('+', p1 + 1) : std::string_view::npos;
    if (x_pos == std::string_view::npos || p1 == std::string_view::npos ||
        p2 == std::string_view::npos || x_pos == 0 || p1 <= x_pos + 1 ||
        p2 <= p1 + 1 || p2 + 1 >= str.size()) {
        return std::unexpected(std::format("Invalid crop '{}', expected WxH+X+Y (e.g. 1920x1080+320+180)", str));
    }
    auto parse_i32 = [](std::string_view s, int32_t &out) {
        auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
        return ec == std::errc() && p == s.data() + s.size();
    };
    CropRect r{};
    if (!parse_i32(str.substr(0, x_pos), r.width) ||
        !parse_i32(str.substr(x_pos + 1, p1 - x_pos - 1), r.height) ||
        !parse_i32(str.substr(p1 + 1, p2 - p1 - 1), r.x) ||
        !parse_i32(str.substr(p2 + 1), r.y) ||
        r.width < 2 || r.height < 2 || r.x < 0 || r.y < 0) {
        return std::unexpected(std::format("Invalid crop '{}': width/height must be >= 2 and x/y >= 0", str));
    }
    return r;
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
  -s, --scale <WxH|preset>    Scale output via GPU Catmull-Rom bicubic (even WxH, or 720p/1080p/1440p/4k)
      --crop <WxH+X+Y>        Zero-copy GPU crop region in display pixels (e.g. 1920x1080+320+180)
  -d, --duration <sec>        Stop recording automatically after <sec> seconds
      --instant-replay [sec]  Enable rolling instant-replay buffer [default: 30s; save via SIGUSR2 or 's']
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

Vulkan Video Encoder (hardware-adaptive; AV1 / HEVC / H.264):
      --codec <name>          av1, hevc, h264 [default: auto (best supported by GPU)]
      --pixfmt <fmt>          nv12 (8-bit + IGN dither) or p010 (10-bit) [default: nv12]
      --rc <mode>             Rate control: auto, vbr, cqp, cbr [default: auto (adaptive VBR)]
      --qp <n>                Quantizer for CQP (0..255 for AV1 [52], 0..51 for HEVC/H264 [22]; selects CQP if --rc=auto)
  -b, --bitrate <rate>        Target bitrate (e.g. 20M, 35M) [default: 20M]
      --max-bitrate <rate>    Max bitrate for VBR mode [default: 32M or 1.5x bitrate]
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
    OPT_CROP,
    OPT_INSTANT_REPLAY,
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

std::expected<Config, std::string> parse_args(int argc, char **argv, const Config *base_defaults) {
    Config cfg = base_defaults ? *base_defaults : Config{};
    // Never inherit a saved recording duration limit when seeding from defaults;
    // explicit -d on the command line will still set it below.
    if (base_defaults) {
        cfg.duration_sec = 0.0;
        cfg.from_saved_defaults = true;
    }

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
        {"crop",              required_argument, nullptr, OPT_CROP},
        {"instant-replay",    optional_argument, nullptr, OPT_INSTANT_REPLAY},
        {"replay-buffer",     optional_argument, nullptr, OPT_INSTANT_REPLAY},
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
                cfg.fps_explicit = true;
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
                cfg.bitrate_explicit = true;
                // Reset default max_bitrate so 1.5x scaling applies unless --max-bitrate is also passed
                if (!cfg.max_bitrate_explicit) cfg.max_bitrate = 0;
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
                cfg.pixfmt_explicit = true;
                break;
            }
            case OPT_RC: {
                auto r = parse_enum<RateControl>(optarg, {
                    {"auto", RateControl::Auto}, {"cqp", RateControl::Cqp},
                    {"cbr", RateControl::Cbr}, {"vbr", RateControl::Vbr}}, "--rc");
                if (!r) return std::unexpected(r.error());
                cfg.rc = *r;
                cfg.rc_explicit = true;
                break;
            }
            case OPT_QP: {
                auto r = parse_int(optarg, 0, 255, "--qp");
                if (!r) return std::unexpected(r.error());
                cfg.qp = *r;
                cfg.qp_explicit = true;
                break;
            }
            case OPT_MAX_BITRATE: {
                auto r = parse_bitrate(optarg);
                if (!r) return std::unexpected(r.error());
                cfg.max_bitrate = *r;
                cfg.max_bitrate_explicit = true;
                break;
            }
            case OPT_QUALITY: {
                auto r = parse_int(optarg, 0, 16, "--quality");
                if (!r) return std::unexpected(r.error());
                cfg.quality = *r;
                cfg.quality_explicit = true;
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
            case OPT_CROP: {
                auto r = parse_crop(optarg);
                if (!r) return std::unexpected(r.error());
                cfg.crop = *r;
                break;
            }
            case OPT_INSTANT_REPLAY: {
                int sec = 30;
                if (optarg && optarg[0] != '\0') {
                    auto r = parse_int(optarg, 5, 600, "--instant-replay");
                    if (!r) return std::unexpected(r.error());
                    sec = *r;
                } else if (optind < argc && argv[optind] && std::isdigit(static_cast<unsigned char>(argv[optind][0]))) {
                    int parsed = 0;
                    std::string_view next(argv[optind]);
                    auto [p, ec] = std::from_chars(next.data(), next.data() + next.size(), parsed);
                    if (ec == std::errc() && p == next.data() + next.size()) {
                        if (parsed < 5 || parsed > 600) {
                            return std::unexpected(std::format("Invalid integer '{}' for --instant-replay (5..600)", next));
                        }
                        sec = parsed;
                        ++optind;
                    }
                }
                cfg.replay_seconds = static_cast<uint32_t>(sec);
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
    } else {
        cfg.output_path = expand_user_path(cfg.output_path);
    }
    return cfg;
}

}  // namespace vkcap
