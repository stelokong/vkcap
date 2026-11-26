#include "user_config.hpp"

#include <charconv>
#include <cstdlib>
#include <format>
#include <fstream>
#include <sstream>
#include <string_view>

namespace vkcap {

namespace {

std::string_view codec_to_str(Codec c) {
    switch (c) {
        case Codec::Av1:  return "av1";
        case Codec::Hevc: return "hevc";
        case Codec::H264: return "h264";
    }
    return "av1";
}
std::string_view pixfmt_to_str(PixelFormat p) {
    return p == PixelFormat::P010 ? "p010" : "nv12";
}
std::string_view rc_to_str(RateControl r) {
    switch (r) {
        case RateControl::Auto: return "auto";
        case RateControl::Cqp:  return "cqp";
        case RateControl::Cbr:  return "cbr";
        case RateControl::Vbr:  return "vbr";
    }
    return "auto";
}
std::string_view tune_to_str(Tune t) {
    switch (t) {
        case Tune::Default:  return "default";
        case Tune::Hq:       return "hq";
        case Tune::Ll:       return "ll";
        case Tune::Ull:      return "ull";
        case Tune::Lossless: return "lossless";
    }
    return "hq";
}
std::string_view usage_to_str(Usage u) {
    switch (u) {
        case Usage::Default:    return "default";
        case Usage::Transcode:  return "transcode";
        case Usage::Stream:     return "stream";
        case Usage::Record:     return "record";
        case Usage::Conference: return "conference";
    }
    return "record";
}
std::string_view content_to_str(Content c) {
    switch (c) {
        case Content::Default:  return "default";
        case Content::Camera:   return "camera";
        case Content::Desktop:  return "desktop";
        case Content::Rendered: return "rendered";
    }
    return "rendered";
}
std::string_view crange_to_str(ColorRange r) {
    return r == ColorRange::Full ? "full" : "limited";
}
std::string_view cspace_to_str(ColorSpace s) {
    return s == ColorSpace::Bt2020 ? "bt2020" : "bt709";
}
std::string_view gamut_to_str(InputGamut g) {
    switch (g) {
        case InputGamut::Auto:      return "auto";
        case InputGamut::Edid:      return "edid";
        case InputGamut::Srgb:      return "srgb";
        case InputGamut::DisplayP3: return "p3";
        case InputGamut::Bt2020:    return "bt2020";
    }
    return "auto";
}
std::string_view acodec_to_str(AudioCodec a) {
    switch (a) {
        case AudioCodec::Opus: return "opus";
        case AudioCodec::Aac:  return "aac";
        case AudioCodec::Flac: return "flac";
    }
    return "opus";
}

std::string trim_sv(std::string_view sv) {
    while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t' || sv.front() == '\r')) sv.remove_prefix(1);
    while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t' || sv.back() == '\r')) sv.remove_suffix(1);
    return std::string(sv);
}

bool parse_bool(std::string_view v) {
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

template <typename T>
bool parse_num(std::string_view v, T &out) {
    auto [p, ec] = std::from_chars(v.data(), v.data() + v.size(), out);
    return ec == std::errc() && p == v.data() + v.size();
}

std::string extract_ext(const std::string &path) {
    std::filesystem::path p(path);
    std::string ext = p.extension().string();
    if (ext == ".mkv" || ext == ".mp4" || ext == ".webm" || ext == ".ivf") {
        return ext;
    }
    return ".mkv";
}

}  // namespace

std::filesystem::path user_defaults_path() {
    if (const char *xdg = std::getenv("XDG_CONFIG_HOME"); xdg && xdg[0] != '\0') {
        return std::filesystem::path(xdg) / "vkcap" / "defaults.conf";
    }
    if (const char *home = std::getenv("HOME"); home && home[0] != '\0') {
        return std::filesystem::path(home) / ".config" / "vkcap" / "defaults.conf";
    }
    return std::filesystem::path(".vkcap-defaults.conf");
}

std::expected<std::optional<Config>, std::string> load_user_defaults(const std::filesystem::path &path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return std::optional<Config>{std::nullopt};
    }

    std::ifstream in(path);
    if (!in) {
        return std::unexpected(std::format("Failed to open saved defaults '{}'", path.string()));
    }

    Config cfg{};
    std::string container_ext = ".mkv";
    std::string output_dir;
    std::string line;
    while (std::getline(in, line)) {
        std::string trimmed = trim_sv(line);
        if (trimmed.empty() || trimmed.front() == '#') continue;
        auto eq = trimmed.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim_sv(std::string_view(trimmed).substr(0, eq));
        std::string v = trim_sv(std::string_view(trimmed).substr(eq + 1));

        if (k == "display") cfg.display = v;
        else if (k == "profile") cfg.profile = v;
        else if (k == "container_ext") container_ext = v.empty() ? ".mkv" : v;
        else if (k == "output_dir") output_dir = v;
        else if (k == "fps") parse_num(v, cfg.fps);
        else if (k == "out_width") parse_num(v, cfg.out_width);
        else if (k == "out_height") parse_num(v, cfg.out_height);
        else if (k == "duration_sec") parse_num(v, cfg.duration_sec);
        else if (k == "capture_cursor") cfg.capture_cursor = parse_bool(v);
        else if (k == "audio") cfg.audio = parse_bool(v);
        else if (k == "capture_system_audio") cfg.capture_system_audio = parse_bool(v);
        else if (k == "capture_mic") cfg.capture_mic = parse_bool(v);
        else if (k == "mic_device") cfg.mic_device = v;
        else if (k == "system_volume_pct") parse_num(v, cfg.system_volume_pct);
        else if (k == "mic_volume_pct") parse_num(v, cfg.mic_volume_pct);
        else if (k == "audio_bitrate") parse_num(v, cfg.audio_bitrate);
        else if (k == "audio_codec") {
            if (v == "aac") cfg.audio_codec = AudioCodec::Aac;
            else if (v == "flac") cfg.audio_codec = AudioCodec::Flac;
            else cfg.audio_codec = AudioCodec::Opus;
        } else if (k == "codec") {
            if (v == "hevc") cfg.codec = Codec::Hevc;
            else if (v == "h264") cfg.codec = Codec::H264;
            else cfg.codec = Codec::Av1;
        } else if (k == "pixfmt") {
            cfg.pixfmt = (v == "p010") ? PixelFormat::P010 : PixelFormat::Nv12;
        } else if (k == "rc") {
            if (v == "cqp") cfg.rc = RateControl::Cqp;
            else if (v == "cbr") cfg.rc = RateControl::Cbr;
            else if (v == "vbr") cfg.rc = RateControl::Vbr;
            else cfg.rc = RateControl::Auto;
        } else if (k == "qp") parse_num(v, cfg.qp);
        else if (k == "bitrate") parse_num(v, cfg.bitrate);
        else if (k == "max_bitrate") parse_num(v, cfg.max_bitrate);
        else if (k == "quality") parse_num(v, cfg.quality);
        else if (k == "tune") {
            if (v == "default") cfg.tune = Tune::Default;
            else if (v == "ll") cfg.tune = Tune::Ll;
            else if (v == "ull") cfg.tune = Tune::Ull;
            else if (v == "lossless") cfg.tune = Tune::Lossless;
            else cfg.tune = Tune::Hq;
        } else if (k == "usage") {
            if (v == "default") cfg.usage = Usage::Default;
            else if (v == "transcode") cfg.usage = Usage::Transcode;
            else if (v == "stream") cfg.usage = Usage::Stream;
            else if (v == "conference") cfg.usage = Usage::Conference;
            else cfg.usage = Usage::Record;
        } else if (k == "content") {
            if (v == "default") cfg.content = Content::Default;
            else if (v == "camera") cfg.content = Content::Camera;
            else if (v == "desktop") cfg.content = Content::Desktop;
            else cfg.content = Content::Rendered;
        } else if (k == "idr_interval") parse_num(v, cfg.idr_interval);
        else if (k == "async_depth") parse_num(v, cfg.async_depth);
        else if (k == "b_depth") parse_num(v, cfg.b_depth);
        else if (k == "device_index") parse_num(v, cfg.device_index);
        else if (k == "color_range") cfg.color_range = (v == "full") ? ColorRange::Full : ColorRange::Limited;
        else if (k == "color_space") cfg.color_space = (v == "bt2020") ? ColorSpace::Bt2020 : ColorSpace::Bt709;
        else if (k == "input_gamut") {
            if (v == "edid") cfg.input_gamut = InputGamut::Edid;
            else if (v == "srgb") cfg.input_gamut = InputGamut::Srgb;
            else if (v == "p3") cfg.input_gamut = InputGamut::DisplayP3;
            else if (v == "bt2020") cfg.input_gamut = InputGamut::Bt2020;
            else cfg.input_gamut = InputGamut::Auto;
        } else if (k == "dither") cfg.dither = parse_bool(v);
        else if (k == "queue_size") parse_num(v, cfg.queue_size);
    }

    if (!output_dir.empty()) {
        cfg.output_path = (std::filesystem::path(expand_user_path(output_dir)) /
                           default_output_filename(container_ext)).string();
    } else {
        cfg.output_path = default_output_filename(container_ext);
    }
    return std::optional<Config>{std::move(cfg)};
}

std::expected<void, std::string> save_user_defaults(const Config &cfg, const std::filesystem::path &path) {
    std::error_code ec;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) {
            return std::unexpected(std::format("Failed to create config directory '{}': {}",
                                               path.parent_path().string(), ec.message()));
        }
    }

    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        return std::unexpected(std::format("Failed to write user defaults to '{}'", path.string()));
    }

    out << "# vkcap personal default settings (independent of built-in profiles)\n";
    out << "display=" << cfg.display << '\n';
    out << "profile=" << cfg.profile << '\n';
    out << "container_ext=" << extract_ext(cfg.output_path) << '\n';
    if (std::filesystem::path p(cfg.output_path); p.has_parent_path()) {
        out << "output_dir=" << p.parent_path().string() << '\n';
    }
    out << "fps=" << cfg.fps << '\n';
    out << "out_width=" << cfg.out_width << '\n';
    out << "out_height=" << cfg.out_height << '\n';
    out << "duration_sec=" << cfg.duration_sec << '\n';
    out << "capture_cursor=" << (cfg.capture_cursor ? 1 : 0) << '\n';
    out << "audio=" << (cfg.audio ? 1 : 0) << '\n';
    out << "capture_system_audio=" << (cfg.capture_system_audio ? 1 : 0) << '\n';
    out << "capture_mic=" << (cfg.capture_mic ? 1 : 0) << '\n';
    out << "mic_device=" << cfg.mic_device << '\n';
    out << "system_volume_pct=" << cfg.system_volume_pct << '\n';
    out << "mic_volume_pct=" << cfg.mic_volume_pct << '\n';
    out << "audio_codec=" << acodec_to_str(cfg.audio_codec) << '\n';
    out << "audio_bitrate=" << cfg.audio_bitrate << '\n';
    out << "codec=" << codec_to_str(cfg.codec) << '\n';
    out << "pixfmt=" << pixfmt_to_str(cfg.pixfmt) << '\n';
    out << "rc=" << rc_to_str(cfg.rc) << '\n';
    out << "qp=" << cfg.qp << '\n';
    out << "bitrate=" << cfg.bitrate << '\n';
    out << "max_bitrate=" << cfg.max_bitrate << '\n';
    out << "quality=" << cfg.quality << '\n';
    out << "tune=" << tune_to_str(cfg.tune) << '\n';
    out << "usage=" << usage_to_str(cfg.usage) << '\n';
    out << "content=" << content_to_str(cfg.content) << '\n';
    out << "idr_interval=" << cfg.idr_interval << '\n';
    out << "async_depth=" << cfg.async_depth << '\n';
    out << "b_depth=" << cfg.b_depth << '\n';
    out << "device_index=" << cfg.device_index << '\n';
    out << "color_range=" << crange_to_str(cfg.color_range) << '\n';
    out << "color_space=" << cspace_to_str(cfg.color_space) << '\n';
    out << "input_gamut=" << gamut_to_str(cfg.input_gamut) << '\n';
    out << "dither=" << (cfg.dither ? 1 : 0) << '\n';
    out << "queue_size=" << cfg.queue_size << '\n';

    if (!out.good()) {
        return std::unexpected(std::format("I/O error writing user defaults to '{}'", path.string()));
    }
    return {};
}

std::string format_config_summary(const Config &cfg) {
    std::ostringstream ss;
    const std::string res_str = (cfg.out_width > 0 && cfg.out_height > 0)
        ? std::format("{}x{}", cfg.out_width, cfg.out_height) : "native";

    std::string rc_str;
    RateControl eff_rc = cfg.rc;
    if (eff_rc == RateControl::Auto) eff_rc = (cfg.bitrate > 0) ? RateControl::Vbr : RateControl::Cqp;
    if (eff_rc == RateControl::Cqp) {
        const int q = cfg.qp >= 0 ? cfg.qp : default_qp_for_codec(cfg.codec);
        rc_str = std::format("CQP {}", q);
    } else if (eff_rc == RateControl::Vbr) {
        rc_str = std::format("VBR {:.1f}M", static_cast<double>(cfg.bitrate > 0 ? cfg.bitrate : 20'000'000) / 1e6);
    } else {
        rc_str = std::format("CBR {:.1f}M", static_cast<double>(cfg.bitrate > 0 ? cfg.bitrate : 20'000'000) / 1e6);
    }

    ss << std::format("  Display : {} | {} @ {} fps | cursor={}\n",
                      cfg.display.empty() ? "auto" : cfg.display,
                      res_str, cfg.fps, cfg.capture_cursor ? "on" : "off");
    ss << std::format("  Video   : {} ({}) | {} | quality={} | tune={} | content={}\n",
                      codec_to_str(cfg.codec), pixfmt_to_str(cfg.pixfmt), rc_str,
                      cfg.quality >= 0 ? cfg.quality : 1,
                      tune_to_str(cfg.tune), content_to_str(cfg.content));
    ss << std::format("  Color   : gamut={} -> {} ({}) | dither={}\n",
                      gamut_to_str(cfg.input_gamut), cspace_to_str(cfg.color_space),
                      crange_to_str(cfg.color_range), cfg.dither ? "on" : "off");
    if (!cfg.audio || (!cfg.capture_system_audio && !cfg.capture_mic)) {
        ss << "  Audio   : off\n";
    } else {
        std::string src;
        if (cfg.capture_system_audio) src += std::format("system ({:.0f}%)", cfg.system_volume_pct);
        if (cfg.capture_mic) {
            if (!src.empty()) src += " + ";
            src += std::format("mic:{} ({:.0f}%)",
                               cfg.mic_device.empty() ? "default" : cfg.mic_device,
                               cfg.mic_volume_pct);
        }
        std::string ac = (cfg.audio_codec == AudioCodec::Flac)
            ? "flac (lossless)"
            : std::format("{} {}k", acodec_to_str(cfg.audio_codec), cfg.audio_bitrate / 1000);
        ss << std::format("  Audio   : {} | {}\n", ac, src);
    }
    ss << std::format("  Output  : {}{}\n", cfg.output_path,
                      cfg.duration_sec > 0.0 ? std::format(" ({}s limit)", cfg.duration_sec) : "");
    return ss.str();
}

}  // namespace vkcap
