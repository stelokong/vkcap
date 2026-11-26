#include "interactive.hpp"
#include "audio_capture.hpp"
#include "kms_capture.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <format>
#include <initializer_list>
#include <string>
#include <string_view>

namespace vkcap {

namespace {

std::string trim(std::string_view sv) {
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) sv.remove_prefix(1);
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) sv.remove_suffix(1);
    return std::string(sv);
}

std::string lower_copy(std::string_view sv) {
    std::string s(sv);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool is_quit(std::string_view s) {
    std::string lo = lower_copy(s);
    return lo == "q" || lo == "quit" || lo == "exit";
}

bool prompt_line(std::istream &in, std::ostream &out, std::string_view prompt, std::string &line_out) {
    out << prompt << std::flush;
    std::string raw;
    if (!std::getline(in, raw)) {
        out << '\n';
        return false;
    }
    line_out = trim(raw);
    if (is_quit(line_out)) {
        return false;
    }
    return true;
}

bool prompt_yes_no(std::istream &in, std::ostream &out, std::string_view prompt, bool default_yes, bool &result) {
    std::string ans;
    while (true) {
        if (!prompt_line(in, out, prompt, ans)) return false;
        if (ans.empty()) {
            result = default_yes;
            return true;
        }
        const std::string lo = lower_copy(ans);
        if (lo == "y" || lo == "yes") { result = true; return true; }
        if (lo == "n" || lo == "no") { result = false; return true; }
        out << "   Please enter 'y' or 'n'.\n";
    }
}

bool prompt_int(std::istream &in, std::ostream &out, std::string_view label,
                int def_val, int min_v, int max_v, int &out_val) {
    std::string ans;
    while (true) {
        if (!prompt_line(in, out, std::format("{} [{}]: ", label, def_val), ans)) return false;
        if (ans.empty()) { out_val = def_val; return true; }
        int v = 0;
        auto [p, ec] = std::from_chars(ans.data(), ans.data() + ans.size(), v);
        if (ec == std::errc() && p == ans.data() + ans.size() && v >= min_v && v <= max_v) {
            out_val = v;
            return true;
        }
        out << std::format("   Please enter an integer between {} and {}.\n", min_v, max_v);
    }
}

template <typename T>
bool prompt_choice(std::istream &in, std::ostream &out, std::string_view heading,
                   std::initializer_list<std::pair<std::string_view, T>> options,
                   size_t default_idx, T &out_val) {
    out << heading << '\n';
    size_t idx = 0;
    for (const auto &[label, _] : options) {
        out << std::format("   [{}] {}{}\n", idx + 1, label, idx == default_idx ? "  (default)" : "");
        ++idx;
    }
    std::string ans;
    while (true) {
        if (!prompt_line(in, out, std::format("   Choice [{}]: ", default_idx + 1), ans)) return false;
        if (ans.empty()) {
            out_val = (options.begin() + static_cast<ptrdiff_t>(default_idx))->second;
            return true;
        }
        size_t n = 0;
        auto [p, ec] = std::from_chars(ans.data(), ans.data() + ans.size(), n);
        if (ec == std::errc() && p == ans.data() + ans.size() && n >= 1 && n <= options.size()) {
            out_val = (options.begin() + static_cast<ptrdiff_t>(n - 1))->second;
            return true;
        }
        const std::string lo = lower_copy(ans);
        for (const auto &[label, val] : options) {
            if (lower_copy(label).starts_with(lo)) {
                out_val = val;
                return true;
            }
        }
        out << std::format("   Please enter 1..{} or an option name.\n", options.size());
    }
}

std::string format_volume_bar(double pct) {
    constexpr int kWidth = 20;
    // Scale bar across 0..200% (10% per step), show '+' when > 200%
    int filled = static_cast<int>(std::round(std::clamp(pct, 0.0, 200.0) / 10.0));
    std::string bar;
    bar.reserve(kWidth + 4);
    bar.push_back('[');
    for (int i = 0; i < kWidth; ++i) {
        bar.push_back(i < filled ? '=' : '-');
    }
    bar.push_back(']');
    if (pct > 200.0) bar.push_back('+');
    return bar;
}

bool prompt_volume_slider(std::istream &in, std::ostream &out, std::string_view label,
                          double default_pct, double &out_pct) {
    out << std::format("   {} {} {:.0f}%\n", label, format_volume_bar(default_pct), default_pct);
    std::string ans;
    while (true) {
        if (!prompt_line(in, out,
                         std::format("   Volume % (0..1000, e.g. 80, 100, 200%) [{:.0f}%]: ", default_pct),
                         ans)) {
            return false;
        }
        if (ans.empty()) {
            out_pct = default_pct;
            return true;
        }
        auto r = parse_volume_pct(ans);
        if (r) {
            out_pct = *r;
            out << std::format("   -> {} {} {:.0f}%\n", label, format_volume_bar(out_pct), out_pct);
            return true;
        }
        out << "   " << r.error() << '\n';
    }
}

bool prompt_display(std::istream &in, std::ostream &out,
                    const std::vector<KmsDisplayInfo> &disps,
                    size_t &disp_idx, std::string &display_out) {
    if (disps.size() == 1) {
        out << std::format("Display: {} ({}x{} @ {:.1f} Hz)\n\n",
                           disps[0].connector_name, disps[0].width, disps[0].height, disps[0].refresh_hz);
        disp_idx = 0;
        display_out = disps[0].connector_name;
        return true;
    }
    out << "Select display:\n";
    for (size_t i = 0; i < disps.size(); ++i) {
        const auto &d = disps[i];
        out << std::format("   [{}] {:<10} {}x{} @ {:.1f} Hz{}\n",
                           i + 1, d.connector_name, d.width, d.height, d.refresh_hz,
                           i == 0 ? "  (default)" : "");
    }
    std::string ans;
    while (true) {
        if (!prompt_line(in, out, "   Display [1]: ", ans)) return false;
        if (ans.empty()) { disp_idx = 0; break; }
        size_t n = 0;
        auto [p, ec] = std::from_chars(ans.data(), ans.data() + ans.size(), n);
        if (ec == std::errc() && p == ans.data() + ans.size() && n >= 1 && n <= disps.size()) {
            disp_idx = n - 1;
            break;
        }
        bool matched = false;
        std::string lo = lower_copy(ans);
        for (size_t i = 0; i < disps.size(); ++i) {
            if (lower_copy(disps[i].connector_name) == lo) {
                disp_idx = i;
                matched = true;
                break;
            }
        }
        if (matched) break;
        out << std::format("   Please enter 1..{} or a connector name (e.g. {}).\n",
                           disps.size(), disps[0].connector_name);
    }
    display_out = disps[disp_idx].connector_name;
    out << '\n';
    return true;
}

bool prompt_mic_device(std::istream &in, std::ostream &out, std::string &mic_device_out) {
    auto mics_res = AudioCapture::enumerate_microphones();
    if (mics_res && !mics_res->empty()) {
        const auto &mics = *mics_res;
        out << "   Select microphone:\n";
        out << "     [1] System default microphone  (default)\n";
        for (size_t i = 0; i < mics.size(); ++i) {
            out << std::format("     [{}] {}  (id={}, {})\n",
                               i + 2, mics[i].description, mics[i].id, mics[i].name);
        }
        std::string ans;
        while (true) {
            if (!prompt_line(in, out, "   Microphone [1]: ", ans)) return false;
            if (ans.empty() || ans == "1") {
                mic_device_out = "default";
                return true;
            }
            size_t n = 0;
            auto [p, ec] = std::from_chars(ans.data(), ans.data() + ans.size(), n);
            if (ec == std::errc() && p == ans.data() + ans.size() && n >= 2 && n <= mics.size() + 1) {
                mic_device_out = mics[n - 2].name;
                return true;
            }
            const std::string lo = lower_copy(ans);
            for (const auto &m : mics) {
                if (lower_copy(m.name).find(lo) != std::string::npos ||
                    lower_copy(m.description).find(lo) != std::string::npos) {
                    mic_device_out = m.name;
                    return true;
                }
            }
            out << std::format("   Please enter 1..{} or part of a microphone name.\n", mics.size() + 1);
        }
    }
    out << "   (Using system default microphone — no PipeWire source list available)\n";
    mic_device_out = "default";
    return true;
}

bool prompt_audio_section(std::istream &in, std::ostream &out, Config &cfg, bool advanced) {
    if (!prompt_yes_no(in, out, "Record system audio (all applications)? [Y/n]: ", true, cfg.capture_system_audio)) {
        return false;
    }
    if (cfg.capture_system_audio) {
        if (!prompt_volume_slider(in, out, "System audio", 100.0, cfg.system_volume_pct)) return false;
    }

    if (!prompt_yes_no(in, out, "Record microphone? [y/N]: ", false, cfg.capture_mic)) {
        return false;
    }
    cfg.mic_device.clear();
    if (cfg.capture_mic) {
        if (!prompt_mic_device(in, out, cfg.mic_device)) return false;
        if (!prompt_volume_slider(in, out, "Microphone  ", 100.0, cfg.mic_volume_pct)) return false;
    }

    cfg.audio = cfg.capture_system_audio || cfg.capture_mic;

    if (advanced && cfg.audio) {
        if (!prompt_choice<AudioCodec>(in, out, "Audio codec:", {
                {"opus (recommended — low latency, best quality/size)", AudioCodec::Opus},
                {"aac  (broad MP4 compatibility)", AudioCodec::Aac},
                {"flac (lossless PCM audio, best in .mkv)", AudioCodec::Flac},
            }, 0, cfg.audio_codec)) {
            return false;
        }
        if (cfg.audio_codec != AudioCodec::Flac) {
            std::string ans;
            while (true) {
                if (!prompt_line(in, out, "Audio bitrate (e.g. 128k, 192k, 256k) [192k]: ", ans)) return false;
                if (ans.empty()) { cfg.audio_bitrate = 192'000; break; }
                auto r = parse_bitrate(ans);
                if (r) { cfg.audio_bitrate = *r; break; }
                out << "   " << r.error() << '\n';
            }
        }
    }
    return true;
}

bool prompt_resolution_and_output(std::istream &in, std::ostream &out,
                                  const KmsDisplayInfo &d, Config &cfg) {
    {
        std::string ans;
        while (true) {
            if (!prompt_line(in, out,
                             std::format("Output resolution [native {}x{}, or WxH e.g. 1920x1080]: ",
                                         d.width, d.height),
                             ans)) {
                return false;
            }
            if (ans.empty() || lower_copy(ans) == "native") {
                cfg.out_width = 0;
                cfg.out_height = 0;
                break;
            }
            auto res = parse_resolution(ans);
            if (res) {
                cfg.out_width = res->first;
                cfg.out_height = res->second;
                break;
            }
            out << "   " << res.error() << '\n';
        }
    }
    {
        std::string def_out = cfg.output_path.empty() ? default_output_filename() : cfg.output_path;
        std::string ans;
        if (!prompt_line(in, out, std::format("Output file (.mkv, .mp4, .webm, .ivf) [{}]: ", def_out), ans)) {
            return false;
        }
        cfg.output_path = ans.empty() ? def_out : ans;
    }
    return true;
}

std::string format_equiv_cli(const Config &cfg, const char *prog_name) {
    std::string cmd = prog_name ? prog_name : "vkcap";
    if (!cfg.display.empty()) cmd += std::format(" -D {}", cfg.display);
    if (!cfg.profile.empty() && cfg.profile != "default") {
        cmd += std::format(" -p {}", cfg.profile);
    } else {
        cmd += std::format(" -f {}", cfg.fps);
        if (cfg.codec == Codec::Hevc) cmd += " --codec hevc";
        else if (cfg.codec == Codec::H264) cmd += " --codec h264";
        if (cfg.pixfmt == PixelFormat::P010) cmd += " --pixfmt p010";
        if (cfg.rc == RateControl::Cqp && cfg.qp >= 0) cmd += std::format(" --rc cqp --qp {}", cfg.qp);
        else if (cfg.rc == RateControl::Vbr && cfg.bitrate > 0) cmd += std::format(" --rc vbr -b {}k", cfg.bitrate / 1000);
        else if (cfg.rc == RateControl::Cbr && cfg.bitrate > 0) cmd += std::format(" --rc cbr -b {}k", cfg.bitrate / 1000);
    }
    if (!cfg.audio || (!cfg.capture_system_audio && !cfg.capture_mic)) {
        cmd += " --no-sound";
    } else {
        if (!cfg.capture_system_audio) cmd += " --no-system-audio";
        else if (std::abs(cfg.system_volume_pct - 100.0) > 0.5) cmd += std::format(" --system-volume {:.0f}%", cfg.system_volume_pct);
        if (cfg.capture_mic) {
            if (cfg.mic_device.empty() || cfg.mic_device == "default") cmd += " --mic";
            else cmd += std::format(" --mic={}", cfg.mic_device);
            if (std::abs(cfg.mic_volume_pct - 100.0) > 0.5) cmd += std::format(" --mic-volume {:.0f}%", cfg.mic_volume_pct);
        }
        if (cfg.audio_codec == AudioCodec::Aac) cmd += " --audio-codec aac";
        else if (cfg.audio_codec == AudioCodec::Flac) cmd += " --audio-codec flac";
        if (cfg.audio_codec != AudioCodec::Flac && cfg.audio_bitrate != 192'000) {
            cmd += std::format(" --audio-bitrate {}k", cfg.audio_bitrate / 1000);
        }
    }
    if (cfg.out_width > 0 && cfg.out_height > 0) cmd += std::format(" -s {}x{}", cfg.out_width, cfg.out_height);
    if (!cfg.capture_cursor) cmd += " --no-cursor";
    if (cfg.duration_sec > 0.0) cmd += std::format(" -d {}", cfg.duration_sec);
    cmd += std::format(" -o {}", cfg.output_path);
    return cmd;
}

}  // namespace

std::expected<bool, std::string> run_interactive_setup(
    Config &cfg,
    const char *prog_name,
    std::istream &in,
    std::ostream &out,
    const std::filesystem::path &defaults_path) {
    // 0. Check if the user already has saved personal defaults
    bool had_saved_defaults = false;
    auto saved_res = load_user_defaults(defaults_path);
    if (saved_res && saved_res->has_value()) {
        had_saved_defaults = true;
        const Config &saved = **saved_res;
        out << "\n=== vkcap Saved Defaults ===\n"
            << format_config_summary(saved) << '\n';
        bool use_saved = true;
        if (!prompt_yes_no(in, out, "Use your saved default settings and start recording? [Y/n]: ", true, use_saved)) {
            return false;
        }
        if (use_saved) {
            cfg = saved;
            out << '\n';
            return true;
        }
        out << '\n';
    }

    auto disps_res = KmsCapture::enumerate_displays();
    if (!disps_res) {
        return std::unexpected(disps_res.error());
    }
    const auto &disps = *disps_res;
    if (disps.empty()) {
        return std::unexpected("No connected active DRM/KMS displays found in /dev/dri/card*");
    }

    out << "=== vkcap Interactive Setup ===\n"
        << "Press Enter to accept any [default], or type 'q' to cancel.\n\n";

    // Choose Quick Setup (premade profile) vs Advanced Setup (every single setting)
    int mode = 1;
    if (!prompt_choice<int>(in, out, "Setup mode:", {
            {"Quick setup    (choose display, premade profile & audio/session options)", 1},
            {"Advanced setup (configure every video, color, audio & encoder parameter)", 2},
        }, 0, mode)) {
        return false;
    }
    out << '\n';

    size_t disp_idx = 0;
    if (!prompt_display(in, out, disps, disp_idx, cfg.display)) return false;

    if (mode == 1) {
        // --- Quick Setup: Profile defines video/encoder parameters ---
        const auto profiles = builtin_profiles();
        out << "Select recording profile (defines FPS, codec, quality & rate control):\n";
        for (size_t i = 0; i < profiles.size(); ++i) {
            out << std::format("   [{}] {:<15} {}\n       ({})\n",
                               i + 1, profiles[i].name, profiles[i].summary, profiles[i].details);
        }
        std::string ans;
        size_t prof_idx = 0;
        while (true) {
            if (!prompt_line(in, out, "   Profile [1]: ", ans)) return false;
            if (ans.empty()) { prof_idx = 0; break; }
            size_t n = 0;
            auto [p, ec] = std::from_chars(ans.data(), ans.data() + ans.size(), n);
            if (ec == std::errc() && p == ans.data() + ans.size() && n >= 1 && n <= profiles.size()) {
                prof_idx = n - 1;
                break;
            }
            if (apply_profile(cfg, ans).has_value()) {
                prof_idx = profiles.size();
                break;
            }
            out << std::format("   Please enter 1..{} or a profile name.\n", profiles.size());
        }
        if (prof_idx < profiles.size()) {
            (void)apply_profile(cfg, profiles[prof_idx].name);
        }
        out << "\nSession options (not set by profile):\n";
        if (!prompt_audio_section(in, out, cfg, false)) return false;
        if (!prompt_yes_no(in, out, "Capture hardware mouse cursor? [Y/n]: ", true, cfg.capture_cursor)) return false;
        if (!prompt_resolution_and_output(in, out, disps[disp_idx], cfg)) return false;
    } else {
        // --- Advanced Setup: Configure every parameter ---
        cfg.profile.clear();  // Custom advanced configuration (does not touch any profile)

        out << "-- Video Codec & Format --\n";
        if (!prompt_choice<Codec>(in, out, "Video codec:", {
                {"av1  (recommended for RDNA 4 / VCN 5)", Codec::Av1},
                {"hevc (H.265)", Codec::Hevc},
                {"h264 (AVC)", Codec::H264},
            }, 0, cfg.codec)) return false;

        if (!prompt_choice<PixelFormat>(in, out, "Pixel format:", {
                {"nv12 (8-bit + static IGN dithering)", PixelFormat::Nv12},
                {"p010 (10-bit, zero banding)", PixelFormat::P010},
            }, 0, cfg.pixfmt)) return false;

        int fps_val = 60;
        if (!prompt_int(in, out, "Framerate (FPS)", 60, 1, 1000, fps_val)) return false;
        cfg.fps = static_cast<uint32_t>(fps_val);

        if (!prompt_yes_no(in, out, "Capture hardware mouse cursor? [Y/n]: ", true, cfg.capture_cursor)) return false;

        {
            std::string ans;
            while (true) {
                if (!prompt_line(in, out, "Auto-stop duration in seconds (0 = manual Ctrl+C) [0]: ", ans)) return false;
                if (ans.empty() || ans == "0") { cfg.duration_sec = 0.0; break; }
                double sec = 0.0;
                auto [p, ec] = std::from_chars(ans.data(), ans.data() + ans.size(), sec);
                if (ec == std::errc() && p == ans.data() + ans.size() && sec >= 0.0) {
                    cfg.duration_sec = sec;
                    break;
                }
                out << "   Please enter a non-negative number of seconds.\n";
            }
        }

        out << "\n-- Rate Control & Encoder Tuning --\n";
        if (!prompt_choice<RateControl>(in, out, "Rate control mode:", {
                {"cqp (constant quality — recommended)", RateControl::Cqp},
                {"vbr (variable bitrate)", RateControl::Vbr},
                {"cbr (constant bitrate)", RateControl::Cbr},
            }, 0, cfg.rc)) return false;

        if (cfg.rc == RateControl::Cqp) {
            const int def_qp = default_qp_for_codec(cfg.codec);
            const int max_qp = max_qp_for_codec(cfg.codec);
            if (!prompt_int(in, out,
                            std::format("Quantizer QP (0..{}, lower = higher quality)", max_qp),
                            def_qp, 0, max_qp, cfg.qp)) {
                return false;
            }
            cfg.bitrate = 0;
            cfg.max_bitrate = 0;
        } else {
            cfg.qp = -1;
            std::string ans;
            while (true) {
                if (!prompt_line(in, out, "Target bitrate (e.g. 20M, 35M) [25M]: ", ans)) return false;
                if (ans.empty()) { cfg.bitrate = 25'000'000; break; }
                auto r = parse_bitrate(ans);
                if (r) { cfg.bitrate = *r; break; }
                out << "   " << r.error() << '\n';
            }
            if (cfg.rc == RateControl::Vbr) {
                while (true) {
                    if (!prompt_line(in, out, "Max bitrate (Enter for 1.5x target): ", ans)) return false;
                    if (ans.empty()) { cfg.max_bitrate = 0; break; }
                    auto r = parse_bitrate(ans);
                    if (r) { cfg.max_bitrate = *r; break; }
                    out << "   " << r.error() << '\n';
                }
            } else {
                cfg.max_bitrate = 0;
            }
        }

        if (!prompt_int(in, out, "Encoder quality level (0..3, 1 = RDNA 4 sweet spot)", 1, 0, 3, cfg.quality)) return false;

        if (!prompt_choice<Tune>(in, out, "Encoder tune:", {
                {"hq       (high quality)", Tune::Hq},
                {"default", Tune::Default},
                {"ll       (low latency)", Tune::Ll},
                {"ull      (ultra-low latency)", Tune::Ull},
                {"lossless", Tune::Lossless},
            }, 0, cfg.tune)) return false;

        if (!prompt_choice<Usage>(in, out, "Encoder usage:", {
                {"record", Usage::Record},
                {"stream", Usage::Stream},
                {"transcode", Usage::Transcode},
                {"conference", Usage::Conference},
                {"default", Usage::Default},
            }, 0, cfg.usage)) return false;

        if (!prompt_choice<Content>(in, out, "Content hint:", {
                {"rendered (games / 3D)", Content::Rendered},
                {"desktop  (UI / text)", Content::Desktop},
                {"camera", Content::Camera},
                {"default", Content::Default},
            }, 0, cfg.content)) return false;

        if (!prompt_int(in, out, "Keyframe/IDR interval in frames (0 = auto 2s GOP)", 0, 0, 100000, cfg.idr_interval)) return false;
        if (!prompt_int(in, out, "GPU async encode depth (1..64, RDNA 4 max safe = 4)", 4, 1, 64, cfg.async_depth)) return false;
        if (!prompt_int(in, out, "Max B-frame depth (1..8, RDNA 4 AV1 = 1)", 1, 1, 8, cfg.b_depth)) return false;
        int dev_idx = 0;
        if (!prompt_int(in, out, "Vulkan GPU device index", 0, 0, 16, dev_idx)) return false;
        cfg.device_index = static_cast<uint32_t>(dev_idx);
        int qsz = 4;
        if (!prompt_int(in, out, "GPU-to-encoder queue depth (1..64)", 4, 1, 64, qsz)) return false;
        cfg.queue_size = static_cast<size_t>(qsz);

        out << "\n-- Color Management --\n";
        if (!prompt_choice<InputGamut>(in, out, "Input framebuffer gamut:", {
                {"auto   (EDID wide-gamut on Wayland -> target)", InputGamut::Auto},
                {"edid   (always convert from display EDID)", InputGamut::Edid},
                {"srgb   (passthrough / monitor in sRGB clamp)", InputGamut::Srgb},
                {"p3     (Display P3)", InputGamut::DisplayP3},
                {"bt2020 (BT.2020)", InputGamut::Bt2020},
            }, 0, cfg.input_gamut)) return false;

        if (!prompt_choice<ColorSpace>(in, out, "Output color space:", {
                {"bt709  (SDR standard)", ColorSpace::Bt709},
                {"bt2020 (wide gamut / HDR container)", ColorSpace::Bt2020},
            }, 0, cfg.color_space)) return false;

        if (!prompt_choice<ColorRange>(in, out, "Color range:", {
                {"limited (TV / 16-235 — standard video)", ColorRange::Limited},
                {"full    (PC / 0-255)", ColorRange::Full},
            }, 0, cfg.color_range)) return false;

        if (cfg.pixfmt == PixelFormat::Nv12) {
            if (!prompt_yes_no(in, out, "Enable static spatial IGN dithering for 8-bit NV12? [Y/n]: ", true, cfg.dither)) {
                return false;
            }
        }

        out << "\n-- Audio & Volume --\n";
        if (!prompt_audio_section(in, out, cfg, true)) return false;

        out << "\n-- Output --\n";
        if (!prompt_resolution_and_output(in, out, disps[disp_idx], cfg)) return false;
    }

    out << "\nSelected configuration:\n"
        << format_config_summary(cfg)
        << "\nEquivalent CLI:\n  " << format_equiv_cli(cfg, prog_name) << "\n\n";

    // Optional: save as personal defaults (never touches built-in profiles)
    bool save_def = false;
    const std::string save_prompt = had_saved_defaults
        ? "Overwrite your saved defaults with these settings? (choosing 'n' keeps previous defaults) [y/N]: "
        : "Save these settings as your personal defaults for future runs? [y/N]: ";
    if (!prompt_yes_no(in, out, save_prompt, false, save_def)) return false;
    if (save_def) {
        auto sres = save_user_defaults(cfg, defaults_path);
        if (!sres) {
            out << "Warning: could not save defaults: " << sres.error() << '\n';
        } else {
            out << "Saved personal defaults to " << defaults_path.string() << '\n';
        }
    }
    out << '\n';
    return true;
}

}  // namespace vkcap
