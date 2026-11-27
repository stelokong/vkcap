#include "audio_capture.hpp"
#include "cli.hpp"
#include "color_math.hpp"
#include "interactive.hpp"
#include "kms_capture.hpp"
#include "privileges.hpp"
#include "types.hpp"
#include "user_config.hpp"
#include "video_encoder.hpp"
#include "vulkan_compute.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <libdrm/drm_fourcc.h>
#include <sstream>
#include <unistd.h>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

using namespace vkcap;

namespace {

void check(bool cond, const char *msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

void test_cli_and_privileges() {
    (void)init_least_privilege();

    const char *argv[] = {
        "vkcap", "-D", "DP-1", "-o", "out.mp4", "-f", "120", "-s", "2560x1440",
        "--pixfmt", "p010", "--rc", "vbr", "-b", "35M", "--no-cursor"
    };
    auto cfg = parse_args(static_cast<int>(std::size(argv)), const_cast<char **>(argv));
    check(cfg.has_value(), "parse_args valid flags");
    check(cfg->display == "DP-1", "display DP-1");
    check(cfg->output_path == "out.mp4", "output_path");
    check(cfg->fps == 120, "fps 120");
    check(cfg->out_width == 2560 && cfg->out_height == 1440, "scale");
    check(cfg->pixfmt == PixelFormat::P010, "pixfmt p010");
    check(!cfg->capture_cursor, "no-cursor");

    // Verify resolution shorthand presets and tilde path expansion
    {
        auto r1 = parse_resolution("1080p");
        check(r1.has_value() && r1->first == 1920 && r1->second == 1080, "parse_resolution 1080p");
        auto r2 = parse_resolution("4K");
        check(r2.has_value() && r2->first == 3840 && r2->second == 2160, "parse_resolution 4K");
        if (const char *home = std::getenv("HOME"); home && home[0] != '\0') {
            check(expand_user_path("~/Videos/test.mkv") == std::string(home) + "/Videos/test.mkv",
                  "expand_user_path ~/");
        }
        check(expand_user_path("relative.mkv") == "relative.mkv", "expand_user_path relative unchanged");

        auto cr = parse_crop("1920x1080+320+180");
        check(cr.has_value() && cr->width == 1920 && cr->height == 1080 && cr->x == 320 && cr->y == 180,
              "parse_crop valid");
        check(!parse_crop("1920x1080").has_value(), "reject crop without +X+Y");
        check(!parse_crop("0x1080+0+0").has_value(), "reject zero-width crop");

        const char *argv_ir[] = {"vkcap", "--instant-replay", "--crop", "1280x720+100+50", "-o", "rep.mkv"};
        auto cir = parse_args(static_cast<int>(std::size(argv_ir)), const_cast<char **>(argv_ir));
        check(cir.has_value() && cir->replay_seconds == 30, "--instant-replay defaults to 30s");
        check(cir->crop.valid() && cir->crop.width == 1280 && cir->crop.x == 100, "--crop parsed");

        const char *argv_ir2[] = {"vkcap", "--instant-replay", "60", "-o", "rep.mkv"};
        auto cir2 = parse_args(static_cast<int>(std::size(argv_ir2)), const_cast<char **>(argv_ir2));
        check(cir2.has_value() && cir2->replay_seconds == 60, "--instant-replay 60");

        // Verify base_defaults seeding + explicit CLI override
        Config base_def;
        base_def.fps = 120;
        base_def.display = "DP-2";
        base_def.capture_mic = true;
        base_def.mic_volume_pct = 180.0;
        base_def.duration_sec = 45.0;  // Should be cleared when seeding replay defaults
        const char *argv_ir3[] = {"vkcap", "--instant-replay", "--fps", "90"};
        auto cir3 = parse_args(static_cast<int>(std::size(argv_ir3)), const_cast<char **>(argv_ir3), &base_def);
        check(cir3.has_value() && cir3->replay_seconds == 30, "base_defaults + --instant-replay");
        check(cir3->display == "DP-2" && cir3->capture_mic && std::abs(cir3->mic_volume_pct - 180.0) < 1e-6,
              "base_defaults preserved");
        check(cir3->fps == 90 && cir3->duration_sec == 0.0, "explicit --fps overrides base_defaults and duration cleared");

        // Verify hardware- and display-aware adaptive profile scaling
        {
            // Fractional & low refresh-rate handling (59.998Hz -> 60, 29.97Hz -> 30, 74.97Hz -> 75, 119.88Hz -> 120)
            check(recommended_standard_fps(59.998) == 60, "59.998 Hz -> 60 fps");
            check(recommended_standard_fps(59.94) == 60, "59.94 Hz -> 60 fps");
            check(recommended_standard_fps(29.97) == 30, "29.97 Hz -> 30 fps");
            check(recommended_standard_fps(30.0) == 30, "30 Hz -> 30 fps");
            check(recommended_standard_fps(50.0) == 50, "50 Hz -> 50 fps");
            check(recommended_gameplay_fps(59.998) == 60, "gameplay 59.998 Hz -> 60 fps");
            check(recommended_gameplay_fps(74.97) == 75, "gameplay 74.97 Hz -> 75 fps");
            check(recommended_gameplay_fps(89.9) == 90, "gameplay 89.9 Hz -> 90 fps");
            check(recommended_gameplay_fps(119.88) == 120, "gameplay 119.88 Hz -> 120 fps");

            // 720p @ 59.998Hz on a GPU without AV1 (HEVC + 10-bit supported)
            ProfileTarget hevc_720p{};
            hevc_720p.width = 1280; hevc_720p.height = 720; hevc_720p.refresh_hz = 59.998;
            hevc_720p.has_av1 = false; hevc_720p.has_hevc = true; hevc_720p.has_h264 = true;
            hevc_720p.supports_10bit = true;
            Config c_def{}, c_gp{}, c_m{}, c_d{};
            check(apply_profile(c_def, "default", hevc_720p).has_value(), "apply default hevc 720p");
            check(c_def.codec == Codec::Hevc && c_def.fps == 60 &&
                  c_def.bitrate <= 10'000'000 && c_def.bitrate >= 6'000'000,
                  "720p non-AV1 selects HEVC @ 60fps with ~8M VBR");
            check(apply_profile(c_gp, "gameplay", hevc_720p).has_value(), "apply gameplay hevc 720p");
            check(c_gp.fps == 60 && c_gp.codec == Codec::Hevc, "59.998Hz display caps gameplay profile to 60 fps");
            check(apply_profile(c_m, "master", hevc_720p).has_value(), "apply master hevc 720p");
            check(c_m.codec == Codec::Hevc && c_m.pixfmt == PixelFormat::P010 && c_m.qp == 18,
                  "HEVC master uses valid HEVC QP 18 + P010");
            check(apply_profile(c_d, "desktop", hevc_720p).has_value(), "apply desktop hevc 720p");
            check(c_d.codec == Codec::Hevc && c_d.qp == 26 && c_d.idr_interval == 240,
                  "HEVC desktop uses valid HEVC QP 26 + 4s GOP");

            // 1080p @ 30Hz TV panel: standard & gameplay both step down to 30 fps, 4s GOP = 120
            ProfileTarget tv_30hz{};
            tv_30hz.width = 1920; tv_30hz.height = 1080; tv_30hz.refresh_hz = 29.97;
            Config c30_def{}, c30_comp{};
            check(apply_profile(c30_def, "default", tv_30hz).has_value(), "apply default 30Hz");
            check(c30_def.fps == 30 && c30_def.bitrate < 10'000'000, "30Hz monitor steps down to 30 fps & lower bitrate");
            check(apply_profile(c30_comp, "compact", tv_30hz).has_value(), "apply compact 30Hz");
            check(c30_comp.fps == 30 && c30_comp.idr_interval == 120, "30Hz 4s GOP scales to 120 frames");

            // 4K @ 144Hz on AV1 GPU with max_hw_bitrate cap
            ProfileTarget av1_4k{};
            av1_4k.width = 3840; av1_4k.height = 2160; av1_4k.refresh_hz = 144.0;
            av1_4k.max_hw_bitrate = 80'000'000;
            Config c4k_def{}, c4k_gp{};
            check(apply_profile(c4k_def, "default", av1_4k).has_value(), "apply default 4k");
            check(c4k_def.codec == Codec::Av1 && c4k_def.bitrate >= 32'000'000 && c4k_def.bitrate <= 42'000'000,
                  "4K AV1 default scales bitrate up (~37M)");
            check(apply_profile(c4k_gp, "gameplay", av1_4k).has_value(), "apply gameplay 4k");
            check(c4k_gp.fps == 120 && c4k_gp.max_bitrate == 80'000'000, "4K120 ceiling clamped to max_hw_bitrate");

            // Driver that lacks VBR (CQP-only) and lacks 10-bit P010
            ProfileTarget cqp_only{};
            cqp_only.width = 1920; cqp_only.height = 1080; cqp_only.refresh_hz = 60.0;
            cqp_only.has_av1 = false; cqp_only.has_hevc = true; cqp_only.has_h264 = true;
            cqp_only.supports_10bit = false; cqp_only.supports_vbr = false; cqp_only.supports_cqp = true;
            cqp_only.max_quality_levels = 1;
            Config cc_def{}, cc_10{};
            check(apply_profile(cc_def, "default", cqp_only).has_value(), "apply default cqp-only");
            check(cc_def.rc == RateControl::Cqp && cc_def.qp == 22, "VBR-unsupported GPU falls back to CQP");
            check(apply_profile(cc_10, "gameplay-10bit", cqp_only).has_value(), "apply 10bit on 8bit-only HEVC");
            check(cc_10.pixfmt == PixelFormat::Nv12 && cc_10.dither && cc_10.quality == 0,
                  "8-bit-only HEVC falls back to NV12+dither and clamps quality to 0");
        }
    }

    auto disps = KmsCapture::enumerate_displays();
    check(disps.has_value(), "enumerate_displays succeeds without CAP_SYS_ADMIN");

    // Verify all builtin profiles apply cleanly
    for (const auto &p : builtin_profiles()) {
        Config pc;
        check(apply_profile(pc, p.name).has_value(), "apply builtin profile");
        check(pc.profile == p.name, "profile name recorded");
    }
    {
        Config bad;
        check(!apply_profile(bad, "nonexistent-profile").has_value(), "reject unknown profile");
    }

    // Verify explicit flags override profile settings regardless of flag order
    {
        const char *argv_p[] = {"vkcap", "-f", "240", "-p", "gameplay-10bit", "-o", "over.mkv"};
        auto c = parse_args(static_cast<int>(std::size(argv_p)), const_cast<char **>(argv_p));
        check(c.has_value(), "profile + override parses");
        check(c->profile == "gameplay-10bit", "profile gameplay-10bit");
        check(c->pixfmt == PixelFormat::P010, "profile sets P010");
        check(c->rc == RateControl::Vbr && c->bitrate == 40'000'000 && c->max_bitrate == 60'000'000, "profile sets 40M/60M VBR");
        check(c->fps == 240, "explicit -f 240 overrides profile 120 fps");
        check(c->output_path == "over.mkv", "explicit -o overrides default");
    }

    // Verify audio CLI defaults and overrides
    {
        const char *argv_def[] = {"vkcap", "-o", "def.mkv"};
        auto c = parse_args(static_cast<int>(std::size(argv_def)), const_cast<char **>(argv_def));
        check(c.has_value(), "default audio flags");
        check(c->audio && c->capture_system_audio && !c->capture_mic, "system audio ON, mic OFF by default");

        const char *argv_ns[] = {"vkcap", "--no-sound", "-o", "ns.mkv"};
        auto c2 = parse_args(static_cast<int>(std::size(argv_ns)), const_cast<char **>(argv_ns));
        check(c2.has_value() && !c2->audio && !c2->capture_system_audio && !c2->capture_mic, "--no-sound disables audio");

        const char *argv_mic[] = {
            "vkcap", "--mic", "alsa_input.usb-test",
            "--system-volume", "85%", "--mic-volume", "200",
            "--audio-codec", "flac", "--audio-bitrate", "256k", "-o", "m.mkv"
        };
        auto c3 = parse_args(static_cast<int>(std::size(argv_mic)), const_cast<char **>(argv_mic));
        check(c3.has_value() && c3->audio && c3->capture_system_audio && c3->capture_mic, "--mic enables mic + system");
        check(c3->mic_device == "alsa_input.usb-test", "--mic target captured");
        check(std::abs(c3->system_volume_pct - 85.0) < 1e-6, "--system-volume 85%");
        check(std::abs(c3->mic_volume_pct - 200.0) < 1e-6, "--mic-volume 200%");
        check(c3->audio_codec == AudioCodec::Flac, "--audio-codec flac");
        check(c3->audio_bitrate == 256'000, "--audio-bitrate 256k");
    }

    // Verify overhauled interactive wizard: Quick Setup + Save Defaults -> Instant Saved Start -> Decline -> Advanced Setup (don't overwrite)
    if (!disps->empty()) {
        const std::filesystem::path tmp_conf = "/tmp/vkcap_test_defaults.conf";
        std::filesystem::remove(tmp_conf);

        // 1) Quick setup with custom volumes (80% sys, 200% mic) + save as personal defaults
        {
            auto mics_res = AudioCapture::enumerate_microphones();
            const bool has_mics = mics_res && !mics_res->empty();
            Config ic;
            // mode=1(quick), disp=1, profile=2(gameplay 120fps), sys_audio=y, sys_vol=80%,
            // mic=y, [mic_choice=1 if mics listed], mic_vol=200%, cursor=n, res=1920x1080, out=wiz_out.mp4, save=y
            std::string script = std::string("1\n1\n2\ny\n80%\ny\n") +
                                 (has_mics ? "1\n" : "") +
                                 "200%\nn\n1920x1080\nwiz_out.mp4\ny\n";
            std::istringstream in(script);
            std::ostringstream out;
            auto w = run_interactive_setup(ic, "vkcap", in, out, tmp_conf);
            check(w.has_value() && *w, "quick wizard confirmed");
            check(ic.display == (*disps)[0].connector_name, "quick display 1");
            check(ic.profile == "gameplay" && ic.fps == 120, "quick profile gameplay 120fps");
            check(ic.audio && ic.capture_system_audio && ic.capture_mic, "quick sys + mic");
            check(std::abs(ic.system_volume_pct - 80.0) < 1e-6, "quick sys volume 80%");
            check(std::abs(ic.mic_volume_pct - 200.0) < 1e-6, "quick mic volume 200%");
            check(!ic.capture_cursor, "quick no cursor");
            check(ic.out_width == 1920 && ic.out_height == 1080, "quick scale");
            check(ic.output_path == "wiz_out.mp4", "quick output");
            check(std::filesystem::exists(tmp_conf), "defaults file saved");
        }

        // 2) Next run with saved defaults: single 'y' starts immediately without asking basic/advanced
        {
            Config ic2;
            std::istringstream in("y\n");
            std::ostringstream out;
            auto w2 = run_interactive_setup(ic2, "vkcap", in, out, tmp_conf);
            check(w2.has_value() && *w2, "saved defaults accepted immediately");
            check(ic2.fps == 120 && std::abs(ic2.mic_volume_pct - 200.0) < 1e-6, "saved defaults loaded accurately");
            check(ic2.output_path.ends_with(".mp4") && ic2.output_path != "wiz_out.mp4", "fresh timestamped .mp4 filename");
            check(out.str().find("Setup mode:") == std::string::npos, "does not ask basic/advanced when using saved defaults");
        }

        // 3) Decline saved defaults -> Advanced setup -> decline saving (must preserve previous saved defaults!)
        {
            Config ic3;
            // use_saved=n, mode=2(advanced), disp=1, codec=2(hevc), pixfmt=2(p010), fps=90, cursor=y, dur=10,
            // rc=1(vbr), bitrate=30M, max_br=45M, quality=2, tune=1(hq), usage=1(record), content=2(desktop),
            // idr=180, async=4, b_depth=1, dev=0, qsz=6, gamut=3(srgb), cs=1(bt709), range=2(full),
            // sys_audio=y, sys_vol=125, mic=n, acodec=3(flac), res=native, out=adv.mkv, save=n
            std::istringstream in(
                "n\n2\n1\n2\n2\n90\ny\n10\n"
                "1\n30M\n45M\n2\n1\n1\n2\n"
                "180\n4\n1\n0\n6\n3\n1\n2\n"
                "y\n125\nn\n3\nnative\nadv.mkv\nn\n");
            std::ostringstream out;
            auto w3 = run_interactive_setup(ic3, "vkcap", in, out, tmp_conf);
            check(w3.has_value() && *w3, "advanced wizard confirmed");
            check(ic3.codec == Codec::Hevc && ic3.pixfmt == PixelFormat::P010, "advanced codec/pixfmt");
            check(ic3.fps == 90 && ic3.duration_sec == 10.0, "advanced fps/duration");
            check(ic3.rc == RateControl::Vbr && ic3.bitrate == 30'000'000 && ic3.max_bitrate == 45'000'000, "advanced vbr");
            check(ic3.quality == 2 && ic3.content == Content::Desktop && ic3.idr_interval == 180, "advanced tuning");
            check(ic3.input_gamut == InputGamut::Srgb && ic3.color_range == ColorRange::Full, "advanced color");
            check(std::abs(ic3.system_volume_pct - 125.0) < 1e-6 && !ic3.capture_mic, "advanced audio vol");
            check(ic3.audio_codec == AudioCodec::Flac, "advanced audio codec flac");
            check(ic3.output_path == "adv.mkv", "advanced output");

            // Previous saved defaults must still be intact (fps=120, mic_vol=200%)
            auto reloaded = load_user_defaults(tmp_conf);
            check(reloaded.has_value() && reloaded->has_value(), "previous defaults preserved");
            check((*reloaded)->fps == 120 && (*reloaded)->codec == Codec::Av1, "previous defaults not overwritten");
        }

        // 4) Cancel via 'q'
        {
            Config ic_cancel;
            std::istringstream in_q("q\n");
            std::ostringstream out_q;
            auto wq = run_interactive_setup(ic_cancel, "vkcap", in_q, out_q, tmp_conf);
            check(wq.has_value() && !*wq, "interactive wizard cancel via 'q'");
        }

        std::filesystem::remove(tmp_conf);
    }

    // Verify codec-specific QP range validation (0..255 for AV1, 0..51 for HEVC/H264)
    {
        const char *bad_hevc[] = {"vkcap", "--codec", "hevc", "--qp", "80", "-o", "x.mkv"};
        check(!parse_args(static_cast<int>(std::size(bad_hevc)), const_cast<char **>(bad_hevc)).has_value(),
              "reject --qp 80 for hevc");
        const char *bad_h264[] = {"vkcap", "--qp", "52", "--codec", "h264", "-o", "x.mkv"};
        check(!parse_args(static_cast<int>(std::size(bad_h264)), const_cast<char **>(bad_h264)).has_value(),
              "reject --qp 52 for h264 regardless of arg order");
        const char *ok_hevc[] = {"vkcap", "--codec", "hevc", "--qp", "51", "-o", "x.mkv"};
        check(parse_args(static_cast<int>(std::size(ok_hevc)), const_cast<char **>(ok_hevc)).has_value(),
              "accept --qp 51 for hevc");
        const char *ok_av1[] = {"vkcap", "--codec", "av1", "--qp", "120", "-o", "x.mkv"};
        check(parse_args(static_cast<int>(std::size(ok_av1)), const_cast<char **>(ok_av1)).has_value(),
              "accept --qp 120 for av1");
    }

    // Verify color gamut matrix math (identity, round-trip inverse, white preservation)
    {
        auto id = compute_gamut_matrix(kPrimariesBt709, kPrimariesBt709);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float expected = (r == c) ? 1.0f : 0.0f;
                check(std::abs(id[static_cast<size_t>(r * 3 + c)] - expected) < 1e-4f, "BT.709->BT.709 identity");
            }
        }
        // Test wide-gamut QD-OLED primaries (like DP-1) -> BT.709 and inverse round-trip
        ColorPrimaries qdoled{0.6973, 0.2969, 0.2363, 0.7197, 0.1436, 0.0557, 0.3125, 0.3291, true};
        check(is_wide_gamut(qdoled), "QD-OLED detected as wide gamut");
        check(!is_wide_gamut(kPrimariesBt709), "BT.709 not wide gamut");
        auto fwd = compute_gamut_matrix(kPrimariesBt709, qdoled);
        auto inv = compute_gamut_matrix(qdoled, kPrimariesBt709);
        // White point (1,1,1) must map to (1,1,1)
        for (int r = 0; r < 3; ++r) {
            float row_sum = inv[static_cast<size_t>(r * 3)] + inv[static_cast<size_t>(r * 3 + 1)] + inv[static_cast<size_t>(r * 3 + 2)];
            check(std::abs(row_sum - 1.0f) < 1e-3f, "gamut matrix preserves white point");
        }
        // inv * fwd == Identity
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                float s = 0.0f;
                for (int k = 0; k < 3; ++k) s += inv[static_cast<size_t>(r * 3 + k)] * fwd[static_cast<size_t>(k * 3 + c)];
                float expected = (r == c) ? 1.0f : 0.0f;
                check(std::abs(s - expected) < 1e-4f, "gamut matrix round-trip inverse");
            }
        }
    }

    std::cout << std::format("[PASS] test_cli_and_privileges (found {} KMS displays, {} profiles)\n",
                             disps->size(), builtin_profiles().size());
}

void verify_av1_container(const std::string &path, int expected_w, int expected_h, int min_frames, bool expect_audio = false) {
    AVFormatContext *fmt = nullptr;
    check(avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) == 0, "avformat_open_input");
    check(avformat_find_stream_info(fmt, nullptr) >= 0, "avformat_find_stream_info");

    int vidx = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    check(vidx >= 0, "find video stream");
    if (expect_audio) {
        int aidx = av_find_best_stream(fmt, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
        check(aidx >= 0, "find audio stream");
        check(fmt->streams[aidx]->codecpar->codec_id == AV_CODEC_ID_OPUS ||
              fmt->streams[aidx]->codecpar->codec_id == AV_CODEC_ID_AAC, "audio codec is Opus/AAC");
    }
    AVStream *st = fmt->streams[vidx];
    check(st->codecpar->codec_id == AV_CODEC_ID_AV1, "codec is AV1");
    check(st->codecpar->width == expected_w && st->codecpar->height == expected_h, "stream dims");

    const AVCodec *dec = avcodec_find_decoder(AV_CODEC_ID_AV1);
    check(dec != nullptr, "AV1 decoder available");
    AVCodecContext *dctx = avcodec_alloc_context3(dec);
    check(avcodec_parameters_to_context(dctx, st->codecpar) >= 0, "params to decoder");
    check(avcodec_open2(dctx, dec, nullptr) == 0, "open AV1 decoder");

    AVPacket *pkt = av_packet_alloc();
    AVFrame *frm = av_frame_alloc();
    int decoded = 0;

    while (av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index == vidx) {
            check(avcodec_send_packet(dctx, pkt) == 0, "send packet");
            while (avcodec_receive_frame(dctx, frm) == 0) {
                check(frm->width == expected_w && frm->height == expected_h, "decoded dims");
                decoded++;
            }
        }
        av_packet_unref(pkt);
    }
    avcodec_send_packet(dctx, nullptr);
    while (avcodec_receive_frame(dctx, frm) == 0) decoded++;

    av_frame_free(&frm);
    av_packet_free(&pkt);
    avcodec_free_context(&dctx);
    avformat_close_input(&fmt);
    check(decoded >= min_frames, "decoded expected AV1 frame count");
}

void test_kms_gpu_pipeline(PixelFormat pixfmt,
                           uint32_t src_w, uint32_t src_h,
                           uint32_t dst_w, uint32_t dst_h,
                           uint32_t num_frames,
                           const std::string &ext,
                           bool with_cursor,
                           bool with_wide_gamut = false,
                           bool with_audio = false) {
    const std::string out_path = std::format(
        "/tmp/vkcap_test_{}_{}x{}_{}{}{}.{}",
        pixfmt == PixelFormat::P010 ? "p010" : "nv12",
        dst_w, dst_h, with_cursor ? "cur" : "nocur",
        with_wide_gamut ? "_wg" : "", with_audio ? "_av" : "", ext);
    std::filesystem::remove(out_path);

    Config cfg;
    cfg.output_path = out_path;
    cfg.fps = 60;
    cfg.pixfmt = pixfmt;
    cfg.capture_cursor = with_cursor;

    const ColorPrimaries qdoled{0.6973, 0.2969, 0.2363, 0.7197, 0.1436, 0.0557, 0.3125, 0.3291, true};
    auto gm = compute_gamut_matrix(qdoled, kPrimariesBt709);
    auto vk = *VulkanCompute::create(cfg, with_wide_gamut, gm);
    check(vk->configure_output(dst_w, dst_h).has_value(), "configure_output");
    auto enc = *VideoEncoder::open(cfg, dst_w, dst_h, vk->hw_device_ref(), vk->hw_frames_ref(), with_audio);
    if (with_audio) {
        check(enc->audio_enabled(), "audio stream enabled in encoder");
    }

    constexpr uint32_t kPool = 4;
    std::vector<VulkanCompute::TestDmaBuf> bufs;
    for (uint32_t i = 0; i < kPool; ++i) {
        bufs.push_back(*vk->create_test_dmabuf(src_w, src_h, i, false));
    }
    VulkanCompute::TestDmaBuf cur_buf{};
    if (with_cursor) {
        cur_buf = *vk->create_test_dmabuf(64, 64, 0, true);
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < num_frames; ++i) {
        KmsFrame kf{};
        auto &db = bufs[i % kPool];
        kf.primary.inode_key = db.inode_key;
        kf.primary.drm_fourcc = DRM_FORMAT_ARGB8888;
        kf.primary.width = src_w;
        kf.primary.height = src_h;
        kf.primary.drm_modifier = db.modifier;
        kf.primary.has_modifier = true;
        kf.primary.num_planes = 1;
        kf.primary.planes[0].fd = ::dup(db.fd);
        kf.primary.planes[0].stride = db.stride;
        kf.primary.planes[0].size = db.stride * src_h;

        if (with_cursor) {
            kf.cursor.visible = true;
            kf.cursor.x = static_cast<int32_t>(100 + i * 4);
            kf.cursor.y = static_cast<int32_t>(100 + i * 2);
            kf.cursor.width = 64;
            kf.cursor.height = 64;
            kf.cursor.surface.inode_key = cur_buf.inode_key;
            kf.cursor.surface.drm_fourcc = DRM_FORMAT_ARGB8888;
            kf.cursor.surface.width = 64;
            kf.cursor.surface.height = 64;
            kf.cursor.surface.drm_modifier = cur_buf.modifier;
            kf.cursor.surface.has_modifier = true;
            kf.cursor.surface.num_planes = 1;
            kf.cursor.surface.planes[0].fd = ::dup(cur_buf.fd);
            kf.cursor.surface.planes[0].stride = cur_buf.stride;
            kf.cursor.surface.planes[0].size = cur_buf.stride * 64;
        }

        auto fres = vk->process_frame(kf);
        check(fres.has_value(), "VulkanCompute::process_frame");
        AVFrame *avf = *fres;
        check(enc->encode_frame(avf, static_cast<int64_t>(i)).has_value(), "VideoEncoder::encode_frame");
        av_frame_free(&avf);
    }
    if (with_audio) {
        const int afsz = enc->audio_frame_size();
        std::vector<float> tone(static_cast<size_t>(afsz) * 2, 0.0f);
        for (int f = 0; f < 50; ++f) {
            for (int s = 0; s < afsz; ++s) {
                float v = 0.2f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(f * afsz + s) / 48000.0f);
                tone[static_cast<size_t>(s) * 2] = v;
                tone[static_cast<size_t>(s) * 2 + 1] = v;
            }
            check(enc->encode_audio_samples(tone.data(), afsz, static_cast<int64_t>(f) * afsz).has_value(),
                  "VideoEncoder::encode_audio_samples");
        }
    }
    check(enc->finish().has_value(), "VideoEncoder::finish");
    const auto t1 = std::chrono::steady_clock::now();

    for (auto &db : bufs) vk->destroy_test_dmabuf(db);
    if (with_cursor) vk->destroy_test_dmabuf(cur_buf);

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double fps = static_cast<double>(num_frames) / sec;
    auto cst = vk->stats();
    double comp_ms = (static_cast<double>(cst.total_compute_ns) / static_cast<double>(cst.frames_processed)) / 1e6;

    verify_av1_container(out_path, static_cast<int>(dst_w), static_cast<int>(dst_h), static_cast<int>(num_frames), with_audio);
    std::filesystem::remove(out_path);

    std::cout << std::format(
        "[PASS] kms_{}_{}x{}->{}x{}_{} ({}): {} frames in {:.3f}s ({:.1f} fps, compute {:.3f} ms/f)\n",
        pixfmt == PixelFormat::P010 ? "p010" : "nv12",
        src_w, src_h, dst_w, dst_h, with_cursor ? "hwcur" : "nocur", ext,
        num_frames, sec, fps, comp_ms);
}

void test_instant_replay_and_crop() {
    const std::string out_path = "/tmp/vkcap_test_replay_crop.mkv";
    std::filesystem::remove(out_path);

    Config cfg;
    cfg.output_path = out_path;
    cfg.fps = 60;
    cfg.idr_interval = 30;  // 0.5s GOP for crisp 2s replay test
    cfg.replay_seconds = 2;
    cfg.pixfmt = PixelFormat::Nv12;
    cfg.capture_cursor = false;

    auto vk = *VulkanCompute::create(cfg);
    check(vk->configure_output(1280, 720).has_value(), "replay configure_output 1280x720");
    auto enc = *VideoEncoder::open(cfg, 1280, 720, vk->hw_device_ref(), vk->hw_frames_ref(), true);
    check(enc->replay_mode(), "encoder in replay mode");
    check(!std::filesystem::exists(out_path), "replay mode creates no file before save trigger");

    constexpr uint32_t kPool = 3;
    std::vector<VulkanCompute::TestDmaBuf> bufs;
    for (uint32_t i = 0; i < kPool; ++i) {
        bufs.push_back(*vk->create_test_dmabuf(1920, 1080, i, false));
    }

    // Feed 210 frames (3.5s) + matching Opus audio into a 2s rolling buffer
    const int afsz = enc->audio_frame_size();
    std::vector<float> tone(static_cast<size_t>(afsz) * 2, 0.05f);
    int a_idx = 0;
    for (uint32_t i = 0; i < 210; ++i) {
        KmsFrame kf{};
        auto &db = bufs[i % kPool];
        kf.primary.inode_key = db.inode_key;
        kf.primary.drm_fourcc = DRM_FORMAT_ARGB8888;
        kf.primary.width = 1920;
        kf.primary.height = 1080;
        kf.primary.drm_modifier = db.modifier;
        kf.primary.has_modifier = true;
        kf.primary.num_planes = 1;
        kf.primary.planes[0].fd = ::dup(db.fd);
        kf.primary.planes[0].stride = db.stride;
        kf.primary.planes[0].size = db.stride * 1080;
        kf.crop = {320, 180, 1280, 720};

        auto fres = vk->process_frame(kf);
        check(fres.has_value(), "replay crop process_frame");
        AVFrame *avf = *fres;
        check(enc->encode_frame(avf, static_cast<int64_t>(i)).has_value(), "replay encode_frame");
        av_frame_free(&avf);

        // Emit ~50 audio frames/sec (every 1.2 video frames)
        if ((i % 6) != 5) {
            check(enc->encode_audio_samples(tone.data(), afsz, static_cast<int64_t>(a_idx) * afsz).has_value(),
                  "replay encode_audio_samples");
            ++a_idx;
        }
    }

    auto rstats = enc->replay_stats();
    check(rstats.buffered_bytes > 0 && rstats.buffered_sec >= 1.8, "replay buffer populated & bounded");

    const auto snap_t0 = std::chrono::steady_clock::now();
    auto saved = enc->save_replay_async(out_path);
    const double snap_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - snap_t0).count();
    check(saved.has_value() && *saved == out_path, "save_replay_async queued");

    check(enc->finish().has_value(), "replay encoder finish (waits for background save)");
    for (auto &db : bufs) vk->destroy_test_dmabuf(db);

    check(std::filesystem::exists(out_path), "replay file written to disk");
    verify_av1_container(out_path, 1280, 720, 115, true);
    std::filesystem::remove(out_path);

    std::cout << std::format(
        "[PASS] instant_replay_crop (1920x1080 -> crop 1280x720, 2s ring): snapshot {:.3f} ms, buffered {:.1f}s ({:.2f} MiB)\n",
        snap_ms, rstats.buffered_sec, static_cast<double>(rstats.buffered_bytes) / (1024.0 * 1024.0));
}

}  // namespace

int main() {
    test_cli_and_privileges();
    test_kms_gpu_pipeline(PixelFormat::Nv12, 1920, 1080, 1920, 1080, 60, "mp4", true, true, true);
    test_kms_gpu_pipeline(PixelFormat::P010, 1920, 1080, 1920, 1080, 60, "mkv", true, false, true);
    test_kms_gpu_pipeline(PixelFormat::Nv12, 3840, 2160, 2560, 1440, 60, "webm", true);
    test_kms_gpu_pipeline(PixelFormat::Nv12, 3840, 2160, 3840, 2160, 180, "mkv", false);
    test_instant_replay_and_crop();
    std::cout << "All KMS unit and GPU integration tests passed.\n";
    return 0;
}
