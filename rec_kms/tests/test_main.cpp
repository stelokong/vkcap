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
        "vkcap-kms", "-D", "DP-1", "-o", "out.mp4", "-f", "120", "-s", "2560x1440",
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
        const char *argv_p[] = {"vkcap-kms", "-f", "240", "-p", "gameplay-10bit", "-o", "over.mkv"};
        auto c = parse_args(static_cast<int>(std::size(argv_p)), const_cast<char **>(argv_p));
        check(c.has_value(), "profile + override parses");
        check(c->profile == "gameplay-10bit", "profile gameplay-10bit");
        check(c->pixfmt == PixelFormat::P010, "profile sets P010");
        check(c->rc == RateControl::Vbr && c->bitrate == 35'000'000, "profile sets 35M VBR");
        check(c->fps == 240, "explicit -f 240 overrides profile 120 fps");
        check(c->output_path == "over.mkv", "explicit -o overrides default");
    }

    // Verify audio CLI defaults and overrides
    {
        const char *argv_def[] = {"vkcap-kms", "-o", "def.mkv"};
        auto c = parse_args(static_cast<int>(std::size(argv_def)), const_cast<char **>(argv_def));
        check(c.has_value(), "default audio flags");
        check(c->audio && c->capture_system_audio && !c->capture_mic, "system audio ON, mic OFF by default");

        const char *argv_ns[] = {"vkcap-kms", "--no-sound", "-o", "ns.mkv"};
        auto c2 = parse_args(static_cast<int>(std::size(argv_ns)), const_cast<char **>(argv_ns));
        check(c2.has_value() && !c2->audio && !c2->capture_system_audio && !c2->capture_mic, "--no-sound disables audio");

        const char *argv_mic[] = {
            "vkcap-kms", "--mic", "alsa_input.usb-test",
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
        const std::filesystem::path tmp_conf = "/tmp/vkcap_kms_test_defaults.conf";
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
            auto w = run_interactive_setup(ic, "vkcap-kms", in, out, tmp_conf);
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
            auto w2 = run_interactive_setup(ic2, "vkcap-kms", in, out, tmp_conf);
            check(w2.has_value() && *w2, "saved defaults accepted immediately");
            check(ic2.fps == 120 && std::abs(ic2.mic_volume_pct - 200.0) < 1e-6, "saved defaults loaded accurately");
            check(ic2.output_path.ends_with(".mp4") && ic2.output_path != "wiz_out.mp4", "fresh timestamped .mp4 filename");
            check(out.str().find("Setup mode:") == std::string::npos, "does not ask basic/advanced when using saved defaults");
        }

        // 3) Decline saved defaults -> Advanced setup -> decline saving (must preserve previous saved defaults!)
        {
            Config ic3;
            // use_saved=n, mode=2(advanced), disp=1, codec=2(hevc), pixfmt=2(p010), fps=90, cursor=y, dur=10,
            // rc=2(vbr), bitrate=30M, max_br=45M, quality=2, tune=1(hq), usage=1(record), content=2(desktop),
            // idr=180, async=4, b_depth=1, dev=0, qsz=6, gamut=3(srgb), cs=1(bt709), range=2(full),
            // sys_audio=y, sys_vol=125, mic=n, acodec=3(flac), res=native, out=adv.mkv, save=n
            std::istringstream in(
                "n\n2\n1\n2\n2\n90\ny\n10\n"
                "2\n30M\n45M\n2\n1\n1\n2\n"
                "180\n4\n1\n0\n6\n3\n1\n2\n"
                "y\n125\nn\n3\nnative\nadv.mkv\nn\n");
            std::ostringstream out;
            auto w3 = run_interactive_setup(ic3, "vkcap-kms", in, out, tmp_conf);
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
            auto wq = run_interactive_setup(ic_cancel, "vkcap-kms", in_q, out_q, tmp_conf);
            check(wq.has_value() && !*wq, "interactive wizard cancel via 'q'");
        }

        std::filesystem::remove(tmp_conf);
    }

    // Verify codec-specific QP range validation (0..255 for AV1, 0..51 for HEVC/H264)
    {
        const char *bad_hevc[] = {"vkcap-kms", "--codec", "hevc", "--qp", "80", "-o", "x.mkv"};
        check(!parse_args(static_cast<int>(std::size(bad_hevc)), const_cast<char **>(bad_hevc)).has_value(),
              "reject --qp 80 for hevc");
        const char *bad_h264[] = {"vkcap-kms", "--qp", "52", "--codec", "h264", "-o", "x.mkv"};
        check(!parse_args(static_cast<int>(std::size(bad_h264)), const_cast<char **>(bad_h264)).has_value(),
              "reject --qp 52 for h264 regardless of arg order");
        const char *ok_hevc[] = {"vkcap-kms", "--codec", "hevc", "--qp", "51", "-o", "x.mkv"};
        check(parse_args(static_cast<int>(std::size(ok_hevc)), const_cast<char **>(ok_hevc)).has_value(),
              "accept --qp 51 for hevc");
        const char *ok_av1[] = {"vkcap-kms", "--codec", "av1", "--qp", "120", "-o", "x.mkv"};
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
        "/tmp/vkcap_kms_test_{}_{}x{}_{}{}{}.{}",
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

}  // namespace

int main() {
    test_cli_and_privileges();
    test_kms_gpu_pipeline(PixelFormat::Nv12, 1920, 1080, 1920, 1080, 60, "mp4", true, true, true);
    test_kms_gpu_pipeline(PixelFormat::P010, 1920, 1080, 1920, 1080, 60, "mkv", true, false, true);
    test_kms_gpu_pipeline(PixelFormat::Nv12, 3840, 2160, 2560, 1440, 60, "webm", true);
    test_kms_gpu_pipeline(PixelFormat::Nv12, 3840, 2160, 3840, 2160, 180, "mkv", false);
    std::cout << "All KMS unit and GPU integration tests passed.\n";
    return 0;
}
