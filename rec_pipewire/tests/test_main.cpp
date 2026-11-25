#include "cli.hpp"
#include "pipewire_capture.hpp"
#include "types.hpp"
#include "video_encoder.hpp"
#include "vulkan_compute.hpp"

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <libdrm/drm_fourcc.h>
#include <spa/param/video/raw.h>
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

void test_cli_parsing() {
    auto br1 = parse_bitrate("20M");
    check(br1 && *br1 == 20'000'000, "parse_bitrate 20M");
    auto br2 = parse_bitrate("1500k");
    check(br2 && *br2 == 1'500'000, "parse_bitrate 1500k");
    check(!parse_bitrate("0M"), "parse_bitrate rejects 0");
    check(!parse_bitrate("abc"), "parse_bitrate rejects garbage");

    auto res1 = parse_resolution("1920x1080");
    check(res1 && res1->first == 1920 && res1->second == 1080, "parse_resolution 1920x1080");
    check(!parse_resolution("1921x1080"), "parse_resolution rejects odd width");
    check(!parse_resolution("1920"), "parse_resolution rejects missing height");

    const char *argv[] = {
        "vkcap", "-o", "out.mp4", "-f", "120", "-s", "2560x1440",
        "--pixfmt", "p010", "--rc", "vbr", "-b", "35M", "--async-depth", "6",
        "--tune", "hq", "--content", "rendered", "--dmabuf-only"
    };
    auto cfg = parse_args(static_cast<int>(std::size(argv)), const_cast<char **>(argv));
    check(cfg.has_value(), "parse_args valid flags");
    check(cfg->output_path == "out.mp4", "output_path");
    check(cfg->fps == 120, "fps");
    check(cfg->out_width == 2560 && cfg->out_height == 1440, "scale");
    check(cfg->pixfmt == PixelFormat::P010, "pixfmt p010");
    check(cfg->rc == RateControl::Vbr, "rc vbr");
    check(cfg->bitrate == 35'000'000, "bitrate 35M");
    check(cfg->async_depth == 6, "async_depth");
    check(cfg->tune == Tune::Hq, "tune hq");
    check(cfg->content == Content::Rendered, "content rendered");
    check(cfg->dmabuf_only, "dmabuf_only");

    std::cout << "[PASS] test_cli_parsing\n";
}

void test_spa_drm_mapping() {
    check(PipeWireCapture::spa_to_drm_fourcc(SPA_VIDEO_FORMAT_BGRx) == DRM_FORMAT_XRGB8888, "BGRx->XRGB8888");
    check(PipeWireCapture::spa_to_drm_fourcc(SPA_VIDEO_FORMAT_BGRA) == DRM_FORMAT_ARGB8888, "BGRA->ARGB8888");
    check(PipeWireCapture::spa_to_drm_fourcc(SPA_VIDEO_FORMAT_RGBA) == DRM_FORMAT_ABGR8888, "RGBA->ABGR8888");
    check(PipeWireCapture::spa_to_drm_fourcc(SPA_VIDEO_FORMAT_xRGB_210LE) == DRM_FORMAT_XRGB2101010, "xRGB_210LE");
    check(PipeWireCapture::spa_to_drm_fourcc(SPA_VIDEO_FORMAT_xBGR_210LE) == DRM_FORMAT_XBGR2101010, "xBGR_210LE");
    std::cout << "[PASS] test_spa_drm_mapping\n";
}

void verify_av1_container(const std::string &path, int expected_w, int expected_h, int min_frames) {
    AVFormatContext *fmt = nullptr;
    check(avformat_open_input(&fmt, path.c_str(), nullptr, nullptr) == 0, "avformat_open_input");
    check(avformat_find_stream_info(fmt, nullptr) >= 0, "avformat_find_stream_info");

    int vidx = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    check(vidx >= 0, "find video stream");
    AVStream *st = fmt->streams[vidx];
    check(st->codecpar->codec_id == AV_CODEC_ID_AV1, "codec is AV1");
    check(st->codecpar->width == expected_w, "decoded stream width matches");
    check(st->codecpar->height == expected_h, "decoded stream height matches");

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
            check(avcodec_send_packet(dctx, pkt) == 0, "send packet to AV1 decoder");
            while (avcodec_receive_frame(dctx, frm) == 0) {
                check(frm->width == expected_w && frm->height == expected_h, "decoded frame dims");
                decoded++;
            }
        }
        av_packet_unref(pkt);
    }
    avcodec_send_packet(dctx, nullptr);
    while (avcodec_receive_frame(dctx, frm) == 0) {
        decoded++;
    }

    av_frame_free(&frm);
    av_packet_free(&pkt);
    avcodec_free_context(&dctx);
    avformat_close_input(&fmt);

    check(decoded >= min_frames, "decoded expected AV1 frame count");
}

void test_gpu_dmabuf_av1_pipeline(PixelFormat pixfmt,
                                  uint32_t src_w, uint32_t src_h,
                                  uint32_t dst_w, uint32_t dst_h,
                                  uint32_t num_frames,
                                  const std::string &ext,
                                  bool use_shm = false) {
    const std::string out_path = std::format(
        "/tmp/vkcap_test_{}_{}x{}_{}.{}",
        pixfmt == PixelFormat::P010 ? "p010" : "nv12",
        dst_w, dst_h, use_shm ? "shm" : "dmabuf", ext);
    std::filesystem::remove(out_path);

    Config cfg;
    cfg.output_path = out_path;
    cfg.fps = 60;
    cfg.pixfmt = pixfmt;
    cfg.rc = RateControl::Cqp;
    cfg.qp = 24;
    cfg.async_depth = 4;
    cfg.idr_interval = 30;
    cfg.quality = 1;

    auto vk_res = VulkanCompute::create(cfg);
    check(vk_res.has_value(), "VulkanCompute::create");
    auto &vk = *vk_res;

    auto fmts = vk->query_supported_capture_formats();
    check(!fmts.empty(), "query_supported_capture_formats non-empty");
    check(!fmts.front().modifiers.empty(), "GPU supports DRM modifiers");

    check(vk->configure_output(dst_w, dst_h).has_value(), "configure_output");
    auto enc_res = VideoEncoder::open(cfg, dst_w, dst_h, vk->hw_device_ref(), vk->hw_frames_ref());
    check(enc_res.has_value(), "VideoEncoder::open");
    auto &enc = *enc_res;

    // Pre-allocate a 4-buffer ring of real exportable GPU DMA-BUFs (mimicking PipeWire buffer pool)
    constexpr uint32_t kPool = 4;
    std::vector<VulkanCompute::TestDmaBuf> dmabufs;
    std::vector<uint8_t> shm_buf;
    if (!use_shm) {
        for (uint32_t i = 0; i < kPool; ++i) {
            auto db = vk->create_test_dmabuf(src_w, src_h, i);
            check(db.has_value(), "create_test_dmabuf");
            dmabufs.push_back(*db);
        }
    } else {
        shm_buf.resize(static_cast<size_t>(src_w) * src_h * 4, 0x7f);
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < num_frames; ++i) {
        CapturedFrame cf{};
        cf.format.spa_format = SPA_VIDEO_FORMAT_BGRA;
        cf.format.drm_fourcc = DRM_FORMAT_ARGB8888;
        cf.format.width = src_w;
        cf.format.height = src_h;
        cf.num_planes = 1;

        if (!use_shm) {
            auto &db = dmabufs[i % kPool];
            cf.buffer_handle = reinterpret_cast<void *>(static_cast<uintptr_t>(0x1000 + (i % kPool)));
            cf.mem_type = CaptureMemoryType::DmaBuf;
            cf.format.drm_modifier = db.modifier;
            cf.format.has_modifier = true;
            cf.planes[0].fd = db.fd;
            cf.planes[0].stride = static_cast<int32_t>(db.stride);
            cf.planes[0].size = db.stride * src_h;
        } else {
            cf.buffer_handle = reinterpret_cast<void *>(0x9999);
            cf.mem_type = CaptureMemoryType::Shm;
            cf.planes[0].data = shm_buf.data();
            cf.planes[0].stride = static_cast<int32_t>(src_w * 4);
            cf.planes[0].size = static_cast<uint32_t>(shm_buf.size());
        }

        auto frame_res = vk->process_frame(cf);
        check(frame_res.has_value(), "VulkanCompute::process_frame");
        AVFrame *avf = *frame_res;
        int64_t pts = cfg.vfr ? static_cast<int64_t>(i) * (1'000'000 / 60) : static_cast<int64_t>(i);
        check(enc->encode_frame(avf, pts).has_value(), "VideoEncoder::encode_frame");
        av_frame_free(&avf);
    }
    check(enc->finish().has_value(), "VideoEncoder::finish");
    const auto t1 = std::chrono::steady_clock::now();

    for (auto &db : dmabufs) {
        vk->evict_buffer(reinterpret_cast<void *>(static_cast<uintptr_t>(0x1000 + (&db - dmabufs.data()))));
        vk->destroy_test_dmabuf(db);
    }

    double sec = std::chrono::duration<double>(t1 - t0).count();
    double fps = static_cast<double>(num_frames) / sec;
    auto cst = vk->stats();
    double comp_ms = (static_cast<double>(cst.total_compute_ns) / static_cast<double>(cst.frames_processed)) / 1e6;

    verify_av1_container(out_path, static_cast<int>(dst_w), static_cast<int>(dst_h), static_cast<int>(num_frames));
    std::filesystem::remove(out_path);

    std::cout << std::format(
        "[PASS] gpu_{}_{}x{}->{}x{}_{} ({}): {} frames in {:.3f}s ({:.1f} fps, compute {:.3f} ms/f)\n",
        pixfmt == PixelFormat::P010 ? "p010" : "nv12",
        src_w, src_h, dst_w, dst_h, use_shm ? "shm" : "dmabuf", ext,
        num_frames, sec, fps, comp_ms);
}

}  // namespace

int main() {
    test_cli_parsing();
    test_spa_drm_mapping();

    // 1. 1080p60 8-bit NV12 DMA-BUF -> AV1 MP4
    test_gpu_dmabuf_av1_pipeline(PixelFormat::Nv12, 1920, 1080, 1920, 1080, 60, "mp4");
    // 2. 1080p60 10-bit P010 DMA-BUF -> AV1 MKV
    test_gpu_dmabuf_av1_pipeline(PixelFormat::P010, 1920, 1080, 1920, 1080, 60, "mkv");
    // 3. 4K -> 1440p scaled DMA-BUF -> AV1 WebM
    test_gpu_dmabuf_av1_pipeline(PixelFormat::Nv12, 3840, 2160, 2560, 1440, 60, "webm");
    // 4. SHM fallback -> AV1 MKV
    test_gpu_dmabuf_av1_pipeline(PixelFormat::Nv12, 1280, 720, 1280, 720, 30, "mkv", true);
    // 5. 4K native 1:1 throughput benchmark (180 frames)
    test_gpu_dmabuf_av1_pipeline(PixelFormat::Nv12, 3840, 2160, 3840, 2160, 180, "mkv");

    // 6. CFR static-frame repeat test (av_frame_clone of the same AVVkFrame)
    {
        const std::string out_path = "/tmp/vkcap_test_cfr_repeat.mkv";
        std::filesystem::remove(out_path);
        Config cfg;
        cfg.output_path = out_path;
        cfg.fps = 60;
        cfg.vfr = false;
        auto vk = *VulkanCompute::create(cfg);
        check(vk->configure_output(1280, 720).has_value(), "cfr repeat configure_output");
        auto enc = *VideoEncoder::open(cfg, 1280, 720, vk->hw_device_ref(), vk->hw_frames_ref());
        auto db = *vk->create_test_dmabuf(1280, 720, 0);
        CapturedFrame cf{};
        cf.buffer_handle = reinterpret_cast<void *>(0x2000);
        cf.mem_type = CaptureMemoryType::DmaBuf;
        cf.format.spa_format = SPA_VIDEO_FORMAT_BGRA;
        cf.format.drm_fourcc = DRM_FORMAT_ARGB8888;
        cf.format.width = 1280;
        cf.format.height = 720;
        cf.format.drm_modifier = db.modifier;
        cf.format.has_modifier = true;
        cf.num_planes = 1;
        cf.planes[0].fd = db.fd;
        cf.planes[0].stride = static_cast<int32_t>(db.stride);
        cf.planes[0].size = db.stride * 720;

        AVFrame *master = *vk->process_frame(cf);
        for (int i = 0; i < 30; ++i) {
            AVFrame *dup = av_frame_clone(master);
            check(enc->encode_frame(dup, i).has_value(), "cfr repeat encode_frame");
            av_frame_free(&dup);
        }
        av_frame_free(&master);
        check(enc->finish().has_value(), "cfr repeat finish");
        vk->evict_buffer(cf.buffer_handle);
        vk->destroy_test_dmabuf(db);
        verify_av1_container(out_path, 1280, 720, 30);
        std::filesystem::remove(out_path);
        std::cout << "[PASS] gpu_cfr_static_frame_repeat (30 frames)\n";
    }

    std::cout << "All unit and GPU integration tests passed.\n";
    return 0;
}
