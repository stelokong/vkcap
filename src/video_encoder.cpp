#include "video_encoder.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <vector>
#include <vulkan/vulkan.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/opt.h>
}

namespace vkcap {

namespace {

const char *codec_name(Codec c) {
    switch (c) {
        case Codec::Av1:  return "av1_vulkan";
        case Codec::Hevc: return "hevc_vulkan";
        case Codec::H264: return "h264_vulkan";
    }
    return "av1_vulkan";
}

const char *tune_str(Tune t) {
    switch (t) {
        case Tune::Default:  return "default";
        case Tune::Hq:       return "hq";
        case Tune::Ll:       return "ll";
        case Tune::Ull:      return "ull";
        case Tune::Lossless: return "lossless";
    }
    return "hq";
}

const char *usage_str(Usage u) {
    switch (u) {
        case Usage::Default:    return "default";
        case Usage::Transcode:  return "transcode";
        case Usage::Stream:     return "stream";
        case Usage::Record:     return "record";
        case Usage::Conference: return "conference";
    }
    return "record";
}

const char *content_str(Content c) {
    switch (c) {
        case Content::Default:  return "default";
        case Content::Camera:   return "camera";
        case Content::Desktop:  return "desktop";
        case Content::Rendered: return "rendered";
    }
    return "rendered";
}

std::string ff_err(int err) {
    char buf[128];
    av_strerror(err, buf, sizeof(buf));
    return std::string(buf);
}

struct VkEncodeCaps {
    bool probed = false;
    bool has_av1 = false;
    bool has_hevc = false;
    bool has_h264 = false;
    uint32_t max_quality_levels = 2;  // Conservative fallback if query unavailable
};

VkEncodeCaps probe_vk_encode_caps(AVBufferRef *hw_device_ref, Codec codec, PixelFormat pixfmt) {
    VkEncodeCaps caps{};
    if (!hw_device_ref || !hw_device_ref->data) return caps;
    auto *hwdev = reinterpret_cast<AVHWDeviceContext *>(hw_device_ref->data);
    auto *vkctx = reinterpret_cast<AVVulkanDeviceContext *>(hwdev->hwctx);
    if (!vkctx || !vkctx->inst || !vkctx->phys_dev || !vkctx->get_proc_addr) return caps;

    auto pfn_enum = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        vkctx->get_proc_addr(vkctx->inst, "vkEnumerateDeviceExtensionProperties"));
    if (!pfn_enum) return caps;

    uint32_t ext_cnt = 0;
    if (pfn_enum(vkctx->phys_dev, nullptr, &ext_cnt, nullptr) != VK_SUCCESS) return caps;
    std::vector<VkExtensionProperties> exts(ext_cnt);
    pfn_enum(vkctx->phys_dev, nullptr, &ext_cnt, exts.data());
    caps.probed = true;
    for (const auto &e : exts) {
        if (std::strcmp(e.extensionName, VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME) == 0) caps.has_av1 = true;
        else if (std::strcmp(e.extensionName, VK_KHR_VIDEO_ENCODE_H265_EXTENSION_NAME) == 0) caps.has_hevc = true;
        else if (std::strcmp(e.extensionName, VK_KHR_VIDEO_ENCODE_H264_EXTENSION_NAME) == 0) caps.has_h264 = true;
    }

    auto pfn_get_caps = reinterpret_cast<PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR>(
        vkctx->get_proc_addr(vkctx->inst, "vkGetPhysicalDeviceVideoCapabilitiesKHR"));
    if (!pfn_get_caps) return caps;

    VkVideoProfileInfoKHR prof{};
    prof.sType = VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR;
    prof.chromaSubsampling = VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR;
    const VkVideoComponentBitDepthFlagBitsKHR bd = (pixfmt == PixelFormat::P010)
        ? VK_VIDEO_COMPONENT_BIT_DEPTH_10_BIT_KHR
        : VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR;
    prof.lumaBitDepth = bd;
    prof.chromaBitDepth = bd;

    VkVideoEncodeAV1ProfileInfoKHR av1_p{VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_PROFILE_INFO_KHR, nullptr, STD_VIDEO_AV1_PROFILE_MAIN};
    VkVideoEncodeH265ProfileInfoKHR h265_p{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_PROFILE_INFO_KHR, nullptr,
                                           (pixfmt == PixelFormat::P010) ? STD_VIDEO_H265_PROFILE_IDC_MAIN_10 : STD_VIDEO_H265_PROFILE_IDC_MAIN};
    VkVideoEncodeH264ProfileInfoKHR h264_p{VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR, nullptr, STD_VIDEO_H264_PROFILE_IDC_HIGH};

    switch (codec) {
        case Codec::Av1:
            if (!caps.has_av1) return caps;
            prof.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_AV1_BIT_KHR;
            prof.pNext = &av1_p;
            break;
        case Codec::Hevc:
            if (!caps.has_hevc) return caps;
            prof.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H265_BIT_KHR;
            prof.pNext = &h265_p;
            break;
        case Codec::H264:
            if (!caps.has_h264) return caps;
            prof.videoCodecOperation = VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR;
            prof.pNext = &h264_p;
            break;
    }

    VkVideoEncodeAV1CapabilitiesKHR av1_caps{};
    av1_caps.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_AV1_CAPABILITIES_KHR;
    VkVideoEncodeH265CapabilitiesKHR h265_caps{};
    h265_caps.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H265_CAPABILITIES_KHR;
    VkVideoEncodeH264CapabilitiesKHR h264_caps{};
    h264_caps.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR;

    VkVideoEncodeCapabilitiesKHR enc_caps{};
    enc_caps.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR;
    if (codec == Codec::Av1) enc_caps.pNext = &av1_caps;
    else if (codec == Codec::Hevc) enc_caps.pNext = &h265_caps;
    else enc_caps.pNext = &h264_caps;

    VkVideoCapabilitiesKHR vcaps{};
    vcaps.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
    vcaps.pNext = &enc_caps;
    if (pfn_get_caps(vkctx->phys_dev, &prof, &vcaps) == VK_SUCCESS && enc_caps.maxQualityLevels > 0) {
        caps.max_quality_levels = enc_caps.maxQualityLevels;
    }
    return caps;
}

}  // namespace

struct VideoEncoder::Impl {
    Config cfg;
    AVFormatContext *fmt_ctx = nullptr;
    AVCodecContext *enc_ctx = nullptr;
    AVStream *stream = nullptr;
    AVCodecContext *audio_enc_ctx = nullptr;
    AVStream *audio_stream = nullptr;
    AVFrame *audio_frame = nullptr;
    AVPacket *audio_scratch_pkt = nullptr;
    std::mutex audio_enc_mtx;
    bool header_written = false;
    bool finished = false;
    int64_t last_pts = -1;

    std::atomic<uint64_t> frames_submitted{0};
    std::atomic<uint64_t> packets_written{0};
    std::atomic<uint64_t> bytes_written{0};

    static constexpr size_t kMaxQueuedPackets = 256;
    AVPacket *scratch_pkt = nullptr;
    std::mutex pkt_mtx;
    std::condition_variable pkt_cv;
    std::condition_variable pkt_space_cv;
    std::array<AVPacket *, kMaxQueuedPackets> pkt_ring{};
    size_t pkt_head = 0;
    size_t pkt_tail = 0;
    size_t pkt_count = 0;
    std::vector<AVPacket *> free_pkts;
    bool writer_stop = false;
    std::atomic<bool> has_writer_err{false};
    std::string writer_err;
    std::jthread writer_thread;

    void start_writer() {
        free_pkts.reserve(32);
        writer_thread = std::jthread([this](std::stop_token) {
            pthread_setname_np(pthread_self(), "vkcap-mux");
            AVPacket *recycle = nullptr;
            while (true) {
                AVPacket *p = nullptr;
                {
                    std::unique_lock lk(pkt_mtx);
                    if (recycle) {
                        if (free_pkts.size() < 32) {
                            free_pkts.push_back(recycle);
                        } else {
                            av_packet_free(&recycle);
                        }
                        recycle = nullptr;
                    }
                    pkt_cv.wait(lk, [this] { return writer_stop || pkt_count > 0; });
                    if (pkt_count == 0) {
                        if (writer_stop) break;
                        continue;
                    }
                    p = pkt_ring[pkt_head];
                    pkt_head = (pkt_head + 1) % kMaxQueuedPackets;
                    --pkt_count;
                }
                pkt_space_cv.notify_one();

                int w = av_interleaved_write_frame(fmt_ctx, p);
                av_packet_unref(p);
                if (w < 0) {
                    std::lock_guard lk(pkt_mtx);
                    if (free_pkts.size() < 32) {
                        free_pkts.push_back(p);
                    } else {
                        av_packet_free(&p);
                    }
                    if (writer_err.empty()) {
                        writer_err = std::format("av_interleaved_write_frame failed: {}", ff_err(w));
                        has_writer_err.store(true, std::memory_order_release);
                        pkt_space_cv.notify_all();
                    }
                    break;
                }
                recycle = p;
            }
            if (recycle) {
                av_packet_free(&recycle);
            }
        });
    }

    void stop_writer() {
        {
            std::lock_guard lk(pkt_mtx);
            writer_stop = true;
        }
        pkt_cv.notify_all();
        pkt_space_cv.notify_all();
        if (writer_thread.joinable()) {
            writer_thread.join();
        }
        while (pkt_count > 0) {
            av_packet_free(&pkt_ring[pkt_head]);
            pkt_head = (pkt_head + 1) % kMaxQueuedPackets;
            --pkt_count;
        }
        for (AVPacket *p : free_pkts) av_packet_free(&p);
        free_pkts.clear();
    }

    std::expected<void, std::string> drain_packets_from(AVCodecContext *ctx, AVStream *st, AVPacket *&scratch) {
        while (true) {
            if (!scratch) {
                {
                    std::lock_guard lk(pkt_mtx);
                    if (!free_pkts.empty()) {
                        scratch = free_pkts.back();
                        free_pkts.pop_back();
                    }
                }
                if (!scratch) {
                    scratch = av_packet_alloc();
                    if (!scratch) return std::unexpected("av_packet_alloc failed");
                }
            }
            int r = avcodec_receive_packet(ctx, scratch);
            if (r == AVERROR(EAGAIN) || r == AVERROR_EOF) {
                if (has_writer_err.load(std::memory_order_acquire)) {
                    std::lock_guard lk(pkt_mtx);
                    return std::unexpected(writer_err);
                }
                return {};
            }
            if (r < 0) {
                return std::unexpected(std::format("avcodec_receive_packet failed: {}", ff_err(r)));
            }
            AVPacket *p = scratch;
            scratch = nullptr;
            p->stream_index = st->index;
            av_packet_rescale_ts(p, ctx->time_base, st->time_base);
            packets_written.fetch_add(1, std::memory_order_relaxed);
            bytes_written.fetch_add(static_cast<uint64_t>(p->size), std::memory_order_relaxed);

            std::unique_lock lk(pkt_mtx);
            if (!writer_err.empty()) {
                av_packet_free(&p);
                return std::unexpected(writer_err);
            }
            pkt_space_cv.wait(lk, [this] {
                return writer_stop || !writer_err.empty() || pkt_count < kMaxQueuedPackets;
            });
            if (writer_stop || !writer_err.empty()) {
                av_packet_free(&p);
                if (!writer_err.empty()) return std::unexpected(writer_err);
                return {};
            }
            pkt_ring[pkt_tail] = p;
            pkt_tail = (pkt_tail + 1) % kMaxQueuedPackets;
            ++pkt_count;
            // Refill scratch packet under the same lock to halve mutex acquisitions
            if (!free_pkts.empty()) {
                scratch = free_pkts.back();
                free_pkts.pop_back();
            }
            lk.unlock();
            pkt_cv.notify_one();
        }
    }

    std::expected<void, std::string> drain_packets() {
        return drain_packets_from(enc_ctx, stream, scratch_pkt);
    }

    ~Impl() {
        stop_writer();
        if (scratch_pkt) av_packet_free(&scratch_pkt);
        if (audio_scratch_pkt) av_packet_free(&audio_scratch_pkt);
        if (audio_frame) av_frame_free(&audio_frame);
        if (audio_enc_ctx) avcodec_free_context(&audio_enc_ctx);
        if (enc_ctx) avcodec_free_context(&enc_ctx);
        if (fmt_ctx) {
            if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE) && fmt_ctx->pb) {
                avio_closep(&fmt_ctx->pb);
            }
            avformat_free_context(fmt_ctx);
        }
    }
};

VideoEncoder::~VideoEncoder() = default;

std::expected<std::unique_ptr<VideoEncoder>, std::string> VideoEncoder::open(
    const Config &cfg,
    uint32_t width,
    uint32_t height,
    AVBufferRef *hw_device_ref,
    AVBufferRef *hw_frames_ref,
    bool enable_audio) {
    auto enc = std::unique_ptr<VideoEncoder>(new VideoEncoder());
    enc->impl_ = std::make_unique<Impl>();
    Impl *impl = enc->impl_.get();
    impl->cfg = cfg;

    int err = avformat_alloc_output_context2(&impl->fmt_ctx, nullptr, nullptr, cfg.output_path.c_str());
    if (err < 0 || !impl->fmt_ctx) {
        return std::unexpected(std::format("Could not determine output format for '{}': {}",
                                           cfg.output_path, ff_err(err)));
    }

    // Probe GPU Vulkan Video encode extensions & quality levels for cross-vendor / older-GPU compatibility
    Codec chosen_codec = cfg.codec;
    VkEncodeCaps vcaps = probe_vk_encode_caps(hw_device_ref, chosen_codec, cfg.pixfmt);
    if (vcaps.probed && !cfg.codec_explicit) {
        auto codec_supported = [&](Codec c) {
            if (c == Codec::Av1) return vcaps.has_av1;
            if (c == Codec::Hevc) return vcaps.has_hevc;
            return vcaps.has_h264;
        };
        if (!codec_supported(chosen_codec)) {
            if (vcaps.has_hevc) chosen_codec = Codec::Hevc;
            else if (vcaps.has_h264) chosen_codec = Codec::H264;
            if (chosen_codec != cfg.codec) {
                std::cerr << std::format(
                    "[vkcap] GPU does not advertise {}; auto-falling back to {}\n",
                    codec_name(cfg.codec), codec_name(chosen_codec));
                vcaps = probe_vk_encode_caps(hw_device_ref, chosen_codec, cfg.pixfmt);
            }
        }
    }
    impl->cfg.codec = chosen_codec;

    const char *cname = codec_name(chosen_codec);
    const AVCodec *codec = avcodec_find_encoder_by_name(cname);
    if (!codec) {
        return std::unexpected(std::format("FFmpeg encoder '{}' not found", cname));
    }

    impl->stream = avformat_new_stream(impl->fmt_ctx, nullptr);
    if (!impl->stream) {
        return std::unexpected("avformat_new_stream failed");
    }

    impl->enc_ctx = avcodec_alloc_context3(codec);
    if (!impl->enc_ctx) {
        return std::unexpected("avcodec_alloc_context3 failed");
    }

    const int eff_idr = (cfg.idr_interval > 0)
        ? cfg.idr_interval
        : static_cast<int>(std::max<uint32_t>(1u, cfg.fps) * 2u);
    const int eff_qp = (cfg.qp >= 0) ? cfg.qp : (chosen_codec == Codec::Av1 ? 52 : 22);
    const int max_q = static_cast<int>(std::max<uint32_t>(1u, vcaps.max_quality_levels)) - 1;
    const int eff_quality = (cfg.quality >= 0) ? std::min(cfg.quality, max_q) : std::min(1, max_q);

    impl->enc_ctx->width = static_cast<int>(width);
    impl->enc_ctx->height = static_cast<int>(height);
    impl->enc_ctx->time_base = AVRational{1, static_cast<int>(cfg.fps)};
    impl->enc_ctx->framerate = AVRational{static_cast<int>(cfg.fps), 1};
    impl->enc_ctx->sample_aspect_ratio = AVRational{1, 1};
    impl->enc_ctx->pix_fmt = AV_PIX_FMT_VULKAN;
    impl->enc_ctx->gop_size = eff_idr;
    impl->enc_ctx->max_b_frames = std::max(0, cfg.b_depth - 1);

    impl->enc_ctx->color_range = (cfg.color_range == ColorRange::Full) ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
    impl->enc_ctx->colorspace = (cfg.color_space == ColorSpace::Bt2020) ? AVCOL_SPC_BT2020_NCL : AVCOL_SPC_BT709;
    impl->enc_ctx->color_primaries = (cfg.color_space == ColorSpace::Bt2020) ? AVCOL_PRI_BT2020 : AVCOL_PRI_BT709;
    impl->enc_ctx->color_trc = (cfg.color_space == ColorSpace::Bt2020) ? AVCOL_TRC_BT2020_10 : AVCOL_TRC_BT709;
    impl->enc_ctx->chroma_sample_location = AVCHROMA_LOC_LEFT;

    if (impl->fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        impl->enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    impl->enc_ctx->hw_device_ctx = av_buffer_ref(hw_device_ref);
    impl->enc_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);

    AVDictionary *opts = nullptr;
    RateControl eff_rc = cfg.rc;
    if (eff_rc == RateControl::Auto) {
        eff_rc = (cfg.bitrate > 0) ? RateControl::Vbr : RateControl::Cqp;
    }

    switch (eff_rc) {
        case RateControl::Cqp:
            av_dict_set(&opts, "rc_mode", "cqp", 0);
            av_dict_set_int(&opts, "qp", eff_qp, 0);
            break;
        case RateControl::Cbr:
            av_dict_set(&opts, "rc_mode", "cbr", 0);
            impl->enc_ctx->bit_rate = cfg.bitrate > 0 ? cfg.bitrate : 20'000'000;
            impl->enc_ctx->rc_max_rate = impl->enc_ctx->bit_rate;
            impl->enc_ctx->rc_buffer_size = static_cast<int>(impl->enc_ctx->bit_rate);
            break;
        case RateControl::Vbr:
            av_dict_set(&opts, "rc_mode", "vbr", 0);
            impl->enc_ctx->bit_rate = cfg.bitrate > 0 ? cfg.bitrate : 20'000'000;
            impl->enc_ctx->rc_max_rate = cfg.max_bitrate > 0 ? cfg.max_bitrate : (impl->enc_ctx->bit_rate * 3) / 2;
            impl->enc_ctx->rc_buffer_size = static_cast<int>(impl->enc_ctx->rc_max_rate);
            break;
        case RateControl::Auto:
            break;
    }

    av_dict_set_int(&opts, "quality", eff_quality, 0);
    av_dict_set(&opts, "tune", tune_str(cfg.tune), 0);
    av_dict_set(&opts, "usage", usage_str(cfg.usage), 0);
    av_dict_set(&opts, "content", content_str(cfg.content), 0);
    av_dict_set_int(&opts, "idr_interval", eff_idr, 0);
    av_dict_set_int(&opts, "async_depth", cfg.async_depth, 0);
    av_dict_set_int(&opts, "b_depth", cfg.b_depth, 0);

    err = avcodec_open2(impl->enc_ctx, codec, &opts);
    av_dict_free(&opts);
    if (err < 0) {
        return std::unexpected(std::format("avcodec_open2({}) failed: {}", cname, ff_err(err)));
    }

    err = avcodec_parameters_from_context(impl->stream->codecpar, impl->enc_ctx);
    if (err < 0) {
        return std::unexpected(std::format("avcodec_parameters_from_context failed: {}", ff_err(err)));
    }
    impl->stream->time_base = impl->enc_ctx->time_base;
    impl->stream->avg_frame_rate = impl->enc_ctx->framerate;
    impl->stream->r_frame_rate = impl->enc_ctx->framerate;

    // Optional low-latency audio stream (Opus / AAC / FLAC; skipped for video-only .ivf)
    const bool is_ivf = impl->fmt_ctx->oformat->name &&
                        std::strcmp(impl->fmt_ctx->oformat->name, "ivf") == 0;
    const bool is_webm = impl->fmt_ctx->oformat->name &&
                         std::strstr(impl->fmt_ctx->oformat->name, "webm") != nullptr;
    if (enable_audio && !is_ivf) {
        AudioCodec eff_acodec = cfg.audio_codec;
        if (is_webm && eff_acodec != AudioCodec::Opus) {
            std::cerr << "[vkcap] Note: WebM container requires Opus audio; switching audio codec to Opus.\n";
            eff_acodec = AudioCodec::Opus;
        }
        const AVCodec *acodec = nullptr;
        switch (eff_acodec) {
            case AudioCodec::Opus:
                acodec = avcodec_find_encoder_by_name("libopus");
                if (!acodec) acodec = avcodec_find_encoder(AV_CODEC_ID_OPUS);
                break;
            case AudioCodec::Aac:
                acodec = avcodec_find_encoder(AV_CODEC_ID_AAC);
                break;
            case AudioCodec::Flac:
                acodec = avcodec_find_encoder(AV_CODEC_ID_FLAC);
                break;
        }
        if (!acodec) acodec = avcodec_find_encoder_by_name("libopus");
        if (!acodec) acodec = avcodec_find_encoder(AV_CODEC_ID_AAC);
        if (!acodec) {
            return std::unexpected("No Opus, AAC, or FLAC audio encoder available in FFmpeg");
        }

        impl->audio_stream = avformat_new_stream(impl->fmt_ctx, nullptr);
        if (!impl->audio_stream) return std::unexpected("avformat_new_stream(audio) failed");

        impl->audio_enc_ctx = avcodec_alloc_context3(acodec);
        if (!impl->audio_enc_ctx) return std::unexpected("avcodec_alloc_context3(audio) failed");

        impl->audio_enc_ctx->sample_rate = 48000;
        av_channel_layout_default(&impl->audio_enc_ctx->ch_layout, 2);
        impl->audio_enc_ctx->time_base = AVRational{1, 48000};
        if (acodec->id == AV_CODEC_ID_FLAC) {
            impl->audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_S32;
        } else if (std::strcmp(acodec->name, "libopus") == 0) {
            impl->audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_FLT;
            impl->audio_enc_ctx->bit_rate = cfg.audio_bitrate > 0 ? cfg.audio_bitrate : 192'000;
        } else {
            impl->audio_enc_ctx->sample_fmt = AV_SAMPLE_FMT_FLTP;
            impl->audio_enc_ctx->bit_rate = cfg.audio_bitrate > 0 ? cfg.audio_bitrate : 192'000;
        }
        if (impl->fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
            impl->audio_enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        AVDictionary *aopts = nullptr;
        if (std::strcmp(acodec->name, "libopus") == 0) {
            av_dict_set(&aopts, "application", "audio", 0);
            av_dict_set(&aopts, "frame_duration", "20", 0);
            // Complexity 5 cuts real-time Opus CPU usage ~3x vs default 10 with transparent quality at 192 kbps
            av_dict_set(&aopts, "compression_level", "5", 0);
        }
        err = avcodec_open2(impl->audio_enc_ctx, acodec, &aopts);
        av_dict_free(&aopts);
        if (err < 0) {
            return std::unexpected(std::format("avcodec_open2({}) failed: {}", acodec->name, ff_err(err)));
        }

        err = avcodec_parameters_from_context(impl->audio_stream->codecpar, impl->audio_enc_ctx);
        if (err < 0) {
            return std::unexpected(std::format("avcodec_parameters_from_context(audio) failed: {}", ff_err(err)));
        }
        impl->audio_stream->time_base = impl->audio_enc_ctx->time_base;

        impl->audio_frame = av_frame_alloc();
        if (!impl->audio_frame) return std::unexpected("av_frame_alloc(audio) failed");
        impl->audio_frame->nb_samples = impl->audio_enc_ctx->frame_size > 0 ? impl->audio_enc_ctx->frame_size : 960;
        impl->audio_frame->format = impl->audio_enc_ctx->sample_fmt;
        av_channel_layout_copy(&impl->audio_frame->ch_layout, &impl->audio_enc_ctx->ch_layout);
        impl->audio_frame->sample_rate = impl->audio_enc_ctx->sample_rate;
        if (av_frame_get_buffer(impl->audio_frame, 0) < 0) {
            return std::unexpected("av_frame_get_buffer(audio) failed");
        }
    }

    if (!(impl->fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        const std::string resolved_out = expand_user_path(cfg.output_path);
        std::filesystem::path out_p(resolved_out);
        if (out_p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(out_p.parent_path(), ec);
            if (ec) {
                return std::unexpected(std::format("Failed to create output directory '{}': {}",
                                                   out_p.parent_path().string(), ec.message()));
            }
        }
        err = avio_open(&impl->fmt_ctx->pb, resolved_out.c_str(), AVIO_FLAG_WRITE);
        if (err < 0) {
            return std::unexpected(std::format("avio_open('{}') failed: {}", resolved_out, ff_err(err)));
        }
    }

    AVDictionary *mux_opts = nullptr;
    if (impl->fmt_ctx->oformat->name && std::strstr(impl->fmt_ctx->oformat->name, "mp4") != nullptr) {
        av_dict_set(&mux_opts, "movflags", "+faststart", 0);
    }
    err = avformat_write_header(impl->fmt_ctx, &mux_opts);
    av_dict_free(&mux_opts);
    if (err < 0) {
        return std::unexpected(std::format("avformat_write_header failed: {}", ff_err(err)));
    }
    impl->header_written = true;
    impl->start_writer();

    if (cfg.verbose) {
        std::cerr << std::format(
            "[encoder] opened {} {}x{} pixfmt={} qp={} quality={} gop={} async_depth={} -> '{}'\n",
            cname, width, height,
            cfg.pixfmt == PixelFormat::P010 ? "p010" : "nv12",
            eff_qp, eff_quality, eff_idr, cfg.async_depth, cfg.output_path);
    }

    return enc;
}

std::expected<void, std::string> VideoEncoder::encode_frame(AVFrame *frame, int64_t pts_ticks) {
    if (impl_->finished) {
        return std::unexpected("Encoder already finished");
    }
    if (frame) {
        if (pts_ticks <= impl_->last_pts) {
            pts_ticks = impl_->last_pts + 1;
        }
        impl_->last_pts = pts_ticks;
        frame->pts = pts_ticks;
        frame->duration = 1;
        impl_->frames_submitted.fetch_add(1, std::memory_order_relaxed);
    }

    int err = avcodec_send_frame(impl_->enc_ctx, frame);
    if (err < 0) {
        return std::unexpected(std::format("avcodec_send_frame failed: {}", ff_err(err)));
    }
    return impl_->drain_packets();
}

std::expected<void, std::string> VideoEncoder::encode_audio_samples(
    const float *interleaved_stereo,
    int nb_samples,
    int64_t pts_samples) {
    std::lock_guard lk(impl_->audio_enc_mtx);
    if (impl_->finished || !impl_->audio_enc_ctx || !impl_->audio_frame) {
        return {};
    }
    if (av_frame_make_writable(impl_->audio_frame) < 0) {
        return std::unexpected("av_frame_make_writable(audio) failed");
    }
    impl_->audio_frame->nb_samples = nb_samples;
    impl_->audio_frame->pts = pts_samples;

    if (impl_->audio_enc_ctx->sample_fmt == AV_SAMPLE_FMT_FLT) {
        std::memcpy(impl_->audio_frame->data[0], interleaved_stereo,
                    static_cast<size_t>(nb_samples) * 2 * sizeof(float));
    } else if (impl_->audio_enc_ctx->sample_fmt == AV_SAMPLE_FMT_S32) {
        auto *dst = reinterpret_cast<int32_t *>(impl_->audio_frame->data[0]);
        for (int i = 0; i < nb_samples * 2; ++i) {
            float clamped = std::max(-1.0f, std::min(1.0f, interleaved_stereo[i]));
            dst[i] = static_cast<int32_t>(clamped * 2147483520.0f);
        }
    } else {
        auto *l = reinterpret_cast<float *>(impl_->audio_frame->data[0]);
        auto *r = reinterpret_cast<float *>(impl_->audio_frame->data[1]);
        for (int i = 0; i < nb_samples; ++i) {
            l[i] = interleaved_stereo[i * 2];
            r[i] = interleaved_stereo[i * 2 + 1];
        }
    }

    int err = avcodec_send_frame(impl_->audio_enc_ctx, impl_->audio_frame);
    if (err < 0) {
        return std::unexpected(std::format("avcodec_send_frame(audio) failed: {}", ff_err(err)));
    }
    return impl_->drain_packets_from(impl_->audio_enc_ctx, impl_->audio_stream, impl_->audio_scratch_pkt);
}

std::expected<void, std::string> VideoEncoder::finish() {
    if (impl_->finished) return {};
    impl_->finished = true;

    int err = avcodec_send_frame(impl_->enc_ctx, nullptr);
    if (err < 0 && err != AVERROR_EOF) {
        impl_->stop_writer();
        return std::unexpected(std::format("avcodec_send_frame(flush) failed: {}", ff_err(err)));
    }
    auto res = impl_->drain_packets();

    std::expected<void, std::string> ares{};
    if (impl_->audio_enc_ctx) {
        std::lock_guard lk(impl_->audio_enc_mtx);
        int aerr = avcodec_send_frame(impl_->audio_enc_ctx, nullptr);
        if (aerr == 0 || aerr == AVERROR_EOF) {
            ares = impl_->drain_packets_from(impl_->audio_enc_ctx, impl_->audio_stream, impl_->audio_scratch_pkt);
        }
    }

    impl_->stop_writer();
    if (!res) return res;
    if (!ares) return ares;
    if (!impl_->writer_err.empty()) {
        return std::unexpected(impl_->writer_err);
    }

    if (impl_->header_written) {
        err = av_write_trailer(impl_->fmt_ctx);
        if (err < 0) {
            return std::unexpected(std::format("av_write_trailer failed: {}", ff_err(err)));
        }
    }
    return {};
}

bool VideoEncoder::audio_enabled() const noexcept {
    return impl_->audio_enc_ctx != nullptr;
}

int VideoEncoder::audio_frame_size() const noexcept {
    return (impl_->audio_enc_ctx && impl_->audio_enc_ctx->frame_size > 0)
        ? impl_->audio_enc_ctx->frame_size : 960;
}

Codec VideoEncoder::active_codec() const noexcept {
    return impl_->cfg.codec;
}

EncoderStats VideoEncoder::stats() const noexcept {
    return {
        impl_->frames_submitted.load(std::memory_order_relaxed),
        impl_->packets_written.load(std::memory_order_relaxed),
        impl_->bytes_written.load(std::memory_order_relaxed),
    };
}

}  // namespace vkcap
