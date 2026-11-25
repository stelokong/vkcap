#include "video_encoder.hpp"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <format>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
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
    return "default";
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
    return "desktop";
}

std::string ff_err(int err) {
    char buf[128];
    av_strerror(err, buf, sizeof(buf));
    return std::string(buf);
}

}  // namespace

struct VideoEncoder::Impl {
    Config cfg;
    AVFormatContext *fmt_ctx = nullptr;
    AVCodecContext *enc_ctx = nullptr;
    AVStream *stream = nullptr;
    bool header_written = false;
    bool finished = false;
    int64_t last_pts_us = -1;

    std::atomic<uint64_t> frames_submitted{0};
    std::atomic<uint64_t> packets_written{0};
    std::atomic<uint64_t> bytes_written{0};

    // Dedicated async muxer/writer thread with recycled AVPacket pool (zero per-packet malloc/free)
    static constexpr size_t kMaxQueuedPackets = 256;
    AVPacket *scratch_pkt = nullptr;
    std::mutex pkt_mtx;
    std::condition_variable pkt_cv;
    std::condition_variable pkt_space_cv;
    std::deque<AVPacket *> pkt_queue;
    std::vector<AVPacket *> free_pkts;
    bool writer_stop = false;
    std::atomic<bool> has_writer_err{false};
    std::string writer_err;
    std::jthread writer_thread;

    void start_writer() {
        writer_thread = std::jthread([this](std::stop_token) {
            pthread_setname_np(pthread_self(), "vkcap-mux");
            while (true) {
                AVPacket *p = nullptr;
                {
                    std::unique_lock lk(pkt_mtx);
                    pkt_cv.wait(lk, [this] { return writer_stop || !pkt_queue.empty(); });
                    if (pkt_queue.empty()) {
                        if (writer_stop) break;
                        continue;
                    }
                    p = pkt_queue.front();
                    pkt_queue.pop_front();
                }

                int w = av_interleaved_write_frame(fmt_ctx, p);
                av_packet_unref(p);

                {
                    std::lock_guard lk(pkt_mtx);
                    if (free_pkts.size() < 32) {
                        free_pkts.push_back(p);
                    } else {
                        av_packet_free(&p);
                    }
                    if (w < 0 && writer_err.empty()) {
                        writer_err = std::format("av_interleaved_write_frame failed: {}", ff_err(w));
                        has_writer_err.store(true, std::memory_order_release);
                        pkt_space_cv.notify_all();
                        break;
                    }
                }
                pkt_space_cv.notify_one();
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
        for (AVPacket *p : pkt_queue) av_packet_free(&p);
        pkt_queue.clear();
        for (AVPacket *p : free_pkts) av_packet_free(&p);
        free_pkts.clear();
    }

    std::expected<void, std::string> drain_packets() {
        while (true) {
            if (!scratch_pkt) {
                {
                    std::lock_guard lk(pkt_mtx);
                    if (!free_pkts.empty()) {
                        scratch_pkt = free_pkts.back();
                        free_pkts.pop_back();
                    }
                }
                if (!scratch_pkt) {
                    scratch_pkt = av_packet_alloc();
                    if (!scratch_pkt) return std::unexpected("av_packet_alloc failed");
                }
            }
            int r = avcodec_receive_packet(enc_ctx, scratch_pkt);
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
            AVPacket *p = scratch_pkt;
            scratch_pkt = nullptr;
            p->stream_index = stream->index;
            av_packet_rescale_ts(p, enc_ctx->time_base, stream->time_base);
            packets_written.fetch_add(1, std::memory_order_relaxed);
            bytes_written.fetch_add(static_cast<uint64_t>(p->size), std::memory_order_relaxed);

            std::unique_lock lk(pkt_mtx);
            if (!writer_err.empty()) {
                av_packet_free(&p);
                return std::unexpected(writer_err);
            }
            pkt_space_cv.wait(lk, [this] {
                return writer_stop || !writer_err.empty() || pkt_queue.size() < kMaxQueuedPackets;
            });
            if (writer_stop || !writer_err.empty()) {
                av_packet_free(&p);
                if (!writer_err.empty()) return std::unexpected(writer_err);
                return {};
            }
            pkt_queue.push_back(p);
            lk.unlock();
            pkt_cv.notify_one();
        }
    }

    ~Impl() {
        stop_writer();
        if (scratch_pkt) av_packet_free(&scratch_pkt);
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
    AVBufferRef *hw_frames_ref) {
    auto enc = std::unique_ptr<VideoEncoder>(new VideoEncoder());
    enc->impl_ = std::make_unique<Impl>();
    Impl *impl = enc->impl_.get();
    impl->cfg = cfg;

    int err = avformat_alloc_output_context2(&impl->fmt_ctx, nullptr, nullptr, cfg.output_path.c_str());
    if (err < 0 || !impl->fmt_ctx) {
        return std::unexpected(std::format("Could not determine output format for '{}': {}",
                                           cfg.output_path, ff_err(err)));
    }

    const char *cname = codec_name(cfg.codec);
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

    // Resolve RDNA 4 / VCN 5 quality-efficiency defaults when left on auto
    const int eff_idr = (cfg.idr_interval > 0)
        ? cfg.idr_interval
        : static_cast<int>(std::max<uint32_t>(1u, cfg.fps) * 2u);  // 2-second GOP

    const int eff_qp = (cfg.qp >= 0)
        ? cfg.qp
        : (cfg.codec == Codec::Av1 ? 52 : 22);

    int eff_quality = cfg.quality;
    if (eff_quality < 0) {
        // On RX 9060 XT (RDNA 4 / VCN 5), quality=1 matches quality=2 SSIM/PSNR within
        // 0.00001 / 0.3% bitrate while using ~20% less VCN & VRAM bandwidth under 3D game load.
        eff_quality = 1;
    }

    impl->enc_ctx->width = static_cast<int>(width);
    impl->enc_ctx->height = static_cast<int>(height);
    impl->enc_ctx->time_base = cfg.vfr
        ? AVRational{1, 1'000'000}
        : AVRational{1, static_cast<int>(cfg.fps)};
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

    if (!(impl->fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
        err = avio_open(&impl->fmt_ctx->pb, cfg.output_path.c_str(), AVIO_FLAG_WRITE);
        if (err < 0) {
            return std::unexpected(std::format("avio_open('{}') failed: {}", cfg.output_path, ff_err(err)));
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

std::expected<void, std::string> VideoEncoder::encode_frame(AVFrame *frame, int64_t pts_us) {
    if (impl_->finished) {
        return std::unexpected("Encoder already finished");
    }
    if (frame) {
        if (pts_us <= impl_->last_pts_us) {
            pts_us = impl_->last_pts_us + 1;
        }
        impl_->last_pts_us = pts_us;
        frame->pts = pts_us;
        frame->duration = impl_->cfg.vfr
            ? static_cast<int64_t>(1'000'000 / std::max<uint32_t>(1u, impl_->cfg.fps))
            : 1;
        impl_->frames_submitted.fetch_add(1, std::memory_order_relaxed);
    }

    int err = avcodec_send_frame(impl_->enc_ctx, frame);
    if (err < 0) {
        return std::unexpected(std::format("avcodec_send_frame failed: {}", ff_err(err)));
    }
    return impl_->drain_packets();
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
    impl_->stop_writer();
    if (!res) return res;
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

EncoderStats VideoEncoder::stats() const noexcept {
    return {
        impl_->frames_submitted.load(std::memory_order_relaxed),
        impl_->packets_written.load(std::memory_order_relaxed),
        impl_->bytes_written.load(std::memory_order_relaxed),
    };
}

}  // namespace vkcap
