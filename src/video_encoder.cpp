#include "video_encoder.hpp"
#include "types.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <unistd.h>
#include <utility>
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
    bool supports_b_frames = false;
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
    if (pfn_get_caps(vkctx->phys_dev, &prof, &vcaps) == VK_SUCCESS) {
        if (enc_caps.maxQualityLevels > 0) {
            caps.max_quality_levels = enc_caps.maxQualityLevels;
        }
        switch (codec) {
            case Codec::Av1:
                caps.supports_b_frames = (av1_caps.maxBidirectionalCompoundReferenceCount > 0 &&
                                          av1_caps.maxBidirectionalCompoundGroup1ReferenceCount > 0 &&
                                          av1_caps.maxBidirectionalCompoundGroup2ReferenceCount > 0);
                break;
            case Codec::Hevc:
                caps.supports_b_frames = (h265_caps.maxBPictureL0ReferenceCount > 0 &&
                                          h265_caps.maxL1ReferenceCount > 0);
                break;
            case Codec::H264:
                caps.supports_b_frames = (h264_caps.maxBPictureL0ReferenceCount > 0 &&
                                          h264_caps.maxL1ReferenceCount > 0);
                break;
        }
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
    bool is_replay = false;
    bool encode_failed = false;
    int64_t last_pts = -1;

    std::atomic<uint64_t> frames_submitted{0};
    std::atomic<uint64_t> packets_written{0};
    std::atomic<uint64_t> bytes_written{0};

    // Instant replay rolling encoded-bitstream ring (0 extra VRAM, 0 disk I/O while buffering)
    std::mutex replay_mtx;
    std::deque<AVPacket *> replay_v_pkts;
    std::deque<AVPacket *> replay_a_pkts;
    std::deque<std::pair<uint64_t, int64_t>> replay_kf_seqs;  // (video_seq, pts) for O(1) keyframe eviction
    uint64_t replay_v_head_seq = 0;
    uint64_t replay_v_next_seq = 0;
    uint64_t replay_bytes = 0;
    std::atomic<uint64_t> replay_clips_saved{0};
    uint32_t replay_save_seq = 0;

    struct SaveJob {
        std::string out_path;
        AVCodecParameters *v_par = nullptr;
        AVRational v_tb{1, 60};
        AVRational v_fps{60, 1};
        AVCodecParameters *a_par = nullptr;
        AVRational a_tb{1, 48000};
        std::vector<AVPacket *> v_pkts;
        std::vector<AVPacket *> a_pkts;
        double duration_sec = 0.0;
    };
    std::mutex save_mtx;
    std::condition_variable save_cv;
    std::deque<SaveJob> save_queue;
    bool saver_stop = false;
    std::jthread saver_thread;

    static void free_save_job(SaveJob &job) noexcept {
        for (AVPacket *p : job.v_pkts) av_packet_free(&p);
        for (AVPacket *p : job.a_pkts) av_packet_free(&p);
        job.v_pkts.clear();
        job.a_pkts.clear();
        if (job.v_par) avcodec_parameters_free(&job.v_par);
        if (job.a_par) avcodec_parameters_free(&job.a_par);
    }

    static std::expected<uint64_t, std::string> write_replay_job(SaveJob &job) {
        std::filesystem::path out_p(job.out_path);
        if (out_p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(out_p.parent_path(), ec);
            if (ec) {
                return std::unexpected(std::format("Failed to create directory '{}': {}",
                                                   out_p.parent_path().string(), ec.message()));
            }
        }

        AVFormatContext *ofmt = nullptr;
        int err = avformat_alloc_output_context2(&ofmt, nullptr, nullptr, job.out_path.c_str());
        if (err < 0 || !ofmt) {
            return std::unexpected(std::format("Could not determine output format for '{}': {}",
                                               job.out_path, ff_err(err)));
        }

        auto cleanup = [&]() {
            if (ofmt) {
                if (!(ofmt->oformat->flags & AVFMT_NOFILE) && ofmt->pb) {
                    avio_closep(&ofmt->pb);
                }
                avformat_free_context(ofmt);
                ofmt = nullptr;
            }
        };

        AVStream *vst = avformat_new_stream(ofmt, nullptr);
        if (!vst || avcodec_parameters_copy(vst->codecpar, job.v_par) < 0) {
            cleanup();
            return std::unexpected("Failed to initialize replay video stream");
        }
        vst->codecpar->codec_tag = 0;
        vst->time_base = job.v_tb;
        vst->avg_frame_rate = job.v_fps;
        vst->r_frame_rate = job.v_fps;

        AVStream *ast = nullptr;
        const bool is_ivf = ofmt->oformat->name && std::strcmp(ofmt->oformat->name, "ivf") == 0;
        if (job.a_par && !is_ivf) {
            ast = avformat_new_stream(ofmt, nullptr);
            if (!ast || avcodec_parameters_copy(ast->codecpar, job.a_par) < 0) {
                cleanup();
                return std::unexpected("Failed to initialize replay audio stream");
            }
            ast->codecpar->codec_tag = 0;
            ast->time_base = job.a_tb;
        }

        if (!(ofmt->oformat->flags & AVFMT_NOFILE)) {
            err = avio_open(&ofmt->pb, job.out_path.c_str(), AVIO_FLAG_WRITE);
            if (err < 0) {
                cleanup();
                return std::unexpected(std::format("avio_open('{}') failed: {}", job.out_path, ff_err(err)));
            }
        }

        AVDictionary *mux_opts = nullptr;
        if (ofmt->oformat->name && std::strstr(ofmt->oformat->name, "mp4") != nullptr) {
            av_dict_set(&mux_opts, "movflags", "+faststart", 0);
        }
        err = avformat_write_header(ofmt, &mux_opts);
        av_dict_free(&mux_opts);
        if (err < 0) {
            cleanup();
            return std::unexpected(std::format("avformat_write_header(replay) failed: {}", ff_err(err)));
        }

        uint64_t total_bytes = 0;
        size_t vi = 0, ai = 0;
        while (vi < job.v_pkts.size() || (ast && ai < job.a_pkts.size())) {
            bool pick_video = false;
            if (vi < job.v_pkts.size() && (!ast || ai >= job.a_pkts.size())) {
                pick_video = true;
            } else if (vi < job.v_pkts.size() && ast && ai < job.a_pkts.size()) {
                pick_video = (av_compare_ts(job.v_pkts[vi]->dts, job.v_tb,
                                            job.a_pkts[ai]->dts, job.a_tb) <= 0);
            }
            if (pick_video) {
                AVPacket *p = job.v_pkts[vi++];
                p->stream_index = vst->index;
                total_bytes += static_cast<uint64_t>(p->size);
                av_packet_rescale_ts(p, job.v_tb, vst->time_base);
                err = av_interleaved_write_frame(ofmt, p);
            } else {
                AVPacket *p = job.a_pkts[ai++];
                p->stream_index = ast->index;
                total_bytes += static_cast<uint64_t>(p->size);
                av_packet_rescale_ts(p, job.a_tb, ast->time_base);
                err = av_interleaved_write_frame(ofmt, p);
            }
            if (err < 0) {
                cleanup();
                return std::unexpected(std::format("av_interleaved_write_frame(replay) failed: {}", ff_err(err)));
            }
        }

        err = av_write_trailer(ofmt);
        cleanup();
        if (err < 0) {
            return std::unexpected(std::format("av_write_trailer(replay) failed: {}", ff_err(err)));
        }
        return total_bytes;
    }

    void start_saver() {
        saver_thread = std::jthread([this](std::stop_token) {
            pthread_setname_np(pthread_self(), "vkcap-save");
            // Lower scheduling priority so background replay writes never compete with games or capture
            (void)::nice(10);
            while (true) {
                SaveJob job{};
                {
                    std::unique_lock lk(save_mtx);
                    save_cv.wait(lk, [this] { return saver_stop || !save_queue.empty(); });
                    if (save_queue.empty()) {
                        if (saver_stop) break;
                        continue;
                    }
                    job = std::move(save_queue.front());
                    save_queue.pop_front();
                }

                auto res = write_replay_job(job);
                const std::string saved_path = job.out_path;
                const double dur = job.duration_sec;
                free_save_job(job);

                if (res) {
                    replay_clips_saved.fetch_add(1, std::memory_order_relaxed);
                    std::lock_guard log_lk(stderr_mutex());
                    std::cerr << std::format(
                        "\n[vkcap] Saved replay '{}' ({:.1f}s, {:.2f} MiB)\n",
                        saved_path, dur, static_cast<double>(*res) / (1024.0 * 1024.0))
                              << std::flush;
                } else {
                    std::lock_guard log_lk(stderr_mutex());
                    std::cerr << std::format("\n[vkcap] Replay save failed: {}\n", res.error())
                              << std::flush;
                }
            }
        });
    }

    void stop_saver() {
        {
            std::lock_guard lk(save_mtx);
            saver_stop = true;
        }
        save_cv.notify_all();
        if (saver_thread.joinable()) {
            saver_thread.join();
        }
        for (auto &j : save_queue) free_save_job(j);
        save_queue.clear();
    }

    void clear_replay_buffers() {
        std::lock_guard lk(replay_mtx);
        for (AVPacket *p : replay_v_pkts) av_packet_free(&p);
        for (AVPacket *p : replay_a_pkts) av_packet_free(&p);
        replay_v_pkts.clear();
        replay_a_pkts.clear();
        replay_kf_seqs.clear();
        replay_v_head_seq = 0;
        replay_v_next_seq = 0;
        replay_bytes = 0;
    }

    void push_replay_packet_locked(AVPacket *p, bool is_video) {
        if (is_video) {
            const uint64_t seq = replay_v_next_seq++;
            replay_v_pkts.push_back(p);
            if ((p->flags & AV_PKT_FLAG_KEY) != 0) {
                replay_kf_seqs.emplace_back(seq, p->pts);
            }
            replay_bytes += static_cast<uint64_t>(p->size);
            const int64_t window_ticks = static_cast<int64_t>(cfg.replay_seconds) *
                                         static_cast<int64_t>(std::max<uint32_t>(1u, cfg.fps));
            const int64_t newest = p->pts;
            const int64_t cutoff = newest - window_ticks;
            // O(1) amortized eviction: advance keyframe index deque to latest keyframe <= cutoff
            // so the ring always starts on a keyframe covering at least replay_seconds.
            while (replay_kf_seqs.size() >= 2 && replay_kf_seqs[1].second <= cutoff) {
                replay_kf_seqs.pop_front();
            }
            const size_t drop_until = (!replay_kf_seqs.empty() && replay_kf_seqs.front().first > replay_v_head_seq)
                ? std::min(replay_v_pkts.size(),
                           static_cast<size_t>(replay_kf_seqs.front().first - replay_v_head_seq))
                : 0;
            for (size_t i = 0; i < drop_until; ++i) {
                AVPacket *old = replay_v_pkts.front();
                replay_v_pkts.pop_front();
                replay_bytes -= std::min<uint64_t>(replay_bytes, static_cast<uint64_t>(old->size));
                av_packet_unref(old);
                if (free_pkts.size() < 32) free_pkts.push_back(old);
                else av_packet_free(&old);
            }
            replay_v_head_seq += drop_until;
            if (audio_enc_ctx && !replay_v_pkts.empty()) {
                const int64_t a_keep = av_rescale_q(replay_v_pkts.front()->pts,
                                                    enc_ctx->time_base, audio_enc_ctx->time_base) - 24000;
                while (!replay_a_pkts.empty() && replay_a_pkts.front()->pts < a_keep) {
                    AVPacket *old = replay_a_pkts.front();
                    replay_a_pkts.pop_front();
                    replay_bytes -= std::min<uint64_t>(replay_bytes, static_cast<uint64_t>(old->size));
                    av_packet_unref(old);
                    if (free_pkts.size() < 32) free_pkts.push_back(old);
                    else av_packet_free(&old);
                }
            }
        } else {
            replay_a_pkts.push_back(p);
            replay_bytes += static_cast<uint64_t>(p->size);
            // Cap audio buffer if video hasn't started yet
            const int64_t max_a_span = static_cast<int64_t>(cfg.replay_seconds + 5) * 48000LL;
            while (replay_a_pkts.size() > 1 &&
                   (replay_a_pkts.back()->pts - replay_a_pkts.front()->pts) > max_a_span) {
                AVPacket *old = replay_a_pkts.front();
                replay_a_pkts.pop_front();
                replay_bytes -= std::min<uint64_t>(replay_bytes, static_cast<uint64_t>(old->size));
                av_packet_unref(old);
                if (free_pkts.size() < 32) free_pkts.push_back(old);
                else av_packet_free(&old);
            }
        }
    }

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
        const bool is_video = (ctx == enc_ctx);
        while (true) {
            if (!scratch) {
                {
                    std::lock_guard lk(is_replay ? replay_mtx : pkt_mtx);
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
                if (!is_replay && has_writer_err.load(std::memory_order_acquire)) {
                    std::lock_guard lk(pkt_mtx);
                    return std::unexpected(writer_err);
                }
                return {};
            }
            if (r < 0) {
                if (is_video) encode_failed = true;
                return std::unexpected(std::format("avcodec_receive_packet failed: {}", ff_err(r)));
            }
            AVPacket *p = scratch;
            scratch = nullptr;
            packets_written.fetch_add(1, std::memory_order_relaxed);
            bytes_written.fetch_add(static_cast<uint64_t>(p->size), std::memory_order_relaxed);

            if (is_replay) {
                if (is_video) {
                    // Copy only the compressed p->size bytes into a plain CPU-backed AVPacket and
                    // immediately unref `p` so any Vulkan VkDeviceMemory bitstream buffer owned by
                    // av1_vulkan is returned to FFmpeg's pool right away (0 VRAM/GTT held by replay ring).
                    AVPacket *cpu_pkt = nullptr;
                    {
                        std::lock_guard lk(replay_mtx);
                        if (!free_pkts.empty()) {
                            cpu_pkt = free_pkts.back();
                            free_pkts.pop_back();
                        }
                    }
                    if (!cpu_pkt) {
                        cpu_pkt = av_packet_alloc();
                        if (!cpu_pkt) {
                            av_packet_unref(p);
                            scratch = p;
                            return std::unexpected("av_packet_alloc(replay) failed");
                        }
                    }
                    if (av_new_packet(cpu_pkt, p->size) < 0 || av_packet_copy_props(cpu_pkt, p) < 0) {
                        av_packet_free(&cpu_pkt);
                        av_packet_unref(p);
                        scratch = p;
                        return std::unexpected("av_new_packet(replay) failed");
                    }
                    if (p->size > 0) {
                        std::memcpy(cpu_pkt->data, p->data, static_cast<size_t>(p->size));
                    }
                    av_packet_unref(p);
                    scratch = p;

                    std::lock_guard lk(replay_mtx);
                    push_replay_packet_locked(cpu_pkt, true);
                } else {
                    // Audio packets (Opus/AAC/FLAC) are already plain CPU buffers: zero-copy push
                    std::lock_guard lk(replay_mtx);
                    push_replay_packet_locked(p, false);
                    if (!free_pkts.empty()) {
                        scratch = free_pkts.back();
                        free_pkts.pop_back();
                    }
                }
                continue;
            }

            p->stream_index = st->index;
            av_packet_rescale_ts(p, ctx->time_base, st->time_base);

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
        stop_saver();
        clear_replay_buffers();
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
    impl->is_replay = (cfg.replay_seconds > 0);

    int err = avformat_alloc_output_context2(&impl->fmt_ctx, nullptr, nullptr, cfg.output_path.c_str());
    if (err < 0 || !impl->fmt_ctx) {
        return std::unexpected(std::format("Could not determine output format for '{}': {}",
                                           cfg.output_path, ff_err(err)));
    }

    // Probe GPU Vulkan Video encode extensions & quality levels for cross-vendor / older-GPU compatibility
    Codec chosen_codec = cfg.codec;
    VkEncodeCaps vcaps = probe_vk_encode_caps(hw_device_ref, chosen_codec, cfg.pixfmt);
    if (vcaps.probed) {
        if (!vcaps.has_av1 && !vcaps.has_hevc && !vcaps.has_h264) {
            return std::unexpected(no_encode_support_error());
        }
        if (!cfg.codec_explicit) {
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
    const int auto_q = (max_q <= 1) ? max_q : std::max(1, (max_q + 1) / 2);
    const int eff_quality = (cfg.quality >= 0) ? std::min(cfg.quality, max_q) : auto_q;
    const int eff_b_depth = (!cfg.b_depth_explicit && vcaps.probed && !vcaps.supports_b_frames)
        ? 1 : std::max(1, cfg.b_depth);

    impl->enc_ctx->width = static_cast<int>(width);
    impl->enc_ctx->height = static_cast<int>(height);
    impl->enc_ctx->time_base = AVRational{1, static_cast<int>(cfg.fps)};
    impl->enc_ctx->framerate = AVRational{static_cast<int>(cfg.fps), 1};
    impl->enc_ctx->sample_aspect_ratio = AVRational{1, 1};
    impl->enc_ctx->pix_fmt = AV_PIX_FMT_VULKAN;
    impl->enc_ctx->gop_size = eff_idr;
    impl->enc_ctx->max_b_frames = std::max(0, eff_b_depth - 1);

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
        if (cfg.qp >= 0) eff_rc = RateControl::Cqp;
        else if (cfg.bitrate > 0) eff_rc = RateControl::Vbr;
        else eff_rc = RateControl::Cqp;
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
            impl->enc_ctx->rc_buffer_size = static_cast<int>(
                std::min<int64_t>(impl->enc_ctx->bit_rate, std::numeric_limits<int>::max()));
            break;
        case RateControl::Vbr:
            av_dict_set(&opts, "rc_mode", "vbr", 0);
            impl->enc_ctx->bit_rate = cfg.bitrate > 0 ? cfg.bitrate : 20'000'000;
            impl->enc_ctx->rc_max_rate = cfg.max_bitrate > 0 ? cfg.max_bitrate : (impl->enc_ctx->bit_rate * 3) / 2;
            // 2-second VBV buffer gives local recording rate control headroom during high-motion bursts
            impl->enc_ctx->rc_buffer_size = static_cast<int>(
                std::min<int64_t>(impl->enc_ctx->rc_max_rate * 2LL, std::numeric_limits<int>::max()));
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
    av_dict_set_int(&opts, "b_depth", eff_b_depth, 0);

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

    if (impl->is_replay) {
        // Instant replay mode: do not open output file yet; packets stay in rolling RAM ring
        // and are muxed asynchronously on a low-priority thread only when save_replay_async() is called.
        impl->free_pkts.reserve(32);
        impl->start_saver();
        if (cfg.verbose) {
            std::cerr << std::format(
                "[encoder] opened {} {}x{} pixfmt={} qp={} quality={} gop={} async_depth={} (instant-replay {}s)\n",
                cname, width, height,
                cfg.pixfmt == PixelFormat::P010 ? "p010" : "nv12",
                eff_qp, eff_quality, eff_idr, cfg.async_depth, cfg.replay_seconds);
        }
        return enc;
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
        // Fragmented MP4: crash-safe on every keyframe (2s GOP) and instant 0 ms stop on Ctrl+C
        // instead of rewriting multi-GB files via +faststart.
        av_dict_set(&mux_opts, "movflags", "+frag_keyframe+empty_moov+default_base_moof", 0);
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
        impl_->encode_failed = true;
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

    std::expected<void, std::string> res{};
    if (!impl_->encode_failed) {
        int err = avcodec_send_frame(impl_->enc_ctx, nullptr);
        if (err < 0 && err != AVERROR_EOF) {
            impl_->stop_writer();
            impl_->stop_saver();
            impl_->clear_replay_buffers();
            return std::unexpected(std::format("avcodec_send_frame(flush) failed: {}", ff_err(err)));
        }
        res = impl_->drain_packets();
    }

    std::expected<void, std::string> ares{};
    if (impl_->audio_enc_ctx) {
        std::lock_guard lk(impl_->audio_enc_mtx);
        int aerr = avcodec_send_frame(impl_->audio_enc_ctx, nullptr);
        if (aerr == 0 || aerr == AVERROR_EOF) {
            ares = impl_->drain_packets_from(impl_->audio_enc_ctx, impl_->audio_stream, impl_->audio_scratch_pkt);
        }
    }

    impl_->stop_writer();
    impl_->stop_saver();
    impl_->clear_replay_buffers();
    if (!res) return res;
    if (!ares) return ares;
    if (!impl_->writer_err.empty()) {
        return std::unexpected(impl_->writer_err);
    }

    if (impl_->header_written) {
        int terr = av_write_trailer(impl_->fmt_ctx);
        if (terr < 0) {
            return std::unexpected(std::format("av_write_trailer failed: {}", ff_err(terr)));
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

bool VideoEncoder::replay_mode() const noexcept {
    return impl_->is_replay;
}

ReplayStats VideoEncoder::replay_stats() const noexcept {
    if (!impl_->is_replay) return {};
    std::lock_guard lk(impl_->replay_mtx);
    double sec = 0.0;
    if (impl_->replay_v_pkts.size() >= 2) {
        const int64_t span = impl_->replay_v_pkts.back()->pts - impl_->replay_v_pkts.front()->pts;
        sec = static_cast<double>(std::max<int64_t>(0, span)) * av_q2d(impl_->enc_ctx->time_base);
    }
    return {
        impl_->replay_bytes,
        sec,
        impl_->replay_clips_saved.load(std::memory_order_relaxed),
    };
}

std::expected<std::string, std::string> VideoEncoder::save_replay_async(std::string_view out_path_override) {
    if (!impl_->is_replay) {
        return std::unexpected("save_replay_async() called when instant replay mode is disabled");
    }

    Impl::SaveJob job{};
    {
        std::lock_guard lk(impl_->replay_mtx);
        if (impl_->replay_v_pkts.empty()) {
            return std::unexpected("Replay buffer is empty (no encoded frames yet)");
        }

        // Start from the oldest keyframe inside the rolling buffer
        size_t first_kf = impl_->replay_v_pkts.size();
        for (size_t i = 0; i < impl_->replay_v_pkts.size(); ++i) {
            if ((impl_->replay_v_pkts[i]->flags & AV_PKT_FLAG_KEY) != 0) {
                first_kf = i;
                break;
            }
        }
        if (first_kf == impl_->replay_v_pkts.size()) {
            return std::unexpected("Replay buffer has no keyframe yet");
        }

        const int64_t v_base_pts = impl_->replay_v_pkts[first_kf]->pts;
        const int64_t v_base_dts = (impl_->replay_v_pkts[first_kf]->dts != AV_NOPTS_VALUE)
            ? impl_->replay_v_pkts[first_kf]->dts : v_base_pts;
        // Shift both PTS and DTS by the same origin to preserve exact CTS (pts - dts) with B-frames
        const int64_t v_origin = std::min(v_base_pts, v_base_dts);
        const int64_t v_last_pts = impl_->replay_v_pkts.back()->pts;
        job.duration_sec = static_cast<double>(std::max<int64_t>(0, v_last_pts - v_base_pts + 1)) *
                           av_q2d(impl_->enc_ctx->time_base);

        // Zero-copy shallow packet references (AVBufferRef refcount++, no bitstream memcpy)
        job.v_pkts.reserve(impl_->replay_v_pkts.size() - first_kf);
        for (size_t i = first_kf; i < impl_->replay_v_pkts.size(); ++i) {
            AVPacket *ref = av_packet_alloc();
            if (!ref || av_packet_ref(ref, impl_->replay_v_pkts[i]) < 0) {
                if (ref) av_packet_free(&ref);
                Impl::free_save_job(job);
                return std::unexpected("av_packet_ref(video) failed");
            }
            if (ref->pts != AV_NOPTS_VALUE) ref->pts -= v_origin;
            if (ref->dts != AV_NOPTS_VALUE) ref->dts -= v_origin;
            job.v_pkts.push_back(ref);
        }

        if (impl_->audio_enc_ctx && !impl_->replay_a_pkts.empty()) {
            const int64_t a_start_pts = av_rescale_q(v_origin, impl_->enc_ctx->time_base,
                                                     impl_->audio_enc_ctx->time_base);
            job.a_pkts.reserve(impl_->replay_a_pkts.size());
            for (AVPacket *src : impl_->replay_a_pkts) {
                if (src->pts < a_start_pts) continue;
                AVPacket *ref = av_packet_alloc();
                if (!ref || av_packet_ref(ref, src) < 0) {
                    if (ref) av_packet_free(&ref);
                    Impl::free_save_job(job);
                    return std::unexpected("av_packet_ref(audio) failed");
                }
                if (ref->pts != AV_NOPTS_VALUE) ref->pts -= a_start_pts;
                if (ref->dts != AV_NOPTS_VALUE) ref->dts -= a_start_pts;
                job.a_pkts.push_back(ref);
            }
        }
    }

    job.v_par = avcodec_parameters_alloc();
    if (!job.v_par || avcodec_parameters_from_context(job.v_par, impl_->enc_ctx) < 0) {
        Impl::free_save_job(job);
        return std::unexpected("Failed to copy replay video codec parameters");
    }
    job.v_tb = impl_->enc_ctx->time_base;
    job.v_fps = impl_->enc_ctx->framerate;

    if (impl_->audio_enc_ctx) {
        job.a_par = avcodec_parameters_alloc();
        if (!job.a_par || avcodec_parameters_from_context(job.a_par, impl_->audio_enc_ctx) < 0) {
            Impl::free_save_job(job);
            return std::unexpected("Failed to copy replay audio codec parameters");
        }
        job.a_tb = impl_->audio_enc_ctx->time_base;
    }

    if (!out_path_override.empty()) {
        job.out_path = expand_user_path(out_path_override);
    } else {
        std::filesystem::path base_p(expand_user_path(impl_->cfg.output_path));
        std::string ext = base_p.extension().string();
        if (ext != ".mkv" && ext != ".mp4" && ext != ".webm" && ext != ".ivf") ext = ".mkv";
        std::time_t now = std::time(nullptr);
        std::tm tm{};
        localtime_r(&now, &tm);
        char ts[64];
        std::strftime(ts, sizeof(ts), "vkcap_replay_%Y%m%d_%H%M%S", &tm);
        const uint32_t seq = ++impl_->replay_save_seq;
        std::string fname = (seq > 1)
            ? std::format("{}_{}{}", ts, seq, ext)
            : std::format("{}{}", ts, ext);
        if (base_p.has_parent_path()) {
            job.out_path = (base_p.parent_path() / fname).string();
        } else {
            job.out_path = fname;
        }
    }

    const std::string out_path = job.out_path;
    {
        std::lock_guard lk(impl_->save_mtx);
        impl_->save_queue.push_back(std::move(job));
    }
    impl_->save_cv.notify_one();
    return out_path;
}

EncoderStats VideoEncoder::stats() const noexcept {
    return {
        impl_->frames_submitted.load(std::memory_order_relaxed),
        impl_->packets_written.load(std::memory_order_relaxed),
        impl_->bytes_written.load(std::memory_order_relaxed),
    };
}

}  // namespace vkcap
