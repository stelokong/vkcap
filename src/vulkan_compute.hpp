#pragma once

#include "cli.hpp"
#include "types.hpp"

#include <array>
#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vulkan/vulkan.h>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

namespace vkcap {

struct ComputeStats {
    uint64_t frames_processed = 0;
    uint64_t total_compute_ns = 0;
};

struct GpuEncodeSupport {
    std::string device_name;
    bool probed = false;
    CodecEncodeCaps av1{true, true, true, true, true, 2, 0, 0, 0};
    CodecEncodeCaps hevc{true, true, true, true, true, 2, 0, 0, 0};
    CodecEncodeCaps h264{true, false, true, true, true, 2, 0, 0, 0};

    bool has_any_codec() const noexcept {
        return av1.supported || hevc.supported || h264.supported;
    }

    void apply_to(ProfileTarget &target) const {
        target.av1 = av1;
        target.hevc = hevc;
        target.h264 = h264;
        target.gpu_name = device_name;
    }
};

/// Probes the target Vulkan GPU's hardware Vulkan Video encode extensions and quality levels
/// without requiring elevated privileges (used for hardware-aware profile recommendations).
GpuEncodeSupport probe_gpu_encode_support(uint32_t device_index = 0,
                                          bool explicit_dev = false,
                                          std::string_view drm_card_path = {});

class VulkanCompute {
public:
    VulkanCompute() = default;
    ~VulkanCompute();

    VulkanCompute(const VulkanCompute &) = delete;
    VulkanCompute &operator=(const VulkanCompute &) = delete;

    static std::expected<std::unique_ptr<VulkanCompute>, std::string> create(
        const Config &cfg,
        bool convert_gamut = false,
        const std::array<float, 9> &gamut_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1},
        std::string_view drm_card_path = {});

    /// Initializes or re-initializes the FFmpeg AVHWFramesContext for the target video size.
    std::expected<void, std::string> configure_output(uint32_t width, uint32_t height);

    /// Imports/looks up the KMS primary (and optional cursor) DMA-BUF surfaces on the isolated
    /// low-priority compute_dev, submits rgb_to_yuv.comp asynchronously into an idle AVVkFrame
    /// alias, and returns immediately without stalling the capture thread on the GPU fence.
    /// Caller MUST call wait_slot(sf.slot_idx) before encoding or freeing sf.frame.
    struct SubmittedFrame {
        AVFrame *frame = nullptr;
        uint32_t slot_idx = 0;
    };
    std::expected<SubmittedFrame, std::string> submit_frame(const KmsFrame &frame);

    /// Waits for the compute fence of `slot_idx` to complete and releases the slot for reuse.
    /// Safe to call from the encode thread concurrently with submit_frame() on the capture thread.
    void wait_slot(uint32_t slot_idx) noexcept;

    /// Synchronous convenience wrapper: calls submit_frame() + wait_slot().
    std::expected<AVFrame *, std::string> process_frame(const KmsFrame &frame);

    AVBufferRef *hw_device_ref() const noexcept;
    AVBufferRef *hw_frames_ref() const noexcept;
    uint32_t output_width() const noexcept;
    uint32_t output_height() const noexcept;
    ComputeStats stats() const noexcept;

    /// Helper for GPU integration tests: allocates an exportable DMA-BUF image on compute_dev.
    struct TestDmaBuf {
        int fd = -1;
        uint64_t inode_key = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t stride = 0;
        uint64_t modifier = 0;
        VkImage img = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
    };
    std::expected<TestDmaBuf, std::string> create_test_dmabuf(uint32_t width, uint32_t height, uint32_t frame_idx, bool is_cursor = false, bool is_fp16 = false);
    void destroy_test_dmabuf(TestDmaBuf &buf);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vkcap
