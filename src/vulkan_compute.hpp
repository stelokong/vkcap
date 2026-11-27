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
    bool has_av1 = true;
    bool has_hevc = true;
    bool has_h264 = true;
    bool supports_10bit = true;
    bool supports_vbr = true;
    bool supports_cqp = true;
    bool supports_cbr = true;
    uint32_t max_quality_levels = 2;
    uint32_t max_coded_width = 0;
    uint32_t max_coded_height = 0;
    uint64_t max_hw_bitrate = 0;
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
    /// low-priority compute_dev, runs rgb_to_yuv.comp into an idle AVVkFrame alias, and returns
    /// the GPU-resident AVFrame ready for Vulkan Video encoding.
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
    std::expected<TestDmaBuf, std::string> create_test_dmabuf(uint32_t width, uint32_t height, uint32_t frame_idx, bool is_cursor = false);
    void destroy_test_dmabuf(TestDmaBuf &buf);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vkcap
