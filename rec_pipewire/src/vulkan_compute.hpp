#pragma once

#include "cli.hpp"
#include "types.hpp"

#include <expected>
#include <memory>
#include <string>
#include <vector>
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

class VulkanCompute {
public:
    VulkanCompute() = default;
    ~VulkanCompute();

    VulkanCompute(const VulkanCompute &) = delete;
    VulkanCompute &operator=(const VulkanCompute &) = delete;

    static std::expected<std::unique_ptr<VulkanCompute>, std::string> create(const Config &cfg);

    /// Returns Vulkan-supported DRM formats & modifiers ordered by preference.
    std::vector<FormatModifierSet> query_supported_capture_formats() const;

    /// Initializes or re-initializes the FFmpeg AVHWFramesContext for the target video size.
    std::expected<void, std::string> configure_output(uint32_t width, uint32_t height);

    /// Imports/looks up the captured frame, picks an idle AVVkFrame from the pool, runs the
    /// RGB->NV12/P010 compute shader on the async compute queue, and waits ONLY for compute
    /// completion (~0.1-0.3ms) so the caller can return the PipeWire pw_buffer immediately.
    /// Returns the GPU-resident AVFrame ready for the dedicated av1_vulkan encode thread.
    std::expected<AVFrame *, std::string> process_frame(const CapturedFrame &frame);

    /// Evicts any cached Vulkan image/memory imported from this PipeWire buffer handle.
    void evict_buffer(void *buffer_handle);

    AVBufferRef *hw_device_ref() const noexcept;
    AVBufferRef *hw_frames_ref() const noexcept;
    uint32_t output_width() const noexcept;
    uint32_t output_height() const noexcept;
    ComputeStats stats() const noexcept;

    /// Helper for tests: creates an exportable DMA-BUF image filled with a solid/pattern RGBA color.
    struct TestDmaBuf {
        int fd = -1;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t stride = 0;
        uint64_t modifier = 0;
        VkImage img = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
    };
    std::expected<TestDmaBuf, std::string> create_test_dmabuf(uint32_t width, uint32_t height, uint32_t frame_idx);
    void destroy_test_dmabuf(TestDmaBuf &buf);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vkcap
