#include "vulkan_compute.hpp"

#include "rgb_to_yuv_nv12_spv.hpp"
#include "rgb_to_yuv_p010_spv.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <iostream>
#include <libdrm/drm_fourcc.h>
#include <linux/dma-buf.h>
#include <mutex>
#include <poll.h>
#include <spa/param/video/raw.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>

namespace vkcap {

namespace {

struct FormatCandidate {
    uint32_t spa_format;
    uint32_t drm_fourcc;
    VkFormat vk_format;
    VkComponentMapping swizzle;
    bool is_10bit_or_hdr;
};

constexpr VkComponentMapping kSwizzleIdentity = {
    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};

constexpr VkComponentMapping kSwizzleOpaqueAlpha = {
    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_ONE};

constexpr std::array<FormatCandidate, 9> kCandidates = {{
    {SPA_VIDEO_FORMAT_BGRx,       DRM_FORMAT_XRGB8888,        VK_FORMAT_B8G8R8A8_UNORM,           kSwizzleOpaqueAlpha, false},
    {SPA_VIDEO_FORMAT_BGRA,       DRM_FORMAT_ARGB8888,        VK_FORMAT_B8G8R8A8_UNORM,           kSwizzleIdentity,    false},
    {SPA_VIDEO_FORMAT_RGBx,       DRM_FORMAT_XBGR8888,        VK_FORMAT_R8G8B8A8_UNORM,           kSwizzleOpaqueAlpha, false},
    {SPA_VIDEO_FORMAT_RGBA,       DRM_FORMAT_ABGR8888,        VK_FORMAT_R8G8B8A8_UNORM,           kSwizzleIdentity,    false},
    {SPA_VIDEO_FORMAT_xRGB_210LE, DRM_FORMAT_XRGB2101010,     VK_FORMAT_A2R10G10B10_UNORM_PACK32, kSwizzleOpaqueAlpha, true},
    {SPA_VIDEO_FORMAT_ARGB_210LE, DRM_FORMAT_ARGB2101010,     VK_FORMAT_A2R10G10B10_UNORM_PACK32, kSwizzleIdentity,    true},
    {SPA_VIDEO_FORMAT_xBGR_210LE, DRM_FORMAT_XBGR2101010,     VK_FORMAT_A2B10G10R10_UNORM_PACK32, kSwizzleOpaqueAlpha, true},
    {SPA_VIDEO_FORMAT_ABGR_210LE, DRM_FORMAT_ABGR2101010,     VK_FORMAT_A2B10G10R10_UNORM_PACK32, kSwizzleIdentity,    true},
    {SPA_VIDEO_FORMAT_RGBA_F16,   DRM_FORMAT_ABGR16161616F,   VK_FORMAT_R16G16B16A16_SFLOAT,      kSwizzleIdentity,    true},
}};

const FormatCandidate *find_candidate(uint32_t spa_format) {
    for (const auto &c : kCandidates) {
        if (c.spa_format == spa_format) return &c;
    }
    return nullptr;
}

struct PushConstants {
    float yCoeffs[4];
    float cbCoeffs[4];
    float crCoeffs[4];
    float srcOffset[2];
    float srcScale[2];
    float srcBaseUv[2];
    float invSrcSize[2];
    uint32_t dstExtent[2];
    uint32_t flags;     // bit 0: exact1To1, bit 1: enableDither
    uint32_t reserved;
};

struct CachedSrcImage {
    VkImage image = VK_NULL_HANDLE;
    std::array<VkDeviceMemory, 4> mem{};
    uint32_t num_mems = 0;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t spa_format = 0;
    uint64_t modifier = DRM_FORMAT_MOD_INVALID;
    bool has_modifier = false;
};

struct CachedDstAlias {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView y_view = VK_NULL_HANDLE;
    VkImageView uv_view = VK_NULL_HANDLE;
    bool initialized_layout = false;
};

struct CmdSlot {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore ext_wait_sem = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
};

}  // namespace

struct VulkanCompute::Impl {
    Config cfg;
    AVBufferRef *hw_device_ref = nullptr;
    AVBufferRef *hw_frames_ref = nullptr;
    AVVulkanDeviceContext *vkctx = nullptr;

    // Isolated compute-only VkDevice on the same VkPhysicalDevice so av1_vulkan's VCN
    // submissions on vkctx->act_dev NEVER attach implicit dma_resv fences to KWin's DMA-BUFs!
    VkDevice compute_dev = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props{};
    uint32_t compute_qf_idx = 0;
    VkQueue compute_queue = VK_NULL_HANDLE;

    // Function pointers on FFmpeg's instance / vkctx->act_dev
    PFN_vkGetDeviceProcAddr pfn_get_dev_proc = nullptr;
    PFN_vkCreateDevice pfn_create_device = nullptr;
    PFN_vkDestroyDevice pfn_destroy_device = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties pfn_enum_dev_exts = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties2 pfn_get_fmt_props2 = nullptr;
    PFN_vkGetPhysicalDeviceImageFormatProperties2 pfn_get_img_fmt_props2 = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties pfn_get_mem_props = nullptr;
    PFN_vkGetMemoryFdKHR pfn_ff_get_mem_fd = nullptr;
    PFN_vkGetSemaphoreCounterValue pfn_ff_get_sem_val = nullptr;
    PFN_vkWaitSemaphores pfn_ff_wait_sems = nullptr;

    // Function pointers on our isolated compute_dev
    PFN_vkCreateImage pfn_create_image = nullptr;
    PFN_vkDestroyImage pfn_destroy_image = nullptr;
    PFN_vkGetImageMemoryRequirements2 pfn_get_img_mem_req2 = nullptr;
    PFN_vkBindImageMemory2 pfn_bind_image_mem2 = nullptr;
    PFN_vkAllocateMemory pfn_alloc_mem = nullptr;
    PFN_vkFreeMemory pfn_free_mem = nullptr;
    PFN_vkGetMemoryFdPropertiesKHR pfn_get_mem_fd_props = nullptr;
    PFN_vkGetMemoryFdKHR pfn_get_mem_fd = nullptr;
    PFN_vkCreateImageView pfn_create_img_view = nullptr;
    PFN_vkDestroyImageView pfn_destroy_img_view = nullptr;
    PFN_vkCreateSampler pfn_create_sampler = nullptr;
    PFN_vkDestroySampler pfn_destroy_sampler = nullptr;
    PFN_vkCreateShaderModule pfn_create_shader = nullptr;
    PFN_vkDestroyShaderModule pfn_destroy_shader = nullptr;
    PFN_vkCreateDescriptorSetLayout pfn_create_dsl = nullptr;
    PFN_vkDestroyDescriptorSetLayout pfn_destroy_dsl = nullptr;
    PFN_vkCreatePipelineLayout pfn_create_pl = nullptr;
    PFN_vkDestroyPipelineLayout pfn_destroy_pl = nullptr;
    PFN_vkCreateComputePipelines pfn_create_comp_pipe = nullptr;
    PFN_vkDestroyPipeline pfn_destroy_pipe = nullptr;
    PFN_vkCreateDescriptorPool pfn_create_dp = nullptr;
    PFN_vkDestroyDescriptorPool pfn_destroy_dp = nullptr;
    PFN_vkAllocateDescriptorSets pfn_alloc_ds = nullptr;
    PFN_vkUpdateDescriptorSets pfn_update_ds = nullptr;
    PFN_vkCreateCommandPool pfn_create_cp = nullptr;
    PFN_vkDestroyCommandPool pfn_destroy_cp = nullptr;
    PFN_vkAllocateCommandBuffers pfn_alloc_cb = nullptr;
    PFN_vkBeginCommandBuffer pfn_begin_cb = nullptr;
    PFN_vkEndCommandBuffer pfn_end_cb = nullptr;
    PFN_vkResetCommandPool pfn_reset_cp = nullptr;
    PFN_vkCmdPipelineBarrier2 pfn_cmd_barrier2 = nullptr;
    PFN_vkCmdBindPipeline pfn_cmd_bind_pipe = nullptr;
    PFN_vkCmdBindDescriptorSets pfn_cmd_bind_ds = nullptr;
    PFN_vkCmdPushDescriptorSetKHR pfn_cmd_push_desc = nullptr;
    PFN_vkCmdPushConstants pfn_cmd_push = nullptr;
    PFN_vkCmdDispatch pfn_cmd_dispatch = nullptr;
    PFN_vkCmdCopyBufferToImage pfn_cmd_copy_buf_to_img = nullptr;
    PFN_vkCmdClearColorImage pfn_cmd_clear_color = nullptr;
    PFN_vkCreateFence pfn_create_fence = nullptr;
    PFN_vkDestroyFence pfn_destroy_fence = nullptr;
    PFN_vkWaitForFences pfn_wait_fences = nullptr;
    PFN_vkResetFences pfn_reset_fences = nullptr;
    PFN_vkCreateSemaphore pfn_create_sem = nullptr;
    PFN_vkDestroySemaphore pfn_destroy_sem = nullptr;
    PFN_vkImportSemaphoreFdKHR pfn_import_sem_fd = nullptr;
    PFN_vkQueueSubmit2 pfn_queue_submit2 = nullptr;
    PFN_vkDeviceWaitIdle pfn_device_wait_idle = nullptr;
    PFN_vkGetDeviceQueue pfn_get_dev_queue = nullptr;
    PFN_vkCreateBuffer pfn_create_buf = nullptr;
    PFN_vkDestroyBuffer pfn_destroy_buf = nullptr;
    PFN_vkGetBufferMemoryRequirements pfn_get_buf_mem_req = nullptr;
    PFN_vkBindBufferMemory pfn_bind_buf_mem = nullptr;
    PFN_vkMapMemory pfn_map_mem = nullptr;
    PFN_vkUnmapMemory pfn_unmap_mem = nullptr;
    PFN_vkGetImageDrmFormatModifierPropertiesEXT pfn_get_img_drm_mod = nullptr;

    bool compute_has_video_enc_ext = false;
    bool use_push_descriptors = false;

    VkSampler linear_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkPipeline pipe_nv12 = VK_NULL_HANDLE;
    VkPipeline pipe_p010 = VK_NULL_HANDLE;

    static constexpr size_t kNumSlots = 4;
    std::array<CmdSlot, kNumSlots> slots{};
    size_t slot_cursor = 0;

    std::mutex cache_mtx;
    std::unordered_map<void *, CachedSrcImage> src_cache;
    std::unordered_map<VkImage, CachedDstAlias> dst_alias_cache;

    // Reusable SHM staging buffer + device image when compositor falls back to MemPtr/MemFd
    VkBuffer shm_buf = VK_NULL_HANDLE;
    VkDeviceMemory shm_mem = VK_NULL_HANDLE;
    void *shm_mapped = nullptr;
    size_t shm_buf_size = 0;
    CachedSrcImage shm_img{};

    uint32_t out_width = 0;
    uint32_t out_height = 0;
    ComputeStats stats{};

    template <typename T>
    T load_inst(const char *name) const {
        return reinterpret_cast<T>(vkctx->get_proc_addr(vkctx->inst, name));
    }

    template <typename T>
    T load_dev(const char *name) const {
        return reinterpret_cast<T>(pfn_get_dev_proc(compute_dev, name));
    }

    int find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags req_flags) const {
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
            if ((type_bits & (1u << i)) &&
                (mem_props.memoryTypes[i].propertyFlags & req_flags) == req_flags) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    void destroy_src_image(CachedSrcImage &s) {
        if (s.view) pfn_destroy_img_view(compute_dev, s.view, nullptr);
        if (s.image) pfn_destroy_image(compute_dev, s.image, nullptr);
        for (uint32_t i = 0; i < s.num_mems; ++i) {
            if (s.mem[i]) pfn_free_mem(compute_dev, s.mem[i], nullptr);
        }
        s = {};
    }

    void destroy_dst_alias(CachedDstAlias &d) {
        if (d.y_view) pfn_destroy_img_view(compute_dev, d.y_view, nullptr);
        if (d.uv_view) pfn_destroy_img_view(compute_dev, d.uv_view, nullptr);
        if (d.image) pfn_destroy_image(compute_dev, d.image, nullptr);
        if (d.mem) pfn_free_mem(compute_dev, d.mem, nullptr);
        d = {};
    }

    void clear_caches() {
        if (pfn_device_wait_idle && compute_dev) {
            pfn_device_wait_idle(compute_dev);
        }
        std::lock_guard lk(cache_mtx);
        for (auto &[_, img] : src_cache) {
            destroy_src_image(img);
        }
        src_cache.clear();
        for (auto &[_, da] : dst_alias_cache) {
            destroy_dst_alias(da);
        }
        dst_alias_cache.clear();
    }

    ~Impl() {
        if (!compute_dev) {
            av_buffer_unref(&hw_frames_ref);
            av_buffer_unref(&hw_device_ref);
            return;
        }
        clear_caches();
        destroy_src_image(shm_img);
        if (shm_mapped) pfn_unmap_mem(compute_dev, shm_mem);
        if (shm_buf) pfn_destroy_buf(compute_dev, shm_buf, nullptr);
        if (shm_mem) pfn_free_mem(compute_dev, shm_mem, nullptr);

        for (auto &sl : slots) {
            if (sl.ext_wait_sem) pfn_destroy_sem(compute_dev, sl.ext_wait_sem, nullptr);
            if (sl.fence) pfn_destroy_fence(compute_dev, sl.fence, nullptr);
            if (sl.desc_pool) pfn_destroy_dp(compute_dev, sl.desc_pool, nullptr);
            if (sl.pool) pfn_destroy_cp(compute_dev, sl.pool, nullptr);
        }
        if (pipe_nv12) pfn_destroy_pipe(compute_dev, pipe_nv12, nullptr);
        if (pipe_p010) pfn_destroy_pipe(compute_dev, pipe_p010, nullptr);
        if (pipe_layout) pfn_destroy_pl(compute_dev, pipe_layout, nullptr);
        if (dsl) pfn_destroy_dsl(compute_dev, dsl, nullptr);
        if (linear_sampler) pfn_destroy_sampler(compute_dev, linear_sampler, nullptr);

        pfn_destroy_device(compute_dev, nullptr);
        compute_dev = VK_NULL_HANDLE;

        av_buffer_unref(&hw_frames_ref);
        av_buffer_unref(&hw_device_ref);
    }

    std::expected<void, std::string> init_vulkan_objects() {
        pfn_get_dev_proc = load_inst<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
        pfn_create_device = load_inst<PFN_vkCreateDevice>("vkCreateDevice");
        pfn_enum_dev_exts = load_inst<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
        pfn_get_fmt_props2 = load_inst<PFN_vkGetPhysicalDeviceFormatProperties2>("vkGetPhysicalDeviceFormatProperties2");
        pfn_get_img_fmt_props2 = load_inst<PFN_vkGetPhysicalDeviceImageFormatProperties2>("vkGetPhysicalDeviceImageFormatProperties2");
        pfn_get_mem_props = load_inst<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties");

        pfn_ff_get_mem_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(pfn_get_dev_proc(vkctx->act_dev, "vkGetMemoryFdKHR"));
        pfn_ff_get_sem_val = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(pfn_get_dev_proc(vkctx->act_dev, "vkGetSemaphoreCounterValue"));
        pfn_ff_wait_sems = reinterpret_cast<PFN_vkWaitSemaphores>(pfn_get_dev_proc(vkctx->act_dev, "vkWaitSemaphores"));

        pfn_get_mem_props(vkctx->phys_dev, &mem_props);

        // Pick a dedicated compute-only queue family (e.g. RADV async compute qf[1])
        bool found_compute = false;
        for (int i = 0; i < vkctx->nb_qf; ++i) {
            const auto &q = vkctx->qf[i];
            if ((q.flags & VK_QUEUE_COMPUTE_BIT) && !(q.flags & VK_QUEUE_GRAPHICS_BIT)) {
                compute_qf_idx = static_cast<uint32_t>(q.idx);
                found_compute = true;
                break;
            }
        }
        if (!found_compute) {
            for (int i = 0; i < vkctx->nb_qf; ++i) {
                if (vkctx->qf[i].flags & VK_QUEUE_COMPUTE_BIT) {
                    compute_qf_idx = static_cast<uint32_t>(vkctx->qf[i].idx);
                    found_compute = true;
                    break;
                }
            }
        }
        if (!found_compute) {
            return std::unexpected("No Vulkan compute queue family exposed by AVVulkanDeviceContext");
        }

        // Create an isolated compute-only VkDevice on the same VkPhysicalDevice so av1_vulkan's
        // VCN encode submissions on vkctx->act_dev NEVER pollute KWin's imported DMA-BUF dma_resv!
        uint32_t ext_cnt = 0;
        pfn_enum_dev_exts(vkctx->phys_dev, nullptr, &ext_cnt, nullptr);
        std::vector<VkExtensionProperties> avail_exts(ext_cnt);
        pfn_enum_dev_exts(vkctx->phys_dev, nullptr, &ext_cnt, avail_exts.data());
        auto has_ext = [&](const char *name) {
            for (const auto &e : avail_exts) {
                if (std::strcmp(e.extensionName, name) == 0) return true;
            }
            return false;
        };

        std::vector<const char *> req_exts = {
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
            VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
            VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        };
        if (has_ext(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) &&
            has_ext(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME)) {
            req_exts.push_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
            req_exts.push_back(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);
            if (has_ext(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME)) {
                req_exts.push_back(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME);
            }
            compute_has_video_enc_ext = true;
        }
        if (has_ext(VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME)) {
            req_exts.push_back(VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME);
        }
        if (has_ext(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
            req_exts.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
            use_push_descriptors = true;
        }
        const bool has_global_prio = has_ext(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
        if (has_global_prio) {
            req_exts.push_back(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
        }

        // Use low queue priority so active 3D games / Wayland compositor always preempt
        // our ~0.1ms compute shader rather than contending on CUs.
        float q_prio = 0.0f;
        VkDeviceQueueGlobalPriorityCreateInfoKHR gprio{};
        gprio.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR;
        gprio.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR;

        VkDeviceQueueCreateInfo dqci{};
        dqci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqci.pNext = has_global_prio ? &gprio : nullptr;
        dqci.queueFamilyIndex = compute_qf_idx;
        dqci.queueCount = 1;
        dqci.pQueuePriorities = &q_prio;

        VkPhysicalDeviceVulkan13Features f13{};
        f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        f13.synchronization2 = VK_TRUE;

        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        f12.pNext = &f13;
        f12.timelineSemaphore = VK_TRUE;

        VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcr_feat{};
        ycbcr_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES;
        ycbcr_feat.pNext = &f12;
        ycbcr_feat.samplerYcbcrConversion = VK_TRUE;

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.pNext = &ycbcr_feat;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &dqci;
        dci.enabledExtensionCount = static_cast<uint32_t>(req_exts.size());
        dci.ppEnabledExtensionNames = req_exts.data();

        if (pfn_create_device(vkctx->phys_dev, &dci, nullptr, &compute_dev) != VK_SUCCESS) {
            // Fallback without global priority if driver/kernel rejects LOW global priority
            dqci.pNext = nullptr;
            if (pfn_create_device(vkctx->phys_dev, &dci, nullptr, &compute_dev) != VK_SUCCESS) {
                return std::unexpected("Failed to create isolated Vulkan compute device");
            }
        }

        pfn_destroy_device = load_dev<PFN_vkDestroyDevice>("vkDestroyDevice");
        pfn_create_image = load_dev<PFN_vkCreateImage>("vkCreateImage");
        pfn_destroy_image = load_dev<PFN_vkDestroyImage>("vkDestroyImage");
        pfn_get_img_mem_req2 = load_dev<PFN_vkGetImageMemoryRequirements2>("vkGetImageMemoryRequirements2");
        pfn_bind_image_mem2 = load_dev<PFN_vkBindImageMemory2>("vkBindImageMemory2");
        pfn_alloc_mem = load_dev<PFN_vkAllocateMemory>("vkAllocateMemory");
        pfn_free_mem = load_dev<PFN_vkFreeMemory>("vkFreeMemory");
        pfn_get_mem_fd_props = load_dev<PFN_vkGetMemoryFdPropertiesKHR>("vkGetMemoryFdPropertiesKHR");
        pfn_get_mem_fd = load_dev<PFN_vkGetMemoryFdKHR>("vkGetMemoryFdKHR");
        pfn_create_img_view = load_dev<PFN_vkCreateImageView>("vkCreateImageView");
        pfn_destroy_img_view = load_dev<PFN_vkDestroyImageView>("vkDestroyImageView");
        pfn_create_sampler = load_dev<PFN_vkCreateSampler>("vkCreateSampler");
        pfn_destroy_sampler = load_dev<PFN_vkDestroySampler>("vkDestroySampler");
        pfn_create_shader = load_dev<PFN_vkCreateShaderModule>("vkCreateShaderModule");
        pfn_destroy_shader = load_dev<PFN_vkDestroyShaderModule>("vkDestroyShaderModule");
        pfn_create_dsl = load_dev<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout");
        pfn_destroy_dsl = load_dev<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout");
        pfn_create_pl = load_dev<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout");
        pfn_destroy_pl = load_dev<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout");
        pfn_create_comp_pipe = load_dev<PFN_vkCreateComputePipelines>("vkCreateComputePipelines");
        pfn_destroy_pipe = load_dev<PFN_vkDestroyPipeline>("vkDestroyPipeline");
        pfn_create_dp = load_dev<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool");
        pfn_destroy_dp = load_dev<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool");
        pfn_alloc_ds = load_dev<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets");
        pfn_update_ds = load_dev<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets");
        pfn_create_cp = load_dev<PFN_vkCreateCommandPool>("vkCreateCommandPool");
        pfn_destroy_cp = load_dev<PFN_vkDestroyCommandPool>("vkDestroyCommandPool");
        pfn_alloc_cb = load_dev<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers");
        pfn_begin_cb = load_dev<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer");
        pfn_end_cb = load_dev<PFN_vkEndCommandBuffer>("vkEndCommandBuffer");
        pfn_reset_cp = load_dev<PFN_vkResetCommandPool>("vkResetCommandPool");
        pfn_cmd_barrier2 = load_dev<PFN_vkCmdPipelineBarrier2>("vkCmdPipelineBarrier2");
        pfn_cmd_bind_pipe = load_dev<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
        pfn_cmd_bind_ds = load_dev<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
        if (use_push_descriptors) {
            pfn_cmd_push_desc = load_dev<PFN_vkCmdPushDescriptorSetKHR>("vkCmdPushDescriptorSetKHR");
            if (!pfn_cmd_push_desc) use_push_descriptors = false;
        }
        pfn_cmd_push = load_dev<PFN_vkCmdPushConstants>("vkCmdPushConstants");
        pfn_cmd_dispatch = load_dev<PFN_vkCmdDispatch>("vkCmdDispatch");
        pfn_cmd_copy_buf_to_img = load_dev<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
        pfn_cmd_clear_color = load_dev<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
        pfn_create_fence = load_dev<PFN_vkCreateFence>("vkCreateFence");
        pfn_destroy_fence = load_dev<PFN_vkDestroyFence>("vkDestroyFence");
        pfn_wait_fences = load_dev<PFN_vkWaitForFences>("vkWaitForFences");
        pfn_reset_fences = load_dev<PFN_vkResetFences>("vkResetFences");
        pfn_create_sem = load_dev<PFN_vkCreateSemaphore>("vkCreateSemaphore");
        pfn_destroy_sem = load_dev<PFN_vkDestroySemaphore>("vkDestroySemaphore");
        pfn_import_sem_fd = load_dev<PFN_vkImportSemaphoreFdKHR>("vkImportSemaphoreFdKHR");
        pfn_queue_submit2 = load_dev<PFN_vkQueueSubmit2>("vkQueueSubmit2");
        pfn_device_wait_idle = load_dev<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle");
        pfn_get_dev_queue = load_dev<PFN_vkGetDeviceQueue>("vkGetDeviceQueue");
        pfn_create_buf = load_dev<PFN_vkCreateBuffer>("vkCreateBuffer");
        pfn_destroy_buf = load_dev<PFN_vkDestroyBuffer>("vkDestroyBuffer");
        pfn_get_buf_mem_req = load_dev<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements");
        pfn_bind_buf_mem = load_dev<PFN_vkBindBufferMemory>("vkBindBufferMemory");
        pfn_map_mem = load_dev<PFN_vkMapMemory>("vkMapMemory");
        pfn_unmap_mem = load_dev<PFN_vkUnmapMemory>("vkUnmapMemory");
        pfn_get_img_drm_mod = load_dev<PFN_vkGetImageDrmFormatModifierPropertiesEXT>("vkGetImageDrmFormatModifierPropertiesEXT");

        pfn_get_dev_queue(compute_dev, compute_qf_idx, 0, &compute_queue);
        if (!compute_queue) {
            return std::unexpected("vkGetDeviceQueue returned VK_NULL_HANDLE on isolated compute device");
        }

        // Sampler
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (pfn_create_sampler(compute_dev, &sci, nullptr, &linear_sampler) != VK_SUCCESS) {
            return std::unexpected("Failed to create Vulkan sampler");
        }

        // Descriptor set layout: binding 0 = sampler2D, binding 1 = Y image2D, binding 2 = UV image2D
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.flags = use_push_descriptors ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0;
        dslci.bindingCount = static_cast<uint32_t>(bindings.size());
        dslci.pBindings = bindings.data();
        if (pfn_create_dsl(compute_dev, &dslci, nullptr, &dsl) != VK_SUCCESS) {
            return std::unexpected("Failed to create descriptor set layout");
        }

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset = 0;
        pcr.size = sizeof(PushConstants);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        if (pfn_create_pl(compute_dev, &plci, nullptr, &pipe_layout) != VK_SUCCESS) {
            return std::unexpected("Failed to create pipeline layout");
        }

        auto make_pipe = [&](const uint32_t *words, size_t word_count, VkPipeline *out) -> bool {
            VkShaderModuleCreateInfo smci{};
            smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            smci.codeSize = word_count * sizeof(uint32_t);
            smci.pCode = words;
            VkShaderModule sm = VK_NULL_HANDLE;
            if (pfn_create_shader(compute_dev, &smci, nullptr, &sm) != VK_SUCCESS) return false;

            VkComputePipelineCreateInfo cpci{};
            cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            cpci.stage.module = sm;
            cpci.stage.pName = "main";
            cpci.layout = pipe_layout;
            VkResult r = pfn_create_comp_pipe(compute_dev, VK_NULL_HANDLE, 1, &cpci, nullptr, out);
            pfn_destroy_shader(compute_dev, sm, nullptr);
            return r == VK_SUCCESS;
        };

        if (!make_pipe(kRgbToYuvNv12Spv.data(), kRgbToYuvNv12Spv.size(), &pipe_nv12) ||
            !make_pipe(kRgbToYuvP010Spv.data(), kRgbToYuvP010Spv.size(), &pipe_p010)) {
            return std::unexpected("Failed to create Vulkan RGB->YUV compute pipelines");
        }

        for (auto &sl : slots) {
            VkCommandPoolCreateInfo cpci{};
            cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            cpci.queueFamilyIndex = compute_qf_idx;
            if (pfn_create_cp(compute_dev, &cpci, nullptr, &sl.pool) != VK_SUCCESS) {
                return std::unexpected("Failed to create command pool");
            }

            VkCommandBufferAllocateInfo cbai{};
            cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cbai.commandPool = sl.pool;
            cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 1;
            if (pfn_alloc_cb(compute_dev, &cbai, &sl.cmd) != VK_SUCCESS) {
                return std::unexpected("Failed to allocate command buffer");
            }

            VkFenceCreateInfo fci{};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (pfn_create_fence(compute_dev, &fci, nullptr, &sl.fence) != VK_SUCCESS) {
                return std::unexpected("Failed to create fence");
            }

            VkSemaphoreCreateInfo semci{};
            semci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            if (pfn_create_sem(compute_dev, &semci, nullptr, &sl.ext_wait_sem) != VK_SUCCESS) {
                return std::unexpected("Failed to create external wait semaphore");
            }

            if (!use_push_descriptors) {
                std::array<VkDescriptorPoolSize, 2> psizes = {{
                    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
                }};
                VkDescriptorPoolCreateInfo dpci{};
                dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                dpci.maxSets = 1;
                dpci.poolSizeCount = static_cast<uint32_t>(psizes.size());
                dpci.pPoolSizes = psizes.data();
                if (pfn_create_dp(compute_dev, &dpci, nullptr, &sl.desc_pool) != VK_SUCCESS) {
                    return std::unexpected("Failed to create descriptor pool");
                }

                VkDescriptorSetAllocateInfo dsai{};
                dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dsai.descriptorPool = sl.desc_pool;
                dsai.descriptorSetCount = 1;
                dsai.pSetLayouts = &dsl;
                if (pfn_alloc_ds(compute_dev, &dsai, &sl.desc_set) != VK_SUCCESS) {
                    return std::unexpected("Failed to allocate descriptor set");
                }
            }
        }
        return {};
    }

    std::expected<CachedSrcImage, std::string> import_dmabuf(const CapturedFrame &cf) {
        const FormatCandidate *cand = find_candidate(cf.format.spa_format);
        if (!cand) {
            return std::unexpected(std::format("Unsupported capture SPA format {}", cf.format.spa_format));
        }

        const uint32_t w = cf.format.width;
        const uint32_t h = cf.format.height;
        const uint32_t np = cf.num_planes;

        std::array<VkSubresourceLayout, 4> layouts{};
        for (uint32_t i = 0; i < np; ++i) {
            layouts[i].offset = cf.planes[i].offset;
            layouts[i].rowPitch = static_cast<VkDeviceSize>(cf.planes[i].stride);
            layouts[i].size = 0;
            layouts[i].arrayPitch = 0;
            layouts[i].depthPitch = 0;
        }

        VkExternalMemoryImageCreateInfo ext_img{};
        ext_img.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        ext_img.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

        VkImageDrmFormatModifierExplicitCreateInfoEXT mod_info{};
        mod_info.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT;
        mod_info.drmFormatModifier = cf.format.drm_modifier;
        mod_info.drmFormatModifierPlaneCount = np;
        mod_info.pPlaneLayouts = layouts.data();

        if (cf.format.has_modifier) {
            ext_img.pNext = &mod_info;
        }

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.pNext = &ext_img;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = cand->vk_format;
        ici.extent = {w, h, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = cf.format.has_modifier ? VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT : VK_IMAGE_TILING_LINEAR;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        CachedSrcImage out{};
        out.width = w;
        out.height = h;
        out.spa_format = cf.format.spa_format;
        out.modifier = cf.format.drm_modifier;
        out.has_modifier = cf.format.has_modifier;

        if (pfn_create_image(compute_dev, &ici, nullptr, &out.image) != VK_SUCCESS) {
            return std::unexpected(std::format("vkCreateImage failed for DMA-BUF {}x{} mod=0x{:x}",
                                               w, h, cf.format.drm_modifier));
        }

        VkImageMemoryRequirementsInfo2 req_info{};
        req_info.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
        req_info.image = out.image;
        VkMemoryDedicatedRequirements ded_req{};
        ded_req.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
        VkMemoryRequirements2 mem_req{};
        mem_req.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
        mem_req.pNext = &ded_req;
        pfn_get_img_mem_req2(compute_dev, &req_info, &mem_req);

        int dup_fd = ::dup(cf.planes[0].fd);
        if (dup_fd < 0) {
            destroy_src_image(out);
            return std::unexpected("dup() failed on DMA-BUF fd");
        }

        VkMemoryFdPropertiesKHR fd_props{};
        fd_props.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
        if (pfn_get_mem_fd_props(compute_dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
                                 dup_fd, &fd_props) != VK_SUCCESS) {
            ::close(dup_fd);
            destroy_src_image(out);
            return std::unexpected("vkGetMemoryFdPropertiesKHR failed");
        }

        uint32_t valid_bits = mem_req.memoryRequirements.memoryTypeBits & fd_props.memoryTypeBits;
        int mem_idx = find_memory_type(valid_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mem_idx < 0) {
            mem_idx = find_memory_type(valid_bits, 0);
        }
        if (mem_idx < 0) {
            ::close(dup_fd);
            destroy_src_image(out);
            return std::unexpected("No compatible memory type for DMA-BUF import");
        }

        VkImportMemoryFdInfoKHR imp_fd{};
        imp_fd.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        imp_fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        imp_fd.fd = dup_fd;  // Ownership transfers to Vulkan on success

        VkMemoryDedicatedAllocateInfo ded_alloc{};
        ded_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        ded_alloc.image = out.image;
        if (ded_req.prefersDedicatedAllocation || ded_req.requiresDedicatedAllocation) {
            imp_fd.pNext = &ded_alloc;
        }

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &imp_fd;
        mai.allocationSize = std::max<VkDeviceSize>(mem_req.memoryRequirements.size, cf.planes[0].size);
        mai.memoryTypeIndex = static_cast<uint32_t>(mem_idx);

        if (pfn_alloc_mem(compute_dev, &mai, nullptr, &out.mem[0]) != VK_SUCCESS) {
            ::close(dup_fd);
            destroy_src_image(out);
            return std::unexpected("vkAllocateMemory(DMA-BUF import) failed");
        }
        out.num_mems = 1;

        VkBindImageMemoryInfo bimi{};
        bimi.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
        bimi.image = out.image;
        bimi.memory = out.mem[0];
        bimi.memoryOffset = 0;
        if (pfn_bind_image_mem2(compute_dev, 1, &bimi) != VK_SUCCESS) {
            destroy_src_image(out);
            return std::unexpected("vkBindImageMemory2 failed on DMA-BUF");
        }

        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = out.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = cand->vk_format;
        ivci.components = cand->swizzle;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (pfn_create_img_view(compute_dev, &ivci, nullptr, &out.view) != VK_SUCCESS) {
            destroy_src_image(out);
            return std::unexpected("vkCreateImageView failed on imported DMA-BUF");
        }
        return out;
    }

    std::expected<CachedSrcImage *, std::string> get_or_import_src(const CapturedFrame &cf, VkCommandBuffer cmd, bool *out_uploaded_shm) {
        *out_uploaded_shm = false;
        if (cf.mem_type == CaptureMemoryType::DmaBuf) {
            std::lock_guard lk(cache_mtx);
            auto it = src_cache.find(cf.buffer_handle);
            if (it != src_cache.end()) {
                const auto &c = it->second;
                if (c.width == cf.format.width && c.height == cf.format.height &&
                    c.spa_format == cf.format.spa_format && c.modifier == cf.format.drm_modifier &&
                    c.has_modifier == cf.format.has_modifier) {
                    return &it->second;
                }
                destroy_src_image(it->second);
                src_cache.erase(it);
            }
            auto imported = import_dmabuf(cf);
            if (!imported) return std::unexpected(imported.error());
            auto [ins_it, _] = src_cache.emplace(cf.buffer_handle, *imported);
            return &ins_it->second;
        }

        // SHM fallback: copy into staging VkBuffer -> optimal device-local VkImage
        const FormatCandidate *cand = find_candidate(cf.format.spa_format);
        if (!cand) {
            return std::unexpected(std::format("Unsupported SHM SPA format {}", cf.format.spa_format));
        }
        const uint32_t w = cf.format.width;
        const uint32_t h = cf.format.height;
        const uint32_t bpp = (cand->vk_format == VK_FORMAT_R16G16B16A16_SFLOAT) ? 8u : 4u;
        const size_t needed = static_cast<size_t>(w) * h * bpp;

        if (!shm_buf || shm_buf_size < needed) {
            if (shm_mapped) { pfn_unmap_mem(compute_dev, shm_mem); shm_mapped = nullptr; }
            if (shm_buf) { pfn_destroy_buf(compute_dev, shm_buf, nullptr); shm_buf = VK_NULL_HANDLE; }
            if (shm_mem) { pfn_free_mem(compute_dev, shm_mem, nullptr); shm_mem = VK_NULL_HANDLE; }

            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = needed;
            bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (pfn_create_buf(compute_dev, &bci, nullptr, &shm_buf) != VK_SUCCESS) {
                return std::unexpected("Failed to create SHM staging buffer");
            }
            VkMemoryRequirements breq{};
            pfn_get_buf_mem_req(compute_dev, shm_buf, &breq);
            int midx = find_memory_type(breq.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (midx < 0) return std::unexpected("No host-visible memory type for SHM staging");
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, breq.size, static_cast<uint32_t>(midx)};
            if (pfn_alloc_mem(compute_dev, &mai, nullptr, &shm_mem) != VK_SUCCESS ||
                pfn_bind_buf_mem(compute_dev, shm_buf, shm_mem, 0) != VK_SUCCESS ||
                pfn_map_mem(compute_dev, shm_mem, 0, needed, 0, &shm_mapped) != VK_SUCCESS) {
                return std::unexpected("Failed to allocate/map SHM staging buffer");
            }
            shm_buf_size = needed;
        }

        if (!shm_img.image || shm_img.width != w || shm_img.height != h || shm_img.spa_format != cf.format.spa_format) {
            destroy_src_image(shm_img);
            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = cand->vk_format;
            ici.extent = {w, h, 1};
            ici.mipLevels = 1;
            ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (pfn_create_image(compute_dev, &ici, nullptr, &shm_img.image) != VK_SUCCESS) {
                return std::unexpected("Failed to create SHM device image");
            }
            VkImageMemoryRequirementsInfo2 rinfo{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, nullptr, shm_img.image};
            VkMemoryRequirements2 mreq{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, nullptr, {}};
            pfn_get_img_mem_req2(compute_dev, &rinfo, &mreq);
            int midx = find_memory_type(mreq.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mreq.memoryRequirements.size, static_cast<uint32_t>(midx)};
            if (pfn_alloc_mem(compute_dev, &mai, nullptr, &shm_img.mem[0]) != VK_SUCCESS) {
                return std::unexpected("Failed to allocate SHM device image memory");
            }
            shm_img.num_mems = 1;
            VkBindImageMemoryInfo bimi{VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, nullptr, shm_img.image, shm_img.mem[0], 0};
            if (pfn_bind_image_mem2(compute_dev, 1, &bimi) != VK_SUCCESS) {
                return std::unexpected("Failed to bind SHM device image memory");
            }
            VkImageViewCreateInfo ivci{};
            ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ivci.image = shm_img.image;
            ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ivci.format = cand->vk_format;
            ivci.components = cand->swizzle;
            ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (pfn_create_img_view(compute_dev, &ivci, nullptr, &shm_img.view) != VK_SUCCESS) {
                return std::unexpected("Failed to create SHM device image view");
            }
            shm_img.width = w;
            shm_img.height = h;
            shm_img.spa_format = cf.format.spa_format;
        }

        const uint8_t *src_bytes = static_cast<const uint8_t *>(cf.planes[0].data);
        void *temp_mmap = nullptr;
        size_t temp_mmap_len = 0;
        if (!src_bytes && cf.planes[0].fd >= 0) {
            temp_mmap_len = static_cast<size_t>(cf.planes[0].offset) + cf.planes[0].size;
            if (temp_mmap_len > 0) {
                void *p = ::mmap(nullptr, temp_mmap_len, PROT_READ, MAP_SHARED, cf.planes[0].fd, 0);
                if (p != MAP_FAILED) {
                    temp_mmap = p;
                    src_bytes = static_cast<const uint8_t *>(p);
                }
            }
        }
        if (!src_bytes) {
            return std::unexpected("SHM frame has null data pointer and could not be mapped");
        }
        src_bytes += cf.planes[0].offset;
        const size_t row_bytes = static_cast<size_t>(w) * bpp;
        const size_t src_stride = cf.planes[0].stride > 0 ? static_cast<size_t>(cf.planes[0].stride) : row_bytes;
        auto *dst_bytes = static_cast<uint8_t *>(shm_mapped);
        if (src_stride == row_bytes) {
            std::memcpy(dst_bytes, src_bytes, needed);
        } else {
            for (uint32_t y = 0; y < h; ++y) {
                std::memcpy(dst_bytes + y * row_bytes, src_bytes + y * src_stride, row_bytes);
            }
        }
        if (temp_mmap) {
            ::munmap(temp_mmap, temp_mmap_len);
        }

        VkImageMemoryBarrier2 to_dst{};
        to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_dst.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        to_dst.srcAccessMask = VK_ACCESS_2_NONE;
        to_dst.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        to_dst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_dst.image = shm_img.image;
        to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep1{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 0, nullptr, 1, &to_dst};
        pfn_cmd_barrier2(cmd, &dep1);

        VkBufferImageCopy reg{};
        reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        reg.imageExtent = {w, h, 1};
        pfn_cmd_copy_buf_to_img(cmd, shm_buf, shm_img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);

        VkImageMemoryBarrier2 to_sample{};
        to_sample.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_sample.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        to_sample.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        to_sample.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        to_sample.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        to_sample.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_sample.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_sample.image = shm_img.image;
        to_sample.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep2{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 0, nullptr, 1, &to_sample};
        pfn_cmd_barrier2(cmd, &dep2);

        *out_uploaded_shm = true;
        return &shm_img;
    }

    std::expected<CachedDstAlias *, std::string> get_or_create_dst_alias(const AVVkFrame *vkf) {
        std::lock_guard lk(cache_mtx);
        auto it = dst_alias_cache.find(vkf->img[0]);
        if (it != dst_alias_cache.end()) {
            return &it->second;
        }

        if (!pfn_ff_get_mem_fd) {
            return std::unexpected("vkGetMemoryFdKHR unavailable on FFmpeg Vulkan device");
        }

        int mem_fd = -1;
        VkMemoryGetFdInfoKHR gfdi{};
        gfdi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        gfdi.memory = vkf->mem[0];
        gfdi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        if (pfn_ff_get_mem_fd(vkctx->act_dev, &gfdi, &mem_fd) != VK_SUCCESS || mem_fd < 0) {
            return std::unexpected("Failed to export AVVkFrame memory as OPAQUE_FD");
        }

        VkExternalMemoryImageCreateInfo ext_img{};
        ext_img.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        ext_img.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (compute_has_video_enc_ext) {
            usage |= VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR;
        }

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.pNext = &ext_img;
        ici.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT | VK_IMAGE_CREATE_ALIAS_BIT;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = (cfg.pixfmt == PixelFormat::P010)
            ? VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
            : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        ici.extent = {out_width, out_height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        CachedDstAlias da{};
        if (pfn_create_image(compute_dev, &ici, nullptr, &da.image) != VK_SUCCESS) {
            ::close(mem_fd);
            return std::unexpected("Failed to create aliased AVVkFrame image on compute_dev");
        }

        VkImageMemoryRequirementsInfo2 rinfo{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, nullptr, da.image};
        VkMemoryDedicatedRequirements ded_req{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS, nullptr, VK_FALSE, VK_FALSE};
        VkMemoryRequirements2 mreq{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &ded_req, {}};
        pfn_get_img_mem_req2(compute_dev, &rinfo, &mreq);

        int midx = find_memory_type(mreq.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (midx < 0) midx = find_memory_type(mreq.memoryRequirements.memoryTypeBits, 0);

        VkImportMemoryFdInfoKHR imp_fd{};
        imp_fd.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        imp_fd.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        imp_fd.fd = mem_fd;

        VkMemoryDedicatedAllocateInfo ded_alloc{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, nullptr, da.image, VK_NULL_HANDLE};
        if (ded_req.prefersDedicatedAllocation || ded_req.requiresDedicatedAllocation) {
            imp_fd.pNext = &ded_alloc;
        }

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &imp_fd;
        mai.allocationSize = std::max<VkDeviceSize>(mreq.memoryRequirements.size, vkf->size[0]);
        mai.memoryTypeIndex = static_cast<uint32_t>(midx);

        if (pfn_alloc_mem(compute_dev, &mai, nullptr, &da.mem) != VK_SUCCESS) {
            ::close(mem_fd);
            destroy_dst_alias(da);
            return std::unexpected("Failed to import AVVkFrame OPAQUE_FD memory on compute_dev");
        }

        VkBindImageMemoryInfo bimi{VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, nullptr, da.image, da.mem, static_cast<VkDeviceSize>(vkf->offset[0])};
        if (pfn_bind_image_mem2(compute_dev, 1, &bimi) != VK_SUCCESS) {
            destroy_dst_alias(da);
            return std::unexpected("Failed to bind aliased AVVkFrame memory on compute_dev");
        }

        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = da.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = (cfg.pixfmt == PixelFormat::P010) ? VK_FORMAT_R10X6_UNORM_PACK16 : VK_FORMAT_R8_UNORM;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_PLANE_0_BIT, 0, 1, 0, 1};
        if (pfn_create_img_view(compute_dev, &ivci, nullptr, &da.y_view) != VK_SUCCESS) {
            destroy_dst_alias(da);
            return std::unexpected("Failed to create Y plane image view on aliased AVVkFrame");
        }

        ivci.format = (cfg.pixfmt == PixelFormat::P010) ? VK_FORMAT_R10X6G10X6_UNORM_2PACK16 : VK_FORMAT_R8G8_UNORM;
        ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
        if (pfn_create_img_view(compute_dev, &ivci, nullptr, &da.uv_view) != VK_SUCCESS) {
            destroy_dst_alias(da);
            return std::unexpected("Failed to create UV plane image view on aliased AVVkFrame");
        }

        auto [ins_it, _] = dst_alias_cache.emplace(vkf->img[0], da);
        return &ins_it->second;
    }
};

VulkanCompute::~VulkanCompute() = default;

std::expected<std::unique_ptr<VulkanCompute>, std::string> VulkanCompute::create(const Config &cfg) {
    auto vc = std::unique_ptr<VulkanCompute>(new VulkanCompute());
    vc->impl_ = std::make_unique<Impl>();
    Impl *impl = vc->impl_.get();
    impl->cfg = cfg;

    std::string dev_str = std::to_string(cfg.device_index);
    int err = av_hwdevice_ctx_create(&impl->hw_device_ref, AV_HWDEVICE_TYPE_VULKAN, dev_str.c_str(), nullptr, 0);
    if (err < 0) {
        char buf[128];
        av_strerror(err, buf, sizeof(buf));
        return std::unexpected(std::format("Failed to create FFmpeg Vulkan device '{}': {}", dev_str, buf));
    }

    auto *hwdev = reinterpret_cast<AVHWDeviceContext *>(impl->hw_device_ref->data);
    impl->vkctx = reinterpret_cast<AVVulkanDeviceContext *>(hwdev->hwctx);

    auto res = impl->init_vulkan_objects();
    if (!res) {
        return std::unexpected(res.error());
    }

    if (cfg.verbose) {
        VkPhysicalDeviceProperties props{};
        auto pfn_get_props = impl->load_inst<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
        pfn_get_props(impl->vkctx->phys_dev, &props);
        std::cerr << std::format("[vulkan] device[{}]='{}' compute_qf={}\n",
                                 cfg.device_index, props.deviceName, impl->compute_qf_idx);
    }

    return vc;
}

std::vector<FormatModifierSet> VulkanCompute::query_supported_capture_formats() const {
    std::vector<const FormatCandidate *> ordered;
    for (const auto &c : kCandidates) ordered.push_back(&c);
    if (impl_->cfg.pixfmt == PixelFormat::P010) {
        std::stable_sort(ordered.begin(), ordered.end(), [](const FormatCandidate *a, const FormatCandidate *b) {
            return a->is_10bit_or_hdr > b->is_10bit_or_hdr;
        });
    } else {
        std::stable_sort(ordered.begin(), ordered.end(), [](const FormatCandidate *a, const FormatCandidate *b) {
            return a->is_10bit_or_hdr < b->is_10bit_or_hdr;
        });
    }

    std::vector<FormatModifierSet> out;
    for (const FormatCandidate *c : ordered) {
        VkDrmFormatModifierPropertiesListEXT mod_list{};
        mod_list.sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT;
        VkFormatProperties2 fp2{};
        fp2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2;
        fp2.pNext = &mod_list;
        impl_->pfn_get_fmt_props2(impl_->vkctx->phys_dev, c->vk_format, &fp2);

        std::vector<VkDrmFormatModifierPropertiesEXT> props(mod_list.drmFormatModifierCount);
        if (!props.empty()) {
            mod_list.pDrmFormatModifierProperties = props.data();
            impl_->pfn_get_fmt_props2(impl_->vkctx->phys_dev, c->vk_format, &fp2);
        }

        FormatModifierSet fms{};
        fms.spa_format = c->spa_format;
        fms.drm_fourcc = c->drm_fourcc;

        std::vector<uint64_t> tiled_mods;
        bool has_linear = false;
        for (const auto &p : props) {
            if ((p.drmFormatModifierTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) == 0) {
                continue;
            }
            // Verify external DMA-BUF import support for this modifier
            VkPhysicalDeviceImageDrmFormatModifierInfoEXT mod_info{};
            mod_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT;
            mod_info.drmFormatModifier = p.drmFormatModifier;
            mod_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VkPhysicalDeviceExternalImageFormatInfo ext_info{};
            ext_info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
            ext_info.pNext = &mod_info;
            ext_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

            VkPhysicalDeviceImageFormatInfo2 ifi2{};
            ifi2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
            ifi2.pNext = &ext_info;
            ifi2.format = c->vk_format;
            ifi2.type = VK_IMAGE_TYPE_2D;
            ifi2.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
            ifi2.usage = VK_IMAGE_USAGE_SAMPLED_BIT;

            VkExternalImageFormatProperties eifp{};
            eifp.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
            VkImageFormatProperties2 ifp2{};
            ifp2.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
            ifp2.pNext = &eifp;

            if (impl_->pfn_get_img_fmt_props2(impl_->vkctx->phys_dev, &ifi2, &ifp2) == VK_SUCCESS &&
                (eifp.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) {
                if (p.drmFormatModifier == DRM_FORMAT_MOD_LINEAR) {
                    has_linear = true;
                } else {
                    tiled_mods.push_back(p.drmFormatModifier);
                }
            }
        }

        // Prefer GPU-optimal tiled/DCC modifiers first, linear last
        fms.modifiers = std::move(tiled_mods);
        if (has_linear) {
            fms.modifiers.push_back(DRM_FORMAT_MOD_LINEAR);
        }
        fms.allow_modifierless =
            (fp2.formatProperties.linearTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0 ||
            !fms.modifiers.empty();

        if (!fms.modifiers.empty() || fms.allow_modifierless) {
            out.push_back(std::move(fms));
        }
    }
    return out;
}

std::expected<void, std::string> VulkanCompute::configure_output(uint32_t width, uint32_t height) {
    // Ensure even dimensions for 4:2:0 video encoding
    width &= ~1u;
    height &= ~1u;
    if (width == 0 || height == 0) {
        return std::unexpected("Output dimensions must be non-zero even integers");
    }

    if (impl_->hw_frames_ref && impl_->out_width == width && impl_->out_height == height) {
        return {};
    }

    impl_->clear_caches();
    av_buffer_unref(&impl_->hw_frames_ref);

    AVBufferRef *fref = av_hwframe_ctx_alloc(impl_->hw_device_ref);
    if (!fref) {
        return std::unexpected("av_hwframe_ctx_alloc failed");
    }

    auto *fc = reinterpret_cast<AVHWFramesContext *>(fref->data);
    auto *vkfc = reinterpret_cast<AVVulkanFramesContext *>(fc->hwctx);
    fc->format = AV_PIX_FMT_VULKAN;
    fc->sw_format = (impl_->cfg.pixfmt == PixelFormat::P010) ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
    fc->width = static_cast<int>(width);
    fc->height = static_cast<int>(height);
    fc->initial_pool_size = 16 + impl_->cfg.async_depth * 2;
    vkfc->usage = static_cast<VkImageUsageFlagBits>(
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR);

    int err = av_hwframe_ctx_init(fref);
    if (err < 0) {
        av_buffer_unref(&fref);
        char buf[128];
        av_strerror(err, buf, sizeof(buf));
        return std::unexpected(std::format("av_hwframe_ctx_init({}x{}) failed: {}", width, height, buf));
    }

    impl_->hw_frames_ref = fref;
    impl_->out_width = width;
    impl_->out_height = height;
    return {};
}

std::expected<AVFrame *, std::string> VulkanCompute::process_frame(const CapturedFrame &cf) {
    if (!impl_->hw_frames_ref) {
        return std::unexpected("VulkanCompute output not configured");
    }

    const auto t0 = std::chrono::steady_clock::now();

    // FFmpeg's AVBufferPool is LIFO, so av_hwframe_get_buffer() often returns a recently
    // unref'd AVVkFrame whose av1_vulkan GPU encode has not finished yet. Probe up to 8
    // frames to pick one whose timeline semaphore is ALREADY signaled so our compute
    // shader never stalls waiting on VCN encode while holding KWin's DMA-BUF!
    std::array<AVFrame *, 8> busy_frames{};
    size_t nb_busy = 0;
    AVFrame *dst_frame = nullptr;

    for (size_t attempt = 0; attempt < busy_frames.size(); ++attempt) {
        AVFrame *cand = av_frame_alloc();
        if (!cand) break;
        int err = av_hwframe_get_buffer(impl_->hw_frames_ref, cand, 0);
        if (err < 0) {
            av_frame_free(&cand);
            if (attempt == 0) {
                char buf[128];
                av_strerror(err, buf, sizeof(buf));
                return std::unexpected(std::format("av_hwframe_get_buffer failed: {}", buf));
            }
            break;
        }
        auto *vkf_cand = reinterpret_cast<AVVkFrame *>(cand->data[0]);
        uint64_t cur_val = 0;
        if (!impl_->pfn_ff_get_sem_val ||
            (impl_->pfn_ff_get_sem_val(impl_->vkctx->act_dev, vkf_cand->sem[0], &cur_val) == VK_SUCCESS &&
             cur_val >= vkf_cand->sem_value[0])) {
            dst_frame = cand;
            break;
        }
        busy_frames[nb_busy++] = cand;
    }

    if (!dst_frame && nb_busy > 0) {
        dst_frame = busy_frames[--nb_busy];
        auto *vkf_wait = reinterpret_cast<AVVkFrame *>(dst_frame->data[0]);
        if (impl_->pfn_ff_wait_sems) {
            VkSemaphoreWaitInfo swi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, nullptr, 0, 1,
                                    &vkf_wait->sem[0], &vkf_wait->sem_value[0]};
            impl_->pfn_ff_wait_sems(impl_->vkctx->act_dev, &swi, UINT64_MAX);
        }
    }
    for (size_t i = 0; i < nb_busy; ++i) {
        av_frame_free(&busy_frames[i]);
    }
    if (!dst_frame) {
        return std::unexpected("Failed to acquire AVVkFrame from pool");
    }

    auto *vkf = reinterpret_cast<AVVkFrame *>(dst_frame->data[0]);

    auto dst_alias_res = impl_->get_or_create_dst_alias(vkf);
    if (!dst_alias_res) {
        av_frame_free(&dst_frame);
        return std::unexpected(dst_alias_res.error());
    }
    CachedDstAlias *dst_alias = *dst_alias_res;

    CmdSlot &sl = impl_->slots[impl_->slot_cursor];
    impl_->slot_cursor = (impl_->slot_cursor + 1) % Impl::kNumSlots;

    impl_->pfn_reset_cp(impl_->compute_dev, sl.pool, 0);
    impl_->pfn_reset_fences(impl_->compute_dev, 1, &sl.fence);

    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (impl_->pfn_begin_cb(sl.cmd, &cbbi) != VK_SUCCESS) {
        av_frame_free(&dst_frame);
        return std::unexpected("vkBeginCommandBuffer failed");
    }

    bool uploaded_shm = false;
    auto src_img_res = impl_->get_or_import_src(cf, sl.cmd, &uploaded_shm);
    if (!src_img_res) {
        impl_->pfn_end_cb(sl.cmd);
        av_frame_free(&dst_frame);
        return std::unexpected(src_img_res.error());
    }
    CachedSrcImage *src = *src_img_res;

    // Export DMA-BUF implicit sync fence as a sync_file fd if available.
    // Fast-path: if the fence is already signaled (non-blocking poll), skip the
    // vkImportSemaphoreFdKHR + kernel drm_syncobj wait entirely!
    int sync_fd = -1;
    if (cf.mem_type == CaptureMemoryType::DmaBuf && cf.planes[0].fd >= 0) {
        dma_buf_export_sync_file ex{};
        ex.flags = DMA_BUF_SYNC_READ;
        ex.fd = -1;
        if (::ioctl(cf.planes[0].fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &ex) == 0 && ex.fd >= 0) {
            pollfd pfd{ex.fd, POLLIN, 0};
            if (::poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN) != 0) {
                ::close(ex.fd);
            } else {
                sync_fd = ex.fd;
            }
        }
    }

    bool has_ext_sem = false;
    if (sync_fd >= 0) {
        VkImportSemaphoreFdInfoKHR isfi{};
        isfi.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
        isfi.semaphore = sl.ext_wait_sem;
        isfi.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
        isfi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        isfi.fd = sync_fd;
        if (impl_->pfn_import_sem_fd(impl_->compute_dev, &isfi) == VK_SUCCESS) {
            has_ext_sem = true;
        } else {
            ::close(sync_fd);
        }
    }

    VkImageLayout src_sample_layout = uploaded_shm ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo src_di{impl_->linear_sampler, src->view, src_sample_layout};
    VkDescriptorImageInfo y_di{VK_NULL_HANDLE, dst_alias->y_view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo uv_di{VK_NULL_HANDLE, dst_alias->uv_view, VK_IMAGE_LAYOUT_GENERAL};

    std::array<VkWriteDescriptorSet, 3> writes{};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, impl_->use_push_descriptors ? VK_NULL_HANDLE : sl.desc_set, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &src_di, nullptr, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, impl_->use_push_descriptors ? VK_NULL_HANDLE : sl.desc_set, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &y_di, nullptr, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, impl_->use_push_descriptors ? VK_NULL_HANDLE : sl.desc_set, 2, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &uv_di, nullptr, nullptr};
    if (!impl_->use_push_descriptors) {
        impl_->pfn_update_ds(impl_->compute_dev, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    std::array<VkImageMemoryBarrier2, 2> pre_barriers{};
    uint32_t nb_pre = 0;

    if (!uploaded_shm) {
        auto &b = pre_barriers[nb_pre++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        b.srcAccessMask = VK_ACCESS_2_NONE;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        b.dstQueueFamilyIndex = impl_->compute_qf_idx;
        b.image = src->image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }

    if (!dst_alias->initialized_layout) {
        auto &b = pre_barriers[nb_pre++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        b.srcAccessMask = VK_ACCESS_2_NONE;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = dst_alias->image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        dst_alias->initialized_layout = true;
    }

    if (nb_pre > 0) {
        VkDependencyInfo dep_pre{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 0, nullptr, nb_pre, pre_barriers.data()};
        impl_->pfn_cmd_barrier2(sl.cmd, &dep_pre);
    }

    VkPipeline pipe = (impl_->cfg.pixfmt == PixelFormat::P010) ? impl_->pipe_p010 : impl_->pipe_nv12;
    impl_->pfn_cmd_bind_pipe(sl.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    if (impl_->use_push_descriptors) {
        impl_->pfn_cmd_push_desc(sl.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipe_layout, 0,
                                 static_cast<uint32_t>(writes.size()), writes.data());
    } else {
        impl_->pfn_cmd_bind_ds(sl.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipe_layout, 0, 1, &sl.desc_set, 0, nullptr);
    }

    const float crop_x = cf.crop.valid() ? static_cast<float>(cf.crop.x) : 0.0f;
    const float crop_y = cf.crop.valid() ? static_cast<float>(cf.crop.y) : 0.0f;
    const float crop_w = cf.crop.valid() ? static_cast<float>(cf.crop.width) : static_cast<float>(cf.format.width);
    const float crop_h = cf.crop.valid() ? static_cast<float>(cf.crop.height) : static_cast<float>(cf.format.height);

    const bool exact_1to1 =
        (static_cast<uint32_t>(crop_w) == impl_->out_width) &&
        (static_cast<uint32_t>(crop_h) == impl_->out_height);

    const float inv_src_w = 1.0f / static_cast<float>(cf.format.width);
    const float inv_src_h = 1.0f / static_cast<float>(cf.format.height);
    const float step_x = crop_w / static_cast<float>(impl_->out_width);
    const float step_y = crop_h / static_cast<float>(impl_->out_height);

    const float kr = (impl_->cfg.color_space == ColorSpace::Bt2020) ? 0.2627f : 0.2126f;
    const float kb = (impl_->cfg.color_space == ColorSpace::Bt2020) ? 0.0593f : 0.0722f;
    const float kg = 1.0f - kr - kb;

    const bool is_p010 = (impl_->cfg.pixfmt == PixelFormat::P010);
    const bool is_limited = (impl_->cfg.color_range == ColorRange::Limited);
    const float y_scale  = is_limited ? (is_p010 ? (876.0f / 1023.0f) : (219.0f / 255.0f)) : 1.0f;
    const float y_offset = is_limited ? (is_p010 ? (64.0f / 1023.0f)  : (16.0f / 255.0f))  : 0.0f;
    const float c_scale  = is_limited ? (is_p010 ? (896.0f / 1023.0f) : (224.0f / 255.0f)) : 1.0f;
    const float c_offset = is_p010 ? (512.0f / 1023.0f) : (128.0f / 255.0f);

    const float cb_norm = c_scale / (2.0f * (1.0f - kb));
    const float cr_norm = c_scale / (2.0f * (1.0f - kr));

    PushConstants pc{};
    pc.yCoeffs[0]  = kr * y_scale;
    pc.yCoeffs[1]  = kg * y_scale;
    pc.yCoeffs[2]  = kb * y_scale;
    pc.yCoeffs[3]  = y_offset;
    pc.cbCoeffs[0] = -kr * cb_norm;
    pc.cbCoeffs[1] = -kg * cb_norm;
    pc.cbCoeffs[2] = (1.0f - kb) * cb_norm;
    pc.cbCoeffs[3] = c_offset;
    pc.crCoeffs[0] = (1.0f - kr) * cr_norm;
    pc.crCoeffs[1] = -kg * cr_norm;
    pc.crCoeffs[2] = -kb * cr_norm;
    pc.crCoeffs[3] = c_offset;
    pc.srcOffset[0] = crop_x;
    pc.srcOffset[1] = crop_y;
    pc.srcScale[0] = step_x * inv_src_w;
    pc.srcScale[1] = step_y * inv_src_h;
    pc.srcBaseUv[0] = (crop_x + 0.5f * step_x) * inv_src_w;
    pc.srcBaseUv[1] = (crop_y + 0.5f * step_y) * inv_src_h;
    pc.invSrcSize[0] = inv_src_w;
    pc.invSrcSize[1] = inv_src_h;
    pc.dstExtent[0] = impl_->out_width;
    pc.dstExtent[1] = impl_->out_height;
    pc.flags = (exact_1to1 ? 1u : 0u) | (impl_->cfg.dither ? 2u : 0u);
    impl_->pfn_cmd_push(sl.cmd, impl_->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    const uint32_t gx = (impl_->out_width + 15u) / 16u;
    const uint32_t gy = (impl_->out_height + 15u) / 16u;
    impl_->pfn_cmd_dispatch(sl.cmd, gx, gy, 1);

    // Flush compute shader writes to VRAM so vkctx->act_dev's av1_vulkan sees coherent data
    std::array<VkImageMemoryBarrier2, 2> post_barriers{};
    uint32_t nb_post = 0;
    {
        auto &b = post_barriers[nb_post++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        b.dstAccessMask = VK_ACCESS_2_NONE;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = impl_->compute_qf_idx;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        b.image = dst_alias->image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    if (!uploaded_shm) {
        auto &b = post_barriers[nb_post++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        b.dstAccessMask = VK_ACCESS_2_NONE;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = impl_->compute_qf_idx;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        b.image = src->image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    VkDependencyInfo dep_post{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 0, nullptr, nb_post, post_barriers.data()};
    impl_->pfn_cmd_barrier2(sl.cmd, &dep_post);

    if (impl_->pfn_end_cb(sl.cmd) != VK_SUCCESS) {
        av_frame_free(&dst_frame);
        return std::unexpected("vkEndCommandBuffer failed");
    }

    VkSemaphoreSubmitInfo wait_ext{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, nullptr,
                                   sl.ext_wait_sem, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0};
    VkCommandBufferSubmitInfo cbsi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, sl.cmd, 0};

    VkSubmitInfo2 si2{};
    si2.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    si2.waitSemaphoreInfoCount = has_ext_sem ? 1u : 0u;
    si2.pWaitSemaphoreInfos = has_ext_sem ? &wait_ext : nullptr;
    si2.commandBufferInfoCount = 1;
    si2.pCommandBufferInfos = &cbsi;

    VkResult sub_res = impl_->pfn_queue_submit2(impl_->compute_queue, 1, &si2, sl.fence);
    if (sub_res != VK_SUCCESS) {
        av_frame_free(&dst_frame);
        return std::unexpected("vkQueueSubmit2 failed for compute color conversion");
    }

    // Isolated compute_dev wait (~0.12ms at 4K) — completely decoupled from av1_vulkan VCN!
    impl_->pfn_wait_fences(impl_->compute_dev, 1, &sl.fence, VK_TRUE, UINT64_MAX);

    dst_frame->color_range = (impl_->cfg.color_range == ColorRange::Full) ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
    dst_frame->colorspace = (impl_->cfg.color_space == ColorSpace::Bt2020) ? AVCOL_SPC_BT2020_NCL : AVCOL_SPC_BT709;
    dst_frame->color_primaries = (impl_->cfg.color_space == ColorSpace::Bt2020) ? AVCOL_PRI_BT2020 : AVCOL_PRI_BT709;
    dst_frame->color_trc = (impl_->cfg.color_space == ColorSpace::Bt2020) ? AVCOL_TRC_BT2020_10 : AVCOL_TRC_BT709;
    dst_frame->chroma_location = AVCHROMA_LOC_LEFT;

    const auto t1 = std::chrono::steady_clock::now();
    impl_->stats.frames_processed++;
    impl_->stats.total_compute_ns += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());

    return dst_frame;
}

void VulkanCompute::evict_buffer(void *buffer_handle) {
    std::lock_guard lk(impl_->cache_mtx);
    auto it = impl_->src_cache.find(buffer_handle);
    if (it != impl_->src_cache.end()) {
        impl_->destroy_src_image(it->second);
        impl_->src_cache.erase(it);
    }
}

AVBufferRef *VulkanCompute::hw_device_ref() const noexcept { return impl_->hw_device_ref; }
AVBufferRef *VulkanCompute::hw_frames_ref() const noexcept { return impl_->hw_frames_ref; }
uint32_t VulkanCompute::output_width() const noexcept { return impl_->out_width; }
uint32_t VulkanCompute::output_height() const noexcept { return impl_->out_height; }
ComputeStats VulkanCompute::stats() const noexcept { return impl_->stats; }

std::expected<VulkanCompute::TestDmaBuf, std::string> VulkanCompute::create_test_dmabuf(
    uint32_t width, uint32_t height, uint32_t frame_idx) {
    // Allocate an exportable Vulkan image with RADV DRM format modifier (tiled if supported, else linear)
    auto fmts = query_supported_capture_formats();
    uint64_t mod = DRM_FORMAT_MOD_LINEAR;
    for (const auto &f : fmts) {
        if (f.spa_format == SPA_VIDEO_FORMAT_BGRA && !f.modifiers.empty()) {
            mod = f.modifiers.front();
            break;
        }
    }

    VkExternalMemoryImageCreateInfo ext_img{};
    ext_img.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_img.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    VkImageDrmFormatModifierListCreateInfoEXT mod_list{};
    mod_list.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT;
    mod_list.drmFormatModifierCount = 1;
    mod_list.pDrmFormatModifiers = &mod;
    ext_img.pNext = &mod_list;

    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.pNext = &ext_img;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = {width, height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    TestDmaBuf out{};
    out.width = width;
    out.height = height;
    out.modifier = mod;

    if (impl_->pfn_create_image(impl_->compute_dev, &ici, nullptr, &out.img) != VK_SUCCESS) {
        return std::unexpected("create_test_dmabuf: vkCreateImage failed");
    }

    VkImageMemoryRequirementsInfo2 rinfo{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, nullptr, out.img};
    VkMemoryRequirements2 mreq{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, nullptr, {}};
    impl_->pfn_get_img_mem_req2(impl_->compute_dev, &rinfo, &mreq);

    int midx = impl_->find_memory_type(mreq.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (midx < 0) midx = impl_->find_memory_type(mreq.memoryRequirements.memoryTypeBits, 0);

    VkExportMemoryAllocateInfo exp_ai{};
    exp_ai.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exp_ai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkMemoryDedicatedAllocateInfo ded_ai{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, nullptr, out.img, VK_NULL_HANDLE};
    exp_ai.pNext = &ded_ai;

    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &exp_ai, mreq.memoryRequirements.size, static_cast<uint32_t>(midx)};
    if (impl_->pfn_alloc_mem(impl_->compute_dev, &mai, nullptr, &out.mem) != VK_SUCCESS) {
        destroy_test_dmabuf(out);
        return std::unexpected("create_test_dmabuf: vkAllocateMemory failed");
    }

    VkBindImageMemoryInfo bimi{VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, nullptr, out.img, out.mem, 0};
    if (impl_->pfn_bind_image_mem2(impl_->compute_dev, 1, &bimi) != VK_SUCCESS) {
        destroy_test_dmabuf(out);
        return std::unexpected("create_test_dmabuf: vkBindImageMemory2 failed");
    }

    // Query actual modifier & subresource layout stride
    VkImageDrmFormatModifierPropertiesEXT img_mod{};
    img_mod.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT;
    if (impl_->pfn_get_img_drm_mod(impl_->compute_dev, out.img, &img_mod) == VK_SUCCESS) {
        out.modifier = img_mod.drmFormatModifier;
    }

    auto pfn_sub_layout = impl_->load_dev<PFN_vkGetImageSubresourceLayout>("vkGetImageSubresourceLayout");
    VkImageSubresource sub{VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT, 0, 0};
    VkSubresourceLayout slayout{};
    pfn_sub_layout(impl_->compute_dev, out.img, &sub, &slayout);
    out.stride = static_cast<uint32_t>(slayout.rowPitch);

    // Populate the image via staging buffer -> vkCmdCopyBufferToImage with a test pattern
    const VkDeviceSize buf_bytes = static_cast<VkDeviceSize>(width) * height * 4;
    VkBuffer st_buf = VK_NULL_HANDLE;
    VkDeviceMemory st_mem = VK_NULL_HANDLE;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, buf_bytes,
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    if (impl_->pfn_create_buf(impl_->compute_dev, &bci, nullptr, &st_buf) != VK_SUCCESS) {
        destroy_test_dmabuf(out);
        return std::unexpected("create_test_dmabuf: staging buffer failed");
    }
    VkMemoryRequirements sreq{};
    impl_->pfn_get_buf_mem_req(impl_->compute_dev, st_buf, &sreq);
    int smidx = impl_->find_memory_type(sreq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo smai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, sreq.size, static_cast<uint32_t>(smidx)};
    void *mapped = nullptr;
    if (impl_->pfn_alloc_mem(impl_->compute_dev, &smai, nullptr, &st_mem) != VK_SUCCESS ||
        impl_->pfn_bind_buf_mem(impl_->compute_dev, st_buf, st_mem, 0) != VK_SUCCESS ||
        impl_->pfn_map_mem(impl_->compute_dev, st_mem, 0, buf_bytes, 0, &mapped) != VK_SUCCESS) {
        if (st_buf) impl_->pfn_destroy_buf(impl_->compute_dev, st_buf, nullptr);
        if (st_mem) impl_->pfn_free_mem(impl_->compute_dev, st_mem, nullptr);
        destroy_test_dmabuf(out);
        return std::unexpected("create_test_dmabuf: staging alloc/map failed");
    }
    auto *px = static_cast<uint32_t *>(mapped);
    const uint32_t bar_x = (frame_idx * 37u) % width;
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t g = static_cast<uint8_t>((y * 255u) / height);
        for (uint32_t x = 0; x < width; ++x) {
            const uint8_t r = (x >= bar_x && x < bar_x + 64) ? 240u : static_cast<uint8_t>((x * 255u) / width);
            const uint8_t b = 128u;
            px[static_cast<size_t>(y) * width + x] = (0xFFu << 24) | (r << 16) | (g << 8) | b;
        }
    }
    impl_->pfn_unmap_mem(impl_->compute_dev, st_mem);

    CmdSlot &sl = impl_->slots[0];
    impl_->pfn_reset_cp(impl_->compute_dev, sl.pool, 0);
    impl_->pfn_reset_fences(impl_->compute_dev, 1, &sl.fence);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    impl_->pfn_begin_cb(sl.cmd, &cbbi);

    VkImageMemoryBarrier2 b1{};
    b1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b1.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b1.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b1.image = out.img;
    b1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo d1{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 0, nullptr, 1, &b1};
    impl_->pfn_cmd_barrier2(sl.cmd, &d1);

    VkBufferImageCopy reg{};
    reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    reg.imageExtent = {width, height, 1};
    impl_->pfn_cmd_copy_buf_to_img(sl.cmd, st_buf, out.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);

    VkImageMemoryBarrier2 b2{};
    b2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b2.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b2.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b2.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b2.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b2.srcQueueFamilyIndex = impl_->compute_qf_idx;
    b2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
    b2.image = out.img;
    b2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo d2{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 0, nullptr, 1, &b2};
    impl_->pfn_cmd_barrier2(sl.cmd, &d2);
    impl_->pfn_end_cb(sl.cmd);

    VkCommandBufferSubmitInfo cbsi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, sl.cmd, 0};
    VkSubmitInfo2 si2{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0, 0, nullptr, 1, &cbsi, 0, nullptr};
    impl_->pfn_queue_submit2(impl_->compute_queue, 1, &si2, sl.fence);
    impl_->pfn_wait_fences(impl_->compute_dev, 1, &sl.fence, VK_TRUE, UINT64_MAX);

    impl_->pfn_destroy_buf(impl_->compute_dev, st_buf, nullptr);
    impl_->pfn_free_mem(impl_->compute_dev, st_mem, nullptr);

    VkMemoryGetFdInfoKHR gfdi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, nullptr, out.mem, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    if (impl_->pfn_get_mem_fd(impl_->compute_dev, &gfdi, &out.fd) != VK_SUCCESS) {
        destroy_test_dmabuf(out);
        return std::unexpected("create_test_dmabuf: vkGetMemoryFdKHR failed");
    }
    return out;
}

void VulkanCompute::destroy_test_dmabuf(TestDmaBuf &buf) {
    if (buf.fd >= 0) { ::close(buf.fd); buf.fd = -1; }
    if (buf.img) { impl_->pfn_destroy_image(impl_->compute_dev, buf.img, nullptr); buf.img = VK_NULL_HANDLE; }
    if (buf.mem) { impl_->pfn_free_mem(impl_->compute_dev, buf.mem, nullptr); buf.mem = VK_NULL_HANDLE; }
}

}  // namespace vkcap
