#include "vulkan_compute.hpp"

#include "rgb_to_yuv_nv12_spv.hpp"
#include "rgb_to_yuv_p010_spv.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <format>
#include <iostream>
#include <libdrm/drm_fourcc.h>
#include <linux/dma-buf.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <vector>

namespace vkcap {

namespace {

struct DrmFormatMapping {
    uint32_t drm_fourcc;
    VkFormat vk_format;
    VkComponentMapping swizzle;
};

constexpr VkComponentMapping kSwizzleIdentity = {
    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};

constexpr VkComponentMapping kSwizzleOpaqueAlpha = {
    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
    VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_ONE};

constexpr std::array<DrmFormatMapping, 10> kFormatMap = {{
    {DRM_FORMAT_XRGB8888,      VK_FORMAT_B8G8R8A8_UNORM,           kSwizzleOpaqueAlpha},
    {DRM_FORMAT_ARGB8888,      VK_FORMAT_B8G8R8A8_UNORM,           kSwizzleIdentity},
    {DRM_FORMAT_XBGR8888,      VK_FORMAT_R8G8B8A8_UNORM,           kSwizzleOpaqueAlpha},
    {DRM_FORMAT_ABGR8888,      VK_FORMAT_R8G8B8A8_UNORM,           kSwizzleIdentity},
    {DRM_FORMAT_XRGB2101010,   VK_FORMAT_A2R10G10B10_UNORM_PACK32, kSwizzleOpaqueAlpha},
    {DRM_FORMAT_ARGB2101010,   VK_FORMAT_A2R10G10B10_UNORM_PACK32, kSwizzleIdentity},
    {DRM_FORMAT_XBGR2101010,   VK_FORMAT_A2B10G10R10_UNORM_PACK32, kSwizzleOpaqueAlpha},
    {DRM_FORMAT_ABGR2101010,   VK_FORMAT_A2B10G10R10_UNORM_PACK32, kSwizzleIdentity},
    {DRM_FORMAT_ABGR16161616F, VK_FORMAT_R16G16B16A16_SFLOAT,      kSwizzleIdentity},
    {DRM_FORMAT_XBGR16161616F, VK_FORMAT_R16G16B16A16_SFLOAT,      kSwizzleOpaqueAlpha},
}};

const DrmFormatMapping *find_drm_format(uint32_t fourcc) {
    for (const auto &m : kFormatMap) {
        if (m.drm_fourcc == fourcc) return &m;
    }
    return nullptr;
}

struct PushConstants {
    float yCoeffs[4];
    float cbCoeffs[4];
    float crCoeffs[4];
    float cursorRect[4];
    float invCursorSize[2];
    float srcOffset[2];
    float srcStepPx[2];
    float srcBasePx[2];
    float invSrcSize[2];
    uint32_t dstExtent[2];
};

struct SrcBinding {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
};

struct CachedSrcImage {
    uint64_t inode_key = 0;
    uint64_t last_used_frame = 0;
    VkImage image = VK_NULL_HANDLE;
    std::array<VkDeviceMemory, 4> mem{};
    uint32_t num_mems = 0;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t drm_fourcc = 0;
    uint64_t modifier = DRM_FORMAT_MOD_INVALID;
    bool has_modifier = false;
};

struct CachedDstAlias {
    VkImage key_img = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    std::array<VkDeviceMemory, 2> mem{};
    uint32_t num_mems = 0;
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

// Matches a DRM /dev/dri/card* node to its Vulkan physical device index via VK_EXT_physical_device_drm.
// Skips creating a temporary VkInstance on single-GPU systems so no extra Vulkan layers are loaded.
uint32_t resolve_vulkan_device_index(uint32_t default_idx, bool explicit_dev, std::string_view drm_card_path) {
    if (explicit_dev || drm_card_path.empty()) return default_idx;

    size_t card_count = 0;
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator("/dev/dri", ec)) {
        std::string name = entry.path().filename().string();
        if (name.starts_with("card") && name.size() > 4 && std::isdigit(static_cast<unsigned char>(name[4]))) {
            ++card_count;
        }
    }
    if (card_count <= 1) return default_idx;

    struct stat st{};
    std::string path_str(drm_card_path);
    if (::stat(path_str.c_str(), &st) != 0) return default_idx;
    const int64_t want_major = static_cast<int64_t>(major(st.st_rdev));
    const int64_t want_minor = static_cast<int64_t>(minor(st.st_rdev));

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkInstance tmp_inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ici, nullptr, &tmp_inst) != VK_SUCCESS || !tmp_inst) {
        return default_idx;
    }

    uint32_t matched_idx = default_idx;
    uint32_t dev_count = 0;
    if (vkEnumeratePhysicalDevices(tmp_inst, &dev_count, nullptr) == VK_SUCCESS && dev_count > 0) {
        std::vector<VkPhysicalDevice> devs(dev_count);
        if (vkEnumeratePhysicalDevices(tmp_inst, &dev_count, devs.data()) == VK_SUCCESS) {
            for (uint32_t i = 0; i < dev_count; ++i) {
                uint32_t ext_cnt = 0;
                vkEnumerateDeviceExtensionProperties(devs[i], nullptr, &ext_cnt, nullptr);
                std::vector<VkExtensionProperties> exts(ext_cnt);
                vkEnumerateDeviceExtensionProperties(devs[i], nullptr, &ext_cnt, exts.data());
                bool has_drm = false;
                for (const auto &e : exts) {
                    if (std::strcmp(e.extensionName, VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME) == 0) {
                        has_drm = true;
                        break;
                    }
                }
                if (!has_drm) continue;

                VkPhysicalDeviceDrmPropertiesEXT drm_props{};
                drm_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT;
                VkPhysicalDeviceProperties2 props2{};
                props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                props2.pNext = &drm_props;
                vkGetPhysicalDeviceProperties2(devs[i], &props2);

                if ((drm_props.hasPrimary && drm_props.primaryMajor == want_major && drm_props.primaryMinor == want_minor) ||
                    (drm_props.hasRender && drm_props.renderMajor == want_major && drm_props.renderMinor == want_minor)) {
                    matched_idx = i;
                    break;
                }
            }
        }
    }
    vkDestroyInstance(tmp_inst, nullptr);
    return matched_idx;
}

}  // namespace

struct VulkanCompute::Impl {
    Config cfg;
    bool convert_gamut = false;
    std::array<float, 9> gamut_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    AVBufferRef *hw_device_ref = nullptr;
    AVBufferRef *hw_frames_ref = nullptr;
    AVVulkanDeviceContext *vkctx = nullptr;

    VkDevice compute_dev = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props{};
    uint32_t compute_qf_idx = 0;
    VkQueue compute_queue = VK_NULL_HANDLE;

    PFN_vkGetDeviceProcAddr pfn_get_dev_proc = nullptr;
    PFN_vkCreateDevice pfn_create_device = nullptr;
    PFN_vkDestroyDevice pfn_destroy_device = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties pfn_enum_dev_exts = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties2 pfn_get_fmt_props2 = nullptr;
    PFN_vkGetPhysicalDeviceImageFormatProperties2 pfn_get_img_fmt_props2 = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties pfn_get_mem_props = nullptr;
    PFN_vkGetPhysicalDeviceProperties2 pfn_get_phys_props2 = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 pfn_get_phys_feats2 = nullptr;
    PFN_vkGetMemoryFdKHR pfn_ff_get_mem_fd = nullptr;
    PFN_vkGetSemaphoreCounterValue pfn_ff_get_sem_val = nullptr;
    PFN_vkWaitSemaphores pfn_ff_wait_sems = nullptr;

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
    PFN_vkCreatePipelineCache pfn_create_pipe_cache = nullptr;
    PFN_vkDestroyPipelineCache pfn_destroy_pipe_cache = nullptr;
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
    bool compute_has_video_maint1 = false;
    bool use_push_descriptors = false;

    VkSampler linear_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pipe_layout = VK_NULL_HANDLE;
    VkShaderModule shader_mod = VK_NULL_HANDLE;
    VkPipelineCache pipe_cache = VK_NULL_HANDLE;
    // Lazily compiled specialized pipelines: [exact1To1 ? 1 : 0][hasCursor ? 1 : 0]
    VkPipeline pipes[2][2]{};

    // 1x1 transparent dummy cursor image so binding 1 is always valid
    CachedSrcImage dummy_cursor{};

    static constexpr size_t kNumSlots = 4;
    static constexpr size_t kMaxCachedSurfaces = 6;
    static constexpr uint64_t kStaleSurfaceFrames = 180;
    std::array<CmdSlot, kNumSlots> slots{};
    size_t slot_cursor = 0;
    uint64_t frame_seq = 0;

    // Flat L1-cache-resident MRU tables (no mutex or hash overhead on single capture thread)
    std::vector<CachedSrcImage> src_cache;
    std::vector<CachedDstAlias> dst_alias_cache;
    std::array<AVFrame *, 8> probe_shells{};

    uint32_t out_width = 0;
    uint32_t out_height = 0;
    PushConstants base_pc{};
    std::atomic<uint64_t> stat_frames_processed{0};
    std::atomic<uint64_t> stat_total_compute_ns{0};

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
        for (uint32_t i = 0; i < d.num_mems; ++i) {
            if (d.mem[i]) pfn_free_mem(compute_dev, d.mem[i], nullptr);
        }
        d = {};
    }

    void clear_caches() {
        if (pfn_device_wait_idle && compute_dev) {
            pfn_device_wait_idle(compute_dev);
        }
        for (auto &img : src_cache) destroy_src_image(img);
        src_cache.clear();
        for (auto &da : dst_alias_cache) destroy_dst_alias(da);
        dst_alias_cache.clear();
    }

    ~Impl() {
        for (auto *&f : probe_shells) {
            if (f) av_frame_free(&f);
        }
        if (!compute_dev) {
            av_buffer_unref(&hw_frames_ref);
            av_buffer_unref(&hw_device_ref);
            return;
        }
        clear_caches();
        destroy_src_image(dummy_cursor);

        for (auto &sl : slots) {
            if (sl.ext_wait_sem) pfn_destroy_sem(compute_dev, sl.ext_wait_sem, nullptr);
            if (sl.fence) pfn_destroy_fence(compute_dev, sl.fence, nullptr);
            if (sl.desc_pool) pfn_destroy_dp(compute_dev, sl.desc_pool, nullptr);
            if (sl.pool) pfn_destroy_cp(compute_dev, sl.pool, nullptr);
        }
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                if (pipes[i][j]) pfn_destroy_pipe(compute_dev, pipes[i][j], nullptr);
            }
        }
        if (pipe_cache) pfn_destroy_pipe_cache(compute_dev, pipe_cache, nullptr);
        if (shader_mod) pfn_destroy_shader(compute_dev, shader_mod, nullptr);
        if (pipe_layout) pfn_destroy_pl(compute_dev, pipe_layout, nullptr);
        if (dsl) pfn_destroy_dsl(compute_dev, dsl, nullptr);
        if (linear_sampler) pfn_destroy_sampler(compute_dev, linear_sampler, nullptr);

        pfn_destroy_device(compute_dev, nullptr);
        compute_dev = VK_NULL_HANDLE;

        av_buffer_unref(&hw_frames_ref);
        av_buffer_unref(&hw_device_ref);
    }

    std::expected<void, std::string> init_dummy_cursor() {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_B8G8R8A8_UNORM;
        ici.extent = {1, 1, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (pfn_create_image(compute_dev, &ici, nullptr, &dummy_cursor.image) != VK_SUCCESS) {
            return std::unexpected("Failed to create dummy cursor image");
        }
        VkImageMemoryRequirementsInfo2 rinfo{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, nullptr, dummy_cursor.image};
        VkMemoryRequirements2 mreq{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, nullptr, {}};
        pfn_get_img_mem_req2(compute_dev, &rinfo, &mreq);
        int midx = find_memory_type(mreq.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (midx < 0) midx = find_memory_type(mreq.memoryRequirements.memoryTypeBits, 0);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mreq.memoryRequirements.size, static_cast<uint32_t>(midx)};
        if (pfn_alloc_mem(compute_dev, &mai, nullptr, &dummy_cursor.mem[0]) != VK_SUCCESS) {
            return std::unexpected("Failed to allocate dummy cursor memory");
        }
        dummy_cursor.num_mems = 1;
        VkBindImageMemoryInfo bimi{VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, nullptr, dummy_cursor.image, dummy_cursor.mem[0], 0};
        if (pfn_bind_image_mem2(compute_dev, 1, &bimi) != VK_SUCCESS) {
            return std::unexpected("Failed to bind dummy cursor memory");
        }
        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = dummy_cursor.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = VK_FORMAT_B8G8R8A8_UNORM;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (pfn_create_img_view(compute_dev, &ivci, nullptr, &dummy_cursor.view) != VK_SUCCESS) {
            return std::unexpected("Failed to create dummy cursor view");
        }

        // Transition dummy cursor to GENERAL once
        CmdSlot &sl = slots[0];
        pfn_reset_cp(compute_dev, sl.pool, 0);
        pfn_reset_fences(compute_dev, 1, &sl.fence);
        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        pfn_begin_cb(sl.cmd, &cbbi);
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.image = dummy_cursor.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 0, nullptr, 1, &b};
        pfn_cmd_barrier2(sl.cmd, &dep);
        pfn_end_cb(sl.cmd);
        VkCommandBufferSubmitInfo cbsi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, sl.cmd, 0};
        VkSubmitInfo2 si2{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0, 0, nullptr, 1, &cbsi, 0, nullptr};
        pfn_queue_submit2(compute_queue, 1, &si2, sl.fence);
        pfn_wait_fences(compute_dev, 1, &sl.fence, VK_TRUE, UINT64_MAX);
        return {};
    }

    std::expected<void, std::string> init_vulkan_objects() {
        pfn_get_dev_proc = load_inst<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
        pfn_create_device = load_inst<PFN_vkCreateDevice>("vkCreateDevice");
        pfn_enum_dev_exts = load_inst<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
        pfn_get_fmt_props2 = load_inst<PFN_vkGetPhysicalDeviceFormatProperties2>("vkGetPhysicalDeviceFormatProperties2");
        pfn_get_img_fmt_props2 = load_inst<PFN_vkGetPhysicalDeviceImageFormatProperties2>("vkGetPhysicalDeviceImageFormatProperties2");
        pfn_get_mem_props = load_inst<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties");
        pfn_get_phys_props2 = load_inst<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2");
        pfn_get_phys_feats2 = load_inst<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2");

        if (pfn_get_phys_props2) {
            VkPhysicalDeviceSubgroupProperties sg_props{};
            sg_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
            VkPhysicalDeviceProperties2 p2{};
            p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            p2.pNext = &sg_props;
            pfn_get_phys_props2(vkctx->phys_dev, &p2);
            if ((sg_props.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) == 0) {
                return std::unexpected("GPU does not support Vulkan subgroupShuffleUp (VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT)");
            }
        }

        pfn_ff_get_mem_fd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(pfn_get_dev_proc(vkctx->act_dev, "vkGetMemoryFdKHR"));
        pfn_ff_get_sem_val = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(pfn_get_dev_proc(vkctx->act_dev, "vkGetSemaphoreCounterValue"));
        pfn_ff_wait_sems = reinterpret_cast<PFN_vkWaitSemaphores>(pfn_get_dev_proc(vkctx->act_dev, "vkWaitSemaphores"));

        pfn_get_mem_props(vkctx->phys_dev, &mem_props);

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
        VkPhysicalDeviceVideoMaintenance1FeaturesKHR vm1_feat{};
        vm1_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR;
        if (has_ext(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) &&
            has_ext(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME)) {
            req_exts.push_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
            req_exts.push_back(VK_KHR_VIDEO_ENCODE_QUEUE_EXTENSION_NAME);
            compute_has_video_enc_ext = true;
            if (has_ext(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME) && pfn_get_phys_feats2) {
                VkPhysicalDeviceVideoMaintenance1FeaturesKHR vm1_query{};
                vm1_query.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_MAINTENANCE_1_FEATURES_KHR;
                VkPhysicalDeviceFeatures2 feats2{};
                feats2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                feats2.pNext = &vm1_query;
                pfn_get_phys_feats2(vkctx->phys_dev, &feats2);
                if (vm1_query.videoMaintenance1 == VK_TRUE) {
                    req_exts.push_back(VK_KHR_VIDEO_MAINTENANCE_1_EXTENSION_NAME);
                    vm1_feat.videoMaintenance1 = VK_TRUE;
                    compute_has_video_maint1 = true;
                }
            }
        }
        if (has_ext(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
            req_exts.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
            use_push_descriptors = true;
        }
        const bool has_global_prio = has_ext(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
        if (has_global_prio) {
            req_exts.push_back(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
        }

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
        f13.pNext = compute_has_video_maint1 ? &vm1_feat : nullptr;
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
        pfn_create_pipe_cache = load_dev<PFN_vkCreatePipelineCache>("vkCreatePipelineCache");
        pfn_destroy_pipe_cache = load_dev<PFN_vkDestroyPipelineCache>("vkDestroyPipelineCache");
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

        // Bindings: 0 = uSrc, 1 = uCursor, 2 = uDstY, 3 = uDstUV
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.flags = use_push_descriptors ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0;
        dslci.bindingCount = static_cast<uint32_t>(bindings.size());
        dslci.pBindings = bindings.data();
        if (pfn_create_dsl(compute_dev, &dslci, nullptr, &dsl) != VK_SUCCESS) {
            return std::unexpected("Failed to create descriptor set layout");
        }

        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
        if (pfn_create_pl(compute_dev, &plci, nullptr, &pipe_layout) != VK_SUCCESS) {
            return std::unexpected("Failed to create pipeline layout");
        }

        // Keep shader module + pipeline cache so specialized pipelines are compiled lazily
        // (typically only 1 variant for --no-cursor or 1-2 variants with cursor, instead of all 4).
        const bool is_p010 = (cfg.pixfmt == PixelFormat::P010);
        const uint32_t *spv_words = is_p010 ? kRgbToYuvP010Spv.data() : kRgbToYuvNv12Spv.data();
        const size_t spv_count = is_p010 ? kRgbToYuvP010Spv.size() : kRgbToYuvNv12Spv.size();

        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                      spv_count * sizeof(uint32_t), spv_words};
        if (pfn_create_shader(compute_dev, &smci, nullptr, &shader_mod) != VK_SUCCESS) {
            return std::unexpected("Failed to create Vulkan compute shader module");
        }
        VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, nullptr, 0, 0, nullptr};
        (void)pfn_create_pipe_cache(compute_dev, &pcci, nullptr, &pipe_cache);

        src_cache.reserve(kMaxCachedSurfaces);
        dst_alias_cache.reserve(32);

        // Pre-compile the expected 1:1 (or scaled) pipeline variant(s) at startup so frame 0 has zero compile stall
        const bool likely_1to1 = (cfg.out_width == 0 && cfg.out_height == 0);
        if (auto p = get_or_create_pipeline(likely_1to1, false); !p) {
            return std::unexpected(p.error());
        }
        if (cfg.capture_cursor) {
            if (auto p = get_or_create_pipeline(likely_1to1, true); !p) {
                return std::unexpected(p.error());
            }
        }

        for (auto &sl : slots) {
            VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                         VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, compute_qf_idx};
            if (pfn_create_cp(compute_dev, &cpci, nullptr, &sl.pool) != VK_SUCCESS) {
                return std::unexpected("Failed to create command pool");
            }
            VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                             sl.pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
            if (pfn_alloc_cb(compute_dev, &cbai, &sl.cmd) != VK_SUCCESS) {
                return std::unexpected("Failed to allocate command buffer");
            }
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0};
            if (pfn_create_fence(compute_dev, &fci, nullptr, &sl.fence) != VK_SUCCESS) {
                return std::unexpected("Failed to create fence");
            }
            VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, nullptr, 0};
            if (pfn_create_sem(compute_dev, &semci, nullptr, &sl.ext_wait_sem) != VK_SUCCESS) {
                return std::unexpected("Failed to create external wait semaphore");
            }
            if (!use_push_descriptors) {
                std::array<VkDescriptorPoolSize, 2> psizes = {{
                    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
                    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
                }};
                VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr,
                                                0, 1, static_cast<uint32_t>(psizes.size()), psizes.data()};
                if (pfn_create_dp(compute_dev, &dpci, nullptr, &sl.desc_pool) != VK_SUCCESS) {
                    return std::unexpected("Failed to create descriptor pool");
                }
                VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                 sl.desc_pool, 1, &dsl};
                if (pfn_alloc_ds(compute_dev, &dsai, &sl.desc_set) != VK_SUCCESS) {
                    return std::unexpected("Failed to allocate descriptor set");
                }
            }
        }

        return init_dummy_cursor();
    }

    std::expected<VkPipeline, std::string> get_or_create_pipeline(bool exact_1to1, bool has_cursor) {
        VkPipeline &slot = pipes[exact_1to1 ? 1 : 0][has_cursor ? 1 : 0];
        if (slot != VK_NULL_HANDLE) return slot;

        struct SpecData {
            VkBool32 exact1To1;
            VkBool32 hasCursor;
            VkBool32 enableDither;
            VkBool32 convertGamut;
            float gm[9];
        };
        constexpr std::array<VkSpecializationMapEntry, 13> spec_entries = {{
            {0,  offsetof(SpecData, exact1To1),    sizeof(VkBool32)},
            {1,  offsetof(SpecData, hasCursor),    sizeof(VkBool32)},
            {2,  offsetof(SpecData, enableDither), sizeof(VkBool32)},
            {3,  offsetof(SpecData, convertGamut), sizeof(VkBool32)},
            {4,  offsetof(SpecData, gm[0]),        sizeof(float)},
            {5,  offsetof(SpecData, gm[1]),        sizeof(float)},
            {6,  offsetof(SpecData, gm[2]),        sizeof(float)},
            {7,  offsetof(SpecData, gm[3]),        sizeof(float)},
            {8,  offsetof(SpecData, gm[4]),        sizeof(float)},
            {9,  offsetof(SpecData, gm[5]),        sizeof(float)},
            {10, offsetof(SpecData, gm[6]),        sizeof(float)},
            {11, offsetof(SpecData, gm[7]),        sizeof(float)},
            {12, offsetof(SpecData, gm[8]),        sizeof(float)},
        }};

        SpecData sd{};
        sd.exact1To1 = exact_1to1 ? VK_TRUE : VK_FALSE;
        sd.hasCursor = has_cursor ? VK_TRUE : VK_FALSE;
        sd.enableDither = cfg.dither ? VK_TRUE : VK_FALSE;
        sd.convertGamut = convert_gamut ? VK_TRUE : VK_FALSE;
        for (size_t k = 0; k < 9; ++k) sd.gm[k] = gamut_matrix[k];

        VkSpecializationInfo sinfo{static_cast<uint32_t>(spec_entries.size()), spec_entries.data(),
                                   sizeof(SpecData), &sd};
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_COMPUTE_BIT, shader_mod, "main", &sinfo};
        cpci.layout = pipe_layout;

        if (pfn_create_comp_pipe(compute_dev, pipe_cache, 1, &cpci, nullptr, &slot) != VK_SUCCESS) {
            slot = VK_NULL_HANDLE;
            return std::unexpected("Failed to create specialized Vulkan RGB->YUV compute pipeline");
        }
        return slot;
    }

    std::expected<CachedSrcImage, std::string> import_dmabuf(const KmsSurface &surf) {
        const DrmFormatMapping *fm = find_drm_format(surf.drm_fourcc);
        if (!fm) {
            return std::unexpected(std::format("Unsupported DRM fourcc 0x{:08x}", surf.drm_fourcc));
        }

        const uint32_t num_planes = std::clamp<uint32_t>(surf.num_planes, 1u, 4u);
        bool disjoint = false;
        if (surf.has_modifier && num_planes > 1) {
            for (uint32_t i = 1; i < num_planes; ++i) {
                if (surf.planes[i].fd != surf.planes[0].fd) {
                    disjoint = true;
                    break;
                }
            }
        }

        std::array<VkSubresourceLayout, 4> layouts{};
        for (uint32_t i = 0; i < num_planes; ++i) {
            layouts[i].offset = surf.planes[i].offset;
            layouts[i].rowPitch = surf.planes[i].stride;
        }

        VkExternalMemoryImageCreateInfo ext_img{
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, nullptr,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};

        VkImageDrmFormatModifierExplicitCreateInfoEXT mod_info{};
        mod_info.sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT;
        mod_info.drmFormatModifier = surf.drm_modifier;
        mod_info.drmFormatModifierPlaneCount = num_planes;
        mod_info.pPlaneLayouts = layouts.data();
        if (surf.has_modifier) ext_img.pNext = &mod_info;

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.pNext = &ext_img;
        ici.flags = disjoint ? VK_IMAGE_CREATE_DISJOINT_BIT : 0;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fm->vk_format;
        ici.extent = {surf.width, surf.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = surf.has_modifier ? VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT : VK_IMAGE_TILING_LINEAR;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        CachedSrcImage out{};
        out.width = surf.width;
        out.height = surf.height;
        out.drm_fourcc = surf.drm_fourcc;
        out.modifier = surf.drm_modifier;
        out.has_modifier = surf.has_modifier;

        if (pfn_create_image(compute_dev, &ici, nullptr, &out.image) != VK_SUCCESS) {
            return std::unexpected(std::format("vkCreateImage failed for KMS DMA-BUF {}x{} mod=0x{:x}",
                                               surf.width, surf.height, surf.drm_modifier));
        }

        if (!disjoint) {
            VkImageMemoryRequirementsInfo2 req_info{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, nullptr, out.image};
            VkMemoryDedicatedRequirements ded_req{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS, nullptr, VK_FALSE, VK_FALSE};
            VkMemoryRequirements2 mem_req{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &ded_req, {}};
            pfn_get_img_mem_req2(compute_dev, &req_info, &mem_req);

            int dup_fd = ::dup(surf.planes[0].fd);
            if (dup_fd < 0) {
                destroy_src_image(out);
                return std::unexpected("dup() failed on KMS DMA-BUF fd");
            }

            VkMemoryFdPropertiesKHR fd_props{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR, nullptr, 0};
            if (pfn_get_mem_fd_props(compute_dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dup_fd, &fd_props) != VK_SUCCESS) {
                ::close(dup_fd);
                destroy_src_image(out);
                return std::unexpected("vkGetMemoryFdPropertiesKHR failed");
            }

            uint32_t valid_bits = mem_req.memoryRequirements.memoryTypeBits & fd_props.memoryTypeBits;
            int mem_idx = find_memory_type(valid_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (mem_idx < 0) mem_idx = find_memory_type(valid_bits, 0);
            if (mem_idx < 0) {
                ::close(dup_fd);
                destroy_src_image(out);
                return std::unexpected("No compatible memory type for KMS DMA-BUF import");
            }

            VkImportMemoryFdInfoKHR imp_fd{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, nullptr,
                                           VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dup_fd};
            VkMemoryDedicatedAllocateInfo ded_alloc{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, nullptr, out.image, VK_NULL_HANDLE};
            if (ded_req.prefersDedicatedAllocation || ded_req.requiresDedicatedAllocation) {
                imp_fd.pNext = &ded_alloc;
            }

            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp_fd,
                                     std::max<VkDeviceSize>(mem_req.memoryRequirements.size, surf.planes[0].size),
                                     static_cast<uint32_t>(mem_idx)};
            if (pfn_alloc_mem(compute_dev, &mai, nullptr, &out.mem[0]) != VK_SUCCESS) {
                ::close(dup_fd);
                destroy_src_image(out);
                return std::unexpected("vkAllocateMemory(KMS DMA-BUF import) failed");
            }
            out.num_mems = 1;

            VkBindImageMemoryInfo bimi{VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, nullptr, out.image, out.mem[0], 0};
            if (pfn_bind_image_mem2(compute_dev, 1, &bimi) != VK_SUCCESS) {
                destroy_src_image(out);
                return std::unexpected("vkBindImageMemory2 failed on KMS DMA-BUF");
            }
        } else {
            static constexpr std::array<VkImageAspectFlagBits, 4> kMemPlaneAspects = {
                VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
                VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT,
                VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT,
                VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT,
            };
            std::array<VkBindImagePlaneMemoryInfo, 4> plane_binds{};
            std::array<VkBindImageMemoryInfo, 4> bind_infos{};
            std::array<int, 4> imported_fds{-1, -1, -1, -1};
            std::array<VkDeviceMemory, 4> plane_mems{};

            for (uint32_t p = 0; p < num_planes; ++p) {
                VkDeviceMemory chosen_mem = VK_NULL_HANDLE;
                for (uint32_t prev = 0; prev < p; ++prev) {
                    if (imported_fds[prev] == surf.planes[p].fd) {
                        chosen_mem = plane_mems[prev];
                        break;
                    }
                }
                if (chosen_mem == VK_NULL_HANDLE) {
                    VkImagePlaneMemoryRequirementsInfo plane_req{
                        VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO, nullptr, kMemPlaneAspects[p]};
                    VkImageMemoryRequirementsInfo2 req_info{
                        VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, &plane_req, out.image};
                    VkMemoryDedicatedRequirements ded_req{
                        VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS, nullptr, VK_FALSE, VK_FALSE};
                    VkMemoryRequirements2 mem_req{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &ded_req, {}};
                    pfn_get_img_mem_req2(compute_dev, &req_info, &mem_req);

                    int dup_fd = ::dup(surf.planes[p].fd);
                    if (dup_fd < 0) {
                        destroy_src_image(out);
                        return std::unexpected("dup() failed on multi-plane KMS DMA-BUF fd");
                    }
                    VkMemoryFdPropertiesKHR fd_props{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR, nullptr, 0};
                    if (pfn_get_mem_fd_props(compute_dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dup_fd, &fd_props) != VK_SUCCESS) {
                        ::close(dup_fd);
                        destroy_src_image(out);
                        return std::unexpected("vkGetMemoryFdPropertiesKHR failed on multi-plane KMS DMA-BUF");
                    }
                    uint32_t valid_bits = mem_req.memoryRequirements.memoryTypeBits & fd_props.memoryTypeBits;
                    int mem_idx = find_memory_type(valid_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                    if (mem_idx < 0) mem_idx = find_memory_type(valid_bits, 0);
                    if (mem_idx < 0) {
                        ::close(dup_fd);
                        destroy_src_image(out);
                        return std::unexpected("No compatible memory type for multi-plane KMS DMA-BUF import");
                    }
                    VkImportMemoryFdInfoKHR imp_fd{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, nullptr,
                                                   VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, dup_fd};
                    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp_fd,
                                             std::max<VkDeviceSize>(mem_req.memoryRequirements.size, surf.planes[p].size),
                                             static_cast<uint32_t>(mem_idx)};
                    if (pfn_alloc_mem(compute_dev, &mai, nullptr, &chosen_mem) != VK_SUCCESS) {
                        ::close(dup_fd);
                        destroy_src_image(out);
                        return std::unexpected("vkAllocateMemory(multi-plane KMS DMA-BUF) failed");
                    }
                    out.mem[out.num_mems++] = chosen_mem;
                }
                imported_fds[p] = surf.planes[p].fd;
                plane_mems[p] = chosen_mem;
                plane_binds[p] = {VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO, nullptr, kMemPlaneAspects[p]};
                bind_infos[p] = {VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, &plane_binds[p], out.image, chosen_mem, 0};
            }

            if (pfn_bind_image_mem2(compute_dev, num_planes, bind_infos.data()) != VK_SUCCESS) {
                destroy_src_image(out);
                return std::unexpected("vkBindImageMemory2 failed on multi-plane KMS DMA-BUF");
            }
        }

        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = out.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = fm->vk_format;
        ivci.components = fm->swizzle;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (pfn_create_img_view(compute_dev, &ivci, nullptr, &out.view) != VK_SUCCESS) {
            destroy_src_image(out);
            return std::unexpected("vkCreateImageView failed on imported KMS DMA-BUF");
        }
        return out;
    }

    void evict_stale_src_surfaces() {
        while (!src_cache.empty() &&
               (frame_seq - src_cache.back().last_used_frame) > kStaleSurfaceFrames) {
            destroy_src_image(src_cache.back());
            src_cache.pop_back();
        }
    }

    std::expected<SrcBinding, std::string> get_or_import_surface(const KmsSurface &surf) {
        // Flat MRU search: steady-state triple-buffered scanout hits index 0..2 in L1 cache
        for (size_t i = 0; i < src_cache.size(); ++i) {
            if (src_cache[i].inode_key == surf.inode_key) {
                auto &c = src_cache[i];
                if (c.width == surf.width && c.height == surf.height &&
                    c.drm_fourcc == surf.drm_fourcc && c.modifier == surf.drm_modifier &&
                    c.has_modifier == surf.has_modifier) {
                    c.last_used_frame = frame_seq;
                    if (i > 0) {
                        CachedSrcImage tmp = c;
                        for (size_t k = i; k > 0; --k) src_cache[k] = src_cache[k - 1];
                        src_cache[0] = tmp;
                    }
                    return SrcBinding{src_cache[0].image, src_cache[0].view};
                }
                destroy_src_image(c);
                src_cache.erase(src_cache.begin() + static_cast<ptrdiff_t>(i));
                break;
            }
        }

        // Evict least-recently-used tail entry when full
        if (src_cache.size() >= kMaxCachedSurfaces) {
            destroy_src_image(src_cache.back());
            src_cache.pop_back();
        }

        auto imported = import_dmabuf(surf);
        if (!imported) return std::unexpected(imported.error());
        imported->inode_key = surf.inode_key;
        imported->last_used_frame = frame_seq;
        src_cache.insert(src_cache.begin(), *imported);
        return SrcBinding{src_cache[0].image, src_cache[0].view};
    }

    std::expected<CachedDstAlias *, std::string> get_or_create_dst_alias(const AVVkFrame *vkf) {
        const VkImage key = vkf->img[0];
        for (auto &da : dst_alias_cache) {
            if (da.key_img == key) return &da;
        }

        const bool disjoint_planes = (vkf->mem[1] != VK_NULL_HANDLE && vkf->mem[1] != vkf->mem[0]);

        VkExternalMemoryImageCreateInfo ext_img{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, nullptr,
                                                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
        VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (compute_has_video_enc_ext) usage |= VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR;

        VkImageCreateFlags create_flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                                          VK_IMAGE_CREATE_EXTENDED_USAGE_BIT |
                                          VK_IMAGE_CREATE_ALIAS_BIT;
        if (compute_has_video_maint1) {
            create_flags |= VK_IMAGE_CREATE_VIDEO_PROFILE_INDEPENDENT_BIT_KHR;
        }
        if (disjoint_planes) {
            create_flags |= VK_IMAGE_CREATE_DISJOINT_BIT;
        }

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.pNext = &ext_img;
        ici.flags = create_flags;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = (cfg.pixfmt == PixelFormat::P010)
            ? VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
            : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        ici.extent = {out_width, out_height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = vkf->tiling;
        ici.usage = usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        CachedDstAlias da{};
        da.key_img = key;
        if (pfn_create_image(compute_dev, &ici, nullptr, &da.image) != VK_SUCCESS) {
            return std::unexpected("Failed to create aliased AVVkFrame image on compute_dev");
        }

        const uint32_t n_mems = disjoint_planes ? 2u : 1u;
        static constexpr std::array<VkImageAspectFlagBits, 2> kPlaneAspects = {
            VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT};
        std::array<VkBindImagePlaneMemoryInfo, 2> plane_binds{};
        std::array<VkBindImageMemoryInfo, 2> bind_infos{};

        for (uint32_t p = 0; p < n_mems; ++p) {
            int mem_fd = -1;
            VkMemoryGetFdInfoKHR gfdi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, nullptr,
                                      vkf->mem[p], VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
            if (pfn_ff_get_mem_fd(vkctx->act_dev, &gfdi, &mem_fd) != VK_SUCCESS || mem_fd < 0) {
                destroy_dst_alias(da);
                return std::unexpected("Failed to export AVVkFrame memory as OPAQUE_FD");
            }

            VkImagePlaneMemoryRequirementsInfo preq{
                VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO, nullptr, kPlaneAspects[p]};
            VkImageMemoryRequirementsInfo2 rinfo{
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, disjoint_planes ? &preq : nullptr, da.image};
            VkMemoryDedicatedRequirements ded_req{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS, nullptr, VK_FALSE, VK_FALSE};
            VkMemoryRequirements2 mreq{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &ded_req, {}};
            pfn_get_img_mem_req2(compute_dev, &rinfo, &mreq);

            int midx = find_memory_type(mreq.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (midx < 0) midx = find_memory_type(mreq.memoryRequirements.memoryTypeBits, 0);

            VkImportMemoryFdInfoKHR imp_fd{VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR, nullptr,
                                           VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT, mem_fd};
            VkMemoryDedicatedAllocateInfo ded_alloc{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, nullptr, da.image, VK_NULL_HANDLE};
            if (!disjoint_planes && (ded_req.prefersDedicatedAllocation || ded_req.requiresDedicatedAllocation)) {
                imp_fd.pNext = &ded_alloc;
            }

            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp_fd,
                                     std::max<VkDeviceSize>(mreq.memoryRequirements.size, vkf->size[p]),
                                     static_cast<uint32_t>(midx)};
            if (pfn_alloc_mem(compute_dev, &mai, nullptr, &da.mem[p]) != VK_SUCCESS) {
                ::close(mem_fd);
                destroy_dst_alias(da);
                return std::unexpected("Failed to import AVVkFrame OPAQUE_FD memory on compute_dev");
            }
            da.num_mems = p + 1;

            if (disjoint_planes) {
                plane_binds[p] = {VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO, nullptr, kPlaneAspects[p]};
                bind_infos[p] = {VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, &plane_binds[p],
                                 da.image, da.mem[p], static_cast<VkDeviceSize>(vkf->offset[p])};
            } else {
                bind_infos[0] = {VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO, nullptr,
                                 da.image, da.mem[0], static_cast<VkDeviceSize>(vkf->offset[0])};
            }
        }

        if (pfn_bind_image_mem2(compute_dev, n_mems, bind_infos.data()) != VK_SUCCESS) {
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

        dst_alias_cache.push_back(da);
        return &dst_alias_cache.back();
    }
};

VulkanCompute::~VulkanCompute() = default;

std::expected<std::unique_ptr<VulkanCompute>, std::string> VulkanCompute::create(
    const Config &cfg,
    bool convert_gamut,
    const std::array<float, 9> &gamut_matrix,
    std::string_view drm_card_path) {
    auto vc = std::unique_ptr<VulkanCompute>(new VulkanCompute());
    vc->impl_ = std::make_unique<Impl>();
    Impl *impl = vc->impl_.get();
    impl->cfg = cfg;
    impl->convert_gamut = convert_gamut;
    impl->gamut_matrix = gamut_matrix;

    const uint32_t resolved_dev_idx = resolve_vulkan_device_index(
        cfg.device_index, cfg.device_explicit, drm_card_path);
    std::string dev_str = std::to_string(resolved_dev_idx);
    int err = av_hwdevice_ctx_create(&impl->hw_device_ref, AV_HWDEVICE_TYPE_VULKAN, dev_str.c_str(), nullptr, 0);
    if (err < 0) {
        char buf[128];
        av_strerror(err, buf, sizeof(buf));
        return std::unexpected(std::format("Failed to create FFmpeg Vulkan device '{}': {}", dev_str, buf));
    }

    auto *hwdev = reinterpret_cast<AVHWDeviceContext *>(impl->hw_device_ref->data);
    impl->vkctx = reinterpret_cast<AVVulkanDeviceContext *>(hwdev->hwctx);

    auto res = impl->init_vulkan_objects();
    if (!res) return std::unexpected(res.error());

    if (cfg.verbose) {
        VkPhysicalDeviceProperties props{};
        auto pfn_get_props = impl->load_inst<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties");
        pfn_get_props(impl->vkctx->phys_dev, &props);
        std::cerr << std::format("[vulkan] device[{}]='{}' isolated compute_qf={} push_desc={} video_maint1={}\n",
                                 resolved_dev_idx, props.deviceName, impl->compute_qf_idx,
                                 impl->use_push_descriptors ? "yes" : "no",
                                 impl->compute_has_video_maint1 ? "yes" : "no");
    }

    return vc;
}

std::expected<void, std::string> VulkanCompute::configure_output(uint32_t width, uint32_t height) {
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
    if (!fref) return std::unexpected("av_hwframe_ctx_alloc failed");

    auto *fc = reinterpret_cast<AVHWFramesContext *>(fref->data);
    auto *vkfc = reinterpret_cast<AVVulkanFramesContext *>(fc->hwctx);
    fc->format = AV_PIX_FMT_VULKAN;
    fc->sw_format = (impl_->cfg.pixfmt == PixelFormat::P010) ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12;
    fc->width = static_cast<int>(width);
    fc->height = static_cast<int>(height);
    fc->initial_pool_size = static_cast<int>(impl_->cfg.queue_size) + impl_->cfg.async_depth + 4;
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

    // Precompute static color-space/range affine matrix & output extent once
    const float kr = (impl_->cfg.color_space == ColorSpace::Bt2020) ? 0.2627f : 0.2126f;
    const float kb = (impl_->cfg.color_space == ColorSpace::Bt2020) ? 0.0593f : 0.0722f;
    const float kg = 1.0f - kr - kb;

    const bool is_p010 = (impl_->cfg.pixfmt == PixelFormat::P010);
    const bool is_limited = (impl_->cfg.color_range == ColorRange::Limited);
    const float y_scale  = is_limited ? (is_p010 ? (876.0f / 1023.0f) : (219.0f / 255.0f)) : 1.0f;
    const float y_offset = is_limited ? (is_p010 ? (64.0f / 1023.0f)  : (16.0f / 255.0f))  : 0.0f;
    const float c_scale  = is_limited ? (is_p010 ? (896.0f / 1023.0f) : (224.0f / 255.0f)) : 1.0f;
    const float c_offset = is_p010 ? (512.0f / 1023.0f) : (128.0f / 255.0f);
    const float cb_norm  = c_scale / (2.0f * (1.0f - kb));
    const float cr_norm  = c_scale / (2.0f * (1.0f - kr));

    PushConstants &bpc = impl_->base_pc;
    bpc = {};
    bpc.yCoeffs[0]  = kr * y_scale;
    bpc.yCoeffs[1]  = kg * y_scale;
    bpc.yCoeffs[2]  = kb * y_scale;
    bpc.yCoeffs[3]  = y_offset;
    bpc.cbCoeffs[0] = -kr * cb_norm;
    bpc.cbCoeffs[1] = -kg * cb_norm;
    bpc.cbCoeffs[2] = (1.0f - kb) * cb_norm;
    bpc.cbCoeffs[3] = c_offset;
    bpc.crCoeffs[0] = (1.0f - kr) * cr_norm;
    bpc.crCoeffs[1] = -kg * cr_norm;
    bpc.crCoeffs[2] = -kb * cr_norm;
    bpc.crCoeffs[3] = c_offset;
    bpc.dstExtent[0] = width;
    bpc.dstExtent[1] = height;
    return {};
}

std::expected<AVFrame *, std::string> VulkanCompute::process_frame(const KmsFrame &kf) {
    if (!impl_->hw_frames_ref) return std::unexpected("VulkanCompute output not configured");

    ++impl_->frame_seq;
    if ((impl_->frame_seq % 60) == 0) {
        impl_->evict_stale_src_surfaces();
    }

    // LIFO pool probing using reusable empty AVFrame shells to avoid malloc/free thrashing
    size_t nb_busy = 0;
    AVFrame *dst_frame = nullptr;

    for (size_t attempt = 0; attempt < impl_->probe_shells.size(); ++attempt) {
        if (!impl_->probe_shells[attempt]) {
            impl_->probe_shells[attempt] = av_frame_alloc();
            if (!impl_->probe_shells[attempt]) break;
        }
        AVFrame *cand = impl_->probe_shells[attempt];
        int err = av_hwframe_get_buffer(impl_->hw_frames_ref, cand, 0);
        if (err < 0) {
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
            impl_->probe_shells[attempt] = nullptr;
            break;
        }
        nb_busy = attempt + 1;
    }

    if (!dst_frame && nb_busy > 0) {
        --nb_busy;
        dst_frame = impl_->probe_shells[nb_busy];
        impl_->probe_shells[nb_busy] = nullptr;
        auto *vkf_wait = reinterpret_cast<AVVkFrame *>(dst_frame->data[0]);
        if (impl_->pfn_ff_wait_sems) {
            VkSemaphoreWaitInfo swi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, nullptr, 0, 1,
                                    &vkf_wait->sem[0], &vkf_wait->sem_value[0]};
            impl_->pfn_ff_wait_sems(impl_->vkctx->act_dev, &swi, UINT64_MAX);
        }
    }
    for (size_t i = 0; i < nb_busy; ++i) {
        if (impl_->probe_shells[i]) av_frame_unref(impl_->probe_shells[i]);
    }
    if (!dst_frame) return std::unexpected("Failed to acquire AVVkFrame from pool");

    auto *vkf = reinterpret_cast<AVVkFrame *>(dst_frame->data[0]);
    auto dst_alias_res = impl_->get_or_create_dst_alias(vkf);
    if (!dst_alias_res) {
        av_frame_free(&dst_frame);
        return std::unexpected(dst_alias_res.error());
    }
    CachedDstAlias *dst_alias = *dst_alias_res;

    auto src_res = impl_->get_or_import_surface(kf.primary);
    if (!src_res) {
        av_frame_free(&dst_frame);
        return std::unexpected(src_res.error());
    }
    const SrcBinding src = *src_res;

    SrcBinding cur_img{impl_->dummy_cursor.image, impl_->dummy_cursor.view};
    bool has_cursor = false;
    if (kf.cursor.visible && kf.cursor.surface.num_planes > 0) {
        auto cur_res = impl_->get_or_import_surface(kf.cursor.surface);
        if (cur_res) {
            cur_img = *cur_res;
            has_cursor = true;
        }
    }

    const float crop_x = kf.crop.valid() ? static_cast<float>(kf.crop.x) : 0.0f;
    const float crop_y = kf.crop.valid() ? static_cast<float>(kf.crop.y) : 0.0f;
    const float crop_w = kf.crop.valid() ? static_cast<float>(kf.crop.width) : static_cast<float>(kf.primary.width);
    const float crop_h = kf.crop.valid() ? static_cast<float>(kf.crop.height) : static_cast<float>(kf.primary.height);

    const bool exact_1to1 =
        (static_cast<uint32_t>(crop_w) == impl_->out_width) &&
        (static_cast<uint32_t>(crop_h) == impl_->out_height);

    auto pipe_res = impl_->get_or_create_pipeline(exact_1to1, has_cursor);
    if (!pipe_res) {
        av_frame_free(&dst_frame);
        return std::unexpected(pipe_res.error());
    }
    VkPipeline pipe = *pipe_res;

    const auto t0 = std::chrono::steady_clock::now();

    CmdSlot &sl = impl_->slots[impl_->slot_cursor];
    impl_->slot_cursor = (impl_->slot_cursor + 1) % Impl::kNumSlots;

    impl_->pfn_reset_cp(impl_->compute_dev, sl.pool, 0);
    impl_->pfn_reset_fences(impl_->compute_dev, 1, &sl.fence);

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                  VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    if (impl_->pfn_begin_cb(sl.cmd, &cbbi) != VK_SUCCESS) {
        av_frame_free(&dst_frame);
        return std::unexpected("vkBeginCommandBuffer failed");
    }

    int sync_fd = -1;
#ifdef DMA_BUF_IOCTL_EXPORT_SYNC_FILE
    if (kf.primary.planes[0].fd >= 0) {
        dma_buf_export_sync_file ex{DMA_BUF_SYNC_READ, -1};
        if (::ioctl(kf.primary.planes[0].fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE, &ex) == 0 && ex.fd >= 0) {
            pollfd pfd{ex.fd, POLLIN, 0};
            if (::poll(&pfd, 1, 0) > 0 && (pfd.revents & POLLIN) != 0) {
                ::close(ex.fd);
            } else {
                sync_fd = ex.fd;
            }
        }
    }
#endif

    bool has_ext_sem = false;
    if (sync_fd >= 0) {
        VkImportSemaphoreFdInfoKHR isfi{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR, nullptr,
                                        sl.ext_wait_sem, VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
                                        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, sync_fd};
        if (impl_->pfn_import_sem_fd(impl_->compute_dev, &isfi) == VK_SUCCESS) {
            has_ext_sem = true;
        } else {
            ::close(sync_fd);
        }
    }

    VkDescriptorImageInfo src_di{impl_->linear_sampler, src.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo cur_di{impl_->linear_sampler, cur_img.view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo y_di{VK_NULL_HANDLE, dst_alias->y_view, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo uv_di{VK_NULL_HANDLE, dst_alias->uv_view, VK_IMAGE_LAYOUT_GENERAL};

    VkDescriptorSet ds = impl_->use_push_descriptors ? VK_NULL_HANDLE : sl.desc_set;
    std::array<VkWriteDescriptorSet, 4> writes{};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &src_di, nullptr, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cur_di, nullptr, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &y_di, nullptr, nullptr};
    writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &uv_di, nullptr, nullptr};
    if (!impl_->use_push_descriptors) {
        impl_->pfn_update_ds(impl_->compute_dev, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    std::array<VkImageMemoryBarrier2, 3> pre_barriers{};
    uint32_t nb_pre = 0;
    {
        auto &b = pre_barriers[nb_pre++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        b.dstQueueFamilyIndex = impl_->compute_qf_idx;
        b.image = src.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    if (has_cursor) {
        auto &b = pre_barriers[nb_pre++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        b.dstQueueFamilyIndex = impl_->compute_qf_idx;
        b.image = cur_img.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    if (!dst_alias->initialized_layout) {
        auto &b = pre_barriers[nb_pre++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
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

    VkDependencyInfo dep_pre{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 0, nullptr, nb_pre, pre_barriers.data()};
    impl_->pfn_cmd_barrier2(sl.cmd, &dep_pre);

    impl_->pfn_cmd_bind_pipe(sl.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    if (impl_->use_push_descriptors) {
        impl_->pfn_cmd_push_desc(sl.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipe_layout, 0,
                                 static_cast<uint32_t>(writes.size()), writes.data());
    } else {
        impl_->pfn_cmd_bind_ds(sl.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl_->pipe_layout, 0, 1, &sl.desc_set, 0, nullptr);
    }

    const float inv_src_w = 1.0f / static_cast<float>(kf.primary.width);
    const float inv_src_h = 1.0f / static_cast<float>(kf.primary.height);
    const float step_x = crop_w / static_cast<float>(impl_->out_width);
    const float step_y = crop_h / static_cast<float>(impl_->out_height);

    PushConstants pc = impl_->base_pc;
    if (has_cursor) {
        const float cw = static_cast<float>(kf.cursor.width);
        const float ch = static_cast<float>(kf.cursor.height);
        pc.cursorRect[0] = static_cast<float>(kf.cursor.x);
        pc.cursorRect[1] = static_cast<float>(kf.cursor.y);
        pc.cursorRect[2] = cw;
        pc.cursorRect[3] = ch;
        pc.invCursorSize[0] = 1.0f / cw;
        pc.invCursorSize[1] = 1.0f / ch;
    }
    pc.srcOffset[0] = crop_x;
    pc.srcOffset[1] = crop_y;
    pc.srcStepPx[0] = step_x;
    pc.srcStepPx[1] = step_y;
    pc.srcBasePx[0] = crop_x + 0.5f * step_x;
    pc.srcBasePx[1] = crop_y + 0.5f * step_y;
    pc.invSrcSize[0] = inv_src_w;
    pc.invSrcSize[1] = inv_src_h;
    impl_->pfn_cmd_push(sl.cmd, impl_->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // Workgroup is 16x4 threads, each covering 2x2 luma = 32x8 pixels per workgroup
    impl_->pfn_cmd_dispatch(sl.cmd, (impl_->out_width + 31u) / 32u, (impl_->out_height + 7u) / 8u, 1);

    std::array<VkImageMemoryBarrier2, 3> post_barriers{};
    uint32_t nb_post = 0;
    {
        auto &b = post_barriers[nb_post++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = impl_->compute_qf_idx;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        b.image = dst_alias->image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    {
        // Read-only sampled source: srcAccessMask = 0 avoids unnecessary writeback cache flushes
        auto &b = post_barriers[nb_post++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = 0;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = impl_->compute_qf_idx;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        b.image = src.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    if (has_cursor) {
        auto &b = post_barriers[nb_post++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = 0;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = impl_->compute_qf_idx;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        b.image = cur_img.image;
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
    VkSubmitInfo2 si2{VK_STRUCTURE_TYPE_SUBMIT_INFO_2, nullptr, 0,
                      has_ext_sem ? 1u : 0u, has_ext_sem ? &wait_ext : nullptr,
                      1, &cbsi, 0, nullptr};

    if (impl_->pfn_queue_submit2(impl_->compute_queue, 1, &si2, sl.fence) != VK_SUCCESS) {
        av_frame_free(&dst_frame);
        return std::unexpected("vkQueueSubmit2 failed for compute color conversion");
    }

    impl_->pfn_wait_fences(impl_->compute_dev, 1, &sl.fence, VK_TRUE, UINT64_MAX);

    dst_frame->color_range = (impl_->cfg.color_range == ColorRange::Full) ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
    dst_frame->colorspace = (impl_->cfg.color_space == ColorSpace::Bt2020) ? AVCOL_SPC_BT2020_NCL : AVCOL_SPC_BT709;
    dst_frame->color_primaries = (impl_->cfg.color_space == ColorSpace::Bt2020) ? AVCOL_PRI_BT2020 : AVCOL_PRI_BT709;
    dst_frame->color_trc = (impl_->cfg.color_space == ColorSpace::Bt2020) ? AVCOL_TRC_BT2020_10 : AVCOL_TRC_BT709;
    dst_frame->chroma_location = AVCHROMA_LOC_LEFT;

    const auto t1 = std::chrono::steady_clock::now();
    impl_->stat_frames_processed.fetch_add(1, std::memory_order_relaxed);
    impl_->stat_total_compute_ns.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
        std::memory_order_relaxed);

    return dst_frame;
}

AVBufferRef *VulkanCompute::hw_device_ref() const noexcept { return impl_->hw_device_ref; }
AVBufferRef *VulkanCompute::hw_frames_ref() const noexcept { return impl_->hw_frames_ref; }
uint32_t VulkanCompute::output_width() const noexcept { return impl_->out_width; }
uint32_t VulkanCompute::output_height() const noexcept { return impl_->out_height; }
ComputeStats VulkanCompute::stats() const noexcept {
    return {
        impl_->stat_frames_processed.load(std::memory_order_relaxed),
        impl_->stat_total_compute_ns.load(std::memory_order_relaxed),
    };
}

std::expected<VulkanCompute::TestDmaBuf, std::string> VulkanCompute::create_test_dmabuf(
    uint32_t width, uint32_t height, uint32_t frame_idx, bool is_cursor) {
    VkDrmFormatModifierPropertiesListEXT mod_list{VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT, nullptr, 0, nullptr};
    VkFormatProperties2 fp2{VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, &mod_list, {}};
    impl_->pfn_get_fmt_props2(impl_->vkctx->phys_dev, VK_FORMAT_B8G8R8A8_UNORM, &fp2);
    std::vector<VkDrmFormatModifierPropertiesEXT> mods(mod_list.drmFormatModifierCount);
    if (!mods.empty()) {
        mod_list.pDrmFormatModifierProperties = mods.data();
        impl_->pfn_get_fmt_props2(impl_->vkctx->phys_dev, VK_FORMAT_B8G8R8A8_UNORM, &fp2);
    }
    uint64_t mod = DRM_FORMAT_MOD_LINEAR;
    for (const auto &m : mods) {
        if (m.drmFormatModifier != DRM_FORMAT_MOD_LINEAR &&
            (m.drmFormatModifierTilingFeatures & (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) ==
            (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) {
            mod = m.drmFormatModifier;
            break;
        }
    }

    VkImageDrmFormatModifierListCreateInfoEXT mod_ci{VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT, nullptr, 1, &mod};
    VkExternalMemoryImageCreateInfo ext_img{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, &mod_ci,
                                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
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

    VkMemoryDedicatedAllocateInfo ded_ai{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, nullptr, out.img, VK_NULL_HANDLE};
    VkExportMemoryAllocateInfo exp_ai{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &ded_ai,
                                      VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
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

    VkImageDrmFormatModifierPropertiesEXT img_mod{VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT, nullptr, 0};
    if (impl_->pfn_get_img_drm_mod(impl_->compute_dev, out.img, &img_mod) == VK_SUCCESS) {
        out.modifier = img_mod.drmFormatModifier;
    }

    auto pfn_sub_layout = impl_->load_dev<PFN_vkGetImageSubresourceLayout>("vkGetImageSubresourceLayout");
    VkImageSubresource sub{VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT, 0, 0};
    VkSubresourceLayout slayout{};
    pfn_sub_layout(impl_->compute_dev, out.img, &sub, &slayout);
    out.stride = static_cast<uint32_t>(slayout.rowPitch);

    const VkDeviceSize buf_bytes = static_cast<VkDeviceSize>(width) * height * 4;
    VkBuffer st_buf = VK_NULL_HANDLE;
    VkDeviceMemory st_mem = VK_NULL_HANDLE;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, buf_bytes,
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
    impl_->pfn_create_buf(impl_->compute_dev, &bci, nullptr, &st_buf);
    VkMemoryRequirements sreq{};
    impl_->pfn_get_buf_mem_req(impl_->compute_dev, st_buf, &sreq);
    int smidx = impl_->find_memory_type(sreq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkMemoryAllocateInfo smai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, sreq.size, static_cast<uint32_t>(smidx)};
    void *mapped = nullptr;
    impl_->pfn_alloc_mem(impl_->compute_dev, &smai, nullptr, &st_mem);
    impl_->pfn_bind_buf_mem(impl_->compute_dev, st_buf, st_mem, 0);
    impl_->pfn_map_mem(impl_->compute_dev, st_mem, 0, buf_bytes, 0, &mapped);

    auto *px = static_cast<uint32_t *>(mapped);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            if (is_cursor) {
                bool arrow = (x < 16 && y < 16 && x <= y);
                px[static_cast<size_t>(y) * width + x] = arrow ? 0xFFFFFFFFu : 0x00000000u;
            } else {
                const uint32_t bar_x = (frame_idx * 37u) % width;
                uint8_t r = (x >= bar_x && x < bar_x + 64) ? 240u : static_cast<uint8_t>((x * 255u) / width);
                uint8_t g = static_cast<uint8_t>((y * 255u) / height);
                px[static_cast<size_t>(y) * width + x] = (0xFFu << 24) | (r << 16) | (g << 8) | 128u;
            }
        }
    }
    impl_->pfn_unmap_mem(impl_->compute_dev, st_mem);

    CmdSlot &sl = impl_->slots[0];
    impl_->pfn_reset_cp(impl_->compute_dev, sl.pool, 0);
    impl_->pfn_reset_fences(impl_->compute_dev, 1, &sl.fence);
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    impl_->pfn_begin_cb(sl.cmd, &cbbi);

    VkImageMemoryBarrier2 b1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr, 0, 0,
                             VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                             out.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d1{VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 0, nullptr, 0, nullptr, 1, &b1};
    impl_->pfn_cmd_barrier2(sl.cmd, &d1);

    VkBufferImageCopy reg{};
    reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    reg.imageExtent = {width, height, 1};
    impl_->pfn_cmd_copy_buf_to_img(sl.cmd, st_buf, out.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);

    VkImageMemoryBarrier2 b2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, nullptr,
                             VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                             impl_->compute_qf_idx, VK_QUEUE_FAMILY_EXTERNAL,
                             out.img, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
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

    struct stat st{};
    if (::fstat(out.fd, &st) == 0) out.inode_key = static_cast<uint64_t>(st.st_ino);
    return out;
}

void VulkanCompute::destroy_test_dmabuf(TestDmaBuf &buf) {
    if (buf.fd >= 0) { ::close(buf.fd); buf.fd = -1; }
    if (buf.img) { impl_->pfn_destroy_image(impl_->compute_dev, buf.img, nullptr); buf.img = VK_NULL_HANDLE; }
    if (buf.mem) { impl_->pfn_free_mem(impl_->compute_dev, buf.mem, nullptr); buf.mem = VK_NULL_HANDLE; }
}

}  // namespace vkcap
