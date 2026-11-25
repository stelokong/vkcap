# Development Process & Engineering Chronicle

This document chronicles how `vkcap` was designed, built, debugged, and iteratively optimized for an up-to-date Linux Wayland/PipeWire/Vulkan Video stack on **AMD Radeon RX 9060 XT (RDNA 4 / GFX1200 / VCN 5)**, including every pitfall encountered and how empirical measurements shaped the final architecture.

---

## 1. Target Software & Hardware Stack Discovery

Before writing any code, we probed the live system capabilities:
- **OS / Kernel**: Arch Linux, kernel `7.3.5-arch1-1` (`linux/dma-buf.h` with `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`)
- **GPU / Driver**: AMD Radeon RX 9060 XT (`RADV GFX1200`), Mesa `26.3.1`, Vulkan `1.4.363`
- **Vulkan Queues Exposed**:
  - `qf[0]`: 1× Graphics | Compute | Transfer
  - `qf[1]`: 4× Async Compute | Transfer
  - `qf[2]`: 1× Video Decode (`H264`, `H265`, `AV1`, `VP9`)
  - `qf[3]`: 1× Video Encode (`H264`, `H265`, `AV1`)
- **Vulkan Video AV1 Capabilities (`vkGetPhysicalDeviceVideoCapabilitiesKHR`)**:
  - Both **8-bit (`NV12`)** and **10-bit (`P010`)**: `320×128` up to `8192×4352`, rate control modes `0x7` (CQP/CBR/VBR), `qualityLevels = 4` (`0..3`), `maxSingleReferenceCount = 1`.
- **DRM Format Modifiers**: RADV GFX1200 exposed 9 modifiers for `BGRA8`, `RGBA8`, `ABGR2101010`, and `ARGB2101010` — notably all with **`planes = 1`** (GFX12 uses unified/implicit DCC in page tables rather than a separate metadata plane).
- **FFmpeg**: `n9.1` (`libavcodec 63.24.100`, `libavformat 63.16.100`, `libavutil 61.19.100`) with `--enable-vulkan` (`av1_vulkan`, `hevc_vulkan`, `h264_vulkan`).
- **Desktop / Capture Stack**: KDE Plasma 6 Wayland (`kwin_wayland`), `xdg-desktop-portal` ScreenCast v6 (`pipewire-serial`), PipeWire `1.6.11`, `libdrm 2.4.134`, `gio-2.0 2.90.1`, `glslc`.

---

## 2. Early Design Decisions & First-Contact Pitfalls

### Pitfall 1: FFmpeg `scale_vulkan` Cannot Convert `bgra` → `nv12` / `p010`
The reference outline suggested FFmpeg Vulkan Compute for color conversion. Testing `scale_vulkan=format=nv12` and `format=p010` from `bgra` immediately failed in FFmpeg 9.1:
```
[swscaler] Failed initializing scaling graph (Operation not supported): fmt:bgra -> fmt:nv12
```
While FFmpeg's `libplacebo` filter *did* work (`inherit_device=true`), writing our own **embedded GLSL → SPIR-V compute shader (`shaders/rgb_to_yuv.comp`)** gave us:
- Zero runtime shader file dependencies (`glslc` compiles to `.spv` at build time and `cmake/embed_spv.py` embeds it as a `constexpr` array).
- Direct control over Vulkan external DMA-BUF import, explicit `sync_file` fences, chroma siting, Catmull-Rom scaling, and dithering.
- Two compile-time variants (`-DP010=0` with `layout(r8)`/`rg8` and `-DP010=1` with `layout(r16)`/`rg16`) so the shader does not require `shaderStorageImageWriteWithoutFormat`.

### Pitfall 2: SPA vs DRM FourCC Endianness Trap
Reading `/usr/include/spa-0.2/spa/param/video/raw.h` alongside `/usr/include/libdrm/drm_fourcc.h` avoided a classic Linux red/blue channel swap:
- SPA 8-bit RGB names (`BGRx`, `BGRA`, `RGBx`, `RGBA`) are **big-endian byte order** (`BGRA` = bytes `B, G, R, A`), whereas DRM fourccs (`DRM_FORMAT_ARGB8888`, `DRM_FORMAT_XRGB8888`) are **little-endian 32-bit words** (`A:R:G:B` = bytes `B, G, R, A`).
- Thus `SPA_VIDEO_FORMAT_BGRA` maps to `DRM_FORMAT_ARGB8888` and `VK_FORMAT_B8G8R8A8_UNORM`, while 10-bit SPA formats (`xRGB_210LE`, `xBGR_210LE`) are explicitly little-endian (`VK_FORMAT_A2R10G10B10_UNORM_PACK32` / `VK_FORMAT_A2B10G10R10_UNORM_PACK32`).

### Pitfall 3: `VK_KHR_internally_synchronized_queues` & `vkGetDeviceQueue` Returning `VK_NULL_HANDLE`
On our very first GPU test run, `vkcap_tests` segfaulted inside `libvulkan_radeon.so` at `vkQueueSubmit2`. Debugging with `gdb` revealed why:
- FFmpeg 9.1's `av_hwdevice_ctx_create(AV_HWDEVICE_TYPE_VULKAN)` enables `VK_KHR_internally_synchronized_queues` and creates its queues with non-zero `AVVulkanDeviceContext::queue_flags`.
- Per the Vulkan specification, calling `vkGetDeviceQueue` (which implicitly passes `flags = 0`) on a queue created with non-zero flags returns **`VK_NULL_HANDLE`**.
- Switching to `vkGetDeviceQueue2` with `qinfo.flags = vkctx->queue_flags` immediately resolved the crash.

### Pitfall 4: RADV AV1 Hardware Constraints (`async_depth` & `b_depth`)
Benchmarking `av1_vulkan` options on RADV GFX1200 revealed two hard limits:
- `async_depth >= 8` tripped `Assertion slot_index >= 0 failed at libavcodec/vulkan_encode.c:223` (RADV AV1 DPB slot count limits `async_depth` to `<= 4`).
- `b_depth = 2` failed encoder open because RADV GFX1200 reports `maxSingleReferenceCount = 1`.
- Keeping `async_depth = 4` and `b_depth = 1` proved completely stable across all containers (`.mp4`, `.mkv`, `.webm`, `.ivf`).

---

## 3. Iteration 1: Initial Working Pipeline

The initial build (`PortalSession` → `PipeWireCapture` → `BoundedFrameQueue(4)` → single `worker` thread running `VulkanCompute::process_frame()` + `VideoEncoder::encode_frame()`) passed all synthetic GPU tests (`343 fps` at 1080p, `107 fps` at 4K) and recorded at the expected visual quality on the live desktop.

However, live desktop testing under heavy load revealed **serious performance issues under load**:
- The recording lagged and dropped frames under heavy 3D/desktop load.
- Noticeable clue from our own 4K test output: `compute` was reporting **`9.140 ms/f`** (4K) and **`2.759 ms/f`** (1080p) — roughly equal to the VCN encode time rather than the ~0.12 ms expected for a 4K shader on RDNA 4.

---

## 4. Iteration 2: First Optimization Attempt — And the Compositor Regression

In our first optimization pass, we spotted several genuine issues and attempted to make compute fully asynchronous:
- Removed `PW_STREAM_FLAG_MAP_BUFFERS` (which was forcing PipeWire to `mmap()` VRAM DMA-BUFs into CPU address space).
- Drained queued PipeWire buffers to the freshest frame in `on_process()`.
- Switched DMA-BUF sampling to `VK_IMAGE_LAYOUT_GENERAL` to avoid RADV GFX12 DCC decompression transitions.
- Moved `av_interleaved_write_frame()` onto a dedicated async muxer thread (`vkcap-mux`) with a 256-packet queue.
- **Removed `vkWaitForFences`** from `process_frame()`, attaching `pw_buffer`'s release callback to a 6-slot `CmdSlot` ring reaped via `vkGetFenceStatus()`.

### What Went Wrong (The Regression)
Live testing immediately caught a severe regression: **not only did the recording still lag, the desktop/compositor itself became choppy even with low CPU and GPU usage.**

Root-cause analysis showed exactly why:
1. `BoundedFrameQueue` (up to 4 `pw_buffer`s) + `CmdSlot` (up to 6 `pw_buffer`s) allowed `vkcap` to hoard up to **10 PipeWire buffers simultaneously**.
2. Compute submission and `avcodec_send_frame()` still ran on the **same single worker thread**. Whenever `avcodec_send_frame()` blocked for 9–16 ms on VCN, nobody drained `BoundedFrameQueue` or called `reap_completed()`.
3. KWin's screencast stream only allocates a small buffer pool; hoarding its `pw_buffer`s starved KWin and made desktop presentation choppy.

---

## 5. Iteration 3: 3-Stage Decoupling, The AMDGPU Fence Pollution Discovery & Precision CFR

### Step 3A: Eliminating Compositor Lag (Zero PipeWire Buffer Hoarding)
We refactored the pipeline so **`pw_buffer`s never leave `on_process()`**:
- `on_process()` drains to the freshest buffer, converts it into our own VRAM `AVVkFrame`, waits *only* for compute, and **immediately** calls `pw_stream_queue_buffer()` before returning. At most **1** compositor buffer is ever held.
- Fixed PipeWire negotiation to advertise VFR `SPA_FORMAT_VIDEO_framerate = 0/1` with `SPA_FORMAT_VIDEO_maxFramerate = target_fps/1`.
- Moved `av1_vulkan` onto its own dedicated thread (`vkcap-enc`), creating a true 3-stage pipeline:
  1. `vkcap-pw`: PipeWire capture + fast GPU compute
  2. `vkcap-enc`: `av1_vulkan` hardware encode
  3. `vkcap-mux`: `libavformat` container I/O

This completely fixed desktop/compositor lag — **yet the output recording still lagged under load and frame pacing wasn't smooth.**

### Step 3B: The Smoking-Gun Discovery — AMDGPU/RADV Cross-Engine DMA-BUF Fence Pollution
Even after probing FFmpeg's LIFO `AVBufferPool` via `vkGetSemaphoreCounterValue()` to guarantee an already-idle `AVVkFrame`, a 4K diagnostic benchmark (`bench_compute` / `bench_stall`) revealed something shocking:
```
4K DMA-BUF pure compute (NO encoder):   0.116 ms/frame (8607.6 fps)
4K DMA-BUF WITH av1_vulkan active:      9.235 ms/frame (even on 12 distinct, never-used AVVkFrames!)
4K SHM (NO DMA-BUF) WITH av1_vulkan:    2.900 ms/frame (no stall behind av1_vulkan!)
```
Testing `DMA_BUF_IOCTL_EXPORT_SYNC_FILE(DMA_BUF_SYNC_READ)` right after `encode_frame(f0)` (`bench_sync`) proved the root cause:
```
After encode_frame(f0), db.fd SYNC_READ poll=1 waited 6.071 ms!
```
**Why**: Because KWin's DMA-BUFs were imported into the **same `VkDevice`** (`amdgpu_vm`) used by `av1_vulkan`, RADV's per-VM external BO list attached `av1_vulkan`'s VCN encode fence onto every imported DMA-BUF BO whenever VCN submitted a frame. That forced both our own `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` and KWin's buffer reuse to serialize behind VCN encoding!

**The Fix — Dual-`VkDevice` Isolation**:
- Verified `vkf->mem[0]` from FFmpeg's `AVHWFramesContext` exports cleanly as `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`.
- Created an **isolated compute-only `VkDevice` (`compute_dev`)** on the same `VkPhysicalDevice`.
- KWin's DMA-BUFs are imported *only* on `compute_dev`; each pooled `AVVkFrame` is zero-copy aliased once onto `compute_dev` via `OPAQUE_FD`.
- Re-running `bench_sync` and `ctest` confirmed the fix:
  - `db.fd SYNC_READ` wait dropped from **`6.071 ms` → `0.002 ms`** (**3,000× reduction**).
  - 4K compute hold time dropped from **`9.148 ms/f` → `0.153 ms/f`** (**60× faster**).
  - End-to-end 1080p throughput jumped from `340.6` → **`388.3 fps`**, 10-bit P010 from `303.3` → **`342.4 fps`**, 4K from `107.4` → **`111.2 fps`**.

### Step 3C: Precision CFR Frame Pacing
To fix uneven frame pacing caused by jittery compositor timestamps, unset `frame->duration`, and high-refresh monitors (120–240 Hz) outpacing the recording framerate:
- Added a locked **CFR pacer (default)** on `vkcap-enc` using `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)` at exact `1/fps` intervals, `time_base = {1, fps}`, `frame->duration = 1`, and a `LatestFrameMailbox` (cloning references to the latest GPU `AVFrame*` for zero-cost static-screen repeats).
- Kept `--vfr` with dequeue-time timestamps captured *before* compute.
- Live test on **Overwatch at 120 fps** confirmed smooth gameplay, rock-solid frame pacing, and clean hardware-decoded playback in `mpv`.

---

## 6. Iteration 4: Visual Quality & Compression Efficiency Tuning

We ran empirical 1440p rate-distortion sweeps (`SSIM`, `PSNR`, bitrate, FPS) on the RX 9060 XT:
- **AV1 `qp` scale (`0..255`)**: Our initial `qp=22` spent ~84 Mbps at 1440p60 (`0.9765` SSIM). `qp=45–65` achieved `0.9753–0.9759` SSIM (within `0.02 dB` PSNR) at **35–42% lower bitrate**. We set the codec-aware default to **`qp=52`** for AV1 (preserving `qp=22` for HEVC/H.264), with **auto 2-second GOP (`fps * 2`)** to eliminate 1-second keyframe bitrate pulsing at 120 fps.
- **Quality levels (`0..3`)**: `quality=3` dropped to `64.5 fps` at 1440p (unusable for 120 fps); `quality=1` (`200 fps`) and `quality=2` (`166 fps`) were both above 1440p120.
- **Shader Fidelity Upgrades**:
  - Upgraded 1:1 native chroma downsampling to a **3×2 `AVCHROMA_LOC_LEFT` anti-aliased filter** (`[0.25, 0.50, 0.25]` × `[0.50, 0.50]`) and tagged `AVCHROMA_LOC_LEFT` metadata on frames and encoder context.
  - Upgraded scaled capture (`-s WxH`) from bilinear to a **5-tap Catmull-Rom bicubic** filter.
  - Added **static spatial triangular-PDF IGN dithering** for 8-bit NV12 to eliminate banding without breaking AV1 inter-frame skip blocks.

---

## 7. Iteration 5: Final Micro-Optimizations (Minimizing In-Game Impact)

When the user compared the new 60 fps defaults (`~33 Mbps` 1440p60, perceived native quality) against the earlier 120 fps VBR build, they noticed the game felt *slightly* smoother on the earlier build. Inspecting the diff between the two runs pinpointed two real causes:
1. The auto quality heuristic had bumped `quality` from `1` → `2`, which our own sweep showed saved only `0.3%` bitrate (`+0.00001` SSIM) while consuming **~20% more VCN/VRAM bandwidth**.
2. `compute_dev` was created with queue priority `1.0f`, competing with the game's GPU queues instead of yielding to them.

Final optimizations applied:
- **Low-Priority Compute Queue**: Switched `compute_dev` to `VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR` (fallback `priority = 0.0f`) so 3D games and KWin always preempt `vkcap`'s `0.10 ms` compute shader.
- **Restored Auto `quality = 1`**: Preserves identical visual quality with ~20% lower VCN/VRAM contention (`--quality 2` still available on demand).
- **Push Descriptors (`VK_KHR_push_descriptor`)**: Replaced per-frame `vkUpdateDescriptorSets()` and descriptor pools with `vkCmdPushDescriptorSetKHR`.
- **Single-Hash Triangular IGN Dither**: Halved dithering ALU cost in `rgb_to_yuv.comp`.
- **Tighter CFR Compute Throttle**: Raised the skip window from `0.5 × frame_interval` to `0.75 × frame_interval`, cutting redundant compute dispatches by ~33–40% when gaming at 144–240 Hz while recording at 60 fps.
- **Zero `EAGAIN` Packet Churn & Thread Naming**: Added a reusable `scratch_pkt` in `VideoEncoder::drain_packets()` and named threads (`vkcap-pw`, `vkcap-enc`, `vkcap-mux`).

---

## 8. Summary of the Final Optimized Setup

| Component | Final Optimized Configuration |
|---|---|
| **Portal** | `org.freedesktop.portal.ScreenCast` v1–v6, `pipewire-serial` targeting, `restore_token` persistence |
| **PipeWire Capture** | Vulkan-queried tiled/DCC modifiers, VFR `0/1` + `maxFramerate`, no `MAP_BUFFERS`, burst-drain to freshest frame, immediate `pw_stream_queue_buffer()` (≤ 1 buffer held) |
| **Vulkan Compute** | Isolated `compute_dev` on async compute queue (`VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR`), cached DMA-BUF imports, `OPAQUE_FD` zero-copy `AVVkFrame` aliases, `VK_KHR_push_descriptor`, `VK_IMAGE_LAYOUT_GENERAL` (DCC-safe), `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` explicit sync, LIFO idle-frame probing |
| **Shader** | Embedded NV12 + P010 SPIR-V, 2×2 pixels/thread (`8×8` WG), 3×2 `AVCHROMA_LOC_LEFT` chroma AA, 5-tap Catmull-Rom bicubic scaling, single-hash static spatial IGN dithering |
| **Frame Pacing** | Locked CFR via `clock_nanosleep(TIMER_ABSTIME)` + `LatestFrameMailbox` + `0.75×` CFR compute throttle (or `--vfr` with dequeue timestamps) |
| **Encoder (RDNA 4)** | `av1_vulkan` (`VK_KHR_video_encode_av1`), `qp=52` (AV1), `quality=1`, `tune=hq`, `usage=record`, `content=rendered`, 2-second GOP, `async_depth=4`, `b_depth=1`, `AVCHROMA_LOC_LEFT` |
| **Muxer** | Dedicated `vkcap-mux` thread, 256-packet queue, reusable `scratch_pkt`, `.mkv` / `.mp4` (`+faststart`) / `.webm` / `.ivf` |
