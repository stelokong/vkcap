# `vkcap` Performance, Architecture & Design Philosophy

This document explains **how `vkcap` works internally**, **what optimizations it uses** to minimize game/compositor interference and CPU/GPU overhead compared to traditional Linux screen recorders (OBS Studio, `wf-recorder`, `ffmpeg -f kmsgrab`, and portal/PipeWire recorders), and **what `vkcap` intentionally does and does not try to be**.

---

## 1. Design Philosophy: Streamlined by Choice

### What `vkcap` Strives to Do
- **Zero compositor interference**: Never ask the Wayland/X11 compositor to allocate extra screencast buffers, run extra render passes, or hold swapchain images.
- **100% GPU-resident zero-copy video**: Display VRAM → Vulkan Compute → Hardware Video Encoder VRAM. Zero pixel bytes ever cross PCIe or touch system RAM.
- **Near-zero CPU footprint**: Keep steady-state recording to a fraction of a percent of total CPU, with lock-free audio capture, zero-allocation DRM ioctls, and minimal syscalls.
- **High visual & color fidelity**: Accurate EDID wide-gamut color inversion, anti-aliased `AVCHROMA_LOC_LEFT` chroma filtering, Catmull-Rom bicubic scaling, and static spatial dithering—all fused into a single compute dispatch.
- **Clean cross-vendor Vulkan 1.3+ portability**: Support AMD (RADV), NVIDIA, and Intel (ANV) through standard Khronos Vulkan, DRM/KMS, and Vulkan Video extensions without vendor-specific SDKs.

### What `vkcap` Intentionally Does Not Do
Traditional screen recorders accumulate dozens of capture backends (XSHM, XComposite, PipeWire portal, wlroots screencopy, KMS) paired with dozens of encoder backends (VA-API, NVENC, QSV, AMF, x264, x265, SVT-AV1) and a full scene-graph compositor. Maintaining that matrix creates fragile edge cases, duplicated synchronization bugs, extra buffer copies, and unpredictable latency.

`vkcap` intentionally commits to **one single, streamlined pipeline**:
**Linux DRM/KMS → Isolated Vulkan 1.3 Compute → Vulkan Video (`av1_vulkan` / `hevc_vulkan` / `h264_vulkan`) + Native PipeWire Audio.**

By keeping the pipeline narrow and uniform:
- There are no format-conversion fallbacks through CPU RAM (`swscale` / `glReadPixels`).
- Every vendor (AMD, NVIDIA, Intel) exercises the exact same Vulkan compute shader and Vulkan Video path.
- The hot path stays small, auditable, predictable, and far less likely to regress across driver or compositor updates.

---

## 2. Eliminating Game & Compositor Stutter

### 1. Bypassing the Compositor (Direct DRM/KMS Scanout)
Wayland screencast protocols (`xdg-desktop-portal` / PipeWire Screencast) require the compositor (KWin, Mutter, wlroots) to schedule an extra offscreen copy or retain swapchain buffers until the recorder consumes them. Under heavy 3D gaming load (e.g. 240–360 Hz competitive FPS), holding even 2–3 swapchain buffers introduces visible frame-pacing micro-stutter.

`vkcap` reads the already-scanned-out DRM/KMS primary plane (and optional hardware cursor plane) directly via `DRM_IOCTL_MODE_GETFB2` and `DRM_IOCTL_PRIME_HANDLE_TO_FD`. The compositor never knows `vkcap` is running and never waits on it.

### 2. Dual-`VkDevice` Isolation (Eliminating VCN Fence Pollution)
On AMDGPU/RADV, importing external KMS DMA-BUFs into the same `VkDevice` that runs Vulkan Video encoding places those scanout BOs on RADV's per-VM external BO list. Every hardware encode submission then attaches a read/write fence to the compositor's/game's scanout DMA-BUF, causing **6+ ms `SYNC_READ` stalls** on the display swapchain.

`vkcap` solves this with **Dual-`VkDevice` Isolation**:
- FFmpeg's `AVVulkanDeviceContext` (`vkctx->act_dev`) runs Vulkan Video encoding exclusively and **never imports** any scanout DMA-BUF.
- `vkcap` creates a separate, isolated **low-priority async compute `VkDevice`** (`compute_dev`, `VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR`, queue priority `0.0f`) on a dedicated compute-only queue family (when available).
- All external KMS primary and cursor DMA-BUFs are imported exclusively into `compute_dev`.
- FFmpeg's destination `AVVkFrame` surfaces are aliased into `compute_dev` via `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT` (with `VK_KHR_video_maintenance1` / `VK_IMAGE_CREATE_VIDEO_PROFILE_INDEPENDENT_BIT_KHR` properly chained, plus disjoint 2-plane `AVVkFrame` support for cross-vendor drivers).
- This drops scanout fence stalls on RADV from **~6.07 ms to 0.002 ms** while keeping total compute hold time to **~0.09–0.23 ms/frame** (and ~0.4–0.5 ms/frame even under heavy 3D load).

### 3. DMA-BUF Synchronization Fast-Path
Before reading a scanout DMA-BUF, `vkcap` calls `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` (`DMA_BUF_SYNC_READ`, compile-time guarded for older kernels) followed by a non-blocking `poll(&pfd, 1, 0)`. When the scanout buffer is already signaled (>99% of frames), the sync fd is closed immediately without importing a Vulkan semaphore. Only if the buffer is still being rendered is the sync fd imported via `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT` so the GPU waits without stalling the CPU.

---

## 3. Compute Shader Optimizations (`shaders/rgb_to_yuv.comp`)

Hardware cursor compositing, EDID wide-gamut color inversion, bicubic scaling, anti-aliased 4:2:0 chroma filtering, and 8-bit dithering all execute in **one single compute dispatch**—zero intermediate render targets and zero extra VRAM round-trips.

- **SPIR-V Specialization Constants (`constant_id = 0..12`)**:
  - `kExact1To1`, `kHasCursor`, `kEnableDither`, `kConvertGamut`, and the 9 floats of the 3×3 Bradford gamut matrix (`kGm00..kGm22`) are specialized at pipeline creation so the driver's shader compiler completely dead-code-eliminates unused branches and constant-folds the gamut matrix into ISA.
  - Using a `VkPipelineCache`, `vkcap` pre-compiles only the **1–2 pipeline variants** needed for the configured resolution and cursor mode (instead of all 4 upfront), avoiding frame-0 compile stalls while minimizing startup time.
- **`16×4` Workgroups + `subgroupShuffleUp` (SIMD8/Wave32/Wave64 Safe)**:
  - Each `16×4` workgroup processes a `32×8` pixel tile (one `2×2` luma block + one UV sample per thread), producing 32-byte coalesced Y and UV image stores.
  - In 1:1 mode, threads reuse their left neighbor's already-fetched, cursor-blended, gamut-converted right-column texels directly from registers via `subgroupShuffleUp(..., 1u)`, cutting texture fetches by **33%** (from 6 to 4 per `2×2` block).
  - Lanes guard `gl_LocalInvocationID.x == 0u || gl_SubgroupInvocationID == 0u`, ensuring correctness on Intel ANV **SIMD8** subgroups as well as AMD Wave32/Wave64 and NVIDIA warp32.
- **Fast Wide-Gamut EOTF/OETF (8-bit NV12)**:
  - When capturing a wide-gamut Wayland display (e.g. QD-OLED / DCI-P3) to 8-bit NV12, `srgbToLinear()` and `linearToSrgb()` replace 24 transcendental `pow()` calls per thread with FMA/`sqrt` approximations accurate to `< 0.13` LSB.
  - For 10-bit `P010` archival output, exact piecewise `pow()` is retained.
- **Factored 1D Catmull-Rom Bicubic Scaling + Cursor AABB Early-Out**:
  - In scaled mode (`-s WxH` / `1080p` / `1440p` / `4k`), `catmullRom1D()` precomputes only 2 unique X weight sets and 2 unique Y weight sets per `2×2` block (instead of 4×2D), halving weight/coordinate arithmetic and divisions.
  - Both 1:1 and scaled paths apply a `2×2` bounding-box early-out before `blendCursor()`, skipping cursor texture fetches across >99% of the screen.
- **Zero-`sqrt` Static Spatial Triangular-PDF IGN Dithering**:
  - 8-bit NV12 dithering uses the difference of two uncorrelated static spatial Interleaved Gradient Noise hashes (`n1 - n2`), generating a triangular PDF in `[-0.5, +0.5]` LSB using pure full-rate VALU ops (eliminating 5 SFU `sqrt()` instructions per thread) with zero temporal grain that would waste video encoder bitrate.
- **Minimal Barriers**:
  - Read-only source (`src.image` and `cur_img.image`) post-dispatch release barriers use `srcAccessMask = 0` to avoid unnecessary cache-flush hints.

---

## 4. CPU, Syscall & Memory Optimizations

- **Fast-fail KMS probe**: `run_kms_pipeline()` grabs the first KMS frame *before* initializing Vulkan, so missing `CAP_SYS_ADMIN` or inactive displays fail in <1 ms without spinning up Vulkan.
- **Zero-allocation DRM/KMS hot path**: `grab_frame()` uses direct stack-allocated kernel ioctls (`DRM_IOCTL_MODE_GETPLANE`, 64-entry `DRM_IOCTL_MODE_OBJ_GETPROPERTIES` with dynamic fallback only if >64 properties, `DRM_IOCTL_MODE_GETFB2`, `DRM_IOCTL_PRIME_HANDLE_TO_FD`, `DRM_IOCTL_GEM_CLOSE`) inside a **single `ScopedCapSysAdmin` window** per frame (2 `SYS_capset` syscalls/frame total).
- **Single-GPU zero-overhead device selection**: On multi-GPU systems, `VK_EXT_physical_device_drm` automatically matches the Vulkan physical device to the selected `/dev/dri/card*` node; on single-GPU systems, `/dev/dri` inspection short-circuits immediately without creating a temporary `VkInstance` or loading implicit Vulkan layers twice.
- **Multi-plane / disjoint DMA-BUF import**: Single-BO buffers take a zero-overhead single-allocation path; multi-BO compressed modifiers (such as Intel CCS) automatically use `VK_IMAGE_CREATE_DISJOINT_BIT` and `VkBindImagePlaneMemoryInfo`.
- **Flat L1-resident MRU caches**: Both the 12-entry `st_ino` DMA-BUF import cache (`src_cache`) and `AVVkFrame` alias cache (`dst_alias_cache`) are flat contiguous arrays on the single capture thread—no mutexes, no hash maps. Triple-buffered scanout hits indices `0..2` in L1 cache.
- **LIFO `AVBufferPool` probing**: Probes FFmpeg's LIFO hardware frame pool via `vkGetSemaphoreCounterValue()` using reusable empty `AVFrame` shells (`probe_shells`) to grab an idle `AVVkFrame` immediately without stalling on in-flight hardware encodes.
- **Coalesced muxer locking & recycled `AVPacket` pool**: `vkcap-enc`, `vkcap-aenc`, and `vkcap-mux` share a fixed 256-entry packet ring and a 32-entry recycled `AVPacket` pool. Packet recycling and ring push/pop are coalesced under a single `pkt_mtx` acquisition per packet, halving mutex operations across all three threads.
- **Precision CFR pacing**: `CLOCK_MONOTONIC` `clock_nanosleep(TIMER_ABSTIME)` with `PR_SET_TIMERSLACK(1)` and a single monotonic clock read per frame eliminates variable frame-time jitter.

---

## 5. Low-Overhead, Glitch-Free Audio

- **Native PipeWire 20 ms quantum (`960/48000`)**: Matches PipeWire's capture quantum directly to Opus's 20 ms frame duration, halving PipeWire `data-loop.0` wakeups (50 Hz instead of 100 Hz) while feeding cache-line-aligned **lock-free SPSC ring buffers** (`65536` frames—zero locks, zero syscalls, zero heap allocations inside the RT audio callback).
- **Wall-clock mastered A/V sync**: `vkcap-aenc` ticks on the exact same `CLOCK_MONOTONIC` `base_ns` as video frame `0`, padding silence when PipeWire suspends an idle sink and trimming backlog above 80 ms so A/V sync never drifts.
- **Real-time tuned `libopus` (`compression_level = 5`)**: Feeds `AV_SAMPLE_FMT_FLT` directly to `libopus` with zero resampling and complexity `5` (~3× lower CPU usage during loud gameplay than default `10`, while remaining acoustically transparent at 192 kbps). Also supports AAC (`AV_SAMPLE_FMT_FLTP`) and 32-bit lossless FLAC (`AV_SAMPLE_FMT_S32`).
- **Independent gain + soft-limiting**: System audio and microphone have independent `0..1000%` linear gain passed through an `[[unlikely]]`-guarded `tanh` soft-limiter above `0.95` so boosted voice + game explosions never harshly clip.

---

## 6. Verified GPU Throughput

Measured via `ctest --test-dir build --output-on-failure --verbose` on **AMD Radeon RX 9060 XT (RDNA 4 / GFX1200)**, Mesa RADV 26.3.1, FFmpeg 9.x (end-to-end DMA-BUF → Compute → `av1_vulkan` → Mux → Full AV1 Software Decode Verification):

| Workload | End-to-End Throughput | Compute Wall Time (Submit + Fence) |
| :--- | :---: | :---: |
| 1080p NV12 + HW Cursor + Wide-Gamut → MP4 (AV1 + Opus) | **394.1 fps** | **0.140 ms / frame** |
| 1080p P010 (10-bit) + HW Cursor → MKV (AV1 + Opus) | **348.7 fps** | **0.093 ms / frame** |
| 4K → 1440p Catmull-Rom Bicubic + HW Cursor → WebM (AV1) | **227.6 fps** | **0.228 ms / frame** |
| 4K Native NV12 → MKV (AV1) | **105.9 fps** | **0.155 ms / frame** |
