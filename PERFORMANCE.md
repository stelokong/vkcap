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

## 4. CPU, Syscall, Frame-Pacing & Memory Optimizations

- **Fast-fail KMS probe & fresh frame 0**: `run_kms_pipeline()` probes KMS *before* initializing Vulkan so missing `CAP_SYS_ADMIN` or inactive displays fail in <1 ms, then immediately releases the probe's DMA-BUF FDs so frame `0` is grabbed fresh right at `base_ns` (never a stale pre-Vulkan scanout buffer).
- **Zero-allocation DRM/KMS hot path with cursor caching**: `grab_frame_dedup()` uses direct stack-allocated kernel ioctls (`DRM_IOCTL_MODE_GETPLANE`, 64-entry `DRM_IOCTL_MODE_OBJ_GETPROPERTIES` with dynamic fallback only if >64 properties, `DRM_IOCTL_MODE_GETFB2`, `DRM_IOCTL_PRIME_HANDLE_TO_FD`, `DRM_IOCTL_GEM_CLOSE`) inside a **single `ScopedCapSysAdmin` window** per frame. Unchanged hardware cursor `fb_id`s are cached across frames (with periodic 60-frame re-validation), eliminating per-frame cursor `GETFB2`/`PRIME` ioctls.
- **Zero-IOCTL + zero-compute duplicate-frame fast-path**: Using `DRM_IOCTL_CRTC_GET_SEQUENCE`, whenever the CRTC vblank sequence has not advanced since the previous grab and primary `fb_id` + cursor state are identical (e.g. 120 fps capture on a 60 Hz desktop, static menus, or sub-target game FPS), `vkcap` skips all `CAP_SYS_ADMIN` / `GETFB2` / `PRIME` ioctls *and* skips the Vulkan compute dispatch entirely, reusing the previous GPU `AVFrame` via zero-copy `av_frame_clone()`.
- **Asynchronous GPU compute pipelining (`submit_frame` / `wait_slot`)**: `VulkanCompute` maintains `max(queue_size + 3, 6)` independent `CmdSlot`s (command pool, command buffer, fence, semaphore). The capture thread submits compute work and returns immediately without stalling on `vkWaitForFences`; `vkcap-enc` (and `EncodeFrameQueue` on eviction/close) waits on the slot fence right before `avcodec_send_frame()`, keeping capture wakeup timing completely immune to heavy 3D GPU load preempting the low-priority compute queue.
- **Single-GPU zero-overhead device selection**: On multi-GPU systems, `VK_EXT_physical_device_drm` automatically matches the Vulkan physical device to the selected `/dev/dri/card*` node; on single-GPU systems, `/dev/dri` inspection short-circuits immediately without creating a temporary `VkInstance` or loading implicit Vulkan layers twice.
- **Multi-plane / disjoint DMA-BUF import**: Single-BO buffers take a zero-overhead single-allocation path; multi-BO compressed modifiers (such as Intel CCS) automatically use `VK_IMAGE_CREATE_DISJOINT_BIT` and `VkBindImagePlaneMemoryInfo`.
- **Flat L1-resident MRU caches with stale BO eviction**: Both the 12-entry `st_ino` DMA-BUF import cache (`src_cache`, with in-flight async slot eviction guards) and `AVVkFrame` alias cache (`dst_alias_cache`) are flat contiguous arrays on the single capture thread—no mutexes, no hash maps. Triple-buffered scanout hits indices `0..2` in L1 cache, while automatic stale-surface eviction (releasing scanout DMA-BUFs unused for >180 frames) and a compact `AVHWFramesContext` pool (`queue_size + async_depth + 6`) ensure destroyed desktop/game swapchain BOs are never left pinned in VRAM across fullscreen launches.
- **LIFO `AVBufferPool` probing**: Probes FFmpeg's LIFO hardware frame pool via `vkGetSemaphoreCounterValue()` using reusable empty `AVFrame` shells (`probe_shells`) to grab an idle `AVVkFrame` immediately without stalling on in-flight hardware encodes.
- **Coalesced muxer locking & recycled `AVPacket` pool**: `vkcap-enc`, `vkcap-aenc`, and `vkcap-mux` share a fixed 256-entry packet ring and a 32-entry recycled `AVPacket` pool. Packet recycling and ring push/pop are coalesced under a single `pkt_mtx` acquisition per packet, halving mutex operations across all three threads.
- **Zero-stutter, zero-VRAM Instant Replay (`--instant-replay`)**: Copies only the compressed hardware-encoded video bytes (`p->size`, ~2–50 KiB/frame) into plain CPU `AVPacket`s and immediately calls `av_packet_unref()` on the encoder packet so `av1_vulkan`'s multi-megabyte Vulkan `VkDeviceMemory` bitstream buffers are returned to FFmpeg's pool immediately (**0 GPU BOs / 0 VRAM / 0 GTT** held by the replay ring, **0 disk I/O** while buffering; audio packets are pushed zero-copy). On save (`SIGUSR2` or `'s'`/`Enter`), takes an `O(1)` shallow `av_packet_ref()` snapshot (**~0.05 ms**, zero bitstream memcpy, shifting both video `pts`/`dts` and audio `pts` by a unified `v_origin = min(v_base_pts, v_base_dts)` to preserve B-frame CTS offsets) and muxes straight to disk on a low-priority (`nice(10)`) `vkcap-save` thread with zero re-encoding and zero impact on live capture or game FPS.
- **Rational beat-centering CFR frame pacing (`FramePacer`)**:
  - Uses `CLOCK_MONOTONIC` `clock_nanosleep(TIMER_ABSTIME)` with `PR_SET_TIMERSLACK(1)`, per-thread `nice(-5)` priority on `vkcap` and `vkcap-aenc`, overflow-safe quotient/remainder tick math, and an immutable `base_ns` timeline (stalls advance `frame_idx` rather than shifting `base_ns`; `SIGUSR1` pause/resume atomically shifts `pause_offset_ns` across both video and audio).
  - Solves for the smallest reduced rational `p/q` (`q ∈ [1..12]`, comb spacing `≥ 1.0 ms`) matching `refresh_hz / fps`: integer multiples (`59.94/60/120/240/360/480/540 Hz`, `q = 1`) phase-lock to a post-vblank lead (`0.25–1.5 ms`), while non-integer gaming rates (`144/60` = `12/5`, `165/60` = `11/4`, `75/60`, `100/60`, `144/120`, `280/120`) center the entire `q`-frame beat comb halfway between vblank commits to eliminate beat-edge coin-flipping.
  - On fractional clock skew (`59.94 Hz`, `143.98 Hz`, `164.84 Hz`), wraps `phase_offset_ns_` atomically across `±lock_period / 2` in a single tick instead of clamping or slewing across vblank. Ignores stale PSR timestamps, gently relaxes phase to `0` during sustained VRR (3+ unstable intervals), and dynamically re-locks via `query_refresh_hz()` if a fullscreen game changes refresh rate or resolution mid-recording (clamping active `--crop` rects safely).
  - `--verbose` prints end-of-run pacing & audio sync diagnostics (`wake jitter avg/max µs`, `dup_reused`, `stall_skips`, `kms_misses`, `q_drops`, `vb_wraps`, audio crystal skew `ppm`, `underruns`, `overrun_trims`).

---

## 5. Low-Overhead, Zero-Drift, Glitch-Free Audio

- **Drift-free, click-free PI-controlled Catmull-Rom resampling (`AdaptiveAudioBuffer`)**:
  - Each PipeWire stream (`vkcap-system` and `vkcap-mic`) feeds its own cache-line-aligned **lock-free SPSC ring buffer** (`65536` frames—zero locks, zero syscalls, zero heap allocations inside the RT audio callback).
  - Instead of hard silence-padding or hard sample-dropping when the audio hardware crystal drifts relative to `CLOCK_MONOTONIC`, each stream runs an independent **4-point Catmull-Rom cubic fractional resampler** driven by a critically damped PI controller (`ζ ≈ 0.71`) behind a **2-pole cascaded EMA** (suppressing 1024/2048-frame PipeWire quantum sawtooth ripple to `< 10 ppm` for zero audible pitch flutter and zero cumulative A/V drift across arbitrary recording lengths).
- **Quantum-safe ~42.7 ms jitter cushion at 0.0 ms A/V offset**:
  - Requests PipeWire's `960/48000` (20 ms) quantum to match Opus while maintaining a **`2048`-frame (~42.7 ms) cushion** paired with a matching ~42.7 ms wall-clock read offset in `vkcap-aenc` (stamping `pts = a_idx * a_frame_sz` anchored to `base_ns`). This cleanly absorbs 256, 512, 960, 1024, and 2048-frame PipeWire graph quanta with **0.0 ms A/V offset** in the output file.
  - A pre-fill start gate prevents startup/resume underrun loops, and 64-sample (~1.33 ms) linear fade-in/fade-out ramps keep idle PipeWire sink suspends and resumes completely click-free.
- **Real-time tuned `libopus` (`compression_level = 5`)**: Feeds `AV_SAMPLE_FMT_FLT` directly to `libopus` with zero resampling and complexity `5` (~3× lower CPU usage during loud gameplay than default `10`, while remaining acoustically transparent at 192 kbps). Also supports AAC (`AV_SAMPLE_FMT_FLTP`) and 32-bit lossless FLAC (`AV_SAMPLE_FMT_S32`).
- **Independent gain + soft-limiting**: System audio and microphone have independent `0..1000%` linear gain passed through an `[[unlikely]]`-guarded `tanh` soft-limiter above `0.95` so boosted voice + game explosions never harshly clip.

---

## 6. Verified GPU Throughput

Measured via `ctest --test-dir build --output-on-failure --verbose` on **AMD Radeon RX 9060 XT (RDNA 4 / GFX1200)**, Mesa RADV 26.3.1, FFmpeg 9.x (end-to-end DMA-BUF → Compute → `av1_vulkan` → Mux → Full AV1 Software Decode Verification, plus `FramePacer` fractional/rational lock & `AdaptiveAudioBuffer` 960/1024/2048-quantum ±500 ppm zero-click verification):

| Workload | End-to-End Throughput | Compute Wall Time (Submit + Fence) |
| :--- | :---: | :---: |
| 1080p NV12 + HW Cursor + Wide-Gamut → MP4 (AV1 + Opus) | **366–368 fps** | **0.15 ms / frame** |
| 1080p P010 (10-bit) + HW Cursor → MKV (AV1 + Opus) | **332–336 fps** | **0.09–0.10 ms / frame** |
| 4K → 1440p Catmull-Rom Bicubic + HW Cursor → WebM (AV1) | **213–225 fps** | **0.24–0.25 ms / frame** |
| 4K Native NV12 → MKV (AV1) | **110 fps** | **0.15 ms / frame** |
| Instant Replay + GPU Crop (1080p → 720p, 2s ring + Opus) | **0.049 ms snapshot** | Zero-stutter async mux, full decode verified |
