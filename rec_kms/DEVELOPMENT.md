# `vkcap-kms` Development History & Engineering Chronicle

This document chronicles the end-to-end development of `vkcap-kms`—including its origins as a **fork and architectural evolution of `rec_pipewire` (`vkcap`)**, the real-world bottlenecks discovered during live gaming and desktop testing on AMD RDNA 4 (RX 9060 XT), and every design decision, pitfall, and fix that shaped the final implementation.

---

## 1. Lineage: From `rec_pipewire` (`vkcap`) to `rec_kms` (`vkcap-kms`)

`vkcap-kms` began as a direct architectural fork of our earlier PipeWire + `xdg-desktop-portal` Vulkan Video screen recorder (preserved in `../rec_pipewire/`).

### What `rec_pipewire` Solved First (Inherited by `vkcap-kms`)
Before `rec_kms` existed, `rec_pipewire` went through several major engineering iterations on Linux (Arch, KDE Plasma 6 Wayland, AMD Radeon RX 9060 XT / RDNA 4, Mesa RADV 26.3.1, FFmpeg 9.1):
1. **Custom Embedded SPIR-V Compute Shader**: FFmpeg's `scale_vulkan` filter failed on `bgra -> nv12`, so we replaced it with a custom single-pass GLSL compute shader (`rgb_to_yuv.comp`) compiled to SPIR-V at build time via `glslc` and embedded as C++ `constexpr` arrays.
2. **Eliminating AMDGPU/RADV VCN Fence Pollution (Dual-`VkDevice` Isolation)**:
   - Early builds imported PipeWire DMA-BUFs directly into FFmpeg's `AVVulkanDeviceContext` (`vkctx->act_dev`). Under 3D gaming load, both the game and the desktop became choppy even at low GPU utilization.
   - Root cause: On AMDGPU/RADV, importing external DMA-BUFs into the same `VkDevice` running VCN encoding attaches VCN encode fences to those DMA-BUFs via RADV's per-VM external BO list (`6.071 ms` `SYNC_READ` stalls on compositor buffers).
   - Fix: Created a separate low-priority `compute_dev` (`VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR` / priority `0.0f`) for all external DMA-BUF imports and aliased FFmpeg's destination `AVVkFrame` via `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`—dropping scanout fence stalls to `0.002 ms`.
3. **RADV GFX1200 `av1_vulkan` Tuning & Constraints**:
   - Discovered `async_depth = 8` triggers `Assertion slot_index >= 0` in `libavcodec/vulkan_encode.c:223` on RADV GFX1200 → capped default `async_depth = 4`.
   - Discovered RADV AV1 reports `maxSingleReferenceCount = 1` → defaulted `b_depth = 1`.
   - Benchmarked quality vs. VCN/VRAM load: `quality = 1` cut VCN/VRAM pressure by ~20% over `quality = 2` with virtually identical SSIM/PSNR; AV1 CQP `qp = 52` hit the perceived native-quality sweet spot (~33 Mbps at 1440p60).
   - Added 3×2 `AVCHROMA_LOC_LEFT` anti-aliased chroma filtering, 5-tap Catmull-Rom bicubic scaling, and static spatial triangular-PDF IGN dithering for 8-bit NV12.
   - Added LIFO `AVBufferPool` probing via `vkGetSemaphoreCounterValue()` and a dedicated `vkcap-mux` thread with a recycled `AVPacket` pool.

### Why We Forked into `rec_kms`
Even after `rec_pipewire` achieved smooth 120 fps gameplay capture in Overwatch, PipeWire + `xdg-desktop-portal` still had inherent drawbacks:
- Every recording required an interactive portal dialog (or restore-token management) and depended on the Wayland compositor's screencast pipeline.
- Compositor-side screencast quirks and subtle desktop/game pacing differences remained possible whenever the compositor had to manage screencast buffers alongside a high-refresh-rate game.

The goal for `rec_kms` was clear: **fork the proven zero-copy Vulkan Compute + Dual-`VkDevice` + `av1_vulkan` engine into a standalone DRM/KMS capture architecture** that removes PipeWire video capture and `xdg-desktop-portal` entirely, requires the **least privileges possible**, and pushes overhead as close to zero as hardware allows.

---

## 2. Phase 1 — Direct DRM/KMS Architecture & Least-Privilege Model

### Replacing `xdg-desktop-portal` with Direct KMS Display Enumeration
- Built `KmsCapture::enumerate_displays()` (`src/kms_capture.cpp`) to scan `/dev/dri/card*` (handling systems where the primary GPU is `/dev/dri/card1` rather than `card0`), enable `DRM_CLIENT_CAP_UNIVERSAL_PLANES` and `DRM_CLIENT_CAP_ATOMIC`, and map every connected active connector (`DP-1`, `DP-2`, `HDMI-A-1`) to its CRTC, primary plane (`DRM_PLANE_TYPE_PRIMARY`), and hardware cursor plane (`DRM_PLANE_TYPE_CURSOR`).
- Display selection works by connector name (`-D DP-1`), index (`-D 0`), or `auto`, with `-l / --list-displays` requiring **zero privileges**.

### Least-Privilege Capability Scoping (`src/privileges.cpp`)
- Kernel `DRM_IOCTL_MODE_GETFB2` and `DRM_IOCTL_PRIME_HANDLE_TO_FD` require `CAP_SYS_ADMIN` when reading another process's scanout framebuffer.
- Instead of `sudo` or `setcap cap_sys_admin=ep` (which would leave `CAP_SYS_ADMIN` active during Vulkan, FFmpeg, and file I/O), we implemented `init_least_privilege()` and `ScopedCapSysAdmin` using direct `SYS_capget`/`SYS_capset` + `prctl(PR_SET_NO_NEW_PRIVS, 1)` with `sudo setcap cap_sys_admin=p ./build/vkcap-kms`.
- `Effective` is `0` at startup and throughout Vulkan/FFmpeg/PipeWire/file I/O, raising `CAP_SYS_ADMIN` only for the ~5 µs DRM export window.

### Hardware Cursor Plane Compositing & `st_ino` LRU Caching
- Under KMS, compositors place the mouse cursor on a separate `DRM_PLANE_TYPE_CURSOR` hardware plane rather than baking it into the primary framebuffer. We extended `rgb_to_yuv.comp` with binding 1 (`uCursor`) and a 1×1 transparent dummy image so the hardware cursor is blended in premultiplied alpha at exact CRTC coordinates in the same single compute pass (with `--no-cursor` to disable).
- Because `drmPrimeHandleToFD()` returns a fresh fd number each frame and DRM `fb_id`s can be recycled across mode switches or fullscreen direct-scanout transitions, we keyed the 12-entry bounded LRU `VkImage` cache by `fstat(fd).st_ino` (the kernel's unique `dma_buf` inode), guaranteeing ~100% cache hits without stale-buffer bugs or VRAM leaks.

---

## 3. Phase 2 — Deep Hot-Path Optimization

Once live KMS capture was confirmed working, we audited every hot-path layer and implemented a series of targeted optimizations:

1. **Zero-Allocation Stack DRM Ioctls**:
   - Replaced `libdrm` heap-allocating wrappers (`drmModeGetPlane`, `drmModeObjectGetProperties`, `drmModeGetFB2`) in `grab_frame()` with direct stack-allocated ioctls (`DRM_IOCTL_MODE_GETPLANE`, `DRM_IOCTL_MODE_OBJ_GETPROPERTIES` in 1 ioctl instead of 2, `DRM_IOCTL_MODE_GETFB2`, `DRM_IOCTL_PRIME_HANDLE_TO_FD`).
   - Consolidated primary + cursor export into a single `ScopedCapSysAdmin` window per frame, cutting `SYS_capset` syscalls from **8/frame to 2/frame**.
   - Switched PRIME export to `DRM_CLOEXEC` directly (read-only), avoiding failed `DRM_RDWR` retry ioctls.
2. **SPIR-V Specialization Constants & Subgroup Shuffle Shader**:
   - Converted `exact1To1`, `hasCursor`, and `enableDither` from runtime push-constant branches into SPIR-V specialization constants (`constant_id = 0..2`), compiling 4 dead-code-eliminated RDNA 4 ISA pipelines.
   - Changed workgroups from `8×8` to **`16×4` (`32×8` pixel tiles)** and added `GL_KHR_shader_subgroup_shuffle_relative` (`subgroupShuffleUp`): 15 of 16 lanes reuse the left neighbor's right-column texels directly from VGPRs, cutting 1:1 texture fetches by **33%** while achieving 32-byte coalesced Y and UV stores.
   - Eliminated 8–10 per-thread FP divisions in scaled Catmull-Rom mode by passing source-pixel coordinates (`srcBasePx`, `srcStepPx`, `invCursorSize`) and vectorized the 2×2 luma IGN dither (`ignTriangular2x2`).
3. **Zero-Lock L1-Resident Caches & Ring Buffers**:
   - Replaced `std::unordered_map` + `std::mutex` in `VulkanCompute` with flat contiguous MRU vectors (returning a 16-byte `SrcBinding` handle pair by value to prevent pointer invalidation when primary and cursor lookups both update the MRU order—a subtle bug caught and eliminated during implementation).
   - Precomputed static BT.709/BT.2020 color-space push constants once in `configure_output()`.
   - Added reusable `AVFrame *` probe shells (`probe_shells`) for LIFO pool probing and tightened `initial_pool_size` to `queue_size + async_depth * 2 + 6` (~25% VRAM savings).
   - Replaced `std::deque` with fixed-capacity ring buffers in both `EncodeFrameQueue` and `VideoEncoder::pkt_ring`, added `PR_SET_TIMERSLACK(1)` on the capture thread, and enabled `-fno-math-errno -fno-trapping-math` alongside LTO in Release builds.

---

## 4. Phase 3 — Wide-Gamut Color Accuracy Fix

### The Problem
During live testing on the user's wide-gamut `DP-1` display under KDE Plasma 6 (Wayland), recorded videos played back in `mpv` looked noticeably dull and desaturated compared to the live screen.

### Diagnosis
Inspecting `/dev/dri/card1` revealed:
- `DP-1`'s EDID reports wide-gamut QD-OLED primaries: `R=(0.6973, 0.2969)`, `G=(0.2363, 0.7197)`, `B=(0.1436, 0.0557)`, `W=(0.3125, 0.3291)`.
- `CRTC 443` had `CTM = 0` and a pure identity 4096-entry `GAMMA_LUT`.
- Therefore, KWin was applying `sRGB -> EDID_native` gamut compression **inside its compositor shader before scanout**. Grabbing those pixels and tagging the video BT.709 caused `mpv` + KWin to gamut-compress the video a **second time** on playback.

### Solution (`src/color_math.{hpp,cpp}`, `shaders/rgb_to_yuv.comp`)
- Added automatic EDID chromaticity blob parsing (`read_connector_edid_primaries`) in `KmsCapture`.
- Implemented exact CIE 1931 RGB↔XYZ + Bradford chromatic adaptation matrix computation (`compute_gamut_matrix`) with unit-tested identity, white-point preservation, and round-trip inverse checks.
- Passed `kConvertGamut` and all 9 matrix floats as **SPIR-V specialization constants (`constant_id = 3..12`)** so RADV constant-folds them directly into inline `FMA` instructions while keeping push constants at 112 bytes (within Vulkan's 128-byte guaranteed minimum).
- Applied piecewise sRGB EOTF → 3×3 linear matrix → piecewise sRGB OETF right before `subgroupShuffleUp` (zero extra passes, 389 fps at 1080p), controlled via `--input-gamut <auto|edid|srgb|p3|bt2020>`. Confirmed by the user to restore exact live colors.

---

## 5. Phase 4 — Low-Latency PipeWire Audio, Volume Sliders & Overhauled UX

### Native PipeWire Audio & Multi-Codec Muxing (`src/audio_capture.{hpp,cpp}`, `src/video_encoder.cpp`)
- Added native `libpipewire-0.3` audio capture (system desktop audio ON by default via `PW_KEY_STREAM_CAPTURE_SINK = "true"`, optional selectable microphone via `PW_KEY_TARGET_OBJECT`, `--no-sound`/`--no-audio` to disable).
- Used lock-free SPSC ring buffers (`65536` stereo `F32` frames) in PipeWire's real-time callback (`vkcap-audio-pw`) paired with a wall-clock-mastered `vkcap-aenc` thread locked to video `base_ns` so silent/suspended PipeWire sinks never cause A/V drift.
- Added independent arbitrary volume controls (`--system-volume`, `--mic-volume`, `0..1000%`) with `[[unlikely]]`-hinted `tanh` soft-limiting, plus selectable audio codecs (`opus`, `aac`, lossless `flac` via `AV_SAMPLE_FMT_S32`).

### Profiles, Saved Personal Defaults & Interactive Wizard (`src/cli.{hpp,cpp}`, `src/user_config.{hpp,cpp}`, `src/interactive.{hpp,cpp}`)
- Added 7 RDNA 4 tuned premade profiles (`default`, `gameplay`, `gameplay-10bit`, `master`, `compact`, `desktop`, `low-latency`) with pre-scan parsing so explicit CLI flags always override profile settings regardless of flag order.
- Built `~/.config/vkcap-kms/defaults.conf` persistence (`user_config.cpp`) independent of built-in profiles, generating fresh timestamped filenames with the saved container extension on every run.
- Overhauled the interactive wizard so:
  - When saved personal defaults exist, it immediately offers a one-key confirmation to start recording with zero further questions.
  - Otherwise (or if declined), it offers **Quick Setup** (profile + non-profile session options including visual volume sliders and live PipeWire mic picker) or **Advanced Setup** (every video, rate-control, tuning, color, audio, and container knob), followed by an optional prompt to save or preserve previous defaults.
- Caught and fixed codec-specific QP bounds (`0..255` for AV1 `base_q_idx` vs. `0..51` for HEVC/H.264) across both CLI validation and the Advanced Setup wizard, verified by unit tests.
