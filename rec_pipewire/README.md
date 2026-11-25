# vkcap — Zero-Copy Wayland Vulkan Video (AV1) Screen Recorder

`vkcap` is a high-performance Linux screen capture CLI built for modern Wayland desktops, PipeWire, and AMD RDNA 4 GPUs (`VK_KHR_video_encode_av1`). It captures compositor DMA-BUFs directly in VRAM, converts and optionally scales them on an isolated low-priority Vulkan async compute device, and encodes them via FFmpeg's `av1_vulkan` (`VK_KHR_video_encode_av1`) with precision Constant Frame Rate (CFR) pacing.

---

## Build & Test

### Dependencies
- **Build tools**: CMake ≥ 3.24, C++23 compiler (GCC 14+ / Clang 18+), Python 3, `glslc` (from `shaderc`)
- **System libraries** (with pkg-config files):
  - `vulkan` (Vulkan 1.3+ loader & headers; driver supporting `VK_KHR_video_encode_av1`, `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`, `VK_KHR_external_semaphore_fd`)
  - `libpipewire-0.3`, `libspa-0.2`
  - `libdrm`
  - `gio-2.0`, `gio-unix-2.0` (for `xdg-desktop-portal` over session D-Bus)
  - FFmpeg ≥ 7.1 / 8.x / 9.x (`libavcodec`, `libavformat`, `libavutil`) built with `--enable-vulkan`

### Build Commands
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### Run Automated Unit & GPU Integration Tests
```bash
ctest --test-dir build --output-on-failure --verbose
```
The test suite (`vkcap_tests`) verifies CLI/format parsing and runs real end-to-end GPU tests on your Vulkan device (allocating tiled DRM-modifier DMA-BUFs, importing them on the isolated compute device, running both 8-bit NV12 and 10-bit P010 SPIR-V shaders, encoding with `av1_vulkan`, muxing to `.mp4`/`.mkv`/`.webm`, and decoding every output frame via `libavformat`/`libavcodec`).

### Install (Optional)
```bash
cmake --install build --prefix ~/.local
```

---

## Usage

### Common Examples
Interactive `xdg-desktop-portal` display selection, native resolution, 60 fps locked CFR AV1 recording with RDNA 4 tuned defaults (`qp=52`, `quality=1`, `tune=hq`):
```bash
./build/vkcap -o recording.mkv
```

120 fps 10-bit P010 VBR gameplay recording (persisting the portal restore token for subsequent runs):
```bash
./build/vkcap -o gameplay.mp4 -f 120 --pixfmt p010 --rc vbr -b 35M --content rendered --save-restore
```

Reuse the saved portal restore token without re-opening the display picker:
```bash
./build/vkcap -o clip.mkv -f 120 --use-saved-restore --save-restore
```

Downscale 4K capture to 1440p using the GPU 5-tap Catmull-Rom bicubic filter for 20 seconds:
```bash
./build/vkcap -o scaled.mkv -s 2560x1440 -d 20
```

Variable frame rate (VFR) capture with verbose negotiation & telemetry output:
```bash
./build/vkcap -v --vfr -o desktop.mkv
```

Stop recording cleanly at any time with `Ctrl+C` (`SIGINT`) or `SIGTERM`; `vkcap` flushes delayed encoder frames and writes a valid container trailer (including `+faststart` `moov` relocation for `.mp4`).

### CLI Options Reference

| Flag | Default | Description |
|---|---|---|
| `-o, --output <path>` | `vkcap_<timestamp>.mkv` | Output file (`.mkv`, `.mp4`, `.webm`, `.ivf`; container auto-detected) |
| `-f, --fps <n>` | `60` | Target recording framerate (`1..1000`) |
| `-s, --scale <WxH>` | native | GPU Catmull-Rom bicubic output resolution (even `W` and `H`) |
| `-d, --duration <sec>` | unlimited | Automatically stop recording after `<sec>` seconds |
| `--vfr` | off (CFR) | Use dequeue-timestamped VFR instead of locked CFR pacing |
| `--cursor <mode>` | `embedded` | Portal cursor mode: `embedded`, `hidden`, `metadata` |
| `--source-type <type>` | `monitor` | Portal source types: `monitor`, `window`, `all` |
| `--restore-token <tok>` | none | Pass an explicit `xdg-desktop-portal` restore token |
| `--use-saved-restore` | off | Load restore token from `$XDG_STATE_HOME/vkcap/restore_token` |
| `--save-restore` | off | Save returned restore token to `$XDG_STATE_HOME/vkcap/restore_token` |
| `--dmabuf-only` | off | Refuse `MemFd`/`MemPtr` SHM fallback if compositor does not offer DMA-BUF |
| `--codec <name>` | `av1` | Vulkan Video encoder: `av1` (`av1_vulkan`), `hevc`, `h264` |
| `--pixfmt <fmt>` | `nv12` | Encoder pixel format: `nv12` (8-bit + IGN dither) or `p010` (10-bit) |
| `--rc <mode>` | `auto` | Rate control: `auto` (`vbr` if `-b` set, else `cqp`), `cqp`, `cbr`, `vbr` |
| `--qp <0..255>` | `52` (AV1) / `22` | Constant quantizer (`base_q_idx` for AV1) |
| `-b, --bitrate <rate>` | unset | Target bitrate (`20M`, `15000k`); implies `vbr` when `--rc auto` |
| `--max-bitrate <rate>` | `1.5 × bitrate` | Peak bitrate for VBR mode |
| `--quality <0..3>` | `auto` (`1`) | Vulkan Video quality preset (higher = slower) |
| `--tune <mode>` | `hq` | Encoder tuning: `default`, `hq`, `ll`, `ull`, `lossless` |
| `--usage <mode>` | `record` | Usage hint: `default`, `transcode`, `stream`, `record`, `conference` |
| `--content <mode>` | `rendered` | Content hint: `default`, `camera`, `desktop`, `rendered` |
| `--idr-interval <n>` | `fps * 2` (2 s) | Keyframe interval in frames |
| `--async-depth <1..64>` | `4` | Hardware encode pipeline depth (RADV AV1 caps at 4) |
| `--b-depth <1..8>` | `1` | Max B-frame reference depth (RADV GFX1200 supports `1`) |
| `--device <index>` | `0` | Vulkan physical device index |
| `--color-range <r>` | `limited` | `limited` (`tv`) or `full` (`pc`) |
| `--color-space <cs>` | `bt709` | `bt709` or `bt2020` |
| `--no-dither` | dithering on | Disable static spatial IGN dithering for 8-bit NV12 |
| `--queue-size <n>` | `4` | Bounded queue capacity (used in `--vfr` mode) |
| `-v, --verbose` | off | Print portal, PipeWire format/modifier, Vulkan, and encoder diagnostics |

---

## Detailed Technical Description

### End-to-End Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 0: Display Selection (src/portal.cpp)                                  │
│  org.freedesktop.portal.ScreenCast (v1..v6 over GIO GDBus)                   │
│  CreateSession → SelectSources → Start → OpenPipeWireRemote (fd + serial)    │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        │ PipeWire remote fd + object.serial
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: PipeWire Capture + Isolated Compute (vkcap-pw thread)               │
│  1. Drain pw_stream_dequeue_buffer() to freshest buffer (re-queue stale)     │
│  2. CFR throttle: skip compute if < 0.75 × frame_interval since last pass    │
│  3. Pick an ALREADY-IDLE AVVkFrame from FFmpeg's 24-frame pool               │
│     (vkGetSemaphoreCounterValue on FFmpeg's VkDevice)                        │
│  4. On isolated low-priority compute_dev (VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR): │
│     • Cached DMA-BUF import (VkImageDrmFormatModifierExplicitCreateInfoEXT)  │
│     • Explicit sync: DMA_BUF_IOCTL_EXPORT_SYNC_FILE → vkImportSemaphoreFdKHR │
│     • Push descriptors (VK_KHR_push_descriptor) → rgb_to_yuv.comp (~0.10 ms) │
│     • Writes directly into zero-copy OPAQUE_FD alias of AVVkFrame in VRAM    │
│     • Waits ONLY on compute_dev fence (~0.10–0.17 ms)                        │
│  5. IMMEDIATELY calls pw_stream_queue_buffer() — at most 1 pw_buffer held!   │
│  6. Stores GPU AVFrame* in LatestFrameMailbox (CFR) or EncodeFrameQueue (VFR)│
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        │ GPU-resident AVFrame* (VRAM)
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Dedicated Vulkan Video Encode Thread (vkcap-enc)                    │
│  • CFR (default): clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME) at exact   │
│    1/fps ticks; clones latest GPU AVFrame* (zero-cost skip on static screen) │
│  • VFR (--vfr): pops dequeue-timestamped frames from bounded queue           │
│  • Submits to FFmpeg av1_vulkan (VK_KHR_video_encode_av1 on VCN 5)           │
│  • Drains AVPackets via reusable scratch_pkt → pushes to bounded mux queue   │
└───────────────────────────────────────┬──────────────────────────────────────┘
                                        │ AVPacket* (256-packet bounded queue)
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Dedicated Async Muxer Thread (vkcap-mux)                            │
│  • Calls av_interleaved_write_frame() (.mkv, .mp4 +faststart, .webm, .ivf)   │
│  • Disk/NVMe flushes never stall GPU compute or VCN video encoding           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1. Portal Display Selection (`src/portal.cpp`)
`vkcap` communicates directly with `org.freedesktop.portal.ScreenCast` via `gio-2.0` (`GDBusConnection`) with zero GUI toolkit dependencies:
- Subscribes to `org.freedesktop.portal.Request::Response` on the predicted request object path *before* issuing each D-Bus method call (`CreateSession`, `SelectSources`, `Start`) to avoid signal races.
- Probes `AvailableSourceTypes`, `AvailableCursorModes`, and portal `version` so unsupported cursor/persist options degrade gracefully instead of closing the session.
- Supports ScreenCast v6's monotonically increasing 64-bit `pipewire-serial` (targeted via `PW_KEY_TARGET_OBJECT`) with automatic fallback to the legacy 32-bit `node_id`.
- Extracts the PipeWire remote socket fd from `OpenPipeWireRemote` via `GUnixFDList`, transfers ownership to `pw_context_connect_fd`, and cleanly invokes `org.freedesktop.portal.Session.Close` on shutdown.

### 2. Zero-Copy PipeWire DMA-BUF Capture (`src/pipewire_capture.cpp`)
- **Vulkan-Driven Format & Modifier Negotiation**: Queries `vkGetPhysicalDeviceFormatProperties2` (`VkDrmFormatModifierPropertiesListEXT`) and `vkGetPhysicalDeviceImageFormatProperties2` (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`) for 8-bit (`BGRx`, `BGRA`, `RGBx`, `RGBA`), 10-bit (`xRGB_210LE`, `ARGB_210LE`, `xBGR_210LE`, `ABGR_210LE`), and FP16 (`RGBA_F16`) formats. Non-linear tiled/DCC modifiers are sorted ahead of `DRM_FORMAT_MOD_LINEAR` so the compositor can share its compressed VRAM surfaces without staging copies.
- **Two-Phase Modifier Fixation**: Advertises modifiers with `SPA_POD_PROP_FLAG_MANDATORY | SPA_POD_PROP_FLAG_DONT_FIXATE` plus a modifierless fallback, and completes fixation when `SPA_VIDEO_FLAG_MODIFIER_FIXATION_REQUIRED` arrives.
- **VFR Compositor Negotiation**: Advertises `SPA_FORMAT_VIDEO_framerate = 0/1` and `SPA_FORMAT_VIDEO_maxFramerate = target_fps/1` (matching KWin/Mutter/wlroots damage-driven streams).
- **No CPU DMA-BUF Mapping**: Intentionally omits `PW_STREAM_FLAG_MAP_BUFFERS` from `pw_stream_connect` so PipeWire never `mmap()`s VRAM DMA-BUFs into CPU address space. If a compositor falls back to `SPA_DATA_MemFd`, `VulkanCompute` maps it transiently on demand.
- **Zero Compositor Starvation**: `on_process()` drains any burst of queued buffers to the freshest frame, runs the ~0.10 ms GPU compute conversion, and **immediately** calls `pw_stream_queue_buffer()` before returning. At most **1** compositor buffer is ever held, preventing KWin from running out of buffers even under 100% 3D load.

### 3. Isolated Dual-`VkDevice` Vulkan Compute (`src/vulkan_compute.cpp`)
A key design feature of `vkcap` is **Dual-`VkDevice` Isolation** on the same `VkPhysicalDevice`:
- **Why Two Logical Devices**: On AMDGPU/RADV, submissions on a `VkDevice` can attach implicit `dma_resv` fences to external BOs resident in that device's VM. If PipeWire DMA-BUFs are imported into the same `VkDevice` running `av1_vulkan`, every VCN video encode attaches its 6–10 ms fence onto KWin's DMA-BUFs — causing both `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` and compositor reuse to stall behind video encoding.
- **How Isolation Works**:
  1. FFmpeg owns `vkctx->act_dev`, which allocates the 24-frame `AVHWFramesContext` pool (`NV12` or `P010LE`) and runs `av1_vulkan` on the VCN video encode queue. It **never** imports KWin's DMA-BUFs.
  2. `VulkanCompute` creates a lightweight, compute-only `compute_dev` on the async compute queue family with **`VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR`** (fallback `priority = 0.0f`) so active 3D games and KWin always preempt `vkcap`'s compute pass.
  3. Each PipeWire `pw_buffer` DMA-BUF is imported once into `compute_dev` (`VkImageDrmFormatModifierExplicitCreateInfoEXT` + `vkImportMemoryFdKHR`), cached by `pw_buffer*`, and evicted on PipeWire's `remove_buffer` callback.
  4. Each pooled `AVVkFrame` memory object is zero-copy aliased once onto `compute_dev` via `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT` (`CachedDstAlias`), targeting the exact same VRAM pages.
- **Explicit GPU Sync & DCC Preservation**:
  - Exports KWin's implicit write fence via `DMA_BUF_IOCTL_EXPORT_SYNC_FILE(DMA_BUF_SYNC_READ)` and imports it into a temporary binary `VkSemaphore` (`VK_SEMAPHORE_IMPORT_TEMPORARY_BIT`) so `compute_dev` waits for compositing on the GPU.
  - Samples DMA-BUF images in `VK_IMAGE_LAYOUT_GENERAL` (with `VK_KHR_unified_image_layouts`) to avoid RADV GFX12 DCC decompression transitions.
  - Uses `VK_KHR_push_descriptor` (`vkCmdPushDescriptorSetKHR`) to eliminate per-frame `vkUpdateDescriptorSets()` overhead.
  - Probes up to 8 frames from FFmpeg's LIFO `AVBufferPool` using `vkGetSemaphoreCounterValue()` to select an already-idle `AVVkFrame`, ensuring compute never waits on VCN.

### 4. Embedded SPIR-V Compute Shader (`shaders/rgb_to_yuv.comp`)
Compiled at build time into two embedded `constexpr` SPIR-V arrays (`-DP010=0` with `r8`/`rg8` and `-DP010=1` with `r16`/`rg16`):
- **Workgroup Layout**: `8×8` threads per workgroup, each computing one `2×2` luma block and one `4:2:0` chroma sample (`16×16` pixels per workgroup, 2 Wave32 wavefronts on RDNA 4).
- **1:1 Native Capture**: Exact integer `texelFetch` for luma combined with a **3×2 cosited-left (`AVCHROMA_LOC_LEFT`) anti-aliased chroma filter** (`[0.25, 0.50, 0.25]` horizontal × `[0.50, 0.50]` vertical) to prevent chroma stair-stepping and color fringing on 1 px colored text and game HUD outlines.
- **Scaled Capture (`-s WxH`)**: **5-tap Catmull-Rom bicubic** resampling using hardware bilinear texture units for sharp downscaled text and detail.
- **8-bit Anti-Banding Dithering**: Single-hash **static spatial triangular-PDF Interleaved Gradient Noise (IGN)** dithering (`±0.5` LSB) for 8-bit NV12. Because it is spatially static rather than temporally random, static UI/backgrounds remain identical across frames so AV1 encodes them as near-zero-bit skip blocks.
- **Color Math**: BT.709 and BT.2020 luma/chroma matrices, Limited (`16..235` / `64..940`) and Full range, with 10-bit P010 quantized cleanly into the upper 10 bits `[15:6]` of `R10X6` / `R16` UNORM.

### 5. Precision Frame Pacing, Encoding & Async Muxing (`src/pipeline.cpp`, `src/video_encoder.cpp`)
- **Locked CFR Pacer (Default)**: The dedicated `vkcap-enc` thread wakes on exact `1/fps` deadlines via `clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME)`, encodes integer `time_base = {1, fps}` ticks with `frame->duration = 1`, and clones the newest GPU frame from `LatestFrameMailbox` (zero compute cost on static screens). Meanwhile, Stage 1 throttles compute to `0.75 × frame_interval` so 144–240 Hz gameplay doesn't trigger redundant 60 fps conversions.
- **RDNA 4 / VCN 5 Tuned Encoder Defaults**: `av1_vulkan` with `rc_mode=cqp`, `qp=52` (AV1 `base_q_idx` `0..255`), `quality=1`, `tune=hq`, `usage=record`, `content=rendered`, 2-second GOP (`idr_interval = fps * 2`), `async_depth=4`, `b_depth=1`, and `AVCHROMA_LOC_LEFT` metadata.
- **Async Muxer (`vkcap-mux`)**: `VideoEncoder` drains packets using a reusable `scratch_pkt` (zero `EAGAIN` allocation churn) into a 256-packet queue written asynchronously via `av_interleaved_write_frame()`, with `+faststart` enabled for `.mp4`.

---

## Verified Performance (AMD Radeon RX 9060 XT / RADV GFX1200)

Measured via `ctest` (`vkcap_tests`, Release build, Mesa 26.3.1, Vulkan 1.4.363, FFmpeg n9.1, all output AV1 frames decode-verified):

| Test Scenario | Compute Hold Time | End-to-End Throughput |
|---|---|---|
| 1080p NV12 DMA-BUF → AV1 MP4 | **0.101 ms/f** | **361.9 fps** |
| 1080p P010 10-bit DMA-BUF → AV1 MKV | **0.108 ms/f** | **344.0 fps** |
| 4K → 1440p Catmull-Rom Bicubic → AV1 WebM | **0.249 ms/f** | **211.9 fps** |
| 4K Native NV12 DMA-BUF → AV1 MKV | **0.168 ms/f** | **100.6 fps** |
| 720p SHM Fallback → AV1 MKV | **0.413 ms/f** | **798.5 fps** |
| DMA-BUF `SYNC_READ` wait after `encode_frame` | **0.002 ms** | Zero VCN fence pollution |

---

## Repository Layout
- `CMakeLists.txt` — C++23 build, `glslc` SPIR-V compilation, and `vkcap_tests` registration
- `cmake/embed_spv.py` — Embeds compiled `.spv` binaries as C++ `constexpr` arrays
- `shaders/rgb_to_yuv.comp` — Dual-compiled NV12/P010 compute shader
- `src/cli.{hpp,cpp}` — CLI parser, validation, and RDNA 4 defaults
- `src/portal.{hpp,cpp}` — `xdg-desktop-portal` ScreenCast v1–v6 D-Bus client
- `src/pipewire_capture.{hpp,cpp}` — PipeWire DMA-BUF stream & modifier negotiation
- `src/vulkan_compute.{hpp,cpp}` — Isolated `compute_dev`, DMA-BUF import cache, `OPAQUE_FD` aliasing, compute dispatch
- `src/video_encoder.{hpp,cpp}` — FFmpeg `av1_vulkan` encoder & dedicated async muxer thread
- `src/pipeline.{hpp,cpp}` — 3-stage pipeline orchestration, CFR pacer, and live telemetry
- `src/types.hpp`, `src/main.cpp` — Shared capture types and CLI entry point
- `tests/test_main.cpp` — Unit tests and decode-verified GPU integration benchmarks
- `DEVELOPMENT.md` — Detailed chronicle of the development process, pitfalls, and optimization iterations
