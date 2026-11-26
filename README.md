# `vkcap` — Zero-Copy Linux DRM/KMS → Vulkan Video Screen Recorder

`vkcap` is a compositor-independent, zero-copy Linux screen and audio recorder. It grabs display scanout framebuffers directly from Linux DRM/KMS, composites the hardware cursor plane on the GPU, corrects wide-gamut display colors automatically from the monitor's EDID, converts RGB → YUV (`NV12` or 10-bit `P010`) in a specialized low-priority Vulkan compute shader, captures low-latency system and microphone audio via PipeWire, and encodes directly on the GPU via **Vulkan Video** (`AV1`, `HEVC`, or `H.264`) into `.mkv`, `.mp4`, `.webm`, or `.ivf`.

`vkcap` is **not vendor-locked or RDNA 4-specific** — it works on **any GPU and Vulkan driver** (AMD Mesa RADV, NVIDIA, Intel Mesa ANV) that implements Vulkan 1.3+, DMA-BUF import, and the standard Khronos Vulkan Video encode extensions for your chosen codec. Its default presets, shader wave layout, and encoder parameters are primarily optimised for **AMD RDNA 4 (VCN 5)**, while every codec and tuning knob remains fully configurable for any Vulkan Video GPU.

Because video capture bypasses PipeWire Screencast and `xdg-desktop-portal` entirely, it **never stalls or interferes with the desktop compositor** and works identically under Wayland (KDE Plasma, Sway, Hyprland, GNOME), X11, gamescope, and fullscreen direct-scanout games.

---

## 1. Setup & Installation

### Requirements
- **Linux kernel 5.20+** with DRM/KMS and `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`
- **Any GPU with a Vulkan 1.3+ driver** implementing:
  - **Vulkan Video encode**: `VK_KHR_video_queue`, `VK_KHR_video_encode_queue`, and the codec extension matching your `--codec` choice (`VK_KHR_video_encode_av1` for AV1, `VK_KHR_video_encode_h265` for HEVC, or `VK_KHR_video_encode_h264` for H.264)
  - **Zero-copy DMA-BUF import**: `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`, `VK_KHR_external_memory_fd`, `VK_KHR_external_semaphore_fd`
  - Works on AMD (Mesa RADV), NVIDIA, and Intel (Mesa ANV). Default profiles and compute/encoder tuning are primarily optimised for **AMD RDNA 4 / VCN 5** on Mesa RADV 26+; on GPUs without hardware AV1 encode, pass `--codec hevc` or `--codec h264`.
- **Build dependencies**: CMake 3.24+, C++23 compiler (GCC 13+ / Clang 17+), `glslc` (shaderc), Python 3, `pkg-config`
- **Libraries**: `vulkan`, `libdrm`, `libpipewire-0.3`, `libavcodec >= 61`, `libavformat >= 61`, `libavutil >= 59` (FFmpeg 7.1+ / 9.x with Vulkan Video and `libopus` / `aac` / `flac`)

### Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the unit and decode-verified GPU integration tests (no elevated privileges needed):
```bash
ctest --test-dir build --output-on-failure --verbose
```

### One-Time Least-Privilege Setup
Listing displays (`--list-displays`), listing profiles (`--list-profiles`), and listing microphones (`--list-mics`) require **no privileges at all**.

Exporting another process's KMS scanout framebuffer (`DRM_IOCTL_MODE_GETFB2` + `DRM_IOCTL_PRIME_HANDLE_TO_FD`) is restricted by the Linux kernel to `CAP_SYS_ADMIN`. Instead of running as `root` or via `sudo`, grant **only** `CAP_SYS_ADMIN` in the **Permitted** (`=p`) file capability set once after building:

```bash
sudo setcap cap_sys_admin=p ./build/vkcap
```

> **Note:** Rebuilding the binary overwrites it and clears its file capabilities, so re-run `setcap` after any rebuild.
>
> At runtime, `vkcap` starts with **Effective = 0**, locks `PR_SET_NO_NEW_PRIVS`, strips all other capabilities, and raises `CAP_SYS_ADMIN` into Effective **only** for the ~5 µs `GETFB2`/`PRIME_HANDLE_TO_FD` window inside `grab_frame()`. Vulkan, PipeWire, FFmpeg, config persistence, and output video creation all run unprivileged as your normal user.

---

## 2. Customisation & Usage

### Interactive Setup Wizard (Zero-Argument Launch)
Running `./build/vkcap` with **no arguments** in a terminal (or explicitly with `-i` / `--interactive`) launches the guided setup wizard:

1. **Saved Personal Defaults** (`~/.config/vkcap/defaults.conf`, respecting `$XDG_CONFIG_HOME`):
   - If you previously saved your personal defaults, `vkcap` displays a summary and asks:
     `Use your saved default settings and start recording? [Y/n]:`
   - Answering **Yes** (or pressing **Enter**) starts recording **immediately** with no further questions and a fresh timestamped filename (preserving your preferred container extension).
   - Answering **No** enters the full setup menu. At the end, you can choose whether to overwrite your saved defaults or keep the previous ones intact. Personal defaults are stored separately and **never modify any built-in profile**.
2. **Setup Mode** (shown on first run, or if you declined saved defaults):
   - **`[1] Quick setup`**: Choose **Display** → **Premade Profile** (locks in FPS, codec, bit depth, rate control, quality, and tune) → **System audio** (`[Y/n]` + volume slider `0..1000%`) → **Microphone** (`[y/N]` + PipeWire device picker + volume slider `0..1000%`) → **Hardware mouse cursor** → **Output resolution** → **Output file**.
   - **`[2] Advanced setup`**: Configure every single parameter across 5 sections — video codec (`av1`/`hevc`/`h264`), pixel format (`nv12`/`p010`), FPS, hardware cursor, auto-stop duration, rate control (`cqp`/`vbr`/`cbr`), codec-bounded QP or bitrate/max-bitrate, quality level, tune, usage, content hint, IDR interval, async encode depth, B-frame depth, GPU device index, queue size, input gamut (`auto`/`edid`/`srgb`/`p3`/`bt2020`), output color space/range, IGN dithering, system & mic audio toggles/device/volume sliders, audio codec (`opus`/`aac`/`flac`), audio bitrate, resolution scaling, and output container.
3. Before recording begins, the wizard prints the **full summary** and the **exact equivalent CLI command** for easy scripting or repeat use. Type `q` or `Ctrl+D` at any prompt to cancel.

### Premade Profiles (`-p, --profile <name>`)
View all built-in profiles anytime with `./build/vkcap --list-profiles`. Profiles default to AV1 with parameters tuned primarily for RDNA 4, and any explicit CLI flag (such as `--codec hevc` or `--codec h264` on GPUs without hardware AV1 encode) overrides the chosen profile regardless of flag order:

| Profile | Intended Use | Key Parameters |
| :--- | :--- | :--- |
| `default` | Near-native quality, low overhead (RDNA 4-tuned default) | 60 fps, AV1, 8-bit NV12 + IGN dither, CQP 52, `quality=1`, `hq` |
| `gameplay` | High-framerate 120 fps gameplay | 120 fps, AV1, 8-bit NV12 + IGN dither, CQP 56, `quality=1`, `hq` |
| `gameplay-10bit` | 120 fps 10-bit gameplay (zero banding) | 120 fps, AV1, 10-bit P010, VBR 35M (max 52.5M), `quality=1`, `hq` |
| `master` | Archival / editing master | 60 fps, AV1, 10-bit P010, CQP 36, `quality=1`, `hq` |
| `compact` | Small files for Discord / sharing / long sessions | 60 fps, AV1, 8-bit NV12 + IGN dither, CQP 78, 4s GOP, `hq` |
| `desktop` | Crisp UI, text, and tutorials at low bitrate | 60 fps, AV1, 8-bit NV12 + IGN dither, CQP 62, 4s GOP, `content=desktop` |
| `low-latency` | Minimum GPU/encoder footprint for max game FPS | 60 fps, AV1, 8-bit NV12 + IGN dither, CQP 64, `quality=0`, `ll`, `async_depth=2` |

### Common CLI Examples
```bash
# Interactive wizard (or instant start from saved personal defaults)
./build/vkcap

# List connected displays and PipeWire microphones (no privileges needed)
./build/vkcap --list-displays
./build/vkcap --list-mics

# Record DP-1 with default settings (AV1 + Opus 192k system audio)
./build/vkcap -D DP-1 -o recording.mkv

# Record using HEVC or H.264 (works on any GPU with Vulkan Video H.265/H.264 encode)
./build/vkcap -D DP-1 --codec hevc -o recording.mkv

# 120 fps gameplay profile + specific USB mic boosted to 200% volume
./build/vkcap -D DP-1 -p gameplay --mic usb --mic-volume 200% --system-volume 85% -o gameplay.mp4

# Video-only 10-bit master archival recording without hardware cursor
./build/vkcap -D DP-1 -p master --no-sound --no-cursor -o archive.mkv

# Downscale 4K display to 1440p with GPU Catmull-Rom bicubic filter + lossless FLAC audio
./build/vkcap -D DP-1 -s 2560x1440 --audio-codec flac -o scaled.mkv

# Override input gamut manually (e.g. if monitor is in hardware sRGB clamp mode)
./build/vkcap -D DP-1 --input-gamut srgb -o clip.mkv
```

Press `Ctrl+C` (`SIGINT`) or send `SIGTERM` to flush both video and audio encoders and finalize the container (including MP4 `+faststart`).

### Complete CLI Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-p, --profile <name>` | `default` | Apply a premade profile (`--list-profiles` to view); explicit flags override |
| `--list-profiles` | — | List all premade profiles and exit |
| `-i, --interactive` | auto if no args | Launch the interactive guided setup wizard |
| `-l, --list-displays` | — | List connected active DRM/KMS displays and exit |
| `-D, --display <name\|idx>` | `auto` | Select display by connector (e.g. `DP-1`, `HDMI-A-1`) or list index |
| `-o, --output <path>` | `vkcap_<ts>.mkv` | Output file (`.mkv`, `.mp4`, `.webm`, `.ivf`) |
| `-f, --fps <n>` | `60` | Locked CFR capture framerate (`1..1000`) |
| `-s, --scale <WxH>` | native | GPU Catmull-Rom bicubic scaling (even width & height) |
| `-d, --duration <sec>` | unlimited | Stop recording automatically after `<sec>` seconds |
| `--no-cursor` | off | Disable GPU compositing of the hardware KMS cursor plane |
| `--no-sound`, `--no-audio` | audio ON | Disable all audio recording (video-only) |
| `--no-system-audio` | system ON | Disable desktop/application audio (e.g. for mic-only) |
| `--mic [dev]` | off | Enable microphone recording (node name, ID, index, substring, or `default`) |
| `--list-mics` | — | List available PipeWire microphones and exit |
| `--system-volume <pct>` | `100%` | System audio volume percentage (`0..1000`, e.g. `80%`, `150`) |
| `--mic-volume <pct>` | `100%` | Microphone volume percentage (`0..1000`, e.g. `200%`) |
| `--audio-codec <name>` | `opus` | Audio codec: `opus`, `aac`, `flac` |
| `--audio-bitrate <rate>` | `192k` | Audio bitrate for `opus` / `aac` (48 kHz stereo) |
| `--codec <name>` | `av1` | Video codec: `av1`, `hevc`, `h264` |
| `--pixfmt <fmt>` | `nv12` | Surface format: `nv12` (8-bit + IGN dither) or `p010` (10-bit) |
| `--rc <mode>` | `auto` | Rate control: `auto`, `cqp`, `cbr`, `vbr` |
| `--qp <n>` | `52` (AV1) / `22` | CQP quantizer (`0..255` for AV1, `0..51` for HEVC/H.264) |
| `-b, --bitrate <rate>` | — | Target bitrate (e.g. `20M`, `35M`); selects VBR when `--rc=auto` |
| `--max-bitrate <rate>` | `1.5× bitrate` | Max bitrate for VBR mode |
| `--quality <0..3>` | `1` (auto) | Encoder quality level (`1` is the RDNA 4 quality/VRAM sweet spot; adjust per GPU) |
| `--tune <mode>` | `hq` | `default`, `hq`, `ll`, `ull`, `lossless` |
| `--usage <mode>` | `record` | `default`, `transcode`, `stream`, `record`, `conference` |
| `--content <mode>` | `rendered` | `default`, `camera`, `desktop`, `rendered` |
| `--idr-interval <n>` | `fps * 2` | Keyframe interval in frames (2-second GOP default) |
| `--async-depth <1..64>` | `4` | Vulkan Video encode pipeline depth (default `4`, RADV GFX1200 safe max) |
| `--b-depth <1..8>` | `1` | Max B-frame reference depth (default `1`, matching RADV AV1 `maxSingleReferenceCount`) |
| `--device <index>` | `0` | Vulkan physical device index |
| `--input-gamut <g>` | `auto` | Input framebuffer gamut: `auto`, `edid`, `srgb`, `p3`, `bt2020` |
| `--color-range <r>` | `limited` | `limited` (`tv`) or `full` (`pc`) |
| `--color-space <cs>` | `bt709` | Output color space: `bt709` or `bt2020` |
| `--no-dither` | off | Disable static spatial IGN dithering for 8-bit NV12 |
| `--queue-size <n>` | `4` | Bounded GPU-to-encoder ring buffer depth |
| `-v, --verbose` | off | Print DRM/KMS plane, Vulkan, audio, and encoder telemetry |
| `-h, --help` | — | Show CLI help |

---

## 3. Technical Architecture — How `vkcap` Runs

```text
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        ZERO-COPY VIDEO PATH                             │
  │                                                                         │
  │  DRM/KMS Primary Plane (FB2) ─┐                                         │
  │  DRM/KMS Cursor Plane  (FB2) ─┼─▶ [vkcap] Direct stack ioctls           │
  │  (single ScopedCapSysAdmin)   │   CLOCK_MONOTONIC + PR_SET_TIMERSLACK   │
  │                               │       │                                 │
  │                               │       ▼ DMA-BUF fd (st_ino MRU cache)   │
  │                               │   Isolated Low-Priority Vulkan Compute  │
  │                               │   Device (compute_dev, priority 0.0)    │
  │                               │   rgb_to_yuv.comp (16×4, subgroupUp,    │
  │                               │   hw cursor, EDID gamut, IGN dither)    │
  │                               │       │                                 │
  │                               │       ▼ OPAQUE_FD alias → AVVkFrame     │
  │                               │   Fixed Ring EncodeFrameQueue (depth 4) │
  │                               │       │                                 │
  │                               │       ▼                                 │
  │                               │   [vkcap-enc] FFmpeg Vulkan Video       │
  │                               │   (av1_vulkan / hevc_vulkan / h264)     │
  │                               └───────┼─────────────────────┐           │
  └───────────────────────────────────────┼─────────────────────┼───────────┘
                                          │                     │
  ┌───────────────────────────────────────┼─────────────────────┼───────────┐
  │                       LOW-LATENCY AUDIO PATH                │           │
  │                                                             │           │
  │  PipeWire Sink Monitor (F32 48kHz) ─▶ Lock-free SPSC Ring ─┐│           │
  │  PipeWire Selected Mic (F32 48kHz) ─▶ Lock-free SPSC Ring ─┼┤           │
  │  [vkcap-audio-pw RT thread]                                ││           │
  │                                                            ▼▼           │
  │  [vkcap-aenc] Wall-clock 20ms tick (same base_ns)     [vkcap-mux]       │
  │  Gain (sys/mic %) + soft-limit → Opus/AAC/FLAC ───▶ Recycled AVPacket   │
  │                                                     Ring → libavformat  │
  └─────────────────────────────────────────────────────────────────────────┘
```

### Stage 0 — Least-Privilege Capability Scoping (`privileges.cpp`)
Immediately on entry to `main()`, `init_least_privilege()` executes:
1. `prctl(PR_SET_NO_NEW_PRIVS, 1)` to prevent privilege escalation across any future `execve`.
2. Direct `SYS_capget` / `SYS_capset` (`_LINUX_CAPABILITY_VERSION_3`) to clear every bit in Effective, Inheritable, and Permitted except `CAP_SYS_ADMIN` in Permitted (if granted via `setcap cap_sys_admin=p`).
3. During steady-state capture, `ScopedCapSysAdmin` raises `CAP_SYS_ADMIN` into Effective **once per frame** solely around the `DRM_IOCTL_MODE_GETFB2` and `DRM_IOCTL_PRIME_HANDLE_TO_FD` ioctls (2 `SYS_capset` calls/frame), then immediately drops Effective back to `0`.

### Stage 1 — Direct DRM/KMS Scanout Capture (`kms_capture.cpp`)
- Scans `/dev/dri/card*` with `DRM_CLIENT_CAP_UNIVERSAL_PLANES` and `DRM_CLIENT_CAP_ATOMIC`.
- Enumerates connected active connectors, CRTCs, primary planes (`DRM_PLANE_TYPE_PRIMARY`), hardware cursor planes (`DRM_PLANE_TYPE_CURSOR`), and parses each connector's `EDID` property blob (bytes `25..34`) to extract CIE 1931 chromaticities.
- In `grab_frame()`, queries the active primary `fb_id` via stack-allocated `DRM_IOCTL_MODE_GETPLANE` and cursor properties (`FB_ID`, `CRTC_X/Y/W/H`) via a single stack-allocated `DRM_IOCTL_MODE_OBJ_GETPROPERTIES` (zero heap allocations).
- Exports primary and (if visible) cursor GEM handles to read-only `DRM_CLOEXEC` DMA-BUF fds via `DRM_IOCTL_MODE_GETFB2` + `DRM_IOCTL_PRIME_HANDLE_TO_FD`, immediately closes the temporary GEM handles with `DRM_IOCTL_GEM_CLOSE`, and calls `fstat(fd).st_ino` to obtain the globally unique kernel `dma_buf` inode key (immune to DRM `fb_id` reuse).

### Stage 2 — Isolated Dual-`VkDevice` Compute & Color Pipeline (`vulkan_compute.cpp`, `rgb_to_yuv.comp`)
- **Dual-`VkDevice` Isolation**: FFmpeg's `AVVulkanDeviceContext` (`vkctx->act_dev`) runs Vulkan Video encoding. Importing external KMS DMA-BUFs directly into `vkctx->act_dev` on AMDGPU/RADV causes RADV's per-VM external BO list to attach VCN encode fences onto the compositor's scanout buffers, stalling the desktop/game. `vkcap` therefore creates a **separate, isolated low-priority `compute_dev`** (`VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR`, queue priority `0.0f`) on the async compute queue family for all external DMA-BUF imports, and aliases FFmpeg's destination `AVVkFrame` into `compute_dev` via `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT` — portable across any Vulkan 1.3 driver and especially impactful on AMDGPU/RADV.
- **Flat L1-Resident MRU Caches**: Both the 12-entry `st_ino` DMA-BUF import cache (`src_cache`) and the `AVVkFrame` alias cache (`dst_alias_cache`) are flat contiguous vectors with zero mutexes or hashing. Triple-buffered scanout hits indices `0..2` in L1 cache, while bounded LRU eviction prevents VRAM leaks across fullscreen direct-scanout transitions.
- **DMA-BUF Synchronization Fast-Path**: Calls `DMA_BUF_IOCTL_EXPORT_SYNC_FILE` (`DMA_BUF_SYNC_READ`) followed by non-blocking `poll(&pfd, 1, 0)`. When the scanout buffer is already signaled (>99% of frames), the sync fd is closed immediately without importing a Vulkan semaphore; otherwise it is imported via `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT` so the GPU waits without stalling the CPU.
- **LIFO `AVBufferPool` Probing**: Probes FFmpeg's LIFO hardware frame pool via `vkGetSemaphoreCounterValue()` using reusable empty `AVFrame` shells (`probe_shells`) so compute immediately grabs an idle `AVVkFrame` without stalling on in-flight hardware encodes.
- **Specialized Compute Shader (`rgb_to_yuv.comp`)**:
  - Uses standard Vulkan 1.3 SPIR-V compiled into 4 specialized pipeline variants via **SPIR-V specialization constants** (`kExact1To1`, `kHasCursor`, `kEnableDither`, `kConvertGamut`, plus the 9 coefficients of the 3×3 Bradford gamut matrix `kGm00..kGm22`). Dead branches (scaling, cursor blending, dithering, gamut conversion) are completely removed from the compiled GPU ISA.
  - Runs in **`16×4` workgroups** (`32×8` pixel tiles, 32-byte coalesced Y and UV stores — tuned for RDNA 4 Wave32/Wave64 and efficient across modern GPUs). In 1:1 mode, 15 of 16 lanes reuse the left neighbor's right-column texels directly from registers via **`subgroupShuffleUp`** (cutting texture fetches by 33% from 6 to 4 per 2×2 block).
  - Composites the hardware cursor plane in premultiplied alpha at exact CRTC coordinates, applies piecewise sRGB EOTF → Bradford 3×3 linear gamut matrix → piecewise sRGB OETF when capturing a wide-gamut Wayland display, computes 3×2 MPEG-2 `AVCHROMA_LOC_LEFT` anti-aliased chroma (or zero-division source-pixel 5-tap Catmull-Rom bicubic scaling), and applies vectorized 2×2 static spatial triangular-PDF IGN dithering for 8-bit NV12.

### Stage 3 — Hardware Video Encoding, PipeWire Audio & Interleaved Muxing (`video_encoder.cpp`, `audio_capture.cpp`, `pipeline.cpp`)
- **Video Pacing (`vkcap` thread)**: Sets `PR_SET_TIMERSLACK(1)` and paces capture on `CLOCK_MONOTONIC` via `clock_nanosleep(TIMER_ABSTIME)` for locked CFR timestamps (`time_base = {1, fps}`, `frame->duration = 1`), pushing completed GPU `AVFrame`s into a fixed-capacity ring buffer (`EncodeFrameQueue`, drop-oldest on extreme backpressure).
- **Vulkan Video Encode (`vkcap-enc` thread)**: Submits GPU-resident `AVVkFrame`s directly to `av1_vulkan` (or `hevc_vulkan`/`h264_vulkan`) via standard Vulkan Video, using RDNA 4-tuned defaults (`qp=52`, `quality=1`, `tune=hq`, `usage=record`, `content=rendered`, 2s GOP, `async_depth=4`, `b_depth=1`) that can be freely overridden on any GPU.
- **Low-Latency Audio (`vkcap-audio-pw` + `vkcap-aenc` threads)**: PipeWire's real-time callback pushes 48 kHz stereo `F32` samples into lock-free SPSC ring buffers (`65536` frames, zero locks/allocs). `vkcap-aenc` wakes every 20 ms aligned to the exact same `base_ns` as video frame `0`, reads/pads/trims for drift-free A/V sync, applies independent system & mic gains with soft-limiting, and encodes directly to Opus (`libopus`, `AV_SAMPLE_FMT_FLT`), AAC (`AV_SAMPLE_FMT_FLTP`), or lossless FLAC (`AV_SAMPLE_FMT_S32`).
- **Dedicated Muxer (`vkcap-mux` thread)**: Drains encoded video and audio packets through a fixed 256-entry ring buffer with a 32-entry recycled `AVPacket` pool into `av_interleaved_write_frame()`, keeping disk I/O completely decoupled from capture and encoding.

---

## Further Reading
- [PERFORMANCE.md](PERFORMANCE.md) — How `vkcap` achieves minimal overhead, high visual quality, and color/audio accuracy compared to other Linux screen recorders.
