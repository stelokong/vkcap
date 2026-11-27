# `vkcap` — Zero-Copy Linux DRM/KMS → Vulkan Video Screen Recorder

`vkcap` is a compositor-independent, zero-copy Linux screen and audio recorder designed for maximum performance, minimal game interference, and accurate colors.

It captures the active display scanout framebuffer directly from Linux **DRM/KMS**, composites the hardware mouse cursor on the GPU, automatically corrects wide-gamut Wayland display colors from the monitor's EDID, converts RGB → YUV (`NV12` or 10-bit `P010`) in a specialized low-priority Vulkan compute shader, captures low-latency system and microphone audio via **PipeWire**, and encodes directly on the GPU via **Vulkan Video** (`AV1`, `HEVC`, or `H.264`) into `.mkv`, `.mp4`, `.webm`, or `.ivf`.

Because video capture bypasses PipeWire Screencast and `xdg-desktop-portal` entirely, it **never stalls or interferes with the desktop compositor** and works identically under Wayland (KDE Plasma, Sway, Hyprland, GNOME), X11, gamescope, and fullscreen direct-scanout games.

`vkcap` is **not vendor-locked**—it works on **any GPU and Vulkan 1.3+ driver** (AMD Mesa RADV, NVIDIA, Intel Mesa ANV) with DMA-BUF import and Vulkan Video encode support. On GPUs without hardware AV1 encode, `vkcap` automatically falls back to HEVC or H.264 when `--codec` is not explicitly specified, and automatically matches the Vulkan GPU to the selected display on multi-GPU systems. Its default profiles and tuning are primarily calibrated for **AMD RDNA 4 (VCN 5)** while remaining fully configurable on any supported GPU.

---

## 1. Quick Start

```bash
# 1. Build (Release with LTO)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. Grant least-privilege KMS capture capability (once after each build)
sudo setcap cap_sys_admin=p ./build/vkcap

# 3. Run the interactive setup wizard (or pass CLI flags directly)
./build/vkcap
```

Or start recording immediately from the command line:

```bash
# Record primary display at 60 fps with system audio (creates ~/Videos if needed)
./build/vkcap -o ~/Videos/clip.mkv
```

Press **`Ctrl+C`** (or send `SIGTERM` / close the terminal via `SIGHUP`) to cleanly flush both video and audio encoders and finalize the container (including MP4 `+faststart`).

---

## 2. Requirements & Building

### System & Hardware Requirements
- **Linux** with DRM/KMS (kernel **5.20+** recommended for `DMA_BUF_IOCTL_EXPORT_SYNC_FILE`; older kernels are supported with automatic fallback).
- **Any GPU with a Vulkan 1.3+ driver** supporting:
  - **Vulkan Video encode**: `VK_KHR_video_queue`, `VK_KHR_video_encode_queue`, and at least one codec extension (`VK_KHR_video_encode_av1`, `VK_KHR_video_encode_h265`, or `VK_KHR_video_encode_h264`).
  - **Zero-copy DMA-BUF import**: `VK_EXT_external_memory_dma_buf`, `VK_EXT_image_drm_format_modifier`, `VK_KHR_external_memory_fd`, `VK_KHR_external_semaphore_fd`.
  - **Subgroup relative shuffle**: `VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT` (supported on AMD RADV, NVIDIA, Intel ANV, and modern desktop GPUs; SIMD8/SIMD16/Wave32/Wave64 safe).
- **PipeWire** (`libpipewire-0.3`) for system and microphone audio capture.

### Build Dependencies
- CMake **3.24+**, C++23 compiler (GCC 13+ / Clang 17+), `glslc` (`shaderc`), Python 3, `pkg-config`
- Development headers/libraries: `vulkan`, `libdrm`, `libpipewire-0.3`, `libavcodec >= 61`, `libavformat >= 61`, `libavutil >= 59` (FFmpeg 7.1+ / 8.x / 9.x built with Vulkan Video and `libopus` / `aac` / `flac`)

### Build & Test
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run unit and decode-verified GPU integration tests (no elevated privileges needed)
ctest --test-dir build --output-on-failure --verbose
```

### One-Time Least-Privilege Setup
Listing displays (`--list-displays`), profiles (`--list-profiles`), and microphones (`--list-mics`) requires **no privileges**.

Exporting another process's KMS scanout framebuffer (`DRM_IOCTL_MODE_GETFB2` + `DRM_IOCTL_PRIME_HANDLE_TO_FD`) is restricted by the Linux kernel to `CAP_SYS_ADMIN`. Instead of running as `root` or `sudo`, grant **only** `CAP_SYS_ADMIN` in the **Permitted** (`=p`) file capability set:

```bash
sudo setcap cap_sys_admin=p ./build/vkcap
```

> **Note:** Rebuilding `./build/vkcap` replaces the binary and clears file capabilities, so re-run `setcap` after any rebuild.
>
> At runtime, `vkcap` immediately sets `PR_SET_NO_NEW_PRIVS`, strips all other capabilities, keeps `Effective = 0` everywhere, and raises `CAP_SYS_ADMIN` into `Effective` **only** for the ~5 µs DRM `GETFB2`/`PRIME_HANDLE_TO_FD` window inside `grab_frame()`. Vulkan, PipeWire, FFmpeg, config I/O, and output video files all run unprivileged as your normal user.

---

## 3. Usage Guide

### Interactive Setup Wizard & Saved Personal Defaults
Running `./build/vkcap` with **no arguments** in a terminal (or explicitly with `-i` / `--interactive`) launches the guided setup wizard:

1. **Saved Personal Defaults** (`~/.config/vkcap/defaults.conf`, respecting `$XDG_CONFIG_HOME`):
   - If you previously saved personal defaults, `vkcap` prints a summary and asks:
     `Use your saved default settings and start recording? [Y/n]:`
   - Pressing **Enter** (or `y`) starts recording **immediately** with a fresh timestamped filename in your saved output directory and container format.
   - Answering `n` enters the full setup menu. At the end, you can choose whether to overwrite your saved defaults or keep the previous ones untouched. Personal defaults never modify built-in profiles.
2. **Setup Mode**:
   - **`[1] Quick setup`**: Select **Display** → **Premade Profile** → **System audio** (`[Y/n]` + volume `0..1000%`) → **Microphone** (`[y/N]` + PipeWire device picker + volume `0..1000%`) → **Hardware mouse cursor** → **Output resolution** (`native`, `WxH`, or `720p`/`1080p`/`1440p`/`4k`) → **Output file** (supports `~/` and auto-creates missing directories).
   - **`[2] Advanced setup`**: Exposes every video, encoder, color management, audio, and container knob.
3. Before recording begins, `vkcap` prints a **full summary** and the **exact equivalent CLI command** (including all non-default advanced flags) for scripting or repeat runs. Type `q` or press `Ctrl+D` at any prompt to cancel.

### Premade Profiles (`-p, --profile <name>`)
View all built-in profiles anytime with `./build/vkcap --list-profiles`. Explicit CLI flags always override profile values regardless of argument order:

| Profile | Intended Use | Key Parameters |
| :--- | :--- | :--- |
| `default` | Near-native quality, low overhead | 60 fps, AV1*, 8-bit NV12 + IGN dither, CQP 52, `quality=1`, `hq` |
| `gameplay` | High-framerate 120 fps gameplay | 120 fps, AV1*, 8-bit NV12 + IGN dither, CQP 56, `quality=1`, `hq` |
| `gameplay-10bit` | 120 fps 10-bit gameplay (zero banding) | 120 fps, AV1*, 10-bit P010, VBR 35M (max 52.5M), `quality=1`, `hq` |
| `master` | Archival / editing master | 60 fps, AV1*, 10-bit P010, CQP 36, `quality=1`, `hq` |
| `compact` | Small files for Discord / sharing / long sessions | 60 fps, AV1*, 8-bit NV12 + IGN dither, CQP 78, 4s GOP, `hq` |
| `desktop` | Crisp UI, text, and tutorials at low bitrate | 60 fps, AV1*, 8-bit NV12 + IGN dither, CQP 62, 4s GOP, `content=desktop` |
| `low-latency` | Minimum GPU/encoder footprint for max game FPS | 60 fps, AV1*, 8-bit NV12 + IGN dither, CQP 64, `quality=0`, `ll`, `async_depth=2` |

*\*If your GPU does not support hardware AV1 encode and `--codec` was not explicitly passed, `vkcap` automatically falls back to `hevc` or `h264` and clamps `quality` to the driver's supported range.*

### Common CLI Examples
```bash
# Launch interactive wizard (or one-keypress start from saved defaults)
./build/vkcap

# List connected displays and PipeWire microphones (no privileges needed)
./build/vkcap --list-displays
./build/vkcap --list-mics

# Record DP-1 with default settings (AV1 + Opus 192k system audio)
./build/vkcap -D DP-1 -o ~/Videos/recording.mkv

# Force HEVC or H.264 explicitly
./build/vkcap -D DP-1 --codec hevc -o ~/Videos/recording.mkv

# 120 fps gameplay profile + specific USB mic boosted to 200% + system audio at 85%
./build/vkcap -D DP-1 -p gameplay --mic usb --mic-volume 200% --system-volume 85% -o ~/Videos/gameplay.mp4

# Downscale 4K display to 1440p using resolution shorthand + lossless FLAC audio
./build/vkcap -D DP-1 -s 1440p --audio-codec flac -o ~/Videos/scaled.mkv

# Zero-copy GPU crop region (WxH+X+Y)
./build/vkcap -D DP-1 --crop 1920x1080+320+180 -o ~/Videos/cropped.mkv

# Instant Replay: inherits your ~/.config/vkcap/defaults.conf (display, FPS, audio, output dir)
# Keep the last 30s (or 60s) in RAM; save via 's'/Enter or `pkill -USR2 vkcap`
./build/vkcap --instant-replay
./build/vkcap --instant-replay 60 -p gameplay -o ~/Videos/replays/clip.mp4

# Keybind controls while running:
#   pkill -USR1 vkcap   -> Pause / Resume recording (continuous A/V timestamps)
#   pkill -USR2 vkcap   -> Save Instant Replay clip (zero-stutter background write)
#   pkill -INT  vkcap   -> Stop & finalize cleanly (second Ctrl+C force-exits)

# Video-only 10-bit master recording without hardware cursor
./build/vkcap -D DP-1 -p master --no-sound --no-cursor -o ~/Videos/archive.mkv

# Override input gamut manually (e.g. if monitor is in hardware sRGB clamp mode)
./build/vkcap -D DP-1 --input-gamut srgb -o ~/Videos/clip.mkv
```

While recording, `vkcap` prints a live single-line status bar with elapsed time, framerate, frame count, drops, bitrate, file size, and GPU compute time:
```text
[vkcap] 01:23 | 60.0 fps | 4980 frames | 0 dropped | 18.40 Mbps (182.4 MiB) | compute 0.45 ms/f
```

---

## 4. Complete CLI Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-p, --profile <name>` | `default` | Apply a premade profile (`--list-profiles` to view); explicit flags override |
| `--list-profiles` | — | List all premade profiles and exit |
| `-i, --interactive` | auto if no args | Launch the interactive guided setup wizard |
| `-l, --list-displays` | — | List connected active DRM/KMS displays and exit |
| `-D, --display <name\|idx>` | `auto` | Select display by connector (e.g. `DP-1`, `HDMI-A-1`) or list index |
| `-o, --output <path>` | `vkcap_<ts>.mkv` | Output file (`.mkv`, `.mp4`, `.webm`, `.ivf`); expands `~/` and creates parent dirs |
| `-f, --fps <n>` | `60` | Locked CFR capture framerate (`1..1000`) |
| `-s, --scale <WxH\|preset>` | native | GPU Catmull-Rom bicubic scaling (even `WxH`, or `720p`, `1080p`, `1440p`, `4k`) |
| `--crop <WxH+X+Y>` | full display | Zero-copy GPU crop region in display pixels (e.g. `1920x1080+320+180`) |
| `-d, --duration <sec>` | unlimited | Stop recording automatically after `<sec>` seconds |
| `--instant-replay [sec]` | off (`30` if flag set) | Rolling instant-replay buffer (`5..600`s) in RAM (inherits `~/.config/vkcap/defaults.conf`); save via `'s'`/`Enter` or `SIGUSR2` |
| `--no-cursor` | off | Disable GPU compositing of the hardware KMS cursor plane |
| `--no-sound`, `--no-audio` | audio ON | Disable all audio recording (video-only) |
| `--no-system-audio` | system ON | Disable desktop/application audio (e.g. for mic-only) |
| `--mic [dev]` | off | Enable microphone recording (node name, ID, index, substring, or `default`) |
| `--list-mics` | — | List available PipeWire microphones and exit |
| `--system-volume <pct>` | `100%` | System audio volume percentage (`0..1000`, e.g. `80%`, `150`) |
| `--mic-volume <pct>` | `100%` | Microphone volume percentage (`0..1000`, e.g. `200%`) |
| `--audio-codec <name>` | `opus` | Audio codec: `opus`, `aac`, `flac` (`.webm` auto-selects `opus`; `.ivf` is video-only) |
| `--audio-bitrate <rate>` | `192k` | Audio bitrate for `opus` / `aac` (48 kHz stereo) |
| `--codec <name>` | `av1` (auto-fallback) | Video codec: `av1`, `hevc`, `h264` |
| `--pixfmt <fmt>` | `nv12` | Surface format: `nv12` (8-bit + IGN dither) or `p010` (10-bit) |
| `--rc <mode>` | `auto` | Rate control: `auto`, `cqp`, `cbr`, `vbr` |
| `--qp <n>` | `52` (AV1) / `22` | CQP quantizer (`0..255` for AV1, `0..51` for HEVC/H.264) |
| `-b, --bitrate <rate>` | — | Target bitrate (e.g. `20M`, `35M`); selects VBR when `--rc=auto` |
| `--max-bitrate <rate>` | `1.5× bitrate` | Max bitrate for VBR mode |
| `--quality <0..16>` | auto (`1`, clamped) | Encoder quality level (clamped to the GPU's `maxQualityLevels`) |
| `--tune <mode>` | `hq` | `default`, `hq`, `ll`, `ull`, `lossless` |
| `--usage <mode>` | `record` | `default`, `transcode`, `stream`, `record`, `conference` |
| `--content <mode>` | `rendered` | `default`, `camera`, `desktop`, `rendered` |
| `--idr-interval <n>` | `fps * 2` | Keyframe interval in frames (2-second GOP default) |
| `--async-depth <1..64>` | `4` | Vulkan Video encode pipeline depth |
| `--b-depth <1..8>` | `1` | Max B-frame reference depth |
| `--device <index>` | auto-matched | Vulkan physical device index (auto-matches KMS card on multi-GPU systems) |
| `--input-gamut <g>` | `auto` | Input framebuffer gamut: `auto`, `edid`, `srgb`, `p3`, `bt2020` |
| `--color-range <r>` | `limited` | `limited` (`tv`) or `full` (`pc`) |
| `--color-space <cs>` | `bt709` | Output color space: `bt709` or `bt2020` |
| `--no-dither` | off | Disable static spatial IGN dithering for 8-bit NV12 |
| `--queue-size <n>` | `4` | Bounded GPU-to-encoder ring buffer depth |
| `-v, --verbose` | off | Print DRM/KMS plane, Vulkan, audio, and encoder telemetry |
| `-h, --help` | — | Show CLI help |

---

## 5. Currently Unsupported / Out of Scope

`vkcap` intentionally focuses on a single, rock-solid, zero-copy recording path. The following features are **not currently supported**:

- **Window-specific capture**: Captures the DRM/KMS primary scanout plane (full display or fixed `--crop WxH+X+Y` region) rather than tracking a moving window.
- **Multi-monitor stitched capture**: Each `vkcap` instance records one selected display.
- **Per-application audio capture / separate audio tracks**: Captures all desktop audio via the PipeWire sink monitor and mixes it with the optional microphone into a single stereo track.
- **Live network streaming (RTMP / SRT / WHIP)**: Local file recording and instant replay only (`.mkv`, `.mp4`, `.webm`, `.ivf`).
- **Portrait / rotated display un-rotation**: Captures scanout planes in native orientation (`rotation` plane property is not yet applied).
- **HDR10 PQ / HLG tone-mapping or HDR static metadata passthrough**: Color management targets SDR output (`BT.709` or `BT.2020`), including automatic EDID wide-gamut inversion for SDR content on wide-gamut Wayland displays.
- **CPU software video encoding fallback**: Requires a GPU with Vulkan Video encode support (`AV1`, `HEVC`, or `H.264`).
- **Legacy PulseAudio/ALSA-only audio capture**: Audio capture uses native PipeWire (`libpipewire-0.3`). Video-only recording (`--no-sound`) works without PipeWire running.

---

## 6. Architecture Overview

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
  │  PipeWire Sink Monitor (20ms F32) ─▶ Lock-free SPSC Ring ──┐│           │
  │  PipeWire Selected Mic (20ms F32) ─▶ Lock-free SPSC Ring ──┼┤           │
  │  [vkcap-audio-pw RT thread]                                ││           │
  │                                                            ▼▼           │
  │  [vkcap-aenc] Wall-clock 20ms tick (same base_ns)     [vkcap-mux]       │
  │  Gain (sys/mic %) + soft-limit → Opus/AAC/FLAC ───▶ Recycled AVPacket   │
  │                                                     Ring → libavformat  │
  └─────────────────────────────────────────────────────────────────────────┘
```

For a deep dive into how `vkcap` minimizes GPU/compositor interference, its shader and pipeline optimizations, and verified GPU benchmarks, see [PERFORMANCE.md](PERFORMANCE.md).
