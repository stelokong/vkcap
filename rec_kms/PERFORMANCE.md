# `vkcap-kms` Performance, Quality & Design Philosophy

This document explains how `vkcap-kms` is engineered to outperform traditional Linux screen recorders (such as OBS Studio, `wf-recorder`, `ffmpeg -f kmsgrab`, and general-purpose PipeWire recorders) across **runtime performance**, **game/compositor frame-pacing isolation**, **visual & color fidelity**, **audio latency**, and **user customisation**.

---

## 1. Performance & Zero-Interference Frame Pacing

### Why Other Recorders Cause Stutter Under 3D Gaming Load
1. **Compositor Coupling (OBS PipeWire / `xdg-desktop-portal` / `wf-recorder`)**:
   - Wayland screencast protocols force the compositor (KWin, Mutter, wlroots) to schedule an extra offscreen copy or hold swapchain buffers until the recorder finishes consuming them. Under heavy GPU load (e.g. a competitive FPS at 240–360 Hz), holding even 2–3 compositor buffers or triggering extra compositor render passes causes visible desktop/game micro-stutter.
   - **`vkcap-kms` solution**: Bypasses the compositor completely by reading the already-scanned-out DRM/KMS primary plane (`DRM_IOCTL_MODE_GETFB2` + `DRM_IOCTL_PRIME_HANDLE_TO_FD`). The compositor never knows `vkcap-kms` is running, never allocates extra screencast buffers, and never waits on `vkcap-kms` to finish a frame.
2. **AMDGPU/RADV VCN Fence Pollution (FFmpeg `kmsgrab` / Single-`VkDevice` Recorders)**:
   - On AMDGPU/RADV, importing external DMA-BUFs into the same `VkDevice` that runs Vulkan Video (`av1_vulkan`) places those external BOs on RADV's per-VM external BO list. Every VCN encode submission then attaches a read/write fence onto the compositor's/game's scanout DMA-BUF, causing **6+ ms stalls** (`SYNC_READ`) on the display swapchain.
   - **`vkcap-kms` solution**: **Dual-`VkDevice` Isolation**. All external KMS DMA-BUFs are imported exclusively into an isolated, low-priority async compute `VkDevice` (`VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR`, queue priority `0.0f`), which writes into FFmpeg's destination `AVVkFrame` aliased via `OPAQUE_FD`. The VCN encoder device never sees the scanout DMA-BUF, dropping scanout fence stalls from **~6.07 ms to 0.002 ms** and keeping total 1080p–4K compute hold time to **0.10–0.17 ms/frame**.
3. **CPU & Syscall Overhead**:
   - Traditional tools allocate heap structs on every DRM ioctl, hash into mutex-guarded maps, copy frames through CPU RAM (`glReadPixels` / `swscale`), or allocate/free `AVFrame` and `AVPacket` objects every frame.
   - **`vkcap-kms` solution**:
     - **100% zero-copy GPU path**: Display VRAM → Vulkan Compute → VCN hardware encoder VRAM. Zero pixel bytes ever cross PCIe or touch system RAM.
     - **Direct stack-allocated DRM ioctls**: `DRM_IOCTL_MODE_GETPLANE`, `DRM_IOCTL_MODE_OBJ_GETPROPERTIES` (1 ioctl instead of libdrm's 2), `DRM_IOCTL_MODE_GETFB2`, and `DRM_IOCTL_PRIME_HANDLE_TO_FD` perform **zero heap allocations per frame**.
     - **Single `ScopedCapSysAdmin` window per frame**: Cuts `SYS_capset` syscalls from 8/frame to 2/frame while keeping `Effective = 0` everywhere else.
     - **Flat L1-cache-resident MRU tables**: `st_ino` DMA-BUF import cache (max 12 entries) and `AVVkFrame` alias cache are flat contiguous arrays with zero mutexes or hashing—triple-buffered scanout hits indices `0..2` in L1 cache.
     - **Zero steady-state heap allocations across threads**: Fixed-capacity ring buffers (`EncodeFrameQueue`, `pkt_ring`), reusable empty `AVFrame` probe shells (`probe_shells`), and a 32-entry recycled `AVPacket` pool.
     - **Precision CFR pacing**: `CLOCK_MONOTONIC` `clock_nanosleep(TIMER_ABSTIME)` with `PR_SET_TIMERSLACK(1)` eliminates variable frame-time jitter.

### Compute Shader Efficiency (`shaders/rgb_to_yuv.comp`)
- **SPIR-V Specialization Constants (`constant_id = 0..12`)**: Compiles dead-code-eliminated RDNA 4 ISA variants for `exact1To1`, `hasCursor`, `enableDither`, `convertGamut`, and the 9 gamut matrix floats. Fullscreen gaming without a hardware cursor runs with zero scaling or cursor instructions and minimal VGPR register pressure for maximum GPU wave occupancy.
- **`16×4` Workgroups + `subgroupShuffleUp`**: Each workgroup processes a `32×8` pixel tile (32-byte coalesced Y and UV stores). In 1:1 mode, 15 of 16 lanes read the left neighbor's right-column texels directly from VGPRs via 1-cycle `subgroupShuffleUp`, cutting texture fetches by **33%** (from 6 to 4 per 2×2 block).
- **Single-Pass Cursor + Color + Scaling + YUV**: Hardware cursor compositing, EDID gamut conversion, Catmull-Rom scaling, chroma filtering, and IGN dithering all execute in **one single compute dispatch** (zero intermediate render targets or extra VRAM bandwidth).

### Verified GPU Throughput (AMD Radeon RX 9060 XT / RDNA 4, Mesa 26.3.1, FFmpeg 9.1)
Measured via `ctest --test-dir build --verbose` (end-to-end DMA-BUF → Compute → `av1_vulkan` → Mux → Decode verification):

| Workload | End-to-End Throughput | Compute Wall Time (Submit + Fence) |
| :--- | :---: | :---: |
| 1080p NV12 + HW Cursor + Wide-Gamut → MP4 (AV1 + Opus) | **389.0 fps** | **0.150 ms / frame** |
| 1080p P010 (10-bit) + HW Cursor → MKV (AV1 + Opus) | **348.6 fps** | **0.103 ms / frame** |
| 4K → 1440p Catmull-Rom Bicubic + HW Cursor → WebM (AV1) | **225.5 fps** | **0.255 ms / frame** |
| 4K Native NV12 → MKV (AV1) | **105.1 fps** | **0.162 ms / frame** |

---

## 2. Visual Quality & Color Accuracy

Most hardware screen recorders sacrifice visual fidelity by using naive 2×2 box-average chroma downsampling, un-dithered 8-bit quantization, bilinear scaling, or ignoring Wayland wide-gamut color management. `vkcap-kms` addresses every one of these:

1. **Automatic EDID Wide-Gamut Color Inversion**:
   - On wide-gamut displays (such as QD-OLED / DCI-P3 panels) under KDE Plasma / Wayland, the compositor gamut-maps sRGB content into the monitor's native wide gamut before scanout.
   - Standard KMS grabbers record those wide-gamut values and tag them as BT.709, causing media players (`mpv`, browsers, YouTube) to double-desaturate the video into dull, washed-out colors.
   - `vkcap-kms` reads the display's exact CIE 1931 EDID primaries, computes the Bradford-adapted 3×3 linear RGB→RGB inverse matrix, and applies **piecewise sRGB EOTF → linear 3×3 matrix → piecewise sRGB OETF** directly inside the compute shader. Recordings match what you see on screen and look accurate on any monitor.
2. **3×2 MPEG-2 `AVCHROMA_LOC_LEFT` Anti-Aliased Chroma Filter**:
   - Standard 2×2 box averaging shifts 4:2:0 chroma 0.5 pixels to the right relative to the `AVCHROMA_LOC_LEFT` standard and introduces harsh aliasing on thin colored UI/game text.
   - `vkcap-kms` computes the exact horizontal `[1, 2, 1]/4` × vertical `[1, 1]/2` filter centered on the left column, eliminating chroma shift and color fringing.
3. **Static Spatial Triangular-PDF IGN Dithering (8-bit NV12)**:
   - Converts 8-bit NV12 smooth gradients (skies, fog, dark UI panels) without visible color banding.
   - Uses a vectorized 2×2 **static spatial** Interleaved Gradient Noise hash (no temporal seed) so it never introduces temporal film-grain noise that would waste AV1 bitrate.
   - For zero-banding archival or 10-bit workflows, `--pixfmt p010` writes true 10-bit `G10X6_B10X6R10X6` surfaces directly into VCN 5.
4. **5-Tap Catmull-Rom Bicubic Scaling**:
   - When `--scale WxH` is active, `vkcap-kms` uses a zero-division source-pixel 5-tap Catmull-Rom filter via hardware bilinear texture units—noticeably sharper and cleaner than standard bilinear downscaling.
5. **RDNA 4 / VCN 5 Tuned Defaults**:
   - Defaults to **AV1 CQP `qp=52`**, `quality=1` (~20% lower VCN/VRAM load than `quality=2` with near-identical SSIM/PSNR), `tune=hq`, `usage=record`, `content=rendered`, 2-second GOP, `async_depth=4`, and `b_depth=1`—delivering effectively native visual quality (~33 Mbps at 1440p60) without starving games of GPU/VRAM bandwidth.

---

## 3. Low-Latency, Glitch-Free Audio

- **Direct Native PipeWire RT Capture (`libpipewire-0.3`)**: Captures 48 kHz stereo `F32` at a 10 ms quantum (`480/48000`) directly into cache-line-aligned **lock-free SPSC ring buffers** (zero locks, zero syscalls, zero heap allocations inside the real-time audio callback—no PulseAudio emulation layer, no xruns under load).
- **Wall-Clock Mastered A/V Sync**: Unlike recorders whose audio drifts or desyncs whenever PipeWire suspends an idle desktop audio sink during silence, `vkcap-aenc` is mastered to the exact same `CLOCK_MONOTONIC` `base_ns` as video frame `0`, emitting exact 20 ms frames (padding silence when idle, trimming drift if > 80 ms) so A/V sync stays locked indefinitely.
- **Per-Source Arbitrary Gain + Soft-Limiting**: System audio and microphone have independent linear gain controls (`0..1000%`, e.g. `200%` mic boost) passed through a smooth `tanh` soft-limiter above `0.95` so loud game explosions + boosted voice never harshly clip.
- **Zero-Resample Opus (plus AAC & Lossless FLAC)**: Feeds `AV_SAMPLE_FMT_FLT` directly to `libopus` (192 kbps default) with zero format conversion or resampling, and supports AAC and 32-bit lossless FLAC.

---

## 4. Customisation & Least-Privilege Security

- **Zero-Root Runtime**: Unlike tools that require running the entire recorder/FFmpeg/UI as `root` or `sudo`, `vkcap-kms` uses only `setcap cap_sys_admin=p` with `PR_SET_NO_NEW_PRIVS` and drops Effective capabilities to `0` everywhere except a ~5 µs DRM ioctl window. Output videos and config files are always owned by your normal user.
- **Three Ergonomic Layers of Control**:
  1. **Saved Personal Defaults (`~/.config/vkcap-kms/defaults.conf`)**: One-keypress instant start on repeat runs without overwriting built-in profiles.
  2. **7 Built-In RDNA 4 Profiles (`-p`)**: Curated presets (`default`, `gameplay`, `gameplay-10bit`, `master`, `compact`, `desktop`, `low-latency`) where any explicit CLI flag cleanly overrides the profile.
  3. **Quick & Advanced Interactive Wizard + Full CLI**: Accessible terminal UI exposing every video, color, audio, volume, and encoder knob—and printing the exact equivalent CLI command every time.
