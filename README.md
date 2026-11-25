# vkcap — Zero-Copy Vulkan Video Screen Recorder
`vkcap` is a high-performance Linux screen recording CLI built for modern Linux desktops. `vkcap-kms` (recommended) utilises KMS DMA-BUFs, Vulkan Compute, and Vulkan Video to provide the highest-performance screen recording on Linux desktops (see rec_kms). `vkcap-pipewire` instead uses PipeWire DMA-BUFs to capture displayed frames (see rec_pipewire).

`vkcap` has been primarily tested on AMD GPUs, but the nature of its architecture allows it to function on non-AMD cards with leading performance.
