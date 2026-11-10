# C3: HBF and HBM both directly attached to the GPU

```
   16 SMs ── NoC ── L2 (8 MB)
                     │ GPU memory controller decodes the HBF window (host/driver placed the pages)
          ┌──────────┴───────────┐
          ▼                      ▼
   HBM3 PHY (own channels)    HBF PHY (own channels)
   [ HBM3 stack ]             [ HBF stack ]
   KV cache, activations      weights (read-only)
```

**What it models.** HBM and HBF stacks side by side on the interposer, each with its own GPU-facing
PHY (FlashAccel "CLI", DASH "DirectOnly", HBF-Sucks "HBF-2"). The CPU/driver decides which pages go
to which tier (here: weights → HBF), and the GPU's memory controller routes by address range.

Because the result depends on what is held equal, there are two variants:

| | `iso_stack/` (equal stacks and links) | `iso_shoreline/` (equal GPU shoreline) |
|---|---|---|
| Idea | add an HBF stack with its own PHYs next to the HBM (DASH's equal-resource rule) | the 8 memory sites of the slice are split between HBM and HBF (FlashAccel / H3 accounting) |
| HBM | 8 ch, 6.4 Gb/s, 409.6 GB/s | 8 ch at 3.2 Gb/s = 204.8 GB/s (= 4 sites; ns timings unchanged) |
| HBF link | 8 x 51.2 = 409.6 GB/s | 4 x 51.2 = 204.8 GB/s |
| GPU-facing total | 819.2 GB/s | 409.6 GB/s (same as C1) |

Everything else equals C2 (`-gpgpu_hbf_attach 1`: own L2→HBF queues and credits, 52.49 ns front end,
same NAND model, same prefetcher and LHB, HBF data cached in L2). So C2-vs-C3 isolates the near tier
(GDDR6 vs HBM3); C3-vs-C4 isolates the attachment.

Modelling note: the half-shoreline HBM keeps 8 Ramulator channels at half the data rate rather than 4
channels, because the GPU slice has 8 memory partitions (`-gpgpu_n_mem 8`) and Ramulator must have one
controller per partition. Bandwidth is exactly halved and the ns latencies stay the same; bank
parallelism stays at 8 channels' worth, which slightly flatters iso_shoreline.

**Run.** `iso_stack/run.sh qwen3_layer13 --tR-us 15 --prefetch on`
