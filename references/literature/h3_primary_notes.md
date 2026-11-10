# H3 (Ha, Kim, Kim, IEEE CAL 25(1):49-52, 2026): notes from the primary text

Source: the IEEE Xplore PDF the user downloaded through IIT Guwahati (2026-10-29). It replaces the
second-hand H3 facts in `h3_and_device_params.md`.

## Architecture (Sec. III-A, Fig. 3)
- HBM stacks sit on the GPU shoreline. Each HBF stack is **daisy-chained behind one HBM stack**. The paper
  explicitly rejects placing HBM and HBF side by side on the shoreline, because "the number of HBMs" would
  shrink. That is our C3-iso-shoreline.
- The **HBM base die** contains an **address decoder & router**, which sends each access to the HBM
  controller (on the HBM base die) or over a D2D link to the HBF base die (HBF controller). The GPU sees one
  **unified address space with divided regions**. Fig. 3(c): HBM #0-#7 = 192 GB at the bottom, then
  HBF #0-#7 at 384 GB each.
- **Link bandwidths (Fig. 3(b)):** BW① (GPU ↔ HBM base, D2D) = BW② (HBM base ↔ HBM core) = BW③ (HBM base ↔ HBF base, D2D) = BW④ (HBF base ↔ HBF core).
  The GPU-facing link of a stack therefore has the bandwidth of one core, and **HBM and HBF traffic share it**.
- The **latency-hiding buffer (LHB)** is a prefetch buffer on the HBM base die. The deep-learning framework
  gives tensor-level prefetch hints, possible because LLM access patterns are deterministic and sequential.
  It is double-buffered: **Capacity = 2 × BW_HBF × Latency_HBF = 2 × 1 TB/s × 20 µs = 40 MB per cube**,
  which takes 8.06 mm² at 3 nm, 6.7% of a 121 mm² base die.
- **Hop latency** "may be added" but is "sufficiently hidden" (no number given). Contention between HBM
  and HBF is "manageable" through tensor-level scheduling.

## Data placement (Sec. III-B, Fig. 4)
- **HBF:** model weights + the **shared pre-computed KV cache** of cache-augmented generation (CAG). Both are read-only.
- **HBM:** generated KV cache + other data.

## Evaluation (Sec. IV)
- **Method:** in-house **analytical** simulator: compute time from operation counts, transfer time from
  bytes / bandwidth, ring all-reduce for tensor parallelism. It is not cycle-level.
- **Workload:** Llama 3.1 405B, FP8 weights (405 GB), context extended to 1M / 10M, ISL = OSL = 1K. The rest
  of the context is shared pre-computed KV, taking about 35% (1M) and 84% (10M) of total capacity.
- **System:** B200 with HBM3e 192 GB, 8 TB/s (24 GB, 1 TB/s per cube). HBF: 3 TB, 8 TB/s (same as HBM).
  TDP 40 W per HBM cube, 160 W per HBF cube, 680 W GPU. DGX-style nodes: 8 GPUs at 1M, 32 GPUs at 10M, the
  minimum for HBM-only.
- **Results:**
  - Max batch: up to **2.6×** (1M) and **18.8×** (10M). One GPU with H3 runs 1M; two GPUs run 10M.
  - Throughput (TPS per request): **1.25×** (1M), **6.14×** (10M).
  - Throughput per power: up to **2.69×**.
  - With HBF at half bandwidth: still better at 1M, and **2.09×** throughput/power at 10M.

## How our C4 maps onto Fig. 3(b)
| H3 | Our C4 (8-channel slice = half an HBM3 stack) | Match |
|---|---|---|
| BW① GPU ↔ HBM base | per-partition PHY token bucket, 8 × 51.2 = 409.6 GB/s, shared by HBM and HBF bytes | ✓ |
| BW② HBM core | Ramulator HBM3, 8 ch, 409.6 GB/s | ✓ (BW① = BW②) |
| BW③ HBM base ↔ HBF base | HBF channel link credits, 8 × 51.2 = 409.6 GB/s | ✓ (BW③ = BW②) |
| BW④ HBF core | NAND media, max_active × 4 KiB / tR = 409.6 GB/s by construction | ✓ |
| Address decoder in HBM base die, unified address space | HBF window [256 GiB, 768 GiB) decoded after the L2; HBF requests bypass the HBM controller | ✓ |
| LHB on HBM base die, 2·BW·tR, tensor-level hints | LHB with per-kernel page hints (first-touch order), lookahead 1 kernel; fixed 3,000 pages = 2 × 409.6 GB/s × 15 µs | ≈. At H3's tR = 20 µs its rule gives 4,000 pages for our slice; our tR-20 runs used 3,000 (LHB hit rate 99.3%) |
| Hop latency, unspecified | +10 ns, swept to +200 ns (effect ≤ 0.13%) | ✓ consistent with "sufficiently hidden" |
| Weights + **shared pre-computed KV** in HBF | weights only (the bundled Qwen3 trace has no CAG KV) | ✗ The H3 use case needs a long-context trace with read-only KV |
| HBF 384 GB per cube | projection uses 512 GiB per stack (OCP) | minor |
