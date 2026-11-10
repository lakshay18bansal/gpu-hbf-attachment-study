# Workloads

## qwen3_1p7b/ (main LLM workload)
- **Source.** HBF-Sim's bundled trace (`accel_sim_hbf/artifact/inputs/qwen3`, unpacked and hash-verified
  with its `scripts/unpack_qwen.py`).
  - Qwen3-1.7B (revision 70d244cc), BF16, HF Transformers with eager attention, batch 1.
  - It is the **second decode step after prefill**, with a 29-token context.
  - All 1,819 kernels; tracer v5, SASS sm_86 (captured on an RTX 4090; replayed on the SM80 slice).
- **Placement** (done by HBF-Sim's `prepare_probe.py`). All 310 parameter tensors (3.44 GB) are packed
  4 KiB-aligned from 256 GiB = 0x4000000000, i.e. in the HBF window. KV cache, activations and scratch
  keep their original addresses. `trace/manifest.json` lists every tensor's original and new address.
  `tools/replace_weights.py` and `tools/place_tensors.py` can move any subset.
- **Kernel lists** (`trace/list_*.g`):

  | List | Kernels | Content |
  |---|---|---|
  | `list_prologue.g` | 18 | embedding gather, rotary tables, masks (tiny) |
  | `list_layer13.g` | 64 | decoder layer 13 (kernel ids 4659-4722). All 28 layers have the identical kernel sequence, shapes and HBF access counts |
  | `list_tail.g` | 9 | final RMSNorm + **lm_head GEMV** (kernel 5627: 151936 × 2048 BF16 = 622 MB, the tied embedding, about 18% of the weight bytes) |
  | `list_full.g` | 1819 | the whole token |
  | `list_k<id>.g` | 1 | created on demand by `qwen3_kernel:<id>` (e.g. 4667 = layer-13 q_proj GEMV, 8 MiB) |

  One token = prologue + 28 × layer 13 + tail. Layer 13 streams 100.67 MB of weights (q 8, k 4, v 4, o 8,
  gate 24, up 24, down 24 MiB + norms) and touches about 1.3 MB of other data.
- **Prefetch hints** are cached in `trace/hints/<list>.hints` (written by `tools/hbf_prefetch_hints.py`
  on the first `--prefetch on` run).
- **Limits.** Batch 1 and context 29 means pure weight streaming. Batch and KV effects need new
  captures (`scripts/capture/`).

## vectoradd/ (test workload)
- `original/`: T4 (sm_75) trace from `accelsim-work/cuda/traces/t4_vectoradd`, in binary `.tracez`.
  C = A + B with 1M floats each (4 MiB per array).
- `work/`: text trace (`tools/bin/traceDsm`).
- `hbf/`: A and B moved into the HBF window by `tools/remap_ranges.py --ranges-from-memcpy`.
  - A goes to 0x4000000000 and B to 0x4000400000 (1,048,576 lane addresses each), including the
    `MemcpyHtoD` lines. With `-gpgpu_perf_sim_memcpy 1` those copies warm the L2, so part of A and B
    hits in the L2 and the HBF requested bytes are less than 8 MiB.
  - C stays in HBM/DRAM.
  - The same file is used by every configuration; C1 simply has no HBF tier.
