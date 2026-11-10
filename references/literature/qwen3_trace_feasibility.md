# Qwen3-1.7B decode trace (bundled with HBF-Sim): is it a good LLM workload for C1-C4?

Scope: the trace bundled in `accel_sim_hbf/artifact/inputs/qwen3`, assessed as the LLM workload for
comparing C1 (HBM only), C2 (HBF + DRAM, HBF-Sim's native setup), C3 (HBF and HBM both on the GPU,
host-decided split) and C4 (HBF behind HBM, H3-style). Nothing inside `accel_sim_hbf/` or
`accelsim-work/` was modified. All work is in the scratchpad:

| What | Path |
|---|---|
| Unpacked trace (copy) | `/tmp/claude-1000/-home-lakshay-btp-work/82ba44e8-6a43-4f90-9b90-71b2a205d670/scratchpad/code-qwen3-trace/replay/` |
| Simulation runs | `/tmp/claude-1000/-home-lakshay-btp-work/82ba44e8-6a43-4f90-9b90-71b2a205d670/scratchpad/code-qwen3-trace/runs/<name>/` (`run.log`, `time.txt`, `gpgpusim.config`) |
| Re-placed trace variant (layer 13, MLP-only in HBF) | `/tmp/claude-1000/-home-lakshay-btp-work/82ba44e8-6a43-4f90-9b90-71b2a205d670/scratchpad/code-qwen3-trace/variants/l13_mlp_in_hbf/` |
| My tools | `/tmp/claude-1000/-home-lakshay-btp-work/82ba44e8-6a43-4f90-9b90-71b2a205d670/scratchpad/qwen3-tools/` (`headers.py`, `layers.py`, `addrscan.py`, `run_subset.sh`, `replace_weights.py`, outputs `headers.json`, `layers.out`, `layer13_addr.json`) |

Legend: **[ran]** = I ran it and measured, **[read]** = read from code or files, **[inferred]** = my reasoning or extrapolation, not measured.

---

## 0. Answer in brief

- **Use one decoder layer of the bundled Qwen3-1.7B decode step as the comparison unit.** It needs no GPU. It
  is a real pretrained LLM at batch 1, and decode time is dominated by streaming weights
  (99.4% of the unique bytes a layer touches are weights). The **28 layers are identical** kernel by kernel
  (same names, grids, blocks, registers, HBF-access counts and trace sizes) [ran]. So one layer, times 28,
  plus the lm_head, is the full model. The HBF-Sim paper's full-model number (352,434,530 cycles, Table 5)
  gives a published cross-check for C2. My extrapolation from measured pieces is about 362 M (section 4.4) [inferred].
- **Representative layer: layer 13 = kernel positions 850-913 in `kernelslist.g` (64 kernels, kernel IDs
  4659-4722), 100,672,000 B of weights.**
- **Cost on this laptop (8-SM `qwen3.config`):** one full layer with all memory in DRAM takes **4 min 06 s of wall time,
  181 MB RSS, 1,218,179 cycles** [ran]. With weights in HBF it is about 10.7 M cycles and **about 25 min** [inferred
  from measured q_proj, 2 min 03 s, and gate_proj, 5 min 54 s]. The full model is about 2.3 h (DRAM) and 13-18 h (HBF;
  the paper reports about 18 h) [inferred / read].
- **Limits:** batch 1, a 29-token context, eager attention, KV cache in a Hugging Face `DynamicCache` that is reallocated
  with `torch.cat` every step. KV is only 3.3 MB for the whole model (0.1% of the traffic), so **this trace can only answer weight-placement
  questions.** KV-in-HBF, larger-batch or long-context studies need a new capture on a rented GPU (section 6).
- **Alternative placements are possible from the same trace.** The bundled inner `manifest.json` stores, for
  every one of the 310 parameter tensors, both the original GPU address and the HBF address, so the remap can be inverted.
  I wrote and tested `replace_weights.py`, which puts any regex-selected set of tensors in the HBF window and moves the rest back to their
  original addresses (section 3).
- **Watch-outs for fair numbers:** (1) every kernel carries a 5,000-cycle launch latency (26% of a DRAM layer).
  (2) The HBF path bypasses L2 but the DRAM path does not. (3) `qwen3.config` is an A100 shrunk to 8 SMs and 8 memory channels.
  (4) The DRAM address layout alone moves cycles by about 2%.

---

## 1. How the trace is packaged and unpacked

| Item | Value | Source |
|---|---|---|
| Archive | `replay.tar.part-00/01/02` (33,554,432 + 33,554,432 + 19,122,176 B = 86.2 MB), "concatenated POSIX tar, 32 MiB parts" | `artifact/inputs/qwen3/manifest.json` [read] |
| Members | 1,819 `kernel-<id>-ctx_0x1cdb3040.traceg.xz` + `kernelslist.g` + an inner `manifest.json` | unpacked [ran] |
| Compression | each kernel trace is a separate **xz** stream of a text `.traceg`; accel-sim.out decompresses on the fly with `popen("xz -dc …")` | `gpu-simulator/trace-parser/trace_parser.cc:432-443` [read] |
| Size | **84.1 MB compressed, 30.41 GB uncompressed** (`xz -l`) | [ran] |
| Trace format | NVBit 1.7.6, Accel-Sim tracer version 5, `-binary version = 86` (sm_86 SASS, captured on an RTX 4090) | kernel headers [ran] |
| Unpack | `make prepare-qwen` → `scripts/unpack_qwen.py`: checks each part's size and SHA-256, concatenates the parts into a temp file, streams the tar, rejects unexpected members, checks every member's SHA-256 against `manifest.json['files']`, extracts to `artifact_runs/inputs/qwen3/` (or `--output`), refuses an existing destination | `scripts/unpack_qwen.py:11-31` [read] |
| My copy | `python3 scripts/unpack_qwen.py --output <scratch>/code-qwen3-trace/replay`: "Verified 1819 kernels", 0.4 s, 86 MB on disk (the simulator never needs the 30 GB decompressed form) | [ran] |

Capture metadata [read]: `capture.json` records Qwen/Qwen3-1.7B rev `70d244cc…`, BF16, eager attention, batch 1, a
27-token prompt and `capture_step 2`. `step.json` records a single input token 41776, `context_tokens 29`, kind `decode`. Under NVBit the captured step
took 1,498 s on the 4090 (`native_step_seconds[2]`); native decode steps take about 23 ms (`native.json`).
Runner: `SUITE=qwen make reproduce-paper` → `experiments/reproduce.py:38-43` → `experiments/23_qwen3_1p7b/run_smoke.py`,
which copies `artifact/configs/qwen3.config`, sets `-gpgpu_hbf_trace_file`, and runs
`accel-sim.out -config <cfg> -trace <dir>/kernelslist.g` through `experiments/common/run_sim.sh` (timeout 172,800 s),
then `validate_replay.py`.

## 2. Kernel inventory

### 2.1 Model dimensions (verified from tensor shapes in the inner `manifest.json`) [ran]

`model_metadata.json` is the Hugging Face API record. It has no `config.json` fields, but it gives 2,031,739,904 BF16 parameters. The recorded
tensor shapes confirm: hidden 2048 (`q_proj [2048,2048]`, 16 heads × 128), 8 KV heads (`k_proj [1024,2048]`),
head_dim 128 (`q_norm [128]`), intermediate 6144 (`gate/up [6144,2048]`, `down [2048,6144]`), 28 layers, vocab
151,936 (`embed_tokens [151936,2048]`). There are 310 parameter tensors and no separate `lm_head.weight`: the embeddings are tied, and the lm_head
kernel reads the embedding matrix.

| Weights per decoder layer | Bytes |
|---|---|
| q_proj, o_proj | 8,388,608 each |
| k_proj, v_proj | 4,194,304 each |
| gate_proj, up_proj, down_proj | 25,165,824 each |
| input_layernorm, post_attention_layernorm | 4,096 each |
| q_norm, k_norm | 256 each |
| **Layer total (identical for all 28)** | **100,672,000** |
| embed_tokens (= lm_head) | 622,329,856 |
| final norm | 4,096 |
| **All tensors** | **3,441,149,952** (packed HBF span with 4 KiB alignment: 3,441,364,992) |

### 2.2 Kernel sequence (1,819 = 18 prologue + 28 × 64 + 9 tail) [ran]

Kernel types across all 1,819: mul 396, copy/convert 341, other elementwise 258, cuBLAS `gemvx` 225, `cat` 113, pow 113,
mean-reduce 113, rsqrt 113, neg 56, `gemv` 29 (`gemvNSP` 28 + `gemmk1` 1), softmax 28, silu 28, arange 4, embedding
gather 1, fill 1 (full demangled list in `qwen3-tools/layers.out`).

| Positions | Content | Weight (HBF) access |
|---|---|---|
| 0 | embedding gather (`indexSelectSmallIndex`) | 4 KB (one row) |
| 1-17 | position ids / causal mask (arange, add, compare, fill, where), RoPE cos/sin (`gemmk1`, cat, cos, sin, mul, copy) | none |
| 18-1809 | 28 decoder layers × 64 kernels (layer L = positions 18+64L … 81+64L) | 100,672,000 B each |
| 1810-1817 | final RMSNorm (8 kernels) | 4 KB |
| 1818 | lm_head `gemvx` grid 37,984 | 622,329,856 B; **5.39 GB of the 30.4 GB raw trace** |

**Layer 13 = positions 850-913, kernel IDs 4659-4722** (files `kernel-4659-…` to `kernel-4722-…`). The role of each
relative position is inferred from the demangled names and the operation order in HF Qwen3 [inferred]:

| Rel. pos | Kernels | Role | Grid | DRAM-only cycles [ran] |
|---|---|---|---|---|
| 0-7 | copy, pow, mean, add-eps, rsqrt, mul, copy, **mul(weight)** | input RMSNorm | 1-8 CTAs | ~5.7-9.5 k each |
| 8 | gemvx | **q_proj** | 512 | 72,939 |
| 9-16 | 8 kernels | q_norm (RMSNorm over head_dim) | | |
| 17 | gemvx | **k_proj** | 256 | 41,732 |
| 18-25 | 8 kernels | k_norm | | |
| 26 | gemvx | **v_proj** | 256 | 41,183 |
| 27-36 | mul, neg, cat, mul, add ×2 | RoPE on q and k | | |
| 37-38 | cat [56,2,1] ×2 | **KV-cache append** (`torch.cat`, 28 → 29 tokens, 59,392 B each) | 112 | ~9.5 k |
| 39-40 | copy [116] ×2 | repeat_kv (8 → 16 heads) | 116 | ~9.8 k |
| 41 | gemvx [8,1,16] | Q·Kᵀ (16 heads) | 128 | 14,607 |
| 42-46 | mul, add, copy, softmax, copy | scale, mask, fp32 softmax | | |
| 47 | gemvNSP [16,1,16] | attn·V | 256 | 18,028 |
| 48 | gemvx | **o_proj** | 512 | 73,306 |
| 49 | add | residual | | |
| 50-57 | 8 kernels | post-attention RMSNorm | | |
| 58 / 60 | gemvx | **gate_proj / up_proj** | 1536 | 194,120 / 195,507 |
| 59, 61 | silu, mul | SwiGLU | | |
| 62 | gemvx | **down_proj** | 512 | 197,838 |
| 63 | add | residual | | |

**Are the layers identical?** Yes. I checked every layer against layer 13 on (name, grid, block, shmem, nregs, HBF-mapped lane
accesses, uncompressed trace size). All 28 match exactly, each layer is 893,402,491 B of raw trace (2.52 MB xz), and
each has 50,335,232 HBF-mapped lane accesses [ran, `layers.py`]. Only the addresses differ.

### 2.3 Which kernels touch HBF vs DRAM (address scan of all of layer 13) [ran, `addrscan.py`, 20 s]

Only 11 of the 64 kernels touch the HBF window: the 7 weight GEMVs plus the 4 RMSNorm weight-multiply kernels (rel. 7, 16, 25, 57).
Everything else (activations, KV cache, softmax buffers, reduction scratch) uses ordinary addresses `0x7f0d…/0x7f0e…`.

| Layer 13 | Lane bytes | Unique bytes (32 B sectors) | Warp instructions |
|---|---|---|---|
| HBF loads (weights) | 100,677,632 | **100,672,000** (each weight byte once) | 1,572,976 |
| DRAM loads | 101,773,776 | 755,488 | 1,587,267 |
| DRAM stores | 584,248 | 584,416 | 17,470 |
| HBF stores | 0 | 0 | 0 |

Findings:
- **The weights are read once per step, and HBF is read-only.** There are no stores to HBF, so the strict-write problems described in `HBF_SIM_REPORT.md` §5.2 do not arise.
- **Every weight load in the cuBLAS `gemvx` kernel is paired with a load of the activation vector x**
  (`LDG.E.U16.STRONG.SM`). The DRAM-side lane bytes (101.8 MB) are therefore as large as the weight bytes, but only 4-12 KB are unique.
  With `-gpgpu_gmem_skip_L1D 1` these all become L2 hits. Any HBM/L2 model in C1-C4 carries this L2 load.
- Weight access pattern: each warp instruction reads 16 contiguous BF16 values from **two rows**, and a q/k/v/o/gate/up row
  is exactly 4,096 B, i.e. **one HBF page**. Each page is therefore assembled from many warps over time. That is why MSHR coalescing and the NAND page
  buffers matter (section 4).
- KV cache per layer: K and V are 8 heads × 29 tokens × 128 × 2 B = 59,392 B each. Whole model: 28 × 118,784 ≈ 3.3 MB.

## 3. Address mapping and alternative placements

[read] `experiments/23_qwen3_1p7b/prepare_probe.py`: `BASE = 274877906944` (256 GiB, `0x4000000000`). Parameter
tensors from `weights.json` (recorded by `run_model.py:19-21` via `p.data_ptr()`) are sorted by original address and
packed upward from BASE, each 4 KiB aligned (`prepare_probe.py:15-17`). Every lane address that falls inside a parameter
tensor is rewritten to `target + offset`. Only `LDG`/`LD.` may touch parameters, otherwise the script raises an error (`:42`). Other addresses are untouched.
`qwen3.config` sets `-gpgpu_hbf_base_addr 274877906944 -gpgpu_hbf_size 549755813888` (the window is 256-768 GiB).

| | Range |
|---|---|
| Original parameter addresses | `0x7f0d3c000000` … `0x7f0f20000000` (8.12 GB span, interleaved with other allocations; e.g. layer 13's q_proj is at `0x7f0dc0c00000`, away from its other tensors) |
| HBF targets | `0x4000000000` … `0x40cd1f1000` (3,441,364,992 B) |
| Layer 13 targets | `0x4053c00000` … `0x405f400000` for the matrices; the norm vectors sit together at `0x40a6c34000-0x40a6c38000` (one 4 KiB page each) |
| KV / activations | stay at their original `0x7f0d…`/`0x7f0e…` addresses on the DRAM path |

**The original, un-remapped trace is not bundled.** Only output files are shipped; the inner manifest stores `input_sha256` of the
original compressed files, which cannot be reproduced byte for byte. **The remap is invertible**, though, because the inner
`manifest.json['weights']` keeps `address`, `target`, `bytes` and `name` for all 310 tensors. `util/traces/remap_global_addresses.py`
is not the right tool: it shifts *all* global addresses by base/span (`--base --span --source-base`, lines 237-253) and
cannot select tensors.

New tool, `qwen3-tools/replace_weights.py` [ran]: `--hbf REGEX` packs the matching tensors into the HBF window from `--base`
and moves every other tensor back to its original address. `--hbf '^$'` restores the original all-GPU-memory
addresses. It keeps the base+stride/delta address compression when the shift is uniform (the same logic as
`prepare_probe.py`). `--first/--last` select kernel positions.
- Test: layer 13 with `--hbf 'mlp\.'` ran in 64 s and produced 2,113,929,216 HBF-resident bytes (= 28 × 3 × 25,165,824). A re-scan shows q_proj and o_proj
  reading `0x7f0dc0c00000-0x7f0dc1400000` and `…-0x7f0db9000000` (their original addresses) while gate, up and down read the HBF window.
- Simulated (HBF on) on kernels 0-8 of the variant: **0 HBF requests**, q_proj 71,494 cycles, the norm-weight kernel back to 5,863 cycles.

Placements possible from this trace:

| Placement | How |
|---|---|
| Everything in HBM/DRAM (C1) | `-gpgpu_hbf_enabled 0` on the bundled trace (weights keep their 256 GiB addresses, `is_hbf_addr()` returns false, `gpu-sim.h:517-519`), or `replace_weights.py --hbf '^$'` for the original addresses. The two differ by **2.0%** on q_proj (72,939 vs 71,494 cycles) because of DRAM row and bank mapping, so pick one and use it in every configuration |
| All weights in HBF, rest in DRAM (C2, the paper's setup) | the bundled trace as is |
| Subsets of weights in HBF (C3/C4 policies: MLP only, big matrices only with norm vectors in HBM, per-layer split, embedding/lm_head in HBM, …) | `replace_weights.py --hbf <regex>` |
| KV cache in HBF | **Not cleanly possible.** KV tensors are not in `weights.json`, the cache is reallocated by `torch.cat` every step (rel. 37-38 write a new buffer), and at a 29-token context it is 0.1% of the bytes. It needs a new capture (section 6) |

Note: the norm vectors are tiny, but in HBF each costs a full tR. The weight-multiply kernel went from 5,864 to 26,792 cycles
(+20.9 k ≈ 15 µs × 1.41 GHz) [ran], i.e. about 84 k cycles per layer for 16 KB of data. A C3/C4 placement policy should keep
small tensors in HBM.

## 4. Simulation cost and first results

### 4.1 Setup

HBF-Sim binary `accel_sim_hbf/gpu-simulator/bin/release/accel-sim.out`, `artifact/configs/qwen3.config` with
`-gpgpu_hbf_trace_file ""` (no lifecycle CSV, unlike the paper run), and the same interconnect file. A subset
`kernelslist.g` holds symlinks to the trace files. Runner: `qwen3-tools/run_subset.sh <name> <first> <last> <hbf 0|1> [-opt val …]`,
inside the Apptainer image under `nice -n 10` and `timeout 600`, with `/usr/bin/time -v` on the host. One simulator ran at a time.

What `qwen3.config` is [read, diff against `configs/tested-cfgs/SM80_A100/gpgpusim.config`]: the stock A100 config with
`n_clusters 108→8`, `n_mem 40→8`, `gmem_skip_L1D 0→1`, `max_concurrent_kernel 128→1`, plus trace opcode latencies.
That makes it an 8-SM GPU with 8 × 48.4 GB/s = **387 GB/s** of DRAM. HBF: 1 stack, 16 channels, 512 sub-arrays, 512 active slots,
tR 22,680 ticks = 15 µs, so the array ceiling is 512 × 4 KiB / 15 µs = **139.8 GB/s**. L2 is bypassed for HBF (`l2_policy 0`).

### 4.2 Runs [ran]

| Run | Kernels (positions) | Memory | Cycles (`gpu_sim_cycle` per kernel) | Wall | Peak RSS |
|---|---|---|---|---|---|
| l13_qproj_dram | 850-858 (input norm + q_proj) | HBF off | q_proj **72,939**; total 124,428 | 0:16.9 | 130 MB |
| l13_qproj_dram_nolaunch | same, `-gpgpu_kernel_launch_latency 0` | HBF off | q_proj 67,050; small kernels 668-4,528; total 78,396 | 0:16.5 | 130 MB |
| qproj_alone_dram | 858 only (cold caches) | HBF off | 72,989 (+0.07% vs in sequence) | 0:16.8 | — |
| l13_qproj_hbf | 850-858 | weights in HBF | q_proj **860,480** (11.8×); norm-weight mul 26,792; total 932,897 | **2:03** | 154 MB |
| l13_gate_hbf | 908 only (gate_proj, 25 MiB) | weights in HBF | **2,536,674** (13.1× the DRAM 194,120) | **5:54** | 152 MB |
| **l13_full_dram** | **850-913 (whole layer)** | HBF off | **1,218,179** (= 864 µs at 1.41 GHz) | **4:06** | 177 MB |
| var_mlphbf_qproj | variant, q_proj back in DRAM | HBF on | q_proj 71,494, 0 HBF requests | 0:18 | 131 MB |
| qproj_80sm_dram | 858, `-gpgpu_n_clusters 80` | HBF off | 134,849 (slower than 8 SMs; row-buffer locality 0.95 → 0.62-0.80, mrq latency 54 → 179) | 4:14 | 703 MB |
| qproj_80sm_hbf | 858, 80 SMs | weights in HBF | **did not finish in 600 s** (480 of 512 CTAs issued) | >10:00 | 695 MB |
| newsim_l13_norm_builtin | 850-857 on **accelsim-work's newer** `bin/release/accel-sim.out`, `qwen3.config` minus HBF options | built-in DRAM | 9,512 / 5,667 / 6,720 / … (within 0.2% of HBF-Sim's DRAM run) | 0:01 | 94 MB |

HBF counters for q_proj [ran]: 262,272 requests, MSHR coalescing 73.8%, 68,588 page operations of which 66,539 (97.0%) were
NAND page-buffer hits, so 2,049 array reads (= 2,048 rows + 1 norm page). Read amplification 33.5×, max queue depth 7.
gate_proj: 786,432 requests, 74.1% coalescing, 203,612 page operations (97.0% buffer hits).

### 4.3 What the numbers say

- **Effective bandwidth:** gate_proj from DRAM gets 25.17 MB / ((194,120 − 5,000) / 1.41 GHz) ≈ 188 GB/s (49% of 387 GB/s). From HBF it gets
  25.17 MB / 1.80 ms ≈ **14.0 GB/s, 10% of the 139.8 GB/s array ceiling**. The paper's full model is 3.44 GB / 249.95 ms = 13.8 GB/s,
  which agrees. HBF is held back because too few distinct pages are in flight (8 SMs, one kernel at a time, each row/page assembled by many warps), not by media bandwidth
  [inferred; the 80-SM HBF test that would confirm this did not finish within the 10-minute cap].
- **Launch latency:** every kernel's `gpu_sim_cycle` includes `-gpgpu_kernel_launch_latency 5000` (compare the two q_proj runs above).
  Per layer that is 64 × 5,000 = 320,000 of 1,218,179 DRAM cycles (26%). It dilutes memory differences between C1-C4, so report it
  separately, or set it to 0 consistently.
- **Kernel isolation is safe for the GEMVs.** q_proj alone (cold) vs in sequence differs by 0.07%, so GEMV kernels can be replayed one at a time in parallel jobs when RAM allows (about 150 MB each).
- **More SMs is not free.** With 80 SMs on the same 8-channel memory, DRAM q_proj got slower (row conflicts), and the simulator ran about 10× slower per
  simulated cycle (533 cycles/s vs about 4,900). Scaling the GPU up changes both the result and the cost.
- **The trace runs on our newer Accel-Sim (the Ramulator branch's simulator).** Its `trace_parser.cc:627` also pipes `.xz` through `xz -dc`,
  and the 8 small kernels I ran match HBF-Sim's cycles to within 0.2%. I did not run a gemvx kernel or Ramulator HBM3 on it [not verified].

### 4.4 Extrapolation [inferred]

| Unit | DRAM / HBM-only (8 SMs) | Weights in HBF (8 SMs) |
|---|---|---|
| One layer, cycles | 1.218 M (measured) | ≈ 10.7 M: q 0.86 M + k, v ≈ 0.43 M each + o ≈ 0.86 M + gate, up, down ≈ 2.54 M each + 4 × 21 k norm weights + 0.40 M other kernels |
| One layer, wall | 4 min (measured) | ≈ 25 min (≈ 7,200 cycles/s measured on the HBF runs) |
| Full model, cycles | ≈ 34.1 M + lm_head ≈ 4.8 M ≈ 39 M (≈ 28 ms) | ≈ 300 M + lm_head ≈ 63 M ≈ **362 M**, vs the paper's **352,434,530** (Table 5, §5.5), a 2.7% difference |
| Full model, wall | ≈ 2.3 h (lm_head trace ≈ 6× a layer) | ≈ 13.6 h at the measured rate; the paper/`ARTIFACT_EVALUATION.md` report about 18 h with the CSV lifecycle trace on |
| Peak RSS | 130-180 MB (8 SMs), about 700 MB (80 SMs) | the same |

So on this config HBF-resident weights make decode about 9× slower than DRAM-resident weights. This is a property of the HBF parameters plus the GPU
config, not a general result.

## 5. Recommendation

**Yes: one Qwen3-1.7B decoder layer (positions 850-913) is the right unit for comparing C1-C4 on this laptop.**
1. It is a real model at a real decode step, and weight streaming dominates it (100.67 MB of weights vs 1.34 MB of other unique bytes). That is the case HBF is proposed for: large, read-only weights.
2. All 28 layers are identical, so 28 × layer + lm_head + prologue gives the full-model number, and C2 can be checked
   against the published 352.4 M cycles / 249.95 ms.
3. It fits the laptop: a DRAM/HBM layer takes about 4 min and an HBF layer about 25 min (or split it per GEMV kernel and run 2-4 jobs at a time; each needs about 150 MB).
4. One trace file serves all four configurations. Placements come from `replace_weights.py`, and both simulators read the format.

For accurate numbers, keep the following fixed across C1-C4:
- the same GPU config (start with `qwen3.config` for comparability with the paper; any larger config should be a separately labelled study);
- the same address layout (decide between the bundled 256 GiB addresses and the original ones, which differ by 2% in DRAM timing);
- the same launch-latency treatment, reported both with and without the 5,000 cycles;
- an explicit L2 policy for HBF requests (the paper uses bypass; C4's "behind HBM" path should state whether HBF data is cached in L2 or in an HBM-side buffer).

Metrics available per layer: cycles and µs per layer (and ×28 for the model), effective weight bandwidth (GB/s), cycles per GEMV kernel
(q/k/v/o/gate/up/down), HBF counters (requests, MSHR coalescing, page operations, buffer hits, read amplification, queue depth,
channel utilisation; the utilisation figure over-reports, e.g. 105%, see `HBF_SIM_REPORT.md` §7 #2), DRAM row-buffer locality and bandwidth utilisation, and
tokens/s = 1 / (step time) at batch 1.

## 6. Limits of this trace and what a new capture would need

| Limit of the bundled trace | Consequence |
|---|---|
| batch 1 | cuBLAS `gemvx`/`gemvNSP` kernels; batch > 1 switches to GEMM (tensor-core) kernels with weight reuse, i.e. a different access pattern |
| context 29 (27 prompt + 2) | KV traffic is about 0.1%; attention is negligible |
| eager attention, HF `DynamicCache` with `torch.cat` | the KV cache is copied into a new buffer each step; this is unlike vLLM's paged KV |
| one decode step, cold simulated caches | fine for weight streaming (no reuse across steps at batch 1 anyway) |
| no KV/activation address records | KV cannot be placed selectively |

A new capture on a rented GPU (an RTX 4090 keeps the same SASS and software as the paper: NVIDIA PyTorch 25.06, Transformers 4.57.6) should use
`experiments/23_qwen3_1p7b/run_model.py`, which already supports `--probe-layer L --capture-step S`. That traces only
one decoder layer between `cudaProfilerStart/Stop` (lines 24-31, with the tracer's `ACTIVE_FROM_START=0`), which is far cheaper than
the 1,498 s full-step capture. Suggested changes:
1. batch B: a list of prompts, padded;
2. a long context: a long prompt or a later `--capture-step`, e.g. a 1 K-4 K context. KV per token per layer is 2 × 8 × 128 × 2 B = 4 KiB, so at 4 K tokens it is 16 MiB per layer (0.46 GB for the model per sequence);
3. a `StaticCache` (`cache_implementation="static"`), so KV buffers are preallocated at fixed addresses and appended in place;
4. a dump of the K/V buffer addresses (`data_ptr`, bytes, layer) next to `weights.json`, so `replace_weights.py` can be extended to place KV;
5. keep the same post-processing (`post-traces-processing`, `prepare_probe.py`), or skip the remap and use `replace_weights.py` later.

## 7. Comparison with our existing OPT-125m layer trace

| | Qwen3-1.7B bundled | OPT-125m (`accelsim-work/llm/traces/t4_opt-125m__model.decoder.layers.10.self_attn`) |
|---|---|---|
| Capture | RTX 4090, HF Transformers eager, one full decode step | T4 (sm_75, `gpu.txt`), vLLM eager (`llm/src/trace_layer.py`), the **self_attn module of layer 10 only** over prefill and every decode call (`max_new_tokens 32`) |
| Kernels | 1,819 (64 per layer) | 159 |
| Format | `.traceg.xz`; runs on HBF-Sim **and** the newer Accel-Sim [ran] | `.tracez` (237 MB); **HBF-Sim's pinned parser has no `.tracez` support** (`grep tracez` finds nothing in `accel_sim_hbf/gpu-simulator/trace-parser/`), only the newer Accel-Sim reads it |
| Weight addresses recorded | yes, per tensor (enables any placement) | no; HBF placement would need guessing |
| Weights per call | 100.7 MB per layer | about 4.7 MB (qkv + out_proj at hidden 768) [inferred from OPT-125m dimensions]; small enough to sit in a large GPU's L2, so memory placement would be hidden |
| Existing results | the paper's Table 5 | our Ramulator validation (`RAMULATOR_GUIDE.md` §4-5: whole layer 1.91 M cycles, about 130 GB/s, not bandwidth-bound) |

**Verdict:** the OPT-125m trace stays useful for validating the Ramulator HBM path (C1). It is a poor HBF workload (no MLP, small
weights, no address map, a format HBF-Sim cannot read). Use the Qwen3 layer as the main workload.

## 8. Open points

- HBF throughput with more SMs, which is needed to know whether the HBF slowdown is set by the GPU or by the media. The 80-SM single-kernel run needs more than 10 min; run it unattended.
- C1 on Ramulator HBM3 with this trace: format compatibility is verified, but no gemvx run has been done on the Ramulator build yet.
- The down_proj HBF cycles are assumed equal to gate_proj; down_proj has 12 KiB rows (3 pages per row), so its pattern differs and it should be measured.
