# Merge feasibility: one Accel-Sim binary with Ramulator HBM3 and HBF-Sim's HBF tier

Date: 2026-10-28. Scope: choose how to build one `accel-sim.out` that contains (a) our Ramulator 2.1
bridge (realistic HBM3) and (b) HBF-Sim's HBF tier, so that configs C1 to C4 run on the same binary.
Nothing in `accel_sim_hbf/` or `accelsim-work/` was modified. All prototypes and runs are in
`<scratch>/merge/`
(called `$M` below).

Labels: **[ran]** = I executed it and the numbers come from its output. **[read]** = I read it in the
source (file:line given). **[inferred]** = my reasoning, not checked by a run.

---

## 0. Verdict

**Use Option B: port HBF-Sim onto the newer Accel-Sim tree that already has the Ramulator bridge.**
I prototyped it in scratch. It compiled on the first attempt and passed every check I ran (section 6):

| Check (merged prototype binary) | Result | Reference | Verdict |
|---|---|---|---|
| vectorAdd T4 trace, RTX2060S, built-in DRAM, HBF off | 54,282 cycles, 258,016 DRAM reads | newer-tree binary: 54,282 / 258,016 | identical |
| vectorAdd, Ramulator `hbm3_rtx2060s`, HBF off | 47,452 cycles | RAMULATOR_GUIDE.md section 6: 47,452 | identical |
| vectorAdd H100 trace, SM90_H100 (2 chiplet partitions), HBF off | 8,947 cycles, 68,416 DRAM reads | newer-tree binary: 8,947 / 68,416 | identical |
| HBF-Sim LUD trace smoke (remapped kernel-8, trace-001 config) | 9,121 cycles; 158 requests, 2 page reads, 3 programs, 91 MSHR hits, 12,288 media bytes; lifecycle trace `valid: true` | HBF-Sim reference: 9,137 cycles; same HBF counters, same 322 trace rows | HBF counters identical; cycles -0.18% |
| Qwen3-1.7B kernels 3809+3810, `qwen3.config`, HBF on | 27,295 cycles total; all HBF counters identical | HBF-Sim binary: 27,299 | -0.015% |
| Qwen3 kernels 3834+3843+3852 (weights in HBF) | 83,033 cycles, 320 HBF requests, 19 page reads | HBF-Sim binary: 83,048, 320, 20 | -0.018% |
| H100 + HBF `route_all` (all traffic through HBF, both chiplet halves) | completes, 393,216 HBF requests (= 12 MiB / 32 B), 0 mapping errors, 0 incomplete pages | (no reference) | conservation holds |
| HBM3 (Ramulator) + HBF together on Qwen3 kernels | runs; see the clock pitfall in section 7.1 | – | works, but tR needs rescaling |

The small cycle differences (-0.015% to -0.18%) are **not** caused by the merge: they show up when
HBF is off or never touched. Same trace, same config, HBF never used: LUD gives 9,047 cycles (HBF
binary) vs 9,031 (newer binary), and the Qwen3 kernel without HBF gives 6,427 vs 6,423 **[ran]**. So the
differences come from the 5 upstream commits between the two GPGPU-Sim bases.

Why B and not A, in one line each:

1. **Trace front end.** Only the newer front end reads `.tracez` (all our traces) and Hopper
   (binary version 90, all our H100 traces). The HBF-Sim front end rejects `.tracez`
   ("Unsupported file type!") and quits on H100 traces with `unsupported binary version: 90`. It
   still returns exit code 0 in that case, so the failure is silent **[ran]**. The newer front end
   reads HBF-Sim's `.traceg` / `.traceg.xz` (LUD v0, Qwen3 v5) without changes **[ran]**.
2. **Upstream ancestry.** `6c3cf4ff` (HBF-Sim's pinned base) is an ancestor of `91880c53` (our base),
   5 commits apart. `git apply -3` therefore merges the HBF patch onto our tree using real 3-way
   merges. Only 10 conflict blocks remain, all additive (section 1).
3. **Validation already done.** All our Ramulator validation (built-in vs Ramulator within 0.1% on the OPT layer, scrambled-option proof,
   HBM3 numbers) was measured on `91880c53`. Option B keeps it valid unchanged, as shown by the
   identical 47,452 / 54,282 cycles. HBF-Sim's own validation is counter-level (requests,
   completions, page ops, MSHR hits). Those counters stay identical after the port; only absolute
   cycles move by the base delta.
4. **Option A would cost more in the end.** It has fewer textual conflicts (6 blocks), but the bridge
   depends on symbols that do not exist at `6c3cf4ff`. We would also have to convert every trace,
   lose H100/Hopper, lose `simple_dram_clock_multiplier`, and re-validate Ramulator (section 2).

---

## 1. The two GPGPU-Sim bases and where the two changes touch

| | HBF-Sim tree | Our tree |
|---|---|---|
| Accel-Sim front end | `accel_sim_hbf/gpu-simulator` (fork of accel-sim-framework at `d7f397a`, 2025-11-26, plus HBF commit `b140511`) | `accelsim-work/accel-sim-framework` at `d930ad6` (Accel-Sim 2.0, clean) |
| GPGPU-Sim | `6c3cf4ff` + `hbf/patches/integration-6c3cf4ff.patch` (1,483 lines, 18 files, 64 hunks) + 16 `hbf/*` files (3,742 lines) | `91880c53` + Ramulator bridge (uncommitted: 9 files, +115/-16, plus `dram_interface.h`, `ramulator_dram.{h,cc}`) |
| Ancestry | `git merge-base --is-ancestor 6c3cf4ff 91880c53` → true; 5 commits: `73774727` (ptxinfo parser fix), `03c1fe44` (L2 write-hit stat), `e10018b6` (Accel-Sim 2.0 Hopper model), `213ab663`, `91880c53` (CI/docs) **[ran]** | |
| Upstream delta 6c3cf4ff→91880c53 | 61 files, +6,124/-748. Memory-side files: `l2cache.cc` +459, `l2cache.h` +310, `gpu-sim.cc` +458, `gpu-sim.h` +78, `addrdec.{h,cc}` +101, `mem_fetch.*` +144, `shader.cc` +1,564 **[ran: git diff --stat]** | |

Hunks of the HBF patch that matter for the merge **[read: patch hunk headers]**: `gpu-sim.cc` 5 hunks
(include; 183 lines of `-gpgpu_hbf_*` option registration in `memory_config::reg_options`; `gpgpu_sim`
constructor creates `hbf_system_t`; `m_hbf_partition_rr_next` init; HBF tick + rotating `hbf_cycle()`
in `gpgpu_sim::cycle()`), `gpu-sim.h` 5 hunks (`hbf_*` fields, `is_hbf_addr()`, members), `l2cache.cc` 8 hunks
(constructor arg, `busy()`, new `hbf_cycle()`, `is_hbf_addr` skips in `simple_dram_model_cycle` and
`dram_cycle`, stats print, L2 bypass in `cache_cycle`), `l2cache.h` 4 hunks, `shader.cc` 1 hunk.
Bridge hunks: `dram.h` (dram_t implements `dram_interface`), `l2cache.cc` 4 hunks (include, construct
`ramulator_dram_t`, `set_max_outstanding`, `has_own_latency` in `dram_cycle`), `l2cache.h` 2, `gpu-sim.cc` 3
(include, 2 options, DRAM clock = Ramulator tCK in `init_clock_domains`), `gpu-sim.h` 2, `addrdec.{h,cc}`
(`channel_address()`), `CMakeLists.txt` (optional Ramulator link).

### 1.1 Option B conflicts (HBF patch applied with `git apply -3` onto 91880c53 + bridge) **[ran]**

14 of 18 patch files applied cleanly (all build files, `instructions.cc`, `ptx_loader.cc`,
`cuda_runtime_api.cc`, `shader.cc`, `src/gpgpu-sim/{CMakeLists.txt,Makefile}`). This includes the
`dram_cycle` / `simple_dram_model_cycle` skips and `hbf_cycle()`. Conflicts:

| # | File | Conflict | Cause | Resolution used in the prototype |
|---|---|---|---|---|
| 1 | gpu-sim.cc includes | `ramulator_dram.h` vs `hbf_system.h` | adjacent additions | keep both |
| 2 | gpu-sim.cc `reg_options` | LRC options (new base) vs 45 HBF options | adjacent additions | keep both |
| 3 | gpu-sim.cc `gpgpu_sim()` | new base: `std::vector` partitions + chiplet `LatencyQueue`s + 8-arg partition constructor; HBF: `new hbf_system_t`, raw arrays, 5-arg constructor | upstream refactor (`e10018b6`) | new-base code + create `m_hbf_system` first + pass it as 9th arg |
| 4 | gpu-sim.h `memory_config` fields | `lrc_*` vs `hbf_*` + `is_hbf_addr()` | adjacent additions | keep both |
| 5 | gpu-sim.h `gpgpu_sim` members | vectors + chiplet queues vs `**` arrays + `m_hbf_system`, `m_hbf_partition_rr_next` | upstream refactor | vectors + the two HBF members |
| 6 | l2cache.cc constructor signature | chiplet queue refs vs `hbf_system_t*` | upstream refactor | both (HBF arg last) |
| 7 | l2cache.cc `cache_cycle` | new `interchip_free` / peer-chiplet logic vs HBF L2 bypass | upstream refactor | HBF bypass first. It pops the queue the request came from (`from_icnt ? icnt : peer`) and checks `m_L2_dram_queue->full()`. Then comes the new-base L2 path. HBF's `return;` is kept (prototype `l2cache.cc:696-713`) |
| 8 | l2cache.h includes | `<algorithm>,<cstdint>,<deque>` vs `abstract_hardware_model.h, dram.h, hbf_system.h` | adjacent | keep both |
| 9 | l2cache.h constructor decl | as #6 | | both, `hbf_system = NULL` default last |
| 10 | l2cache.h `m_dram` | `dram_interface *m_dram` (bridge) vs `dram_t *m_dram; hbf_system_t *m_hbf_system` | both touch the member | `dram_interface *m_dram; hbf_system_t *m_hbf_system;` |

Two semantic issues git did not flag, both fixed in #7: (a) in the new base a request can come from
the peer chiplet (`get_chiplet_req()`). HBF's original bypass always popped `m_icnt_L2_queue`, which
would pop the wrong queue on H100 (2 chiplet partitions). (b) The original bypass pushes into
`m_L2_dram_queue` without a full check (`accel_sim_hbf/.../l2cache.cc:594-600` **[read]**). The FIFO
asserts `m_length < m_max_len` (`delayqueue.h:69` **[read]**), so a full queue would abort.

I also dropped the patch's build-system hunks (bison/flex snapshot fallback for `cuobjdump_to_ptxplus`,
`libcuda`, `cuda-sim`, `intersim2`, `gpgpusim_check.cmake`). Those snapshots were generated from the
6c3cf4ff grammars, and `73774727` changed `ptxinfo.{l,y}` after that. The container has bison 3.8.2
and flex 2.6.4, which the newer tree already uses (build log **[ran]**).

Effort spent on the prototype: about 1 hour, including the build (about 8 minutes at `-j4`).

### 1.2 Option A conflicts (bridge applied with `git apply -3` onto 6c3cf4ff + HBF patch) **[ran]**

6 blocks in 5 files: `addrdec.h` (1), `gpu-sim.cc` (2), `gpu-sim.h` (1), `l2cache.cc` (1), `l2cache.h` (1).
The H100 config hunk does not apply (no `SM90_H100` in that tree). The blocks look small, but they
show that the bridge context depends on code that is missing at `6c3cf4ff`:

- `l2cache.cc` constructor: the bridge's context computes `m_chiplet_id` with
  `get_n_chiplet_partition()`, which does not exist at 6c3cf4ff (chiplets arrived in `e10018b6`).
- `gpu-sim.{h,cc}`: the bridge's context includes `simple_dram_clock_multiplier`, which is absent at
  6c3cf4ff. Only `simple_dram_model` exists there.
- `addrdec.h`: `get_n_chiplet_partition()` again.

These are easy to edit out. The real cost of A is outside GPGPU-Sim, though:

| Missing in Option A | Consequence |
|---|---|
| `.tracez` reader (`trace_parser.cc:89-96, 501-520` and `warp_trace_stream.h` exist only in the newer front end) | every trace must go through `traceDsm` (section 3.3) |
| Hopper opcode map (`ISA_Def/hopper_opcode.h`, `trace_driven.cc:199-200` in the newer tree) and the Hopper timing model (`e10018b6`: shader.cc +1,564, l2cache +769, ...) | H100 traces (our Vast.ai route, binary version 90) cannot run at all |
| SM90_H100 / SM90_H200 configs | no H100 machine model |
| our Ramulator validation | it was measured on 91880c53, so it would all have to be redone on 6c3cf4ff |

**Effort estimate.** Option A: 2 to 3 days to get a working binary, plus a full Ramulator
re-validation, and it still has no Hopper or `.tracez` support (porting `e10018b6` back is
equivalent to doing Option B). Option B: already prototyped. About 1 to 2 days remain for the
clean-ups in section 7 and the scripts **[inferred]**.

---

## 2. Trace format compatibility

### 2.1 What each front end reads **[read]**

| | HBF-Sim front end (`accel_sim_hbf/gpu-simulator/trace-parser/trace_parser.cc`) | Newer front end (`accelsim-work/accel-sim-framework/gpu-simulator/trace-parser/trace_parser.cc`) |
|---|---|---|
| `.traceg` (text) | yes (`PipeReader`, `cat`, :435) | yes (:628) |
| `.traceg.xz` | yes (`xz -dc`, :432) | yes (:625) |
| `.tracez` (zstd + index, `tracer_tool/tracez_format.h`) | **no** → `throw "Unsupported file type!"` | yes (`TracezReader`, :89-96, :501-520; `warp_trace_stream.h`) |
| tracer version < 3 (tb ids per line, e.g. LUD) | yes (:144) | yes (:165) |
| version 5 (Qwen3) | yes | yes |
| version 6 (`imm1 imm2 Val/NoVal [values]`, our traces) | parses: it reads `imm` and ignores the rest of the line (:223) | yes (:331-345) |
| `cluster id / cluster cta / cluster rank` lines inside a TB | **no**: treated as an instruction → `std::out_of_range` | yes |
| binary versions | 35, 60, 61, 70, 75, 80, 86 (`trace_driven.cc:104-121`); others → `exit(0)` | the same plus **90 (Hopper)** (`trace_driven.cc:198-218`) |

Neither front end accepts binary version **89** (sm_89: RTX 4090, L4, L40S). Kernels traced on such a GPU
run only if their SASS was compiled for sm_80/86 (as in the Qwen3 trace, captured on a 4090 but
`binary version = 86`). Keep this in mind when renting a GPU: use T4 (75), A100 (80), or H100 (90).

### 2.2 Cross tests **[ran]** (one simulator at a time, `nice -n 10`, each under 4 minutes)

| Test | Binary | Trace | Result |
|---|---|---|---|
| T1 | HBF-Sim | our T4 vectorAdd `.tracez` | abort: `Unsupported file type!` |
| T2a | HBF-Sim | same, converted with `traceDsm` to `.traceg` (24.3 MB) | abort: `vector::_M_range_check` (on the `cluster id = 0,0,0` line) |
| T2b | HBF-Sim | same, with `cluster ...` lines removed | **54,737 cycles, 262,144 DRAM reads**, 33 s |
| T3a | newer | original `.tracez` | 54,282 cycles, 258,016 DRAM reads |
| T3b | newer | the converted `.traceg` (cluster lines removed) | **54,282 / 258,016, identical to T3a**: the conversion keeps everything that affects timing |
| T4 | newer | Qwen3 kernels 3809+3810 `.traceg.xz` (v5, bin 86), `qwen3.config` without the `hbf` lines | 6,423 + 5,263 cycles |
| T5 | HBF-Sim | same trace, same config without HBF / with HBF | 6,427 + 5,263 / 27,299 + 5,263 |
| T6 | both | LUD `kernel-8.traceg` (v0, bin 70), SM7_QV100, not remapped | newer 9,031, HBF-Sim 9,047; both 158 L2 accesses |
| T9 | HBF-Sim | our H100 vectorAdd, converted | prints `unsupported binary version: 90`, **exit code 0** |

Conclusions: the newer tree reads every trace we have and every HBF-Sim trace without conversion.
The HBF-Sim tree can read our T4 traces only after `traceDsm` plus `sed '/^cluster /d'`, and it can
never read H100 traces. Removing the cluster lines is safe only for kernels without thread-block
clusters (all non-Hopper traces).

### 2.3 Converters that exist

- `accelsim-work/accel-sim-framework/util/tracer_nvbit/others/traceDsm_tool/traceDsm.cpp`:
  `.tracez` → `.traceg` next to the input (`--annotate` gives human-readable output). Built in scratch
  with `g++ -std=c++17 -O2 -I<tracer_tool> traceDsm.cpp -lzstd` inside the container **[ran]**. It
  writes cluster lines.
- `tracer_tool/traces-processing/post-traces-processing <dir> --text`: writes `.traceg` instead of
  `.tracez` from raw tracer output (`post-traces-processing.cpp:181-195` **[read]**).
- HBF remappers: `accel_sim_hbf/util/traces/remap_global_addresses.py` assumes the **v<3** line
  grammar (dest count at token 6, `traceg_op_and_addresses` :139-166 **[read]**). It is right for LUD but
  would misparse our v6 lines. `experiments/23_qwen3_1p7b/prepare_probe.py` is the per-tensor
  remapper used for Qwen3. It handles the v5/v6 grammar (`PC mask dest...`), address modes 0/1/2,
  `.traceg.xz` in and out, and keeps trailing tokens, so it also works on our v6 traces after
  `traceDsm`. It needs `metadata/weights.json` (tensor name, address, bytes) from the capture
  **[read]**. This is the remapper to reuse for weights → HBF placement (C2-C4).

---

## 3. Are the 64-bit address fixes relevant in trace-driven mode?

**No, they do not matter for trace-driven runs. They are harmless and should be kept for
execution-driven tests.**

- The `instructions.cc` hunks change `ld_exec`, `st_impl`, `mma_ld_impl`, `mma_st_impl` and `sst_impl`
  from `src.u32` to `src.u64`. These are PTX functional-simulation routines **[read: patch lines 486-536]**.
- In trace mode, `trace_shader_core_ctx::func_exec_inst()` does not run PTX. It only updates thread
  status and calls `generate_mem_accesses()` (`accelsim-work/.../trace-driven/trace_driven.cc:1142-1156`).
  Addresses come straight from the trace into `set_addr(i, trace.memadd_info->addrs[i])` (:372) as
  `new_addr_type = unsigned long long` (`abstract_hardware_model.h:127`) **[read]**.
- Evidence: trace addresses are already far above 4 GB, e.g. `0x7d8162a00000` (vectorAdd) and Qwen3
  weights at 0x7f.. (about 127 TB), remapped to 274,877,906,944 = 256 GiB. The HBF window is hit
  correctly in trace mode (128 HBF requests on kernel 3809) **[ran]**.
- The `shader.cc` hunk (classify `undefined_space` with `whichspace()`) matters only for generic `ld [addr]`
  in PTX mode. In trace mode the front end always sets `global/shared/local` (`trace_driven.cc:461-489`),
  so the hunk never triggers **[read]**.
- `ptx_loader.cc` and `cuda_runtime_api.cc` (temp files in `.hbf_tmp/`) are execution-driven only.

One HBF front-end change *does* matter in trace mode, and Option B does **not** contain it yet:
`accel_sim_hbf/gpu-simulator/trace-driven/trace_driven.cc` (commit `b140511`) classifies generic
`LD/ST` as **global** (instead of shared) when the trace has no shmem/local base address. Newer
front end: `trace_driven.cc:464-470` still says shared. This affects only old traces without
`-shmem base_addr` (tracer v<5). Our traces and Qwen3 carry the base addresses, and LUD gave
identical L2 traffic in both front ends (T6), so it is optional. It is a 5-line port.

---

## 4. Config files

**[ran: diff of comment-free configs]**

| Config | HBF-Sim tree | Newer tree | Same? |
|---|---|---|---|
| SM75_RTX2060_S, SM75_RTX2060, SM80_A100, SM86_RTX3070, SM7_QV100, SM7_GV100, SM7_TITANV, SM6_TITANX, SM3_KEPLER_TITAN | yes | yes | `gpgpusim.config` and `trace.config` identical for RTX2060_S, A100, RTX3070, QV100 |
| SM90_H100, SM90_H200 | **no** | yes (H100 uses `-gpgpu_n_chiplet_partition 2`; our bridge diff sets `-gpgpu_simple_dram_model 0`) | – |
| `SM7_QV100/gpgpusim_hbf.config` (QV100 + 45 HBF options; ships `-gpgpu_hbf_l2_policy 1`) | yes | add when porting | – |
| `artifact/configs/qwen3.config` | yes | – | = SM80_A100 + its trace.config folded in, with 8 clusters × 1 SM, `-gpgpu_n_mem 8`, `-gpgpu_gmem_skip_L1D 1`, `-gpgpu_max_concurrent_kernel 1`, DRAM clock 1512 MHz, and 45 HBF options |

Cross-tree use: GPGPU-Sim aborts on an unknown option (`option_parser.cc:236-240`: `Unknown Option`,
`exit(1)`) **[read]**. A config containing `-gpgpu_hbf_*` therefore fails on a non-HBF binary, and
`-gpgpu_ramulator_*` fails on the HBF binary. With the merged binary every config of both trees
works. The merged binary loaded `qwen3.config` and the trace-001 LUD config unchanged **[ran]**.

---

## 5. Notes on running the existing binaries

- The HBF-Sim `accel-sim.out` is linked against `libcudart.so` and needs
  `LD_LIBRARY_PATH=accel_sim_hbf/gpu-simulator/gpgpu-sim/lib/gcc-13.3.0/cuda-12080/release`.
  Without it you get `undefined symbol: _ZTI10shd_warp_t`. HBF's `experiments/common/env.sh` sets it.
  The newer CMake-built binaries need nothing beyond `libramulator.so` (absolute rpath).
- Helpers I used: `$M/bin/run_hbfbin.sh`, `$M/bin/run_newbin.sh`, `$M/bin/run_merged.sh` (container,
  `timeout 600`, `nice -n 10`).

---

## 6. Prototype of Option B **[ran]**

Location: `$M/gs_optB` (git clone of our GPGPU-Sim; commits: `5e9da71b` = our bridge,
`bb2a3324` = HBF port), `$M/fw_optB` (copy of the newer front end), binary `$M/fw_optB/build/accel-sim.out`.
Patches: `$M/optB_hbf_port_onto_ramulator.patch` (HBF port on top of the bridge, 4,857 lines incl.
the 16 HBF files), `$M/optB_full_vs_91880c53.patch` (bridge + HBF vs clean 91880c53). Build script:
`$M/scripts/build_optB.sh` (`cmake -DGPGPUSIM_USE_RAMULATOR=ON -DRAMULATOR_ROOT=accelsim-work/ramulator`,
`-j4`). Conflict resolver: `$M/scripts/resolve.py`; resolved blocks: `$M/blocks/`.

Results in section 0. Run folders: `$M/runs/t3_newbin_tracez` (vectorAdd: builtin / HBM3 / traceg),
`t5_hbfbin_qwen` (Qwen3 kernels: HBF-Sim vs merged, HBM3+HBF), `t7_lud_smoke` (LUD smoke, `hbf2.csv`
validated with `validate_hbf_trace.py --channels 4 --subarrays 32 --expected-requests 158
--require-stable-subarray --require-nonempty` → `valid: true`, 158 INGRESS / 158 COMPLETED / 2 READ /
3 PROGRAM, the same as the reference), `t8_h100` (H100 without HBF and with HBF `route_all`, tR = 20 ticks,
182 s).

Not done: the execution-driven HBF smoke (`make reproduce-smoke`, 8 CUDA runs). It needs the
GPGPU-Sim `libcudart` built from the merged tree. The 19_media_saturation suite (9 runs, cycle-exact
reference on the old base) was not run either. Run both as the acceptance test in the real port
(section 8).

---

## 7. Things to fix or add after the merge (needed for C1-C4 to be meaningful)

### 7.1 HBF timing is in DRAM ticks, and Ramulator changes the DRAM clock (must fix)

The bridge sets the DRAM clock domain to Ramulator's tCK (`gpu-sim.cc:1559` in the prototype; HBM3
6400 Mbps → 3205.1 MHz) **[read+ran]**. HBF counts `tR/tPROG/tBERS` in DRAM ticks (`hbf_subarray.cc:22`).
Its link budget is `hbf_channel_bw_gbps*1000/dram_freq_mhz` (`hbf_controller.cc:40,78-79`) **[read]**.

- Measured **[ran]** on Qwen3 kernels with Ramulator HBM3 (8 ch) + HBF, `qwen3.config` unchanged:
  the simulator prints `tR=22680 ticks (7.08 us)`, so **NAND became 2.1× faster without any warning**
  (49,292 cycles). With `-gpgpu_hbf_tR 48077 -gpgpu_hbf_tPROG 641026 -gpgpu_hbf_tBERS 6410256`
  (15/200/2000 µs at 3205.128 MHz) the result is `15.00 us` and 82,809 cycles.
- Fix: add `-gpgpu_hbf_tR_ns/tPROG_ns/tBERS_ns` (and the latencies in ns), converted with
  `get_dram_freq()` at construction. Alternatively, let HBF keep its own clock domain (see below).
- `gpgpu_sim_config::get_dram_freq()` returns `unsigned` (`gpu-sim.h:614`) from a `double` in Hz. Above
  4.29 GHz (e.g. our "HBM3 ×2" 12,800 Mbps config, tCK 156 ps → 6.41 GHz) the conversion is undefined
  and HBF's per-tick bandwidth would be wrong. Change it to return `double`, or compute in MHz **[read; not run]**.
- Simulation speed: HBF is ticked on every DRAM edge (`gpu-sim.cc:2513`), now 3.2 GHz instead of
  850 to 1512 MHz. The prior study already found HBF runs slow (17 minutes for a 4 MiB read). A separate
  HBF clock (e.g. 1 GHz, ticked with the GPGPU-Sim clock-domain mechanism) keeps cost and timing
  independent of the HBM model **[inferred]**.
- The "HBF Channel Utilization" stat divides DRAM ticks by core cycles (prior report, issue #2). With
  the 3.2 GHz Ramulator clock it printed `max=179.452%` **[ran]**. It should divide by DRAM ticks.

### 7.2 HBF shares the partition's DRAM credits and queue (decide per config)

`hbf_cycle()` admits through `can_issue_to_dram()` / `borrow_credit()`. That is the same pool Ramulator
resizes with `set_max_outstanding()` (prototype `l2cache.cc:105,167`). An HBF read holds a credit for
the whole NAND operation, and HBF and DRAM requests share one `L2_dram_queue` per sub-partition
(`continue` skips) **[read]**.

- **C2 (HBF + DRAM, HBF-Sim native)** and **C4 (HBF behind HBM)**: shared resources at the
  partition are at least plausible, because in H3 the GPU only sees the HBM PHY. C4 needs more than
  credits, though: HBF returns must also consume the HBM channel's data bus, and HBF requests should
  pay the HBM base-die decode and forwarding latency. Minimal hook: per-partition token bucket and
  fixed latency on the HBF path, or inject HBF transfers into Ramulator's channel as data-only
  transactions **[inferred]**.
- **C3 (HBF and HBM on separate GPU PHYs)**: the coupling is not physical. Give HBF its own
  `arbitration_metadata` instance (separate credits, e.g. `-gpgpu_hbf_partition_credits`) and its own
  L2→HBF queue, so an HBF request at the head does not block HBM requests behind it **[inferred]**.
- Requests on our H100 config also go through the inter-chiplet queues. The prototype's bypass handles
  the peer queue. Write-forwarding to the peer L2 is skipped for HBF-bypassed writes, which is right for
  `l2_policy 0` **[read]**.

### 7.3 Smaller items

- Port the 5-line generic-load fallback from `b140511` (section 3) for old traces.
- Copy `hbf/gpgpusim_hbf.config` as `SM7_QV100/gpgpusim_hbf.config` and `qwen3.config` into the
  merged tree. Note that the shipped HBF config sets `-gpgpu_hbf_l2_policy 1`, while every paper run uses `0`.
- Always pass `-gpgpu_deadlock_detect 0` with HBF (2 ms erase).
- `hbf_cycle` admits at most one request per partition per DRAM tick. That is 8 per tick on
  `qwen3.config`, and it scales with the DRAM clock, so it also changes under Ramulator **[read; inferred]**.
- With Ramulator HBM3, the same 3 Qwen3 kernels and 320 HBF requests gave 35 page reads, vs 19 with
  the built-in DRAM model. The MSHR coalescing outcome depends on arrival timing. I did not dig into
  this, but it means HBF page traffic is sensitive to the DRAM-tier model **[ran; not explained]**.

---

## 8. Step-by-step plan for the real port (Option B)

All builds go inside the container: `apptainer exec --bind ~/btp_work <sif> bash -c '...'`.
Work in a **new branch** of our GPGPU-Sim repo (or a copy), not in the HBF-Sim tree.

1. **Freeze the bridge.** In `accelsim-work/accel-sim-framework/gpu-simulator/gpgpu-sim`, commit the
   uncommitted Ramulator bridge on branch `ramulator`. Create branch `ramulator-hbf` from it.
2. **Apply the HBF integration 3-way**, excluding build-system hunks:
   `git apply -3 --exclude='cuobjdump_to_ptxplus/*' --exclude='libcuda/*Make*' --exclude='libcuda/CMakeLists.txt'
   --exclude='src/cuda-sim/CMakeLists.txt' --exclude='src/cuda-sim/Makefile' --exclude='src/intersim2/*'
   --exclude='gpgpusim_check.cmake' accel_sim_hbf/hbf/patches/integration-6c3cf4ff.patch`.
   Keep `libcuda/cuda_runtime_api.cc`, `src/cuda-sim/{instructions,ptx_loader}.cc`, `shader.cc`, and
   `src/gpgpu-sim/{CMakeLists.txt,Makefile}`.
3. **Resolve the 10 blocks** as in section 1.1. The exact resolved texts are in `$M/blocks/`, or
   cherry-pick prototype commit `bb2a3324` from `$M/gs_optB` (minus build files, which it already excludes).
4. **Copy the 16 model files** `accel_sim_hbf/hbf/hbf_*.{cc,h}` into `src/gpgpu-sim/`.
5. **Fix section 7.1** (HBF ns timings or own clock; `get_dram_freq()` type; utilization denominator).
   Optionally do section 7.2 (separate HBF credit pool and queue behind an option, default = HBF-Sim
   behaviour) and the H3 link hook for C4.
6. **Front end** (accel-sim-framework, same branch): optionally port the `b140511` generic-load
   fallback in `trace-driven/trace_driven.cc`. Copy `gpgpusim_hbf.config` into
   `configs/tested-cfgs/SM7_QV100/` and add `qwen3.config` under a new name (e.g.
   `SM80_A100_8SM_HBF`).
7. **Build** with `scripts/build_accelsim_ramulator.sh`. It already passes `-DGPGPUSIM_USE_RAMULATOR=ON`,
   and the HBF sources are compiled through `src/gpgpu-sim/CMakeLists.txt`. Output:
   `gpu-simulator/bin/ramulator/accel-sim.out`. For HBF's execution-driven smoke tests, also build
   GPGPU-Sim's `libcudart` (`make -C gpu-simulator/gpgpu-sim` after `source setup_environment`).
8. **Acceptance tests** (one at a time):
   - a. `scripts/simulate.sh cuda RTX2060S t4_vectoradd` → 54,282 cycles; `--dram hbm3_rtx2060s` → 47,452.
     `scripts/check_ramulator.sh hbm3_rtx2060s` passes.
   - b. OPT-125m 10 kernels, built-in and HBM3 → 138,087 / 137,151 (RAMULATOR_GUIDE section 6).
   - c. H100 vectorAdd → 8,947.
   - d. HBF trace smoke: run the trace-001 LUD config → 158/158 pairs, `validate_hbf_trace.py` valid,
     HBF counters equal to `experiments/11_trace_smoke/results/trace-001`.
   - e. HBF functional smoke (8 runs) and `19_media_saturation` (9 runs). Compare **HBF counters
     exactly**, and accept cycle deltas only within the base delta (≤0.2% seen so far).
   - f. Qwen3 first 50 kernels with `qwen3.config` on both binaries. Compare total cycles (expect
     ≤0.1%) and HBF read requests and completions.
9. **Run scripts for C1-C4** (another task): reuse `simulate.sh` with `--dram` (Ramulator config) +
   `--gpgpusim-opt "-gpgpu_hbf_enabled 1 ..."`. Weights go into the HBF range with `prepare_probe.py`.

---

## 9. Verified vs inferred (summary)

Verified by running: the ancestry and conflict counts of both options; all cross-format tests T1-T9;
the Option B build; the identical cycles for HBF-off runs (T4 vectorAdd built-in/HBM3, H100); HBF
counters identical to HBF-Sim on LUD and Qwen3 kernels; H100 + HBF `route_all` conservation; the
Ramulator-clock tR distortion (7.08 µs) and its fix by rescaling; the utilization stat above 100%.

Verified by reading: which formats and versions each parser accepts; that trace mode bypasses PTX
functional execution; the unknown-option abort; the HBF bypass missing a full check; how HBF derives
its clock and bandwidth; the `unsigned get_dram_freq()`.

Inferred, not run: the `get_dram_freq` overflow above 4.29 GHz; the effort estimates; the C3/C4
modelling recommendations (separate credits, H3 link sharing); that a separate HBF clock would speed
up simulation; whether the execution-driven smoke and media suites reproduce under Option B.
