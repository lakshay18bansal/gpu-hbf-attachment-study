# HBF-Sim (jager197/accel-sim-hbf): how it works, what I ran, and how it differs from our Ramulator integration

Repo: https://github.com/jager197/accel-sim-hbf (branch `main`, head `3415a31`), cloned to
`<btp_work>/accel_sim_hbf`. Everything below was read from the source and checked by
running it; file:line references point at the cloned tree.

---

## 0. Summary

- **What it is.** HBF-Sim adds a **High Bandwidth Flash (NAND) memory tier** to GPGPU-Sim / Accel-Sim.
  It is not a DRAM replacement: the GPU keeps its normal DRAM (`dram_t`), and a separate address range
  (by default 256 GB to 768 GB) is routed to a cycle-level NAND model on the GPU's DRAM clock.
- **How it is attached.** A 1,483-line patch against a *pinned* GPGPU-Sim commit (`6c3cf4ff`) plus 16
  model files copied into `src/gpgpu-sim/`. The hook is in the **memory partition**: after the L2,
  each partition checks `is_hbf_addr(addr)` and sends HBF requests to one GPU-wide `hbf_system_t`
  and all other requests to its own `dram_t`. Returns come back through the same `dram_L2_queue`.
- **How HBF is modelled.** It is a page-level queueing model:
  `system (stacks) → cube → controller (MSHR + write buffer + scheduler + FTL + page cache) →
  16 host channels (link bandwidth credits, write limits) → 16,384 sub-arrays (IDLE/READ/PROGRAM/ERASE
  state machines with tR = 15 µs, tPROG = 200 µs, tBERS = 2 ms)`, with at most `max_active` NAND
  operations in flight at once. Every 32 B GPU request costs a whole 4 KiB page operation unless it
  coalesces in the page-keyed MSHR.
- **What I ran.** Build, all unit/tool tests, the functional smoke (8 GPU runs), the LUD trace smoke,
  the 9-run media-saturation suite (all match the published reference cycle-for-cycle), and my own
  study (15 runs: DRAM vs HBF, NAND knob sensitivity, write path). Key results: a 4 MiB streaming
  read is 39× slower than DRAM and scales with tR (it is latency-bound, so `max_active` 64→1024
  changes nothing); a fully written 1 MiB buffer is 188× slower than DRAM and **aborts under the
  shipped strict write policy**, because each in-flight HBF write holds one of the partition's DRAM credits
  for 200 µs, which spreads a page's sectors beyond the 1.18 µs aggregation timeout.
- **Versus our Ramulator work.** Ramulator *replaces* the DRAM model behind each partition through a
  virtual `dram_interface` (1 partition = 1 Ramulator channel, every address). HBF *adds* a second
  tier by address range, shared by all partitions (32 partitions → 1 cube → 16 channels), with no
  external library, no YAML, no power model and no validation against hardware. The two approaches
  are complementary and could coexist (section 7).

---

## 1. The repository

| Path | What it holds |
|---|---|
| `hbf/` | The model (16 files, ~3,700 lines), `gpgpusim_hbf.config`, the integration patch, C++ tests |
| `hbf/patches/integration-6c3cf4ff.patch` | All changes to GPGPU-Sim (18 files) |
| `gpu-simulator/` | Stock Accel-Sim front end (trace-driven `accel-sim.out`); `gpgpu-sim/` is cloned by bootstrap |
| `experiments/` | Paper suites: validation, trace smoke, MSHR/MQSim comparison, media scaling, placement, hotspots, isolation, cost, Qwen3-1.7B |
| `artifact/` | Pinned versions, reference CSVs, paper configs, bundled LUD and Qwen3 traces |
| `util/traces/` | `remap_global_addresses.py` (moves trace addresses into the HBF range), `validate_hbf_trace.py` |

**How it installs (`make bootstrap`, `setup_hbf.sh`).** Clone upstream `gpgpu-sim_distribution` at
exactly `6c3cf4ff…`, refuse to proceed if the checkout differs, `git apply` the patch, copy the 16
files from `hbf/` into `gpgpu-sim/src/gpgpu-sim/`, copy the HBF config to
`configs/tested-cfgs/SM7_QV100/gpgpusim_hbf.config`, copy pre-generated bison/flex parser sources,
then `verify_install.sh` checks hashes (18 patched paths, 16 sources, 25 parser snapshots,
45 registered options). The copy in `hbf/` is canonical; editing the copy inside `gpgpu-sim/` makes
the next bootstrap fail rather than silently overwrite.

**What the patch changes (18 files).**

| Group | Files | Purpose |
|---|---|---|
| Build | `cuobjdump_to_ptxplus/Makefile`, `libcuda/{Makefile,CMakeLists.txt}`, `src/cuda-sim/{Makefile,CMakeLists.txt}`, `src/intersim2/*`, `gpgpusim_check.cmake`, `src/gpgpu-sim/{Makefile,CMakeLists.txt}` | Use pinned parser snapshots instead of host bison/flex; add the 7 HBF `.cc` files |
| Functional fixes | `src/cuda-sim/instructions.cc`, `ptx_loader.cc`, `libcuda/cuda_runtime_api.cc` | `ld/st/mma` used `src.u32` for addresses, truncating anything above 4 GB; now `u64` (HBF lives at 256 GB). Temp files go to `.hbf_tmp/` |
| Timing hook | `src/gpgpu-sim/shader.cc` | A generic-space `ld [raw_addr]` (`undefined_space`) was silently dropped by the LD/ST unit; now classified with `whichspace()` |
| Integration | `gpu-sim.{h,cc}`, `l2cache.{h,cc}` | 45 `-gpgpu_hbf_*` options, `is_hbf_addr()`, creating `hbf_system_t`, `memory_partition_unit::hbf_cycle()`, L2 bypass, skip HBF addresses in `dram_cycle()` |

---

## 2. How Accel-Sim is connected to HBF

### 2.1 How a program puts data in HBF

There is no allocator. HBF is a fixed physical address window:
`is_hbf_addr(a) = hbf_base_addr <= a < hbf_base_addr + hbf_size` (gpu-sim.h in the patch;
default 256 GB + 512 GB). `-gpgpu_hbf_route_all 1` sends *every* L2 miss to HBF (test mode).

- **Execution-driven** (CUDA binary linked against GPGPU-Sim's `libcudart`): the kernel dereferences a
  raw integer address, e.g. `experiments/workloads/channel_probe.cu`:
  `volatile unsigned *p = (unsigned *)(hbf_base + tid*4096); out[tid] = p[0];`
  The functional simulator keeps HBF contents in its normal sparse memory (untouched memory reads as 0),
  so HBF is **timing-only**: the NAND model never holds data.
- **Trace-driven** (`accel-sim.out -trace`): `util/traces/remap_global_addresses.py` rewrites the
  `LDG/STG` addresses of a `.traceg` into the HBF window (`--base 256GB --span 512GB`). For Qwen3,
  only the parameter (weight) allocation ranges were remapped; KV cache, activations and scratch stay in DRAM.

### 2.2 A request's journey (default `l2_policy 0`, i.e. L2 bypass)

```
SM LD/ST unit ──icnt──► memory_sub_partition::cache_cycle()           (l2cache.cc)
                           │  is_hbf_addr && hbf_l2_policy==0:
                           │  skip L2 tag lookup, push straight to L2_dram_queue
                           ▼
                        L2_dram_queue (per sub-partition)
                           │
gpgpu_sim::cycle(), DRAM clock edge  (gpu-sim.cc)
   1. hbf_system->cycle()               ← every stack ticks once per DRAM tick
   2. for each partition p, rotating start (fairness):
        p->hbf_cycle():
          a) drain completions: hbf_system->pop_return_for(global_spid)
               → dram_L2_queue (reads)   or  set_done+delete (L1/L2 write-backs)
               → return_credit()
          b) admit ≤1 request: head of an L2_dram_queue that is HBF, has a DRAM
             arbitration credit, and whose stack is not full()
               → hbf_system->push(mf); borrow_credit()
   3. for each partition: dram_cycle()  ← HBF heads are skipped with `continue`
                           ▼
hbf_system_t::push  → choose stack (page interleave / contiguous), rewrite the address
                      to stack-local if >1 stack (original saved in a map)
hbf_cube_t::push    → hbf_controller_t::push   (the NAND model, section 3)
   ... NAND service ...
hbf_controller_t returnq → hbf_cube_t::collect_returns() → per-source-subpartition deques
                           ▼
dram_L2_queue → (not waiting_for_fill, so) L2_icnt_queue → icnt → SM
```

Things to notice:

- **Clock.** HBF runs on the GPU's DRAM clock edge (850 MHz on QV100) exactly like `dram_t`. NAND
  timings are therefore given in DRAM ticks: tR 15 µs × 850 MHz = 12,750 ticks. If the DRAM clock
  changes, every NAND timing must be re-derived by hand (the Qwen3 config uses 1512 MHz and tR 22,680).
- **No `-dram_latency`.** HBF requests skip the `m_dram_latency_queue` (that is only in `dram_cycle`).
  The L2 ROP latency still applies on the way in.
- **Admission rate.** At most one HBF admission per partition per DRAM tick (32 per tick on QV100).
- **Credits are shared with DRAM.** HBF admission uses the partition's DRAM arbitration credits
  (`can_issue_to_dram`, `borrow_credit`); the shared pool is
  `frfcfs_queue (64) + return_queue (192) − 1 = 255` per partition by default. Each HBF request holds
  its credit for the whole NAND operation (~15 µs), so many in-flight HBF reads in one partition can
  block that partition's DRAM traffic.
- **Head-of-line coupling.** One `L2_dram_queue` per sub-partition carries both tiers. `dram_cycle`
  `continue`s past an HBF head and `hbf_cycle` past a DRAM head, so a DRAM request queued behind an HBF
  request waits until HBF admits it (and vice versa).
- **L2 policy.** `0` = bypass (code default, used in every paper experiment); `1` = normal L2
  tag/MSHR/fill path. The shipped `gpgpusim_hbf.config` sets `1` while its comment says bypass is the
  default; every experiment script overrides it to `0`.
- **busy()/stats.** Partition 0 reports HBF busy and prints HBF statistics (one shared system).
- **`-gpgpu_deadlock_detect 0`** is required: a 2 ms erase looks like a deadlock to GPGPU-Sim.

---

## 3. How HBF is modelled

### 3.1 Hierarchy

| Class (file) | Models | Key behaviour |
|---|---|---|
| `hbf_system_t` (hbf_system.{h,cc}) | All HBF stacks of the GPU | Stack = `page % n_stacks` (interleave) or contiguous slices; per-stack capacity must be whole erase blocks; restores the GPU address on return |
| `hbf_cube_t` (hbf_cube.{h,cc}) | One stack | Private `memory_config` copy (per-stack capacity); sorts completions into per-source-subpartition queues so one partition can't head-of-line-block another |
| `hbf_controller_t` (hbf_controller.{h,cc}) | The logic die | MSHR, write buffer, scheduler, FTL, page cache, trace, statistics |
| `hbf_channel_t` (hbf_channel.h) | One OCP host channel (UCIe link) | Owns a contiguous slice of sub-arrays; token-bucket link bandwidth (192 GB/s → bytes per DRAM tick); outstanding-write limit (64) |
| `hbf_subarray_t` (hbf_subarray.{h,cc}) | One NAND die/plane | `IDLE→READING/PROGRAMMING/ERASING→IDLE`, one countdown timer; 2 page buffers (re-read hit = 10 ticks) |
| `hbf_ftl_t` (hbf_ftl.{h,cc}) | Flash translation layer | Logical page → (sub-array, block, page); per-sub-array active block; erase state; HBF mode (no GC) or SSD mode (greedy GC) |
| `hbf_page_cache_t` (hbf_page_cache.{h,cc}) | Logic-die SRAM | LRU of 4 KiB pages (256 entries, hit 50 ticks); write-invalidate |
| `hbf_trace_t` (hbf_trace.{h,cc}) | Observability | CSV per request: `INGRESS, READ/PROGRAM/ERASE, COMPLETED` |

Default (shipped) geometry per stack: 16 channels × 1,024 sub-arrays = 16,384 sub-arrays, but only
**64 operations in flight** (`max_active`, a power limit), 4 KiB pages, 64 pages/block, 512 GB.

### 3.2 Address → NAND location

```
relative = addr − hbf_base;  page = relative / 4096
channel   = page % 16                      (placement 0, interleave)
          | page / pages_per_channel       (placement 1, contiguous)
          | table[page]                    (placement 2, explicit file; missing page = fatal)
local     = page / 16  (or page − ch·pages_per_channel)
subarray  = channel.first + local % subarrays_per_channel
```

The FTL keeps that sub-array (channel affinity is asserted everywhere; a violation is a fatal error).
- **Read of a never-written page**: no allocation; a stable pseudo address
  `(subarray, 0x80000000|page/64, page%64)` so pure-read workloads don't consume capacity.
- **Write**: take the next free page of the sub-array's active block, allocating a block when needed
  (fresh blocks are erased, recycled ones are not); the old mapping is invalidated only after the new
  page is reserved; capacity is enforced (`hbf_size / (64·4096)` blocks), exhaustion is fatal.

### 3.3 Read path (`push` and `schedule_operations`, hbf_controller.cc:301-444, 830-980)

1. `push(mf)`: page = addr/4096. If an MSHR entry for that page exists → append (**MSHR hit**, no new
   NAND op). Otherwise create an `OP_WAITING` entry with its FTL location and queue it.
2. If a buffered write to the same page exists, it is force-flushed first and the read is attached to
   it (read-after-write ordering).
3. `schedule_operations()` each tick, while `active < max_active`: pick the first `OP_WAITING` entry
   whose sub-array is idle (FCFS; read-priority/write-drain variants), skipping busy sub-arrays so one
   hot die can't block the queue.
   - page-cache hit → ready after 50 ticks, no sub-array used;
   - else `start_read` → tR (or 10 ticks if the page is in one of the sub-array's 2 page buffers).
4. When the sub-array returns to IDLE, the entry becomes `delivery_pending`; delivery pushes as many
   requests as the return queue (192) and the channel's link credit allow (each read consumes its byte
   count from the token bucket), fills the page cache, and frees the entry when all are delivered.

Optional **read aggregation** (`read_mode 1`): hold a new page entry until it has `threshold` bytes
requested or `window` ticks passed, to coalesce more sectors per page read.

### 3.4 Write path

1. Writes go to a **write buffer** keyed by page, with exact byte coverage (a 4,096-entry byte bitmap
   built from the request's byte mask).
2. An entry leaves the buffer when the page is fully covered, when the buffer has 32 pages, when it is
   older than `write_agg_timeout` (1,000 ticks), when a read hits it, or when the device is idle for
   100 ticks. A flush is deferred if the channel already has 64 outstanding writes or an MSHR entry for
   the page is in flight.
3. **Strict policy (`write_timeout_policy 1`, the shipped default)**: flushing a page that is not fully
   covered is a **fatal error** (OCP requires full 4 KiB programs). Policy 0 programs a padded partial
   page and counts the padding (write amplification).
4. Scheduling a write: if the target block is not erased → `ERASE` (tBERS, the entry is re-queued
   when done) → `PROGRAM` (tPROG). Writes complete only after PROGRAM.

### 3.5 Media management

- `media_mode 0` (HBF, default): no garbage collection; a block whose pages are all invalid goes back
  to the free pool dirty and is erased on reuse; optional zone-remap *accounting* for wear.
- `media_mode 1` (SSD baseline): greedy GC with page relocation. **Its latency is only counted**
  (`m_gc_cycles_remaining`, `n_gc_stall_cycles` in hbf_ftl.cc:415-443); nothing in the controller
  checks `is_in_gc()`, so GC never actually blocks a sub-array. The authors label SSD mode
  diagnostic-only, so this does not affect their HBF results, but SSD-mode timing is optimistic.

### 3.6 Backpressure and limits

| Limit | Default | Where |
|---|---|---|
| Unique pages in MSHR + write buffer per stack | 256 (`max_outstanding`) | `hbf_controller_t::full()` → `hbf_cycle` stops admitting to that stack |
| NAND ops in flight per stack | 64 (`max_active`) | `schedule_operations` |
| Link bandwidth per channel | 192 GB/s (token bucket, ≤ 1 tick + 128 B burst) | delivery loop |
| Outstanding writes per channel | 64 | `flush_write_buffer_entry` |
| Return queue | 192 (`-gpgpu_dram_return_queue_size`) | `m_returnq`, cube routing queues |
| DRAM arbitration credits per partition | 255 shared | `can_issue_to_dram` |

### 3.7 What the model does not do

No power/energy model; no ONFI/UCIe command-level timing (the link is a byte budget, the die a single
countdown); no read-retry, ECC, read disturb or multi-plane operations; no data stored; NAND timings are
"HBF-grade" product assumptions (15/200/2000 µs, about 5x faster than MQSim's commodity NAND), not
measurements; validation is against conservation invariants, MQSim and closed-form bounds, not silicon.

---

## 4. What I ran (all in the Apptainer CUDA 12.8 container, `containers/accelsim.sif`)

| Step | Command | Result |
|---|---|---|
| Install | `make bootstrap` | PASS: 18 patched paths, 16 sources, 25 parser snapshots, 45 options |
| Build | `make build JOBS=8` | `gpu-simulator/bin/release/accel-sim.out` (~1 min) |
| Tests | `make test` (needs `.venv` from `requirements.txt` for matplotlib) | PASS: release checks, 10 Python tool tests, 3 C++ model test programs (stack mapping, write coverage, core FTL/sub-array) |
| Functional smoke | `make reproduce-smoke RUN_TAG=smoke-001` | 8 GPU runs, all equal to `artifact/reference` (below) |
| Trace smoke | `make trace-smoke RUN_TAG=trace-001` | LUD kernel-8 trace, 158/158 request/completion pairs, trace valid |
| Media suite | `experiments/19_media_saturation/run.py --tag media-001 --jobs 4` | 9 runs, every cycle count equal to the reference |
| My study | `study/run_study.sh s1` | 12 cases, section 5 |

### 4.1 Functional smoke (`experiments/10_validation/results/smoke-001`)

| Case | Cycles (mine) | Reference | HBF requests | Page reads | MSHR hits |
|---|---|---|---|---|---|
| DRAM baseline (vectorAdd, HBF off) | 5,569 | 5,569 | – | – | – |
| channel_probe, 64 pages, 1/2/4/8/16 channels | 5,621 (all five) | 5,621 | 64 | 64 | 0 |
| shared_staging, demand read | 5,508 | 5,508 | 32 | 1 | 31 |
| shared_staging, aggregation read | 5,520 | 5,517 | 32 | 1 | 31 |

The aggregation +3 cycles is the difference already documented in `artifact/VERIFICATION.md`.
These runs use tR = 20 ticks, 32 sub-arrays: they check routing and coalescing, not performance —
which is why 1 to 16 channels give identical cycles. Read amplification in channel_probe is 128x
(one 32 B sector per 4 KiB page read).

### 4.2 LUD trace smoke (`experiments/11_trace_smoke/results/trace-001`)

79 global addresses remapped into HBF; 158 requests (96 reads, 62 writes); 91 MSHR hits (57.6%);
2 page reads, 3 page programs; 1,984 host write bytes became 12,288 programmed bytes (write traffic
ratio 6.19, 10,304 padding bytes) because LUD writes partial pages, so this smoke runs with the
compatibility write policy. 9,137 cycles.

### 4.3 Media saturation (real 15 µs NAND, 80 SMs, 32 partitions)

Each lane reads one word from a different page, so every request is a unique page read.

| Case | Channels | Sub-arrays = slots | Cycles (= reference) | Array GB/s | Ceiling `slots·4KiB/tR` | Returned GB/s |
|---|---|---|---|---|---|---|
| grow_c1 | 1 | 32 | 548,840 | 8.738 | 8.738 | 0.068 |
| grow_c2 | 2 | 64 | 277,192 | 17.472 | 17.476 | 0.137 |
| grow_c4 | 4 | 128 | 141,416 | 34.920 | 34.953 | 0.273 |
| grow_c8 | 8 | 256 | 73,497 | 69.774 | 69.905 | 0.545 |
| grow_c16 | 16 | 512 | 39,537 | 139.285 | 139.810 | 1.088 |
| supply_c16 (256 producers) | 16 | 512 | 74,227 | 69.032 | 139.810 | 0.539 |
| supply_c16 (512 producers) | 16 | 512 | 39,783 | 138.291 | 139.810 | 1.080 |
| size_c16 (2048 pages) | 16 | 512 | 73,497 | 139.547 | 139.810 | 1.090 |
| fixed_c16 (32 slots total) | 16 | 32 | 548,840 | 8.738 | 8.738 | 0.068 |

Reading: the model's read throughput is exactly `min(slots, sub-arrays) × 4 KiB / tR` once enough
requests are in flight. Channels matter only because they carry sub-arrays (fixed_c16 = grow_c1).
With one sector per page, the GPU sees 1/128 of the array bandwidth: HBF needs page-level locality.

---

## 5. My own study (`study/`, results in `study/results/s1/`)

Workload `study/workloads/stream.cu`: 80 blocks × 256 threads, grid-stride loop. Every case uses the
installed QV100 `gpgpusim_hbf.config` (80 SMs, 32 partitions, 850 MHz DRAM clock, tR 15 µs,
tPROG 200 µs, 16 channels, 16,384 sub-arrays, `max_active` 64) with `-gpgpu_gmem_skip_L1D 1`; only the
listed knobs change. Runner: `study/run_study.sh`, `study/run_followup.sh`.

### 5.1 Reads (4 MiB = 1,024 pages, 131,072 × 32 B requests)

| Case | Change | Cycles | vs DRAM | Page reads | MSHR hits | Reported read latency (core cyc) |
|---|---|---|---|---|---|---|
| r_dram | HBF off (buffer in DRAM) | 23,141 | 1× | – | – | – |
| r_hbf | `l2_policy 0` (bypass) | 899,439 | 38.9× | 1,024 | 99.2% | 16,991 (= 15.0 µs = tR) |
| r_hbf_l2cached | `l2_policy 1` | 899,542 | 38.9× | 1,024 | 99.2% | 16,991 |
| r_hbf_active256 | `max_active 256` | 899,439 | 38.9× | 1,024 | 99.2% | 16,991 |
| r_hbf_active1024 | `max_active 1024`, `max_outstanding 1024` | 899,439 | 38.9× | 1,024 | 99.2% | 16,991 |
| r_hbf_tr3us | tR 3 µs | 193,070 | 8.3× | 1,024 | 99.2% | 3,407 |
| r_hbf_tr75us | tR 75 µs (commodity NAND) | 4,431,281 | 191× | 1,024 | 99.2% | 84,911 |
| r_hbf_nomshr | `mshr_enabled 0` | 3,052,380 | 132× | 1,024 | 0% | **201** (99.2% page-cache hits) |

What this shows about the model:

- **Latency-bound, not bandwidth-bound.** Each grid-stride iteration touches only 20 pages (80 KiB),
  so at most ~20 NAND reads are ever in flight (max queue depth 8). More slots change nothing;
  runtime scales with tR (3 µs: ×0.21, 75 µs: ×4.93). Compare the media suite (§4.3), where
  1,024 independent pages are in flight and throughput hits `slots × 4 KiB / tR`. HBF only pays off
  when the GPU exposes hundreds of independent pages.
- **The MSHR is essential.** Without it, every 32 B request is its own entry; the 256-entry
  `max_outstanding` limit then fills with requests for the same few pages, and the kernel is 3.4× slower even
  though the page cache serves 99.2% of them.
- **Do not use "HBF Read Latency" as end-to-end latency.** In r_hbf_nomshr it falls to 201 cycles
  while the kernel gets 3.4× slower, because it is measured from NAND issue (§7, #3).
- **L2 policy is irrelevant for streaming** (all 133,632 L2 accesses missed in cached mode).

### 5.2 Writes (1 MiB = 256 pages)

| Case | Policy / change | Result | Cycles | Programs (partial) | Host → media bytes (ratio) |
|---|---|---|---|---|---|
| w_dram | HBF off | ok | 6,037 | – | – |
| w_hbf_full_strict | strict (shipped default), every byte written | **fatal**: "strict write policy rejected an incomplete NAND page" (page 54) | – | – | – |
| w_hbf_full_strict_bigtimeout | strict, `write_agg_timeout 1,000,000` | ok | 1,137,759 (188×) | 256 (0) | 1,048,576 → 1,048,576 (1.00) |
| w_hbf_full_strict_bigbuf | strict, timeout 1 M + 512-entry buffer | ok, identical | 1,137,759 | 256 (0) | 1.00 |
| w_hbf_full_compat | compat, default timeout | ok | 1,816,867 | 493 (437) | 1.93 |
| w_hbf_partial_strict | strict, 128 B per page | **fatal** (page 0), as designed | – | – | – |
| w_hbf_partial_compat | compat, 128 B per page | ok | 1,816,596 | 498 (498) | 32,768 → 2,039,808 (**62.25**) |

Why the full-page strict run failed, and why writes are 188× slower than DRAM:

1. Every 32 B write that HBF admits holds one of its partition's 255 DRAM arbitration credits until
   the page's PROGRAM (200 µs) completes. With 1,024 write sectors per partition, admission happens
   in about 4 to 5 credit "waves": 1.14 M cycles ≈ 1.0 ms ≈ 5 × tPROG. Max queue depth is only 4.
2. A page's 32 sectors come from 4 thread blocks on different SMs, so they arrive in different
   waves, much more than the 1,000-tick (1.18 µs) aggregation timeout apart. The timeout flushes the
   page incomplete, which strict mode treats as fatal. The 32-entry buffer never filled. Raising only
   the timeout fixes it (bigtimeout = bigbuf).
3. In compat mode the same timeout splits pages into partial programs (437 of 493), costing 1.6× more
   time and 1.93× write amplification. Sub-page writes cost 62× media bytes.

So with the shipped configuration, ordinary GPU write patterns into HBF either abort (strict) or
amplify (compat); the authors' write studies use a 256-entry buffer, a `drain_maxwait` of 170,000 and an explicit
placement table. Part of the slowdown is the credit coupling, which is a property of the integration,
not of NAND.


## 6. Our Ramulator integration vs HBF-Sim's HBF integration

Ours: `accelsim-work/accel-sim-framework/gpu-simulator/gpgpu-sim` (branch `ramulator`, uncommitted),
`ramulator_dram.{h,cc}` + `dram_interface.h`, and the Ramulator tree on branch `accelsim`
(see `accelsim-work/RAMULATOR_GUIDE.md`).

### 6.1 The one-picture difference

```
OUR RAMULATOR INTEGRATION (replace)                 HBF-SIM (add a tier)

   L2 slice ── L2_dram_queue                           L2 slice ── L2_dram_queue
                    │                                                │ is_hbf_addr(addr)?
         memory_partition_unit::dram_cycle()          no ────────────┴──────────── yes
                    │  (-dram_latency skipped:         │                            │
                    │   has_own_latency())   dram_cycle(): dram_t      hbf_cycle(): one GPU-wide
                    ▼                        (GPGPU-Sim DRAM,          hbf_system_t (shared by
     dram_interface* m_dram                   unchanged)                all 32 partitions)
      ├─ dram_t            (built-in)                                       │
      └─ ramulator_dram_t  ──► Ramulator controller i                  stacks → cube → controller
         (channel i == partition i,                                    → 16 channels → sub-arrays
          every address of that partition)
```

### 6.2 Side by side

| Aspect | Our Ramulator ↔ Accel-Sim | HBF-Sim's HBF ↔ Accel-Sim |
|---|---|---|
| Goal | Swap GPGPU-Sim's DRAM timing model for a validated, standards-based DRAM simulator (GDDR6, HBM3) | Add a new, slower, higher-capacity NAND tier next to the existing DRAM |
| Relation to `dram_t` | **Replaces** it (`-gpgpu_ramulator_enabled 1`); `dram_t` is not built into the path | **Coexists**: `dram_t` still serves every non-HBF address |
| Which requests | All L2 misses / write-backs | Only addresses in `[hbf_base, hbf_base+size)` (or all, with `route_all`) |
| Hook | Polymorphism: new abstract `dram_interface` (full/push/cycle/return_queue/stats), `memory_partition_unit` holds `dram_interface*`; 2 small hooks (`has_own_latency`, `max_outstanding`) | New code path: `hbf_cycle()` beside `dram_cycle()`, `is_hbf_addr` checks in `dram_cycle`/`simple_dram_model_cycle`/`cache_cycle`; `hbf_controller_t` *mimics* `dram_t`'s interface but does not implement a common base class |
| Topology | 1:1. GPU partition *i* = Ramulator controller *i*; GPGPU-Sim decides the channel; config must have `CHANNELS == -gpgpu_n_mem` | N:1. All 32 partitions feed one `hbf_system_t`; HBF re-maps pages to its own stacks/channels/sub-arrays, independent of GPU partition hashing |
| Where it sits in time | Inside each partition's `dram_cycle()`; the last channel ticks Ramulator once | Before the `dram_cycle()` loop, in `gpgpu_sim::cycle()`; system ticked once, then partitions in rotating order |
| Clock | DRAM clock domain **set to Ramulator's tCK** (DRAM value of `-gpgpu_clock_domains` ignored) | Uses the GPU's existing DRAM clock; NAND times are hand-converted to DRAM ticks (change the clock ⇒ change tR/tPROG/tBERS) |
| Front-end latency | Ramulator config `FIXED_LATENCY_NS` replaces `-dram_latency` | None (no `-dram_latency`, no replacement); only the L2 ROP latency |
| Request granularity | Splits each `mem_fetch` into 32 B transactions (sector mask aware); completes when all parts return | Keeps the `mem_fetch` whole; coalesces many `mem_fetch` into one 4 KiB page op (MSHR) |
| Backpressure | Pending queue vs read+write buffer sizes; outstanding credit limit = 2 × (read+write buffers) via `set_max_outstanding` | Unique pages (MSHR + write buffer) ≥ 256 per stack; uses the partition's *DRAM* credits unchanged (255 shared, held for 15 µs) |
| Address bits | GPGPU-Sim chooses channel; Ramulator's mapper (`GPUBgCoBaRo`, or `PassThrough` = GPGPU-Sim decode) chooses bank/row/col | HBF ignores GPGPU-Sim's DRAM address decode entirely; page → stack → channel → sub-array → FTL block/page |
| L2 | Unchanged (normal L2 caching) | Default bypasses L2 for HBF addresses (`l2_policy 0`); policy 1 caches normally |
| Device model depth | Command-level DRAM: ACT/PRE/RD/WR/REF, every JEDEC timing constraint, bank/bank-group state, refresh, scheduler (`GPGPUSimFRFCFS`), row policy | Operation-level NAND: one countdown per sub-array; page buffers; power limit `max_active`; link token bucket; FTL; write buffer; MSHR |
| Code location | Separate library (`libramulator.so`, C++20) + ~580 lines bridge in GPGPU-Sim; Ramulator components in `ramulator/src/ramulator/gpgpusim/` | Everything compiled into `libgpgpusim` (C++17); 16 files copied into `src/gpgpu-sim/` + 1,483-line patch on a pinned commit |
| Configuration | External Python → YAML (`configs/accelsim/*.py`, `python3 -m ramulator export`), datasheet presets, `--ram-set` | 45 `-gpgpu_hbf_*` options in `gpgpusim.config` |
| Power | AccelWattch (GPU + calibrated DRAM) and Ayna (HBM3 command energy) | None |
| Validation | Against GPGPU-Sim's own DRAM model with identical parameters (OPT layer +0.1% cycles); "scrambled" proof that unused options are inert; Ramulator unit tests | Invariants (request/completion conservation, trace schema), unit tests, MQSim page-service comparison, closed-form bandwidth bounds; no silicon |
| Workloads used | vectorAdd, OPT-125m layer traces on RTX 2060 S / H100 configs | Synthetic probes, LUD kernel, Qwen3-1.7B decode traces with weights remapped into HBF (QV100 / A100-like configs) |
| Upstream base | Newer accel-sim gpgpu-sim (`91880c53`, has chiplet partitions) | Pinned older gpgpu-sim `6c3cf4ff` |

### 6.3 Similarities

- Both are timing-only memory models under GPGPU-Sim's functional memory.
- Both hook at the same place: **after the L2, inside `memory_partition_unit`, on the DRAM clock edge**,
  and both give completed reads back through `dram_L2_queue` and retire write-backs with `set_done`.
- Both keep the GPU side (SMs, L1, interconnect, L2, partition hashing) untouched and both disable or
  bypass GPGPU-Sim memory options that no longer apply.

### 6.4 Consequences

- HBF-Sim's DRAM tier is still GPGPU-Sim's simple `dram_t`; our Ramulator bridge would give it a
  realistic DRAM tier. The two patches touch the same functions (`memory_partition_unit` constructor,
  `dram_cycle`, `gpu-sim.{h,cc}`) but do not conflict in purpose: Ramulator swaps the object behind
  `m_dram`; HBF adds `hbf_cycle()` and an address filter before it. Porting HBF to our newer gpgpu-sim
  would mean re-applying the patch by hand (the upstream base differs) and keeping its `is_hbf_addr`
  skips in `dram_cycle`.
- Because HBF holds DRAM credits for a full NAND operation, a mixed workload (weights in HBF, KV/activations
  in DRAM, like their Qwen3 study) couples the tiers inside a partition. Ramulator has no such coupling
  because it is the only memory behind the partition.
- Our Ramulator work answers "how good is the DRAM model"; HBF-Sim answers "what if part of the
  footprint lives in flash". Ramulator could in principle model a NAND device, but its Accel-Sim front end
  and controllers are DRAM-command based; HBF-Sim's page/MSHR/FTL structure is what a flash tier needs.

---

## 7. Issues and caveats I found while reading and running it

| # | Finding | Evidence | Impact |
|---|---|---|---|
| 1 | SSD-mode GC latency is counted but never blocks anything | `hbf_ftl.cc:415-443` sets `m_gc_cycles_remaining`; no caller of `is_in_gc()` | SSD-baseline timing optimistic; HBF mode (default) unaffected |
| 2 | Channel "util %" divides DRAM ticks by core cycles | `hbf_controller.cc:1115` `n_active_ticks / (gpu_sim_cycle+gpu_tot_sim_cycle)` | Utilisation under-reported by core/DRAM clock ratio (1132/850 on QV100) |
| 3 | "Avg Op Latency" / "Read Latency" start at NAND issue, not arrival | `latency = current_cycle - issue_cycle` (`:701`) | Excludes queueing; use trace INGRESS→COMPLETED or "First-Request Latency" for end-to-end |
| 4 | Shipped config sets `-gpgpu_hbf_l2_policy 1` but documents bypass as default | `hbf/gpgpusim_hbf.config` last lines vs option help text | A user running the stock config gets the "experimental" cached mode; every paper script forces 0 |
| 5 | Strict write policy is the shipped default and is fatal on any partial page | `flush_write_buffer_entry` → `fail_fast` | Ordinary kernels writing < 4 KiB per page into HBF abort (LUD smoke must use policy 0) |
| 6 | HBF holds DRAM arbitration credits for the whole NAND op; one queue per sub-partition serves both tiers | `hbf_cycle` `borrow_credit`, `continue` in `dram_cycle` | Tier interference in mixed workloads is partly a simulator artefact |
| 7 | Simulation cost | 4 MiB HBF read: ~17 min wall (r_hbf), 78 min at tR 75 µs, vs seconds for DRAM (per-tick scans of the MSHR map and queue) | Plan run time for large footprints (Qwen full replay ≈ 18 host hours per authors) |
| 8 | No power model; timings are assumptions | config comments, `VERIFICATION.md` | Energy claims would need a separate model |

---

## 8. Reproducing everything

```bash
cd <btp_work>/accel_sim_hbf
make bootstrap                                     # host: clone pinned gpgpu-sim, apply patch, verify
SIF=../containers/accelsim.sif
A="apptainer exec --bind $PWD $SIF bash -c"
$A 'export CUDA_INSTALL_PATH=/usr/local/cuda; make build JOBS=8'
$A 'python3 -m venv .venv && .venv/bin/pip install -r requirements.txt'
$A 'source .venv/bin/activate; make test'
$A 'source .venv/bin/activate; make reproduce-smoke RUN_TAG=smoke-002'      # ~10 s
$A 'source .venv/bin/activate; make trace-smoke RUN_TAG=trace-002'          # ~5 s
$A 'source .venv/bin/activate; python3 experiments/19_media_saturation/run.py --tag media-002 --jobs 4'  # ~2 min
$A 'export CUDA_INSTALL_PATH=/usr/local/cuda; JOBS=4 study/run_study.sh s2'   # ~2.3 h (tR 75 µs and no-MSHR are the long ones)
```

Every tag must be new (runners refuse to overwrite). Not run: the MQSim comparison suite
(`make setup-mqsim`), placement/hotspot/isolation/cost suites, and the ~18-hour Qwen3 replay.
