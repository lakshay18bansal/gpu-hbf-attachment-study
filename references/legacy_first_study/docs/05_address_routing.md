# Where addresses are mapped and routed, per configuration (with proof)

Paths are relative to `hbm_hbf_study/`. Simulator sources are in
`sim/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/`, abbreviated `G/`. Line numbers refer
to the current build.

The proof is `results/validation_routing/ROUTING_VALIDATION.md`, produced by `tools/validate_routing.py`.
It checks **every request** of a vectorAdd run in every configuration against an independent
re-implementation of each mapping step, and against the simulator's own counters.

## 1. The path of one memory request (common to all configurations)

```
 (0) trace placement (offline)         tools/remap_ranges.py | prepare_probe.py | tools/place_tensors.py
      weights / inputs -> HBF window [256 GiB, 768 GiB); everything else keeps its address
 (1) SM load/store -> mem_fetch         G/mem_fetch.cc:78      addrdec_tlx(addr) -> partition (chip) + L2 slice
 (2) SM -> L2 slice (crossbar)          G/shader.cc:5618-5625  icnt_push(dest = mf->get_sub_partition_id())
 (3) L2 slice: hit or miss              G/l2cache.cc:782       memory_sub_partition::cache_cycle
 (4) L2 miss -> which memory tier?      G/l2cache.cc:1380      L2interface::push -> memory_queue_push (:1014)
                                        G/l2cache.cc:1005      uses_hbf_queue: attach >= 1 && is_hbf_addr(addr)
                                        G/gpu-sim.h:606        is_hbf_addr: base <= addr < base + size
          ├─ DRAM/HBM tier  -> L2_dram_queue -> G/l2cache.cc:585  dram_cycle -> Ramulator bridge
          └─ HBF tier       -> L2_hbf_queue  -> G/l2cache.cc:377  hbf_direct_cycle (attach 1/2)
                                                G/l2cache.cc:322  hbf_cycle (attach 0, paper legacy)
 (5a) DRAM/HBM: partition i = Ramulator channel i
        G/ramulator_dram.cc:302  make_request: addr_vec = {partition id}; address inside the channel
                                 = channel_address(addr) (G/addrdec.cc:96) mod channel capacity
        G/ramulator_dram.cc:363  fatal() if Ramulator ever puts it in another channel
        Ramulator GPUBgCoBaRo mapper -> pseudo-channel / bank group / bank / row / column
 (5b) HBF: one HBF system shared by all partitions
        G/hbf_system.cc:68,83    locate/push: stack = page mod stacks (1 stack here)
        G/hbf_controller.cc:228  addr_to_page: page = (addr - 256 GiB) / 4 KiB
        G/hbf_controller.cc:235  channel_of_page: channel = page mod channels (interleave)
        G/hbf_ftl.cc:65          sub-array = channel * per_ch + (page / channels) mod per_ch
        G/hbf_controller.cc:333  push: page MSHR merge, LHB hit, or new NAND READ
 (6) reply -> its own L2 slice          G/l2cache.cc:377 (HBF), :585 (DRAM): dram_L2_queue of
                                        mf->get_sub_partition_id(); hbf_system_t::pop_return_for (G/hbf_system.cc:118)
 (7) L2 slice -> SM (crossbar)          G/gpu-sim.cc:2998-3003 icnt_push(source = sub-partition, dest = tpc)
```

The GPU-side address split (which partition, i.e. HBM/DRAM channel, and which L2 slice) is set by
`-gpgpu_mem_addr_mapping dramid@8;…RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS` with 8 partitions × 4 slices.
It lives in `common/gpu/qwen3_slice.config` and is identical in every configuration:
- partition = address bits 8-10;
- L2 slice = 4 × partition + the lowest 2 bank bits.

## 2. Per configuration: which files decide what

| Configuration | Tier decision (step 4) | DRAM/HBM device (5a) | HBF attachment (5b) | Set in |
|---|---|---|---|---|
| **C1** HBM only | `-gpgpu_hbf_enabled 0` → `is_hbf_addr` is false for every address → everything to Ramulator | HBM3 8 ch, `common/memory/hbm3_slice8.py` | none | `C1_hbm_only/config.env` (HBF=0, MEM=hbm3_slice8) |
| **C2** HBF + DRAM | HBF window → HBF, rest → GDDR6 | GDDR6 8 ch, `common/memory/gddr6_slice8.py` | attach 1: own L2→HBF queue + credits, 52.49 ns front end, 8 ch × 51.2 GB/s | `C2_hbf_dram/config.env` |
| **C3** iso-stack | same rule | HBM3 8 ch | attach 1, 8 HBF channels | `C3_hbf_hbm_direct/iso_stack/config.env` |
| **C3** iso-shoreline | same rule | HBM3 at 3.2 Gb/s, `common/memory/hbm3_slice8_half.py` | attach 1, 4 HBF channels | `C3_hbf_hbm_direct/iso_shoreline/config.env` |
| **C4** behind HBM | same rule (the decoder now models the HBM base die) | HBM3 8 ch | attach 2: as attach 1, plus each partition's HBM PHY token bucket (51.2 GB/s) charged by HBM bytes (`G/l2cache.cc:477` phy_take_hbm, called from dram_cycle) and by HBF reply bytes (hbf_direct_cycle); front end 62.49 ns = 52.49 + 10 ns hop | `C4_hbf_behind_hbm/config.env` |
| C2 legacy (paper) | same window | GPGPU-Sim built-in DRAM, `common/memory/builtin_qwen3.config` | attach 0: HBF shares the DRAM queue and credits (HBF-Sim as published) | `C2_hbf_dram/legacy_paper/` |

`scripts/run_config.sh` (lines 108-144) turns `config.env` into simulator options:
- `-gpgpu_hbf_attach`, `-gpgpu_hbf_frontend_latency_ns`, `-gpgpu_hbf_phy_gbps`;
- the derived `max_active`, `num_subarrays` and `credits`;
- `-gpgpu_ramulator_config memory.yaml`.

The HBF window (`-gpgpu_hbf_base_addr 274877906944` = 256 GiB, `-gpgpu_hbf_size` = 512 GiB) and
the page interleave (`-gpgpu_hbf_placement_mode 0`, `-gpgpu_hbf_channel_map 0`) are set in
`common/hbf_base.config` (lines 3-22).

Which data lands in the window is decided **offline, in the trace**:

| Workload | Tool | What goes to HBF |
|---|---|---|
| vectorAdd | `tools/remap_ranges.py --ranges-from-memcpy` | A and B (the host-copied inputs); C stays in DRAM/HBM |
| Qwen3 | HBF-Sim's `experiments/23_qwen3_1p7b/prepare_probe.py` (`weights.json` from `data_ptr()`) | all 310 weight tensors; KV cache and activations stay |
| New captures | `tools/place_tensors.py --hbf weights\|weights+kv\|kv` | chosen tensor classes |

## 3. How we know it really happens

`-gpgpu_mem_access_trace_file f -gpgpu_mem_access_trace_detail 1` logs every request as it leaves
the L2, with the tier it was actually sent to, its partition, its L2 slice and its size. HBF-Sim's
`-gpgpu_hbf_trace_file` logs each HBF request's page, channel, sub-array, arrival and completion.

`tools/validate_routing.py` recomputes every step from the address alone and compares. The results
are in `results/validation_routing/ROUTING_VALIDATION.md`; see section 4 below.

## 4. Validation result: 81 / 81 checks PASS

Ten runs: vectorAdd on C1, C2, C3-iso-stack, C3-iso-shoreline, C4 and C4 with prefetch; the Qwen3 q_proj GEMV
on C3 and C4, which reads weights from HBF **and** activations from HBM; and cap and hop tests.

| Check | What is compared | Result |
|---|---|---|
| decode | partition and L2 slice of each request vs a Python re-implementation of `addrdec.cc` | 0 mismatches over 194,112-262,272 requests per run |
| tier | address in the HBF window ⇔ sent to HBF (C1: none, although 183,872 addresses lie in the window) | 0 misrouted |
| dram | 32 B parts sent to partition *i* = Ramulator channel *i* reads (C1: 183,872 = 183,872); write parts ≤ sectors sent | exact for all 8 channels |
| device | the memory device of each run: HBM3 6.4 Gb/s, HBM3 3.2 Gb/s, GDDR6 14 Gb/s | as intended |
| hbf_count / hbf_csv | trace HBF requests = admitted = returned = controller requests = CSV INGRESS = CSV COMPLETED; no request completed twice | exact |
| hbf_channel / hbf_map | channel = page mod channels (per-channel counters and every CSV row); page = (addr − 256 GiB) / 4 KiB; sub-array formula on every NAND READ | 0 mismatches |
| return | every reply carries the L2 slice of its request's address | 0 mismatches |
| frontend | L2 exit → HBF controller, per request | 74 cycles (52.49 ns) for C2/C3, 88 cycles (62.49 ns) for C4, exactly |
| phy_bytes (C4) | bytes charged to the shared HBM PHY = HBM bytes sent + HBF bytes returned | exact (vectorAdd 1,310,720 + 5,894,912; q_proj 4,096 + 8,388,608) |
| gpu_same | GPU part of the assembled configuration in all runs | one SHA-256 |
| link_cap | HBF link 5 GB/s per channel | busiest channel 4.54 GB/s ≤ 5; 35.5 of 40 GB/s |
| phy_cap | C4 shared PHY 5 GB/s per partition | busiest partition 4.61 GB/s ≤ 5; 36.9 of 40 GB/s |
| hop | C4 hop +190 ns | per-request delay 88 → 356 cycles (+268 = 190 ns exactly); kernel +0.77% (hidden by the LHB) |

The sub-array check runs on prefetch-off runs, where demand reads issue NAND READs (1,439-1,440 per run).
With prefetch on, NAND reads are prefetch reads, which have no requester and therefore no CSV row.

Reproduce: `python3 tools/validate_routing.py --jobs 6` (about 30 min), report in `results/validation_routing/ROUTING_VALIDATION.md`.
