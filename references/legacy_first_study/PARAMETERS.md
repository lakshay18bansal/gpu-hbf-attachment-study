# Where every parameter lives (and how to change it)

The GPU, the HBM/DRAM devices and the HBF device are configured in **separate files**. The validation
suite (`results/validation/VALIDATION.md`, tests D1-D5) checks this separation:
- the built-in DRAM options have no effect on Ramulator runs;
- HBF options have no effect on C1;
- GPU options do change the result.

| What | File | Used by | How to change for one run |
|---|---|---|---|
| **GPU**: SMs, clocks, caches, NoC, number of memory partitions, address hashing | `common/gpu/qwen3_slice.config` | all | `-- -gpgpu_n_clusters 8` (any raw option after `--`) |
| (structural) `-gpgpu_dram_timing_opt nbk=16:...` in the GPU file | `common/gpu/qwen3_slice.config` | all | do not remove: `nbk` must match the 16 banks of `-gpgpu_mem_addr_mapping`, or GPGPU-Sim's per-bank statistics overflow. Its timing values are ignored under Ramulator |
| **HBM3** device (C1, C3-iso-stack, C4) | `common/memory/hbm3_slice8.py` (`EDIT HERE` block) | MEM=hbm3_slice8 | `--ram-set "DATA_RATE_MBPS=12800"` |
| **HBM3 at half rate** (C3-iso-shoreline) | `common/memory/hbm3_slice8_half.py` | MEM=hbm3_slice8_half | `--ram-set ...` |
| **GDDR6** device (C2) | `common/memory/gddr6_slice8.py` | MEM=gddr6_slice8 | `--ram-set ...` |
| GPGPU-Sim built-in DRAM (paper reproduction only) | `common/memory/builtin_qwen3.config` | MEM=builtin | – |
| **HBF device**: page, buffers, MSHR, FTL, tPROG/tBERS, LHB hit time, window base/size | `common/hbf_base.config` | C2, C3, C4 | `-- -gpgpu_hbf_<option> <value>` |
| **NAND read time tR** | command line | C2, C3, C4 | `--tR-us 20` |
| **Per-configuration memory organization** | `<config>/config.env` | that config | `--set VAR=VALUE` |
| Raw simulator options of one configuration | `<config>/config.overrides` | that config | `-- -opt val` |

## `config.env` variables (one file per configuration folder)

| Variable | Meaning | C1 | C2 | C3 iso-stack | C3 iso-shoreline | C4 |
|---|---|---|---|---|---|---|
| `MEM` | Ramulator config in `common/memory/` | hbm3_slice8 | gddr6_slice8 | hbm3_slice8 | hbm3_slice8_half | hbm3_slice8 |
| `RAM_SET` | overrides of that `.py` for this configuration only | "" | "" | "" | "" | "" |
| `HBF` | HBF tier present | 0 | 1 | 1 | 1 | 1 |
| `ATTACH` | 1 = own PHY (direct), 2 = behind HBM (shared PHY) | – | 1 | 1 | 1 | 2 |
| `HBF_CHANNELS` | HBF channels in the slice | – | 8 | 8 | 4 | 8 |
| `HBF_LINK_GBPS` | per HBF channel: link **and** media bandwidth (GB/s) | – | 51.2 | 51.2 | 51.2 | 51.2 |
| `PHY_GBPS` | C4: the HBM link per partition, shared by HBM and HBF bytes (H3's BW①) | – | 0 | 0 | 0 | 51.2 |
| `FRONTEND_NS` | fixed latency of every HBF request (GPU MC + PHY, plus the C4 hop) | – | 52.49 | 52.49 | 52.49 | 62.49 |
| `LHB_PAGES` | prefetch buffer (LHB), in 4 KiB pages (default 3000 = 12 MB) | – | 3000 | 3000 | 3000 | 3000 |

Derived automatically by `scripts/run_config.sh` (do not type these by hand):
- HBF bandwidth of the slice = `HBF_CHANNELS × HBF_LINK_GBPS`;
- NAND parallelism `max_active = ceil(bandwidth × tR / 4 KiB)`;
- sub-arrays per channel = max(32, max_active / channels);
- credits = bandwidth per partition × (tR + 5 µs) / 32 B;
- prefetch window = max_active.

Every run folder keeps the exact `gpgpusim.config`, `memory.yaml`, `memory_summary.txt` and `run_meta.json`, which records `ram_set` and `sets`.

## Common changes

| Goal | Command (or permanent edit) |
|---|---|
| HBM bandwidth ×2 | `C1_hbm_only/run.sh vectoradd --ram-set DATA_RATE_MBPS=12800`, or put `RAM_SET="DATA_RATE_MBPS=12800"` in `config.env`, or change `DATA_RATE_MBPS` in `common/memory/hbm3_slice8.py` (affects C1, C3-iso-stack and C4) |
| HBM CAS latency | `--ram-set ns.nCL=25` (any timing in ns: `ns.<name>`; in cycles: `<name>`) |
| HBM channels / banks | `--ram-set org.bank=8`. `CHANNELS` must stay 8 (= `-gpgpu_n_mem`) |
| HBF bandwidth ×2 | `C3_hbf_hbm_direct/iso_stack/run.sh vectoradd --prefetch on --set HBF_LINK_GBPS=102.4` |
| C4 shared-link bandwidth | `C4_hbf_behind_hbm/run.sh ... --set PHY_GBPS=25.6` |
| C4 base-die/D2D hop +200 ns | `--set FRONTEND_NS=252.49` |
| NAND tR = 20 µs with H3's LHB rule (2 × 409.6 GB/s × 20 µs) | `--tR-us 20 --set LHB_PAGES=4000` |
| GDDR6 bandwidth (C2) | `C2_hbf_dram/run.sh ... --ram-set DATA_RATE_MBPS=28000` |
| Number of SMs | `-- -gpgpu_n_clusters 24` |
