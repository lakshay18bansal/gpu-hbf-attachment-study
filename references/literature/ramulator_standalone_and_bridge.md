# Ramulator 2.1: standalone use, the Accel-Sim bridge, and what a second (HBF) tier would need

Scope: (1) run Ramulator 2.1 on its own from `accelsim-work/ramulator`, (2) explain exactly how our
Accel-Sim build talks to it, (3) list what has to change to put a second memory tier (HBF, or HBF
forwarded through the HBM base die) behind the same memory partition.

Labels used below: **[ran]** = I executed it and the numbers are copied from real output;
**[read]** = taken from source code (file:line) or a document; **[inferred]** = my reasoning, not
checked by running anything.

Nothing inside `<btp_work>/accelsim-work` or `<btp_work>/accel_sim_hbf` was
modified. All runs used `PYTHONDONTWRITEBYTECODE=1`, so no `__pycache__` was written, and
`git -C accelsim-work/ramulator status` lists the same 25 entries before and after. Everything I
built or produced is in
`<scratch>/code-ramulator-bridge/`.
That folder is called `$S` below. `$R` is `<btp_work>/accelsim-work/ramulator`, and
`$G` is `<btp_work>/accelsim-work/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim`.

---

## 0. Short answers

| Question | Answer |
|---|---|
| Is there a `ramulator2` executable? | **No.** Ramulator 2.1 has no C++ `main`/`add_executable` (`$R/CMakeLists.txt` builds only `libramulator.so` (line 73) and the nanobind module (lines 55-96)) **[read]**. The standalone entry point is **Python**: `python3 -m ramulator run <config.py>` (`$R/python/ramulator/__main__.py:1-30`), or a script that builds `ramulator.Simulation(frontend, mem)` and calls `.run()`. The prebuilt `$R/python/ramulator/_ramulator.cpython-312-*.so` works with the host's Python 3.12, so **no build is needed and no container is needed** **[ran]**. |
| Trace frontends | `LoadStoreTrace` (`LD|ST <addr>`), `ReadWriteTrace` (`R|W <c,r,b,...>` address vector), `SimpleO3` (`<bubbles> <load> [store]`, a core model with an LLC), `BHO3` (BlockHammer O3 variant), `LatencyThroughputTrace` (synthetic, no file), `External` (requests are injected from C++), and our `GPGPUSim` (External plus a fixed front-end latency). **There is no gem5 trace frontend**: gem5 is supported only as a live wrapper (`$R/resources/gem5_wrappers/`) **[read]**. |
| Config | A Python script builds component objects. `python3 -m ramulator export x.py -o x.yaml` turns them into fully resolved YAML (`$R/python/ramulator/export.py:7-48`). The YAML has two top-level sections, `frontend:` and `memory_system:` (`controllers: [...]`, each with `dram:` (org/timing/timing_constraints), `scheduler`, `refresh_manager`, `row_policy`, `addr_mapper`, `controller_plugins`) **[ran]**. |
| Output | A stats tree printed by `sim.stats_yaml` / `print_stats()`: per-controller cycles, row hits/misses/conflicts, reads and writes accepted/served/forwarded/coalesced, queue occupancy, `avg_read_latency` (controller ticks), throughput in MB/s, plus plugin stats (GPGPUSimCounter, Ayna energy) **[ran]**. |
| Accel-Sim ↔ Ramulator | **In-process function calls into `libramulator.so`, one call per 32 B transaction. No trace files.** `IMemorySystem::send(Request&)` is called with a `std::function` callback, `IMemorySystem::tick()` is called once per DRAM clock, and the callback retires the `mem_fetch` (`$G/ramulator_dram.cc:336-394, 414-422`) **[read]**. The only files involved are the YAML config read at startup and the stats files written at the end of each kernel. |
| Second tier | The bridge assumes **one** Ramulator memory system (singleton, `ramulator_dram.cc:50-54`) with **one** standard and **one** clock (`:104-106`), exactly `-gpgpu_n_mem` controllers (`:96-100`), and an address that is **wrapped modulo one channel's capacity** (`:305-306`). Each of these has to change, and `dram_interface::full(bool is_write)` (`$G/dram_interface.h:18`) needs the request's address so that backpressure can be applied per tier. Ramulator can model a flash device only as a new `DRAMSpec` (page sense = ACT, page buffer = row buffer, data out = RD). It cannot model the FTL, the 4 KiB page MSHR or program/erase management, and new request types need controller changes (`controller_base.cpp:173-176`). See section 3. |

---

## 1. Running Ramulator 2.1 standalone

### 1.1 Entry points

1. **Python CLI** (`$R/python/ramulator/__main__.py:1-15`) **[read]**:
   - `python3 -m ramulator [run] cfg.py` runs the script through `runpy`. The script itself calls `sim.run()`.
   - `python3 -m ramulator export cfg.py [-o out.yaml] [-f json]` runs the script with `Simulation.__init__/run`
     monkey-patched (`export.py:7-48`), so nothing is simulated and the resolved config is written out.
   - `codegen` regenerates C++ from the Python DRAM DSL. `visualize` starts the trace visualizer.
2. **Python API**: `ramulator.Simulation(frontend, memory_system)`, `.run()`, `.stats` (dict), `.stats_yaml`
   (text), `.finalize()` (`$R/python/ramulator/__init__.py:28-62`) **[read]**.
3. **C++ library** (`libramulator.so`): `Config::parse_config_file(yaml)`, `Factory::create_frontend/
   create_memory_system`, `connect_*`, then `send()`/`tick()` (README §7.2). This is the route Accel-Sim
   uses. I also used it for a small replay driver (section 1.6) **[ran]**.

Environment used **[ran]**: host Python 3.12.3, host g++ 13.3.0. The prebuilt module links to
`$R/libramulator.so` (checked with `ldd`). Command prefix:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<btp_work>/accelsim-work/ramulator/python
```

### 1.2 The simulation loop and what it means for traces

`Simulation::run()` (`$R/src/ramulator/python/bindings.cpp:46-68`) **[read]**:

```
fe_count = mem_ratio-1; mem_count = fe_ratio-1
loop: if ++fe_count >= mem_ratio: frontend.tick()      # frontend frequency : memory frequency
      if frontend.is_finished(): break                 #   = fe_ratio : mem_ratio
      if ++mem_count >= fe_ratio: memory_system.tick()
```

Consequences for the trace frontends (verified by the runs in 1.5):
- **There are no timestamps.** `LoadStoreTrace`/`ReadWriteTrace` offer **one request per frontend tick** and
  retry the same one if `send()` is rejected (`loadstore_trace.cpp:39-48`). The offered load is therefore
  set only by the clock ratio.
- **The run stops when the last request has been *sent*, not when it has been served.** `is_finished()` is
  `m_trace_count >= m_trace_length` (`loadstore_trace.cpp:109-111`). Requests still queued are never
  served. In the 1-channel stream run, 8,192 reads were sent but only 8,128 were served, and a 4-request
  `ReadWriteTrace` run stops after 3 cycles with 0 reads served **[ran]**. For latency studies use the
  C++ driver of section 1.6, which drains.
- The trace **replays cyclically** if the run did not stop (comments at `loadstore_trace.cpp:61`,
  `readwrite_trace.cpp:61`).

### 1.3 Frontends and their trace formats **[read]**

| Frontend (`ramulator.frontend.X`) | Source | Params | Trace line format | Notes |
|---|---|---|---|---|
| `LoadStoreTrace` | `src/ramulator/frontend/impl/memory_trace/loadstore_trace.cpp:51-61` | `clock_ratio`, `path` | `LD <addr>` or `ST <addr>`; the address is decimal or `0x` hex; exactly 2 tokens | Flat address → channel mapper → addr mapper. `size_bytes = tx_bytes` (32 B for HBM3). |
| `ReadWriteTrace` | `.../readwrite_trace.cpp:50-61` | `clock_ratio`, `path` | `R <i0,i1,...>` or `W <...>`: one integer per DRAM level, in the spec's level order | Use it with `PassThroughChannelMapper` + `PassThroughAddrMapper`. HBM3 level order is `Channel,PseudoChannel,Sid,BankGroup,Bank,Row,Column` (`python/ramulator/dram/hbm3.py:14-22`). |
| `SimpleO3` | `.../processor/simpleO3/simpleO3.cpp:14-26`, parser `core.cpp:26-46` | `clock_ratio`, `traces` (one file per core), `num_expected_insts`, `translation`, LLC params (`simpleO3.cpp:49-58`) | `<bubble_count> <load_addr> [store_addr]` | Out-of-order core and LLC. The comment says "decimal or 0x hex", but `core.cpp:36,40` uses `std::stoll(tok)` (base 10), so **use decimal addresses** **[inferred from std::stoll semantics, not run]**. |
| `BHO3` | `.../processor/bhO3/` | (not examined) | (not examined) | BlockHammer variant of the O3 core. |
| `LatencyThroughputTrace` | `.../memory_trace/latency_throughput_trace.cpp:77-87` | `nop_counter`, `num_probe_requests`, `stream_cls`, `warmup_cycles`, ... | none (synthetic) | Used by `tests/latency_throughput`. |
| `External` | `.../frontend/impl/external.cpp` | `clock_ratio` | none | `tick()` does nothing. The host calls `receive_external_requests(type, addr, src, cb, size)`. |
| **`GPGPUSim`** (ours) | `$R/src/ramulator/gpgpusim/gpgpusim_frontend.cpp:76-120` | `clock_ratio`, `fixed_latency_ns` | none | Same as `External`, plus the fixed front-end latency that the Accel-Sim bridge reads with `GPGPUSim::fixed_latency_ns()`. Accel-Sim does **not** use its `receive_external_requests`. It calls `IMemorySystem::send` directly (section 2). |

There is no gem5 trace reader. gem5 drives Ramulator through `resources/gem5_wrappers/ramulator2*.cc`
and `python/ramulator/gem5.py` **[read]**.

### 1.4 How the YAML config is produced, and its sections

Our memory configs are Python: `$R/configs/accelsim/{hbm3,gddr6,gpgpusim_builtin}_rtx2060s.py`, which
use the helpers in `accelsim_common.py` **[read]**:
- `apply_overrides()` (`accelsim_common.py:33-77`): applies the `EDIT HERE` dict and the `RAMULATOR_SET="K=V,..."`
  environment overrides (`ns.X`, `org.X`, timing names).
- `timings_at_rate()` (`:80-102`): resolves the preset (e.g. `HBM3_6400Mbps`) and re-expresses every timing
  at a new data rate. Timings keep their value in ns, except `CLOCK_TIMINGS` (`nBL`, `nCCDS`, `nCCDR`, `nPPD`),
  which keep their cycle count.
- `build_system()` (`:115-189`): `CHANNELS` controllers (one per GPU memory partition), each with
  `GPGPUSimCounter` (required by Accel-Sim) and optionally `Ayna`. Frontend is
  `GPGPUSim(clock_ratio=1, fixed_latency_ns=FIXED_LATENCY_NS)`, memory system is
  `GenericDRAM(clock_ratio=1, channel_mapper=PassThroughChannelMapper)`.
- `scripts/simulate.sh:147-160` copies the `.py`, sets `RAMULATOR_SET` and `GPU_CORE_CLOCK_MHZ`, runs
  `python3 -m ramulator export "$RAM_PY" -o "$OUT/memory_config_USED.yaml"`, and passes
  `-gpgpu_ramulator_enabled 1 -gpgpu_ramulator_config ./memory_config_USED.yaml`.

I exported `hbm3_rtx2060s.py` into `$S/out/hbm3_rtx2060s.yaml` (2,234 lines) **[ran]**. Its structure:

```yaml
frontend:                       # impl: GPGPUSim, clock_ratio: 1, fixed_latency_ns: 52.4934 (=100 cycles @1905 MHz)
memory_system:
  impl: GenericDRAM
  clock_ratio: 1
  channel_mapper: {impl: PassThroughChannelMapper}
  controllers:                  # 16 x (one per GPU channel)
    - impl: HBM34
      wr_low_watermark: 0.2 / wr_high_watermark: 0.8
      read_buffer_size: 64 / write_buffer_size: 64 / priority_buffer_size: 1568
      scheduler: {impl: GPGPUSimFRFCFS}
      refresh_manager: {impl: AllBank, ...}
      row_policy: {impl: Open}
      addr_mapper: {impl: GPUBgCoBaRo}
      dram:
        impl: HBM3
        org: {dq: 32, count: [1, 2, 2, 4, 4, 16384, 256]}  # Ch, PC, SID, BG, Bank, Row, Col
        timing: [6400, 4, 40, 62, 30, 52, 90, 142, 66, 18, 20, 4, 8, 6, 8, 10, 14, 20, 34, 48, 4,
                 1120, 640, 1120, 640, 26, 12480, 390, 312]
        command_cycles: [...]; channel_width: 64; read_latency: 44; data_payload_bytes: 32
        timing_constraints: [[level, [preceding cmds], [following cmds], latency, (window, sibling)], ...]
      controller_plugins: [{impl: GPGPUSimCounter}, {impl: Ayna, power: {...}}]
```

The `timing` vector follows `HBM3.timing_params` (`hbm3.py`): rate 6400, nBL 4, nCL 40, nRCDRD 62,
nRCDWR 30, nRP 52, nRAS 90, nRC 142, nWR 66, nRTP 18, nCWL 20, nCCDS 4, nCCDL 8, nCCDR 6, nRRDS 8,
nRRDL 10, nWTRS 14, nWTRL 20, nRTW 34, nFAW 48, nPPD 4, nRFC 1120, nRFCpb 640, nRFMab 1120,
nRFMpb 640, nRREFD 26, nREFI 12480, nREFIpb 390, tCK_ps 312. One "tick" is a **half CK** (312 ps),
because HBM3 uses `tick_multiplier = 2` (`hbm3.py:10`). `read_latency = nCL + nBL = 44` ticks
(`hbm3.py:11`). The YAML is meant to be regenerated, not edited by hand (README §7.1).

### 1.5 What I ran with the Python trace frontends

Config: `$S/configs/hbm3_standalone.py`. It uses the same HBM3 device and controller settings as
`hbm3_rtx2060s.py` (HBM3_16Gb_8hi, HBM3_6400Mbps, HBM34, 64/64 buffers, GPGPUSimFRFCFS, AllBank, Open,
GPUBgCoBaRo, GPGPUSimCounter), but the frontend is `LoadStoreTrace` (with `CacheLineInterleave`) or
`ReadWriteTrace` (with PassThrough mappers). Knobs: `TRACE`, `FRONTEND`, `CHANNELS`, `FE_RATIO`, `MEM_RATIO`,
`STATS_OUT`. Traces were generated by me into `$S/traces/`. All results are in `$S/out/*.yaml`, and
`$S/run_all.sh` reproduces everything (it takes a few seconds).

```bash
cd $S
TRACE=traces/stream_ld.trace CHANNELS=1 STATS_OUT=out/stream_1ch.yaml \
  nice -n 10 python3 -m ramulator run configs/hbm3_standalone.py
```

Real output of `out/stream_1ch.yaml` (8,192 sequential 32 B loads, 1 channel, 1 request offered per
312 ps tick, i.e. 102 GB/s offered against a 51.2 GB/s channel) **[ran]**:

```
memory_system:
  impl: GenericDRAM
  total_num_read_requests: 8192
  controller:
    impl: HBM34
    id: Channel 0
    cycles: 17665
    row_hits: 7864      row_misses: 128     row_conflicts: 136
    num_read_reqs: 8192 num_read_reqs_served: 8128   num_maintenance_reqs: 2 (refreshes)
    queue_len_avg: 63.8589   read_queue_len_avg: 63.8481
    avg_read_latency: 183.994          # controller ticks (x 0.312 ns = 57.4 ns, mostly queueing)
    read_throughput_MBps: 47191.7      # 92.2 % of the 51.2 GB/s channel peak
    scheduler: {impl: GPGPUSimFRFCFS}  refresh_manager: {impl: AllBank}  addr_mapper: {impl: GPUBgCoBaRo}
```

| Run (all HBM3, `$S/out/<name>.yaml`) | cycles | reads sent / served | row hit / miss / conflict | avg read lat (ticks) | throughput |
|---|---|---|---|---|---|
| `stream_1ch` (FE:MEM 1:1) | 17,665 | 8,192 / 8,128 | 7,864 / 128 / 136 | 184.0 | 47.19 GB/s |
| `stream_1ch_mem4` (FE:MEM 1:4, 25.6 GB/s offered) | 34,784 | 8,192 / 8,191 | 7,919 / 192 / 80 | 73.9 | 24.15 GB/s |
| `random_1ch` (random in 1 GiB) | 55,123 | 8,192 / 8,116 | 2 / 218 / 7,934 | 539.3 | 15.10 GB/s |
| `mix_1ch` (3 LD : 1 ST) | 27,384 | 6,144 R + 2,048 W / 6,080 R + 2,000 W | 7,802 / 146 / 132 | 333.4 | 30.26 GB/s total |
| `stream_16ch_fe8` (16 ch, FE:MEM 8:1) | 1,024 | 512 per channel / ~475 | ~460 / 16 / 0 per channel | ~111 | ~47.6 GB/s per channel |
| `rw_vec` (ReadWriteTrace, 4 requests) | 3 | 3 R + 1 W / **0** | – | 0 | 0 (loop stopped before any completion) |

The last row shows the drain problem of section 1.2.

### 1.6 Bridge-style standalone replay (C++, drains, timestamps, the exact Accel-Sim YAML)

Because of the limits in 1.2, I wrote a 190-line driver, `$S/driver/ramtrace.cpp`. It talks to
`libramulator.so` **the same way `ramulator_dram.cc` does**. It loads an exported YAML (here the
unchanged `hbm3_rtx2060s.py` export), builds `Request{addr_vec={channel}, addr=channel-local address,
size_bytes=32, callback}`, calls `IMemorySystem::send()` directly, retries rejected requests on the next
tick, delays each request by the `GPGPUSim` frontend's `fixed_latency_ns` (168 ticks), ticks once per
controller tick, and **keeps ticking until every request has completed**. It was built with the host
compiler against the existing library, with no rebuild of the Ramulator tree **[ran]**:

```bash
g++ -std=c++20 -O2 -I$R/src driver/ramtrace.cpp -o driver/ramtrace $R/libramulator.so -Wl,-rpath,$R
./driver/ramtrace out/hbm3_rtx2060s.yaml traces/drv_probe.txt out/drv_probe.lat.csv out/drv_probe.stats.yaml
```

Trace format: `<tick> <R|W> <channel> <channel-local byte address>`. This is what the bridge hands to
Ramulator after GPGPU-Sim has chosen the partition.

Isolated probes on channel 0, 3,000 ticks apart (`out/drv_probe.lat.csv`) **[ran]**:

| Probe | sent → done (ticks) | DRAM part (ticks) | Expected from timings |
|---|---|---|---|
| R 0x0 (closed bank: row miss) | 168 → 276 | 108 (33.7 ns) | nRCDRD 62 + nCL 40 + nBL 4 = 106; 2 extra ticks not explained (not investigated) |
| R 0x100 (same row: row hit) | 3168 → 3212 | 44 (13.7 ns) | read_latency = nCL + nBL = 44, exact |
| R 0x400000 (row 64, same bank: conflict) | 6168 → 6326 | 158 (49.3 ns) | nRP 52 + 62 + 44 = 158, exact |
| W 0x800000 | 9168 → 9250 | 82 | callback fires when the WR *issues* (PRE 52 + nRCDWR 30); see 2.6 |
| R 0x800000 (row hit after the write) | 12168 → 12212 | 44 | 44 |

The total GPU-visible latency for a row hit is 212 ticks = 66.1 ns, of which 168 ticks (52.4 ns) is the
fixed front-end latency. Ramulator's own `avg_read_latency` for this run is 89.5 ticks (from controller
arrival to depart).

Saturation, 16 channels **[ran]**:
- `drv_stream16` (16 × 4,096 sequential 32 B reads, all offered at once): **790.50 GB/s = 96.5 %** of the
  819.2 GB/s peak (16 × 64 bit × 6.4 Gb/s ÷ 8). There were 129,456 `send()` rejections (backpressure works),
  and each channel issued 4,096 RD, 128 ACT and 64 PRE.
- `drv_mix16` (3 R : 1 W): 471.66 GB/s, 2 REF per channel. I did not analyse why this is lower. Ramulator
  drains writes in batches between the 0.8/0.2 watermarks (`controller_base.cpp:403-411`); that is the
  likely cause **[inferred]**.

This driver is a cheap way to test any future Ramulator HBM/HBF config, or a request stream dumped from
Accel-Sim, without running the GPU model.

---

## 2. How Accel-Sim talks to Ramulator in our integration

**Answer: in process, one function call per 32 B transaction, driven by GPGPU-Sim's DRAM clock.
No trace file is created.** Accel-Sim links `libramulator.so` (`$G/CMakeLists.txt` diff: `target_link_libraries(gpgpusim
PUBLIC ${RAMULATOR_ROOT}/libramulator.so)` plus `-rpath`). Only `ramulator_dram.cc` is compiled as C++20,
and it is compiled only with `-DGPGPUSIM_USE_RAMULATOR=ON` **[read]**. The binary is
`accel-sim-framework/gpu-simulator/bin/ramulator/accel-sim.out` (`scripts/build_accelsim_ramulator.sh`).

(The diff also adds include paths `${RAMULATOR_ROOT}/build/_deps/{yaml-cpp,fmt}-src/include`, which do
not exist; the sources are in `ramulator/ext/`. This does no harm, because none of the Ramulator headers
the bridge includes pulls in yaml-cpp or fmt **[read]**.)

### 2.1 The pieces

| Piece | Where | Role |
|---|---|---|
| `dram_interface` (abstract) | `$G/dram_interface.h:13-43` | `full(is_write)`, `push(mf)`, `cycle()`, `return_queue_top/pop()`, stats, plus two hooks: `max_outstanding()` (`:38`) and `has_own_latency()` (`:42`) |
| `dram_t : dram_interface` | `$G/dram.h` diff | GPGPU-Sim's own model, now behind the interface (only `override`s added) |
| `ramulator_dram_t : dram_interface` | `$G/ramulator_dram.h:78-106`, `.cc:449-471` | Thin wrapper over `ramulator_channel` |
| `ramulator_channel` (one per partition) | `.cc:164-412` | Pending queue (front-end latency), 32 B split, send, outstanding map, return queue, stats |
| `ramulator_memory` (singleton) | `.cc:48-159, 414-441` | Owns the one Ramulator `IFrontEnd` + `IMemorySystem`, the clock, the tick, and stats dumps |
| Partition hookup | `$G/l2cache.cc:97-104` | `m_dram = new ramulator_dram_t(...)` if `-gpgpu_ramulator_enabled`; `set_max_outstanding(m_dram->max_outstanding())` |
| Options | `$G/gpu-sim.cc:249-257`, `gpu-sim.h:221-237, 443-444` | `-gpgpu_ramulator_enabled`, `-gpgpu_ramulator_config`; abort if built without Ramulator or combined with the simple DRAM model |
| Clock | `$G/gpu-sim.cc:1362-1370` | `dram_period = ramulator_dram_tck_ns() * 1e-9` |
| Channel-local address | `$G/addrdec.cc:96-107`, `addrdec.h:75-78` | `channel_address()`: removes only the channel bits and keeps the sub-partition bit |
| Ramulator-side components | `$R/src/ramulator/gpgpusim/` | `GPGPUSim` frontend, `GPGPUSimCounter` plugin (per-channel ACT/PRE/RD/WR/REF, tCK, buffer sizes), `GPGPUSimFRFCFS` scheduler, `GPUBgCoBaRo` mapper |
| Ramulator core change | `$R/src/ramulator/controller/controller_base.h` diff | Only two read-only getters, `get_read_buffer_size/get_write_buffer_size`. The `dram/impl/*.cpp` diffs are codegen re-formatting: for `HBM3.cpp` I checked that the token multiset is identical apart from line wrapping. |

### 2.2 Initialisation (`ramulator_memory::ramulator_memory`, `.cc:78-159`) **[read]**

1. Parse the YAML (`:85`), then `Factory::create_frontend` and `create_memory_system`, then connect them (`:86-89`).
2. `RG::channels()` is a global registry filled by each `GPGPUSimCounter` plugin's `init()`
   (`gpgpusim_counter.cpp:32`). The bridge clears it first (`:84`) and copies it afterwards (`:90`).
3. Checks: the registry is not empty (`:93-95`); **number of controllers == `-gpgpu_n_mem`** (`:96-100`);
   controller *i* has channel id *i* (`:102-103`); **every channel has the same standard and tCK**
   (`:104-106`); `tx_bytes >= 32` (`:107-108`); the frontend must be `GPGPUSim` (`:110-113`).
4. `m_tck_ns` = controller tick (0.312 ns for HBM3 = 3205.1 MHz). `m_latency_ticks = round(fixed_latency_ns / tCK)`
   (`:115-117`), which is 168 ticks for 52.49 ns.
5. `m_channel_bytes` = tx_bytes × product of all levels below Channel, with Column divided by the prefetch
   (`:119-125`). For HBM3 this is 1024 MB per channel.
6. If the mapper is `PassThroughAddrMapper`, GPGPU-Sim's own bank/row/col decode is used (`:127-135`).
   This works only for Channel/BankGroup/Bank/Row/Column organisations, so not for HBM3.

A real Accel-Sim HBM3 run prints this (`accelsim-work/cuda/simulations/va_check_hbm3_rtx2060s/simulation.log:284-288`) **[read]**:

```
Ramulator DRAM: ./memory_config_USED.yaml
  16 channels of HBM3, 1024 MB each, controller tick 0.3120 ns (3205.1 MHz)
  address mapper GPUBgCoBaRo, read/write buffers 64/64, fixed front-end latency 52.493 ns (168 ticks)
GPGPU-Sim uArch: clock freqs: 1905000000.000000:1905000000.000000:1905000000.000000:3205128138.998793
```

### 2.3 Clocking

- `init_clock_domains()` replaces the DRAM clock of `-gpgpu_clock_domains` with Ramulator's tick
  (`$G/gpu-sim.cc:1362-1370`). `gpgpu_sim::next_clock_domain()` (`gpu-sim.cc:2217-2238`) then raises the
  `DRAM` mask once per tCK.
- On a DRAM edge, `gpgpu_sim::cycle()` calls `memory_partition_unit::dram_cycle()` for every partition
  *i* = 0..n−1 (`gpu-sim.cc:2314-2320`).
- Each partition's `dram_cycle()` calls `m_dram->cycle()` (`l2cache.cc:380`), which becomes
  `ramulator_channel::cycle()` (`ramulator_dram.cc:193-196`). That function first sends ready requests of
  this channel (`send_ready()`). Then, **only for the last channel**, it calls `ramulator_memory::tick()`
  → `m_memsys->tick()` once for all channels, increments `m_clk`, and updates per-channel
  busy/active counters (`:414-422`).
- So one GPGPU-Sim DRAM cycle equals exactly one Ramulator controller tick (one HBM3 half-CK). Within a
  cycle, all channels' sends happen before the shared tick.

### 2.4 Request lifecycle (one L2 miss or write-back)

```
L2 sub-partition miss/write-back  ──► L2_dram_queue (fifo, per sub-partition)
      │  memory_partition_unit::dram_cycle()  [DRAM clock]           l2cache.cc:353
      │  (1) arbitration over sub-partitions, needs a credit:
      │      can_issue_to_dram(spid) (l2cache.cc:246-257) && !m_dram->full()   :392-395
      │      → m_dram_latency_queue, ready_cycle = now + 0 (has_own_latency)   :400-409
      │      borrow_credit(spid); at most ONE request per DRAM cycle            :410-411
      │  (2) latency-queue head → m_dram->push(mf) if !full()                  :415-421
      ▼
ramulator_channel::push(mf)                                                ramulator_dram.cc:177-189
      m_pending.push_back({mf, ready_tick = m_clk + m_latency_ticks})  ← fixed front-end latency
      │  ramulator_channel::cycle() → send_ready()                         :336-372
      │  while front is ready: for each 32 B part p of mf:
      │     Request(addr=0, Read|Write, source=-1, callback=[mf]{part_done(mf)})
      │     make_request(): req.addr = channel_address(part_addr) % channel_bytes, 32B aligned
      │                     req.size_bytes = 32; req.addr_vec = {channel id}  :302-334
      │     m_memsys->send(req)   ── false → stop, retry same part next tick (backpressure) :358-362
      ▼
Ramulator GenericDRAM::send (generic_dram_system.cpp:53-80)
      PassThroughChannelMapper: intra_channel_addr = addr; channel = addr_vec[0]
      ControllerBase::send (controller_base.cpp:134-188)
        addr_mapper (GPUBgCoBaRo) fills PC/SID/BG/Bank/Row/Col from intra_channel_addr
        Read hitting a buffered write → forwarded, depart = now+1                :142-153
        Write to an already-buffered address → coalesced, callback immediately   :163-170
        else enqueue in read_buffer / write_buffer (64/64); full → return false
      HBM34 tick: one column-bus and one row-bus command per tick
        (hbm_controller_base.cpp:16-21, 45-107), scheduler GPGPUSimFRFCFS
      retire_request (controller_base.cpp:221-247):
        Read  → depart = clk + read_latency (44), callback when depart is reached
                (serve_completed_reads, :385-401)
        Write → callback called right away when WR issues
      ▼
callback → ramulator_channel::part_done(mf): --parts; on 0 → retire(mf)   ramulator_dram.cc:374-394
      reads: set_reply(), push to m_returnq
      L1/L2 write-backs: m_mp->set_done(mf) (returns the credit), delete mf
      ▼
memory_partition_unit::dram_cycle() (next DRAM cycle), l2cache.cc:356-378:
      return_queue_top() → dram_L2_queue of the original sub-partition (if not full),
      return_credit(); at most ONE return per DRAM cycle → L2 fill → icnt → SM
```

### 2.5 Splitting into 32 B transactions (`ramulator_dram.cc:278-299`)

- `n_parts = ceil(data_size / 32)`, at least 1. Sector reads from L2 are 32 B, so 1 part. L2 write-backs
  can be up to 128 B (`:7-8`).
- When there is more than one part and the sector mask has exactly `n` bits set, part *p* goes to the *p*-th
  set sector of the 128 B line (`(addr & ~127) + i*32`). Otherwise it goes to `addr + p*32`. A single part
  is aligned down to 32 B.
- The `mem_fetch` is registered in `m_outstanding[mf] = n` **before** its first send (`:346-349`), because
  Ramulator can call the callback from inside `send()` when a write coalesces. The `mem_fetch` retires
  when all `n` callbacks have fired.
- Partial progress is kept in `m_next_part`. If `send()` rejects part *k*, the loop returns and
  continues with part *k* on the next tick (`:359-362, 367-370`).

### 2.6 Address → channel → bank/row/column

- **Channel (memory partition) is decided by GPGPU-Sim**, when the `mem_fetch` is created:
  `addrdec_tlx()` (`$G/addrdec.cc:109-`, called from `mem_fetch.cc:78/125`) applies
  `-gpgpu_mem_addr_mapping` (RTX2060S: `dramid@8;...RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS`) and
  `-gpgpu_memory_partition_indexing` to get `chip`/`sub_partition`. Partition *i* = Ramulator controller *i*.
  If a request still lands on another channel, the run stops (`ramulator_dram.cc:363-365`).
- **Channel-local address**: `channel_address()` (`addrdec.cc:96-107`) removes only the channel bits
  (or in gap mode `(addr>>ADDR_CHIP_S)/n_channel`). Then `% m_channel_bytes` (`ramulator_dram.cc:305-306`)
  **wraps** addresses larger than one channel's capacity. This keeps any trace legal, but it would alias
  a large second-tier address range into HBM rows (section 3).
- **Inside the channel, Ramulator's address mapper decides.** With `GPUBgCoBaRo`
  (`$R/src/ramulator/gpgpusim/gpu_bg_co_ba_ro.cpp:1-58`), starting from the LSB of `addr >> log2(32)`, the
  order is `PseudoChannel`, `BankGroup`, then `Column` (5 bits for HBM3: 256 cols / prefetch 8), then the
  other levels (`Sid`, `Bank`), then `Row`. Every bank-level index is then XORed with the low row bits.
  The probes in 1.6 confirm this: 0x100 is the next column of the same row, and 0x400000 (row 64, above
  the 6 XOR bits) is the same bank in another row.
- With `PassThroughAddrMapper`, GPGPU-Sim's own bank/row/col decode is used instead
  (`ramulator_dram.cc:313-333`). This is the validation mode for GDDR6-like organisations.

### 2.7 Backpressure and credits

- `ramulator_channel::full()` = `m_pending.size() >= read_buffer + write_buffer` (128 for 64/64), and
  `is_write` is ignored (`ramulator_dram.cc:191, 215-217, 457`). The pending queue is the front-end
  latency pipe, so at most 128 `mem_fetch` per channel can be in that pipe.
  **[inferred]** With 168 ticks of latency and an HBM3 channel that needs about 1 request every 2 ticks
  (32 B / (51.2 GB/s × 0.312 ns)), about 84 requests must be in flight to keep the channel busy.
  128 is enough, so this limit does not cap HBM3 bandwidth at 6.4 Gb/s. It might at much higher data rates.
- Partition credits: `set_max_outstanding(2 × (read+write buffers)) = 256` per partition
  (`ramulator_dram.cc:218`, `l2cache.cc:162-168`). Each sub-partition keeps 1 private credit and the
  rest is shared. A credit is borrowed when the request enters the latency queue (`l2cache.cc:410`) and
  returned when the read reaches `dram_L2_queue` (`:367`) or a write-back is `set_done` (`:425-433`).
- Ramulator side: `send()` returns false when the controller's read or write buffer is full
  (`controller_base.cpp:157-181`). The bridge then retries on the next tick, with head-of-line order
  per channel.
- Per-cycle limits in the partition: 1 request L2 → latency queue, 1 latency queue → `push`, and 1
  return per DRAM cycle (`l2cache.cc:356-421`). At 3.2 GHz that is about 102 GB/s per partition in each
  direction **[inferred]**.

### 2.8 Front-end latency

`-dram_latency` is skipped because `has_own_latency()` is true (`l2cache.cc:403-404`, `ramulator_dram.h:102`).
Instead every `mem_fetch` waits `m_latency_ticks` DRAM ticks in `m_pending` (`ramulator_dram.cc:178, 337-338`).
The value comes from the Ramulator config: `FIXED_LATENCY_NS = core_cycles_to_ns(100)` = 52.49 ns at 1905 MHz
(`hbm3_rtx2060s.py:36`, `accelsim_common.py:22-30`). The guide notes that this value is Accel-Sim's
calibrated `-dram_latency` and has not been re-calibrated for Ramulator's own timings
(`RAMULATOR_GUIDE.md` §9).

### 2.9 Stats

- Per kernel, `ramulator_dram_t::print` writes a block in `dram_t`'s format into `simulation.log`
  (`ramulator_dram.cc:239-266`), e.g. `DRAM[0]: Ramulator 2.1 HBM3, tick=0.3120ns, burst=4 ticks,
  2 data bus(es) ... n_act n_pre n_ref_event n_req n_rd n_write n_wr_bk bw_util ... Row_Buffer_Locality`.
- The last channel's print at kernel end writes Ramulator's own stats tree to `ramulator_stats.txt` and
  appends to `ramulator_stats_history.txt` (`:430-441`). This is the same format as `sim.stats_yaml` above.
- AccelWattch gets ACT/PRE/RD/WR counts and busy ticks through `set_dram_power_stats` (`:222-235`).

---

## 3. What a second memory tier needs

### 3.1 Bridge assumptions that break with a second tier

| # | Assumption (file:line) | Why it breaks | Needed change **[inferred]** |
|---|---|---|---|
| 1 | One global `ramulator_memory` (`ramulator_dram.cc:50-54`) built from one `-gpgpu_ramulator_config` | A second tier needs its own device and controllers | Either (a) one Ramulator system holding both tiers' controllers (simplest, see 3.2-A), or (b) a non-singleton class instantiated per tier. For (b), `RG::channels()` is a single global registry (`gpgpusim.h:54-56`): clear it, create system 1, copy it, clear it, create system 2. |
| 2 | Controllers == `-gpgpu_n_mem` (`:96-100`); controller *i* = partition *i* (`:102-103, 363-365`) | HBF controllers are not GPU partitions in C2/C4 | Compare only the HBM controllers with `n_mem`. Map HBF requests to HBF controller `n_mem + f(addr)` (N:1 or 1:1). Relax the "wrong channel" check for that tier. |
| 3 | Same standard and same tCK for all channels (`:104-106`); DRAM clock domain = that tCK (`gpu-sim.cc:1362-1370`) | HBF is not HBM3, and a different clock is possible | Allow different standards. Keep **one tick** by expressing HBF timings in HBM ticks (tR of 15 µs = 48,077 ticks at 312 ps, which fits Ramulator's int timings), **or** add a fifth clock domain in `next_clock_domain()` (`gpu-sim.cc:2217-2238`) and tick the HBF system there. (`GenericDRAM` itself requires all its controllers to share one frequency, README §3.2.) |
| 4 | `m_channel_bytes` from controller 0, and address `% m_channel_bytes` (`:119-125, 305-306`) | A 256 GB HBF window would wrap into HBM rows | Compute the capacity per tier. Subtract the tier base (`hbf_base`) before mapping. Choose the tier by address first, the way HBF-Sim's `is_hbf_addr()` does. |
| 5 | `dram_interface::full(bool is_write)` (`dram_interface.h:18`) is address-blind, and `ramulator_channel::full()` is one queue limit (`:191`) | A full HBF queue must not block HBM requests (and the reverse) | Change it to `full(const mem_fetch*)` (callers at `l2cache.cc:328, 395, 418`) and keep per-tier pending queues. Otherwise one slow HBF request at the head of `m_pending` blocks all later HBM requests of that partition. HBF-Sim has exactly this head-of-line coupling (`HBF_SIM_REPORT.md` §2.2). |
| 6 | One credit pool per partition, `2 × (rd+wr buffers)` (`:218`, `l2cache.cc:162-168`) | An HBF read holds a credit for µs (HBF-Sim: about 15 µs) and starves HBM traffic | Add separate credit pools per tier (a new `max_outstanding(tier)`), or accept the coupling on purpose and report it as part of the design. |
| 7 | 32 B split, one Ramulator request per part (`:278-299, 336-372`) | Flash reads are 4 KiB page operations. 128 separate 32 B requests to one page are fine only if the device model merges them (page buffer = row buffer, see 3.2-B). It needs an MSHR/aggregation step if it does not. | For a Ramulator-HBF spec, the page-buffer "row hit" already merges them. For an HBF-Sim-style model, hand the whole `mem_fetch` to its page MSHR, as HBF-Sim does. |
| 8 | Fixed front-end latency is one number (`:110-117`) | C3/C4 have different paths: a direct HBF PHY, or HBM PHY + base die + HBF link | Use per-tier latency, e.g. `fixed_latency_ns` per frontend/system, and for C4 an extra base-die forwarding latency. |
| 9 | Stats and power per channel (`:222-266`) | They must be separable per tier | Print a second block per tier and write the stats file per system. |
| 10 | The tier is not visible on the GPU side for C3 | In C3 (HBF on its own GPU channels, CPU/driver decides the split), HBF channels are separate partitions with their own L2 slices. GPGPU-Sim's `addrdec_tlx()` interleaves **all** partitions by address bits (`addrdec.cc:109-`). | Change the chip decode so that the HBM range hashes over HBM partitions only and the HBF range over HBF partitions only (a range check before `addrdec_packbits`). The bridge's 1:1 mapping can stay, but rule 3 (same standard) must go. |

### 3.2 Can Ramulator itself model HBF, or an HBM channel that also carries HBF traffic?

**A. HBF as a new Ramulator "DRAM" standard (read-dominated tiers).** **[read + inferred]**
Ramulator standards are Python DSL classes that are code-generated to C++ (README §8.3-8.4,
`python/ramulator/dram/spec.py`). An `HBF` standard could map NAND onto the DRAM state machine:

| NAND | Ramulator concept |
|---|---|
| page sense (tR) | `ACT` with `nRCD` = tR |
| page buffer | row buffer (a row hit = a page-buffer hit) |
| data out over the channel | `RD` with `nCL`/`nBL` for the link |
| discard page buffer | `PRE` |
| program | `WR` with very large `nWR`/new `nPROG` |
| erase-before-write | needs a new command + per-block state |

Levels could be `Channel/Die(Plane)/Block/Page/Column`. That gives command-level timing, bank-level
parallelism, FR-FCFS reordering and per-channel bandwidth for free. For **LLM decode, the weights in HBF
are read-only**, so the read path is what matters most. What Ramulator cannot give without new C++:
- FTL / logical→physical remapping,
- the 4 KiB write aggregation and the full-page program rule,
- garbage collection or erase scheduling,
- a device-wide `max_active` (power) limit,
- a page cache on the logic die.

HBF-Sim models all of these (`HBF_SIM_REPORT.md` §3). So a Ramulator-HBF spec is a reasonable
first-order **read** model. HBF-Sim's model is the better choice when writes or media management matter.

**B. Custom request types.** `Request::Type` reserves 0 = Read and 1 = Write. Other non-negative ids are
"metadata for future extensions" (`$R/src/ramulator/base/request.h:18-24`). The DSL's `supported_requests`
must start with `Read, Write` and may add more (`spec.py:420-430`). However, `ControllerBase::send` **throws** for any other `type_id`
(`controller_base.cpp:173-176`), and `retire_request` handles only 0/1/−1 (`:221-247`). A new type
therefore needs controller changes: decide which buffer it goes into, and complete it like a read.

**C. C4 (H3-style: HBF behind the HBM base die), modelled in Ramulator.** **[inferred; I did not read the
H3 paper in this task, the description comes from the task statement]** What C4 adds is that HBF data
crosses the **same HBM channel (PHY/TSV data bus)** as HBM data, and the base die decides by address
where a request goes. A faithful Ramulator model needs both:
1. **A forwarding command on the HBM channel**: derive `HBM3Fwd(HBM3)` (README §8.3 pattern) with a column
   command `FWD` (and `FWDW`) on the `PseudoChannel` level. It has no bank prerequisite but the same
   data-bus constraints as `RD` (`nBL`, `nCCDS/L`, RD↔FWD turnaround), and a new request type
   `"Forward" → FWD` (needs the controller changes of B). An HBF read then uses (i) the HBF device (system A
   or HBF-Sim) for tR and the link, and **then** (ii) one `FWD` burst per 32 B on the HBM pseudo-channel
   it returns through. This way HBF traffic takes real bandwidth from HBM traffic and shares the HBM
   controller's queues, which is the C4 effect.
2. **The base-die decode in the bridge**: in `ramulator_channel::send_ready()`, tier = f(address). HBM
   requests go straight to the HBM controller. HBF requests go to the HBF model and, on completion, are
   re-injected as `FWD` requests into the HBM controller of the same partition (callback chain:
   HBF done → `send(FWD)` → FWD callback → `part_done`). The extra base-die/HBF-link latency is a delay
   stage inside the bridge.

   A cheaper approximation without new commands: on HBF completion, issue a dummy 32 B **Read** to a
   reserved HBM row region. This takes the data bus but also takes bank time (ACT/row hits), which
   distorts results a little. I would use it only as a first check.
3. Ramulator could also be given a **second device inside one controller**, but nothing in its design
   supports that: one `ControllerBase` owns one `DRAMDevice` (`m_device`), and the per-standard spec is
   compiled in. Two controllers sharing one data bus are not modelled either. Option 1 (a new command
   on the HBM channel) is the natural way to do it in Ramulator.

**D. What this means for the four configurations** **[inferred]**:
- **C1 (HBM only)**: works today (`hbm3_rtx2060s.py` and the bridge as is).
- **C2 (HBF + DRAM)**: HBF behind the partition, chosen by address. Use HBF-Sim's model, or a Ramulator-HBF
  tier with changes 1-9 above. The DRAM tier stays Ramulator (realistic) instead of HBF-Sim's `dram_t`.
- **C3 (HBF and HBM both on the GPU, split decided by host/driver)**: separate partitions. Needs change 10
  (range-aware chip decode) plus heterogeneous controllers (relax 3). Each tier keeps its own credits and
  queues naturally, because the tiers are different partitions.
- **C4 (HBF behind HBM, H3-style)**: C2's routing, plus `FWD` bursts on the HBM channel (3.2-C) and a
  base-die latency.

---

## 4. Files produced

All under
`<scratch>/code-ramulator-bridge/`:

| File | What |
|---|---|
| `configs/hbm3_standalone.py` | Standalone HBM3 config with `LoadStoreTrace` / `ReadWriteTrace` (env knobs) |
| `traces/{stream_ld,random_ld,mix_ldst}.trace`, `traces/rw_vec.trace` | 8,192-line LD/ST traces and a 4-line address-vector trace |
| `driver/ramtrace.cpp` (+ built `driver/ramtrace`) | Bridge-style C++ replay driver: timestamps, front-end latency, drains, per-request latency CSV |
| `traces/drv_{probe,stream16,mix16}.txt` | Driver traces (`tick op channel addr`) |
| `out/hbm3_rtx2060s.yaml` | Export of the unchanged Accel-Sim HBM3 config |
| `out/*.yaml`, `out/*.lat.csv`, `out/*.stats.yaml`, `out/run_all.log` | All raw outputs quoted above |
| `run_all.sh` | Reproduces every run (host only, a few seconds) |
