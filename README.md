# gpu-hbf-attachment-study: four ways to attach High Bandwidth Flash to a GPU

A cycle-level comparison of four GPU memory organizations for LLM inference, with the model
weights in **High Bandwidth Flash (HBF)**. One simulator, built from **Accel-Sim / GPGPU-Sim**
(GPU), **Ramulator 2** (HBM) and **HBF-Sim** (the flash device), runs all four; a configuration
file selects the organization. Workloads: one full **Qwen3-1.7B decode token** (1,819 kernels,
all 3.44 GB of weights in HBF) and **vectorAdd**.

| | Organization | How a flash request travels |
|---|---|---|
| **C1** | GPU + HBM only (baseline) | No flash: every address is HBM |
| **C2** | GPU + DRAM + HBF, **HBF-Sim as published** | The GPU memory controller sends flash requests through the DRAM queue; they share its 257 request slots |
| **C3** | GPU + HBM and HBF **side by side** | The GPU memory controller sends flash requests to a separate HBF port with its own queue and slots |
| **C4** | **H3**: HBF **behind the HBM base die** (Ha et al., IEEE CAL 2026) | The GPU is unchanged and sees only HBM. The HBM base die routes flash addresses over a 10 ns die-to-die hop to the HBF stack; a 16 MiB **latency-hiding buffer (LHB)** in the base die holds the next weights |

## Main result (one Qwen3-1.7B decode token, tR = 15 µs)

| Model | ms/token | tokens/s | vs C1 | Weight bandwidth |
|---|---|---|---|---|
| C1 HBM only | 29.33 | 34.1 | 1.00× | 117 GB/s |
| C2 shared controller | 236.03 | 4.2 | 8.05× | 14.6 GB/s |
| C3 side by side | 91.85 | 10.9 | 3.13× | 37.5 GB/s |
| C4 H3 (LHB 16 MiB) | 29.11 | 34.4 | 0.99× | 118 GB/s |

- Without latency hiding (C2, C3) a token is 3–8× slower: the GPU's request slots, sized for
  DRAM latency, cannot keep enough µs-latency flash reads in flight (`results/qwen3_layer13/*slots*`).
- H3 keeps HBM speed: 95.5% of flash requests hit the LHB and every weight page is read from
  NAND once. **This assumes, as the H3 paper does (Sec. III-C), that the order of weight reads is
  known in advance**; the simulator gives the LHB that order from the trace, so C4 is a best case.
- Once latency is hidden, the flash array's bandwidth (sub-arrays × 4 KiB / tR ≈ 140 GB/s here)
  is the only limit (tR 20 µs, 256 sub-arrays, lm_head).

**All results, sweeps and per-run files: [results/README.md](results/README.md).**

---

## For AI agents and maintainers: where everything is

| What | Where |
|---|---|
| Every result table and how each number is derived | `results/README.md` |
| One row per run (cycles, time, vs C1, HBF GB/s, LHB hits) | `results/summary.csv` |
| Per-run record (config, summary, per-kernel cycles, full log `run.log.xz`) | `results/<group>/<run>/` |
| The exact configuration of any run | `results/<group>/<run>/gpgpusim.config` + `memory.yaml` |
| How a configuration is assembled | `models/run.sh` (header) |
| Shared and per-model settings | `models/base/hbfsim_qwen3.config` (HBF-Sim, verbatim), `models/base/common.config`, `models/C*/config.overrides` |
| HBM device | `models/memory/hbm2e_slice8.py` (exported to `memory.yaml` at every run) |
| The list of experiments (run folder ↔ arguments) | `models/run_experiments.sh` |
| Validation checks and their output | `models/validate/check.py`, `results/vectoradd_validation/VALIDATION.md` |
| Simulator changes, as patches against upstream | `simulator/patches/` |
| Simulator source (modified) | `simulator/accel-sim-framework/gpu-simulator/gpgpu-sim/src/gpgpu-sim/` (`hbf_*` files, `l2cache.cc`, `ramulator_dram.*`, `dram_interface.h`) |

Conventions that hold everywhere:
- **Run folder name** = `<C>__<workload>[__tR<µs>][__lhb<MiB>][__sa<n>][__slots<n>]`; a missing
  tag means the default: tR 15 µs, LHB 16 MiB (C4 only), 512 sub-arrays, the model's own slots.
- **C4 always has its LHB** (it is part of H3); C1–C3 never have one. The LHB size is fixed
  (`--lhb-mib`, default 16) and does not change with tR. It is filled half at a time.
- **Time** = cycles / 1410 MHz (core clock). **vs C1** = cycles / C1 cycles of the same workload.
- **Numbers are read** from the last statistics block of `run.log` (or `run_stats.log`) by
  `models/validate/summarize.py`; never from intermediate kernels.
- **Data placement:** all 310 weight tensors are at 0x4000000000 (256 GiB) and up in the trace,
  which is the HBF window; KV cache and activations stay at their original (HBM) addresses. This
  placement comes with HBF-Sim's Qwen3 trace; `tools/remap_ranges.py` does it for vectorAdd.
- `runs/` (your own runs) is ignored by git; `results/` holds the shipped copies.
- `workloads/` must stay in place while runs are active: the simulator reads each kernel's trace
  when the kernel starts. `models/run.sh` refuses to start if a trace file is missing and marks a
  run that completed fewer kernels than listed as failed (`rc=4`).

## Configuration used

| Part | Setting | Source |
|---|---|---|
| GPU | 8 SMs at 1410 MHz, L1 32 KB, L2 8 MB (32 slices), 8 memory partitions, kernel launch 5,000 cycles | HBF-Sim's Qwen3 config (`models/base/hbfsim_qwen3.config`) |
| GPU request slots | 257 per memory partition (C2 and C4 share them between HBM and flash; C3 has its own flash pool) | HBF-Sim's value (`common.config`) |
| HBM | HBM2E, 8 channels × 128 bit × 3,024 Mb/s = 387 GB/s, CL/RCD/RP 22, RC 72; front end 134.75 ns | Ramulator 2 (`models/memory/hbm2e_slice8.py`) |
| HBF | 1 stack, 512 GiB, 16 channels × 192 GB/s, 512 sub-arrays (all active), 4 KiB pages, 2 page buffers per sub-array, 4,096 admission entries; tR 15 µs, tPROG 200 µs | HBF-Sim (timings set in ns, `common.config`) |
| Flash front end | 134.75 ns for C2, C3, C4 (same as an HBM request) | `C*/config.overrides` |
| C4 link GPU ↔ HBM base die | 48.384 GB/s per partition, shared by HBM and flash data | H3 (BW1 = BW2) |
| C4 base die | router, queues of 64, D2D hop 10 ns per NAND operation | `C4_h3_hbf_behind_hbm/config.overrides` |
| C4 LHB | 16 MiB = 4,096 pages, filled 2,048 pages at a time, hit 33 ns | H3's 40 MB per 1 TB/s, scaled to 387 GB/s |

## Experiments

All are listed, with their exact arguments, in `models/run_experiments.sh`.

| Experiment | Workload | Varied | Result folder |
|---|---|---|---|
| Main comparison | full token | C1–C4 | `results/qwen3_full/C?__qwen3_full` |
| LHB size, full token | full token | C4 LHB 10 / 16 / 40 / 200 MiB | `results/qwen3_full/C4__qwen3_full*` |
| vectorAdd | vectorAdd | C1–C4 | `results/vectoradd/` |
| Flash read time | layer 13 | tR 3 / 5 / 10 / 15 / 20 µs, C1–C4 | `results/qwen3_layer13/*__tR*` |
| LHB size × tR | layer 13 | C4, LHB 2–200 MiB × tR 3–20 µs | `results/qwen3_layer13/C4__*` |
| Flash parallelism | layer 13 | 256 / 512 / 1,024 sub-arrays, C2–C4 | `results/qwen3_layer13/*__sa*` |
| Request slots | layer 13 | C2 with 4,096 slots, C3 with 257 | `results/qwen3_layer13/*__slots*` |
| Validation | vectorAdd | 23 request-level runs, 59 checks | `results/vectoradd_validation/` |

---

## Repository layout

```
.
├── README.md                this file
├── requirements.txt         Python packages (Python >= 3.10)
├── setup/                   install_ubuntu.sh (native dependencies), accelsim.def (Apptainer image)
├── simulator/
│   ├── build.sh             builds Ramulator 2, then the simulator
│   ├── run_sim.sh           runs the simulator (natively or in the container)
│   ├── accel-sim-framework/ Accel-Sim + GPGPU-Sim with the Ramulator bridge, HBF-Sim's flash
│   │                        device, the attach logic for C2/C3/C4 and the H3 base die + LHB
│   ├── ramulator2/          Ramulator 2 + GPGPU-Sim plugins (front end, FR-FCFS, mapper, counter)
│   └── patches/             the same changes as patches against the upstream commits
├── models/
│   ├── base/                HBF-Sim's qwen3.config (verbatim) + common.config (shared changes)
│   ├── memory/              Ramulator HBM2E device
│   ├── C1_hbm_only/ C2_hbf_dram/ C3_hbf_hbm_direct/ C4_h3_hbf_behind_hbm/   config.overrides
│   ├── run.sh               run one workload on one model
│   ├── run_experiments.sh   every experiment of the study, with a job limit
│   ├── status.sh            state of the runs in runs/
│   ├── compare.py           table / CSV of finished runs
│   └── validate/            run_validation.sh, check.py, summarize.py
├── workloads/
│   ├── qwen3_1p7b/trace/    HBF-Sim's Qwen3-1.7B trace (1,819 kernels), kernel lists, LHB order files
│   └── vectoradd/           vectorAdd traces (original, A and B in HBF, write tests)
├── tools/                   trace remapping, LHB order generator, run packaging
├── results/                 shipped results (see results/README.md)
└── references/              literature notes, HBF-Sim notes, the earlier first study
```

## Requirements

- **Hardware:** any x86-64 Linux machine (WSL2 works). No GPU is needed: the simulator replays traces.
- **Resources:** about 0.5 GB of disk for repository and build, plus up to 2 GB per full-token run
  (raw logs); 0.3–0.5 GB of RAM and one core per simulation.
- **Software**, one of:
  - **Native:** Ubuntu 24.04 (22.04 with g++-12), g++ ≥ 12, cmake ≥ 3.17, make, bison, flex, zlib,
    libzstd-dev, xutils-dev, Python ≥ 3.10 with headers, CUDA toolkit 11.x or 12.x (only to compile).
  - **Container:** Apptainer/Singularity, plus Python ≥ 3.10 on the host.

**Run times** (one core each): vectorAdd 2–30 min; layer 13: C1 10 min, C2 1 h, C3 30 min, C4 12 min;
full token: C1 5 h, C2 21 h, C3 10 h, C4 7–9 h.

## Install and build

```bash
git clone https://github.com/lakshay18bansal/gpu-hbf-attachment-study.git
cd gpu-hbf-attachment-study

# Option A: native
setup/install_ubuntu.sh --cuda      # apt packages + CUDA toolkit + .venv (drop --cuda if CUDA is installed)
source .venv/bin/activate

# Option B: container
apptainer build --fakeroot accelsim.sif setup/accelsim.def
export HBF_CONTAINER=$PWD/accelsim.sif
python3 -m pip install --user -r requirements.txt

simulator/build.sh                  # 10–15 min; ends with "build OK"
```
Rebuild after moving the repository: the binary finds Ramulator through an absolute path.

## Run

```bash
models/run.sh C1 vectoradd                         # quick test
models/run.sh C4 qwen3_layer13                     # one decoder layer on H3 (16 MiB LHB)
models/run.sh C3 qwen3_layer13 --tR-us 5           # another flash read time
models/run.sh C4 qwen3_layer13 --lhb-mib 40        # another LHB size
models/run.sh C2 qwen3_layer13 --subarrays 1024    # more flash parallelism
models/run.sh C2 qwen3_full                        # the full token (hours)

models/run_experiments.sh --jobs 4 all             # every experiment (skips finished runs)
models/validate/run_validation.sh --jobs 4         # validation runs + checks (writes VALIDATION.md)

models/status.sh qwen3_full                        # running / done / failed
models/compare.py qwen3_full                       # table of your runs
```

`models/run.sh <C1|C2|C3|C4> <workload> [options] [-- extra simulator options]`

| Option | Meaning |
|---|---|
| workload | `vectoradd`, `vectoradd_nohbf`, `qwen3_full`, `qwen3_layer13`, `qwen3_prologue`, `qwen3_tail`, `qwen3_layer:N`, `qwen3_kernel:ID`, or a path to a `kernelslist.g` |
| `--tR-us X` | NAND page read time in µs (default 15) |
| `--lhb-mib N` | C4 only: LHB size in MiB (default 16) |
| `--subarrays N` | flash sub-arrays working in parallel (default 512; a multiple of 16) |
| `--out DIR`, `--force` | run folder; overwrite it |
| `--timeout SEC`, `--dry-run` | limit the run; only write the configuration |
| `-- -opt val ...` | any GPGPU-Sim/HBF option, applied last (e.g. `-- -gpgpu_dram_max_outstanding 4096`) |

| Variable | Meaning |
|---|---|
| `HBF_CONTAINER` | Apptainer image; build and runs use it when set |
| `HBF_RUNS` | where runs go (default `runs/`) |
| `CUDA_INSTALL_PATH` | CUDA toolkit for the native build (default `/usr/local/cuda`) |

Each run folder holds `run.log`, `rc`, `summary.txt`, `run_meta.json`, `gpgpusim.config`,
`memory.yaml` and `time.txt`. `python3 tools/package_run.py --log-xz runs/<run> -o <dir>` makes the
shareable copy used in `results/`.

## Limitations

- An 8-SM GPU slice (HBF-Sim's Qwen3 setup): its weight demand (~118 GB/s) is below the flash
  array's 140 GB/s, which is why C4 reaches HBM speed. A full-size GPU would need more flash bandwidth.
- tR is an assumption (no HBF silicon data); the sweep covers 3–20 µs.
- The LHB is given the order of weight reads from the trace (H3's assumption), and is modelled as an
  SRAM with a fixed 33 ns hit; C4 values below 1.00× come from this.
- One token, batch 1, 29-token context; energy is not modelled.

## Provenance, licenses and citation

| Component | Upstream | License |
|---|---|---|
| Accel-Sim framework | github.com/accel-sim/accel-sim-framework (dev, `d930ad6`) | BSD-2-Clause (`simulator/accel-sim-framework/LICENSE`) |
| GPGPU-Sim | github.com/accel-sim/gpgpu-sim_distribution (`91880c53`) + the commits in `simulator/patches/gpgpu-sim/` | BSD-style (`.../gpgpu-sim/COPYRIGHT`) |
| pybind11 (Accel-Sim dependency) | github.com/pybind/pybind11 (`3beb376`) | BSD-3-Clause |
| Ramulator 2 | github.com/CMU-SAFARI/ramulator2 (`c6c7a5d`) + GPGPU-Sim plugins (`simulator/patches/ramulator2-gpgpusim.patch`) | MIT (`simulator/ramulator2/LICENSE`) |
| HBF-Sim (flash device, `hbf_*` files) | github.com/jager197/accel-sim-hbf, arXiv 2609.29246 | BSD-2-Clause |
| Qwen3-1.7B trace | HBF-Sim's artifact (`artifact/inputs/qwen3`) | model: Apache-2.0 |

If you use this work, please cite HBF-Sim (Li et al., arXiv 2609.29246), H3 (Ha et al., IEEE CAL
2026), Accel-Sim (Khairy et al., ISCA 2020) and Ramulator 2 (Luo et al., IEEE CAL 2023).
