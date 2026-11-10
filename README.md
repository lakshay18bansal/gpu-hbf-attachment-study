# hbf-gpu-4arch: four ways to attach High Bandwidth Flash to a GPU

A cycle-level comparison of four GPU memory organizations for LLM inference, with the model
weights in **High Bandwidth Flash (HBF)**. It is built on Accel-Sim/GPGPU-Sim, Ramulator 2 and
HBF-Sim. Every model uses HBF-Sim's GPU and HBF device exactly; the workloads are vectorAdd and
one full Qwen3-1.7B decode token (1,819 kernels).

| | Organization | What the GPU sees |
|---|---|---|
| **C1** | GPU + HBM | HBM only (the baseline) |
| **C2** | GPU + HBF + DRAM, *HBF-Sim as shipped* | The GPU memory controller decodes the HBF window. HBF requests share the DRAM queue and request slots, and bypass the L2 |
| **C3** | GPU + HBM and HBF, both directly attached | HBF has its own port, queue and slots, so the two bandwidths add up |
| **C4** | GPU + HBM, **HBF behind the HBM base die (H3)** | HBM only. The HBM base die routes HBF addresses over a die-to-die link to the HBF stack. A latency-hiding buffer (LHB) in the base die prefetches weights |

## Results: Qwen3-1.7B, one decode token (batch 1)

| Model | Cycles | ms/token | tokens/s | vs C1 | Weight bandwidth |
|---|---|---|---|---|---|
| C1 HBM | 41,351,654 | 29.33 | 34.1 | 1.00× | 117 GB/s |
| C4 H3 + LHB | 42,432,066 | 30.09 | 33.2 | 1.03× | 114 GB/s |
| C3 direct | 129,514,394 | 91.85 | 10.9 | 3.13× | 37.5 GB/s |
| C2 HBF-Sim | 332,795,454 | 236.03 | 4.24 | 8.05× | 14.6 GB/s |
| C4 without LHB | 450,446,621 | 319.47 | 3.13 | 10.89× | 10.8 GB/s |

Where these numbers come from:
- Analysis: [results/QWEN_RESULTS.md](results/QWEN_RESULTS.md).
- Validation: [results/vectoradd_validation/VALIDATION.md](results/vectoradd_validation/VALIDATION.md)
  (**67/67 checks**).
- Full explanation of the models, simulators, delays, flaws and comparisons:
  [docs/STUDY_EXPLAINED.md](docs/STUDY_EXPLAINED.md).

---

## Contents
1. [Repository layout](#1-repository-layout)
2. [Requirements](#2-requirements)
3. [Install, step by step](#3-install-step-by-step)
4. [Build](#4-build)
5. [Run vectorAdd (quick test, about 30 min)](#5-run-vectoradd)
6. [Full validation (67 checks, about 3 h)](#6-full-validation)
7. [Run Qwen3-1.7B](#7-run-qwen3-17b)
8. [Outputs, status and comparison](#8-outputs-status-and-comparison)
9. [Options and environment variables](#9-options-and-environment-variables)
10. [Shipped results](#10-shipped-results)
11. [Provenance, licenses and citation](#11-provenance-licenses-and-citation)

---

## 1. Repository layout

```
hbf-gpu-4arch/
├── README.md                this file
├── requirements.txt         Python packages (Python >= 3.10)
├── setup/
│   ├── install_ubuntu.sh    native dependencies (apt, optional CUDA toolkit, .venv)
│   └── accelsim.def         Apptainer image definition (alternative to a native install)
├── simulator/
│   ├── build.sh             builds Ramulator 2, then the simulator
│   ├── run_sim.sh           runs the simulator (natively or in the container)
│   ├── accel-sim-framework/ Accel-Sim + GPGPU-Sim (branch ramulator-hbf, commit 95403064):
│   │                        Ramulator bridge, HBF-Sim's HBF tier, attach modes C2/C3/C4, LHB
│   ├── ramulator2/          Ramulator 2.1 + GPGPU-Sim plugins (front end, FR-FCFS, mapper, counter)
│   └── patches/             the same changes as patches against upstream (for reference)
├── models/                  the four models
│   ├── base/                HBF-Sim's qwen3.config (verbatim) + common.config (the 6 shared changes)
│   ├── memory/              Ramulator HBM2E device (8 ch x 128 bit x 3024 Mbps = 387 GB/s)
│   ├── C1_hbm_only/  C2_hbf_dram/  C3_hbf_hbm_direct/  C4_h3_hbf_behind_hbm/   config.overrides
│   ├── run.sh               run one workload on one model
│   ├── run_qwen_all.sh      the whole Qwen plan with a job limit
│   ├── status.sh            state of every run
│   ├── compare.py           side-by-side table
│   └── validate/            run_validation.sh + check.py (vectorAdd, 67 checks)
├── workloads/
│   ├── vectoradd/           traces: original, hbf (A, B in HBF), hbf_rw, hbf_rw512k (writes), work
│   └── qwen3_1p7b/trace/    HBF-Sim's Qwen3-1.7B decode-token trace (1,819 kernels), kernel lists,
│                            prefetch hints, tensor manifest
├── tools/                   prefetch-hint generator, trace remapping, validation helpers, packaging
├── results/                 shipped results (summaries + final statistics of every run)
├── docs/                    STUDY_EXPLAINED.md (full explanation), MODELS_REFERENCE.md
└── references/              literature review, HBF-Sim paper and implementation notes,
                             legacy_first_study/ (the earlier HBM3/GDDR6 version)
```

Your own runs go to `runs/`, which is ignored by git.

## 2. Requirements

**Hardware:** any x86-64 Linux machine (WSL2 works). **No GPU is needed**: the simulator replays
traces. The CUDA toolkit is only needed to compile it.

**Resources:**
- Disk: about 0.5 GB for the repository and build, plus up to about 2 GB per full Qwen run (raw logs).
- RAM: 0.3–0.5 GB per simulation.
- CPU: one core per simulation; runs are independent, so run as many in parallel as you have cores.

**Software** (one of):
- **Native:**
  - Ubuntu 24.04 (tested; 22.04 works with g++-12);
  - g++ ≥ 12 (C++20), cmake ≥ 3.17, make, bison, flex, zlib, zstd (libzstd-dev), xutils-dev;
  - Python ≥ 3.10 with headers;
  - CUDA toolkit 11.x or 12.x (tested 12.8).
- **Container:** Apptainer (or Singularity) on any Linux, plus Python ≥ 3.10 on the host. The run
  scripts run on the host and call the simulator inside the container.

**Run times** (one core each):

| Workload | C1 | C2 | C3 | C4 + LHB | C4 without LHB |
|---|---|---|---|---|---|
| vectorAdd | 1.5 min | 28 min | 23 min | 2 min | 19 min |
| Qwen prologue (18 kernels) | seconds | seconds | seconds | seconds | seconds |
| Qwen decoder layer 13 (64 kernels) | ~10 min | 1 h | ~20 min | 8 min | 1.5 h |
| **Qwen full token** (1,819 kernels) | **5 h** | **21 h** | **10 h** | **7 h** | **27 h** |

## 3. Install, step by step

### 3.1 Clone
```bash
git clone https://github.com/<your-account>/hbf-gpu-4arch.git
cd hbf-gpu-4arch
```

### 3.2 Option A: native install (Ubuntu)
```bash
setup/install_ubuntu.sh --cuda     # apt packages + CUDA 12.8 toolkit + Python venv (.venv)
                                   # (drop --cuda if a CUDA toolkit is already in /usr/local/cuda)
source .venv/bin/activate          # every new shell: activate before running
```
For a CUDA toolkit somewhere else: `export CUDA_INSTALL_PATH=/path/to/cuda`.

### 3.3 Option B: Apptainer container
```bash
# Apptainer: https://apptainer.org/docs/admin/main/installation.html  (Ubuntu: sudo apt install apptainer)
apptainer build --fakeroot accelsim.sif setup/accelsim.def      # downloads the CUDA base image (several GB)
export HBF_CONTAINER=$PWD/accelsim.sif                         # every new shell
python3 -m pip install --user -r requirements.txt              # host Python >= 3.10
```
With `HBF_CONTAINER` set, `simulator/build.sh` and every run use the container automatically.

## 4. Build
```bash
simulator/build.sh            # or: simulator/build.sh -j8
```
- **Time:** about 10–15 minutes.
- **What it does:** builds `simulator/ramulator2/libramulator.so`, then
  `simulator/build/accel-sim.out`. It ends with `build OK`.
- **Moving the repository:** rebuild afterwards, because the binary finds Ramulator through an
  absolute path.

## 5. Run vectorAdd

Each command is one simulation. Run them one after another, or all at once (they are independent):
```bash
models/run.sh C1 vectoradd      # GPU + HBM
models/run.sh C2 vectoradd      # GPU + HBF + DRAM (HBF-Sim)
models/run.sh C3 vectoradd      # HBF and HBM both direct
models/run.sh C4 vectoradd      # H3 with the LHB (prefetch is on by default for C4)
models/run.sh C4 vectoradd --prefetch off      # H3 without the LHB
```
Each prints a summary at the end and writes `runs/<model>__vectoradd__pf<on|off>/`.

Expected cycles (deterministic; your numbers must match exactly):

| Run | Cycles |
|---|---|
| C1 | 85,434 |
| C2 | 3,377,812 |
| C3 | 2,759,605 |
| C4 | 120,403 |
| C4 `--prefetch off` | 2,272,222 |

```bash
models/compare.py vectoradd     # table of your vectorAdd runs
```

## 6. Full validation
```bash
models/validate/run_validation.sh --jobs 5     # 22 runs + HBF-Sim reference, about 3 h
python3 models/validate/check.py               # writes runs/validation/VALIDATION.md
```
Expect **67/67 PASS**, as in [results/vectoradd_validation/VALIDATION.md](results/vectoradd_validation/VALIDATION.md).

The checks recompute from the per-request traces:
- address decoding, tier routing and HBF page/channel/sub-array mapping;
- request conservation and credits;
- the L2 policy;
- C4's base-die routing, link, hop and LHB;
- writes and the deadlock window;
- C2 against HBF-Sim verbatim;
- each extra option.

`--only ID,ID` reruns selected runs.

## 7. Run Qwen3-1.7B

The trace (`workloads/qwen3_1p7b/trace`) and the C4 prefetch hints are included, so nothing has
to be downloaded or generated.

### 7.1 Smoke test (seconds)
```bash
for c in C1 C2 C3 C4; do models/run.sh $c qwen3_prologue; done
```

### 7.2 The full token, one command per model
These are long (5–27 h). Use `tmux` so they keep running after you close the terminal:
```bash
tmux new -d -s C1      'models/run.sh C1 qwen3_full; exec bash'
tmux new -d -s C2      'models/run.sh C2 qwen3_full; exec bash'
tmux new -d -s C3      'models/run.sh C3 qwen3_full; exec bash'
tmux new -d -s C4      'models/run.sh C4 qwen3_full; exec bash'
tmux new -d -s C4pfoff 'models/run.sh C4 qwen3_full --prefetch off; exec bash'
```
- `tmux ls` lists the sessions.
- `tmux attach -t C2` shows one; **Ctrl+b** then **d** leaves it running.
- Without tmux: `nohup models/run.sh C1 qwen3_full > runs/C1_full.out 2>&1 &`.
- With the container, export `HBF_CONTAINER` before starting tmux, or put it in your shell profile.

Outputs:

| Model | Folder |
|---|---|
| C1 | `runs/C1__qwen3_full__pfoff/` |
| C2 | `runs/C2__qwen3_full__pfoff/` |
| C3 | `runs/C3__qwen3_full__pfoff/` |
| C4 + LHB | `runs/C4__qwen3_full__pfon/` |
| C4 without LHB | `runs/C4__qwen3_full__pfoff/` |

### 7.3 One decoder layer (all 28 layers are identical)
```bash
models/run.sh C1 qwen3_layer13          # likewise C2, C3, C4; any layer: qwen3_layer:<0-27>
```

### 7.4 The whole plan with a job limit
5 full-token runs plus 25 layer-13 sweep runs:
- parity: C2/C3 with the LHB, with L2 caching, C3 with 257 slots;
- credits 4096;
- tR 3/5/20 µs;
- flash bandwidth = HBM.

```bash
tmux new -d -s qwen 'models/run_qwen_all.sh --jobs 8 all; exec bash'    # or: main | sweeps
```
It skips runs that already finished, so it can be restarted, and logs to `runs/qwen_all.log`.

## 8. Outputs, status and comparison
```bash
models/status.sh                 # full-token runs: running / DONE / FAILED / DIED / STUCK?
models/status.sh qwen3_layer13   # other runs by pattern ('' = all)
models/compare.py qwen3_full     # your runs side by side
models/compare.py qwen3_full --dir results/qwen3_full      # the shipped results
```

Every run folder contains:
- `summary.txt`: cycles, time, HBM/HBF traffic, LHB hits, credits;
- `run.log`: full simulator output; `rc`: exit status;
- `time.txt`: wall time and memory;
- `gpgpusim.config`: the exact assembled configuration;
- `memory.yaml`: Ramulator device;
- `run_meta.json`.

`python3 tools/package_run.py runs/<run> -o <dir>` makes the small shareable copy used in
`results/`.

## 9. Options and environment variables

`models/run.sh <C1|C2|C3|C4> <workload> [options] [-- extra simulator options]`

| Option | Meaning |
|---|---|
| workload | `vectoradd`, `vectoradd_nohbf`, `qwen3_full`, `qwen3_layer13`, `qwen3_prologue`, `qwen3_tail`, `qwen3_layer:N`, `qwen3_kernel:ID`, or a path to a `kernelslist.g` |
| `--prefetch on\|off` | H3's prefetcher + LHB (default: on for C4, off for C1–C3) |
| `--tR-us X` | NAND page read time in µs (default 15); LHB and prefetch window resized with it |
| `--out DIR`, `--force` | run folder; overwrite |
| `--timeout SEC`, `--dry-run` | limit the run; only write the config |
| `-- -opt val ...` | any GPGPU-Sim/HBF option, applied last, e.g. `-- -gpgpu_hbf_d2d_hop_ns 20`, `-- -gpgpu_dram_max_outstanding 4096`, `-- -gpgpu_hbf_h3_returns_per_tick 4` |

| Variable | Meaning |
|---|---|
| `HBF_CONTAINER` | Apptainer image; build and runs use it when set |
| `HBF_RUNS` | where runs go (default `runs/`) |
| `CUDA_INSTALL_PATH` | CUDA toolkit for the native build (default `/usr/local/cuda`) |
| `SIM_BIN` | another simulator binary |
| `SIM_TIMEOUT` | seconds |

What each model's configuration contains:
- `models/base/hbfsim_qwen3.config`: HBF-Sim's file, verbatim;
- `models/base/common.config`: the six shared changes;
- `models/C*/config.overrides`: the per-model settings.

[docs/MODELS_REFERENCE.md](docs/MODELS_REFERENCE.md) lists every latency, bandwidth, credit and code location.

## 10. Shipped results
`results/` keeps, for every run, the summary, configuration, Ramulator device, wall time, exit
status, `kernel_cycles.csv` (every kernel), and `run_stats.log`. That log is the option dump,
per-kernel cycles and the complete final statistics block, and `summarize.py`/`compare.py` read it.

| Folder | Content |
|---|---|
| `qwen3_full/` | the 5 full-token runs (`COMPARE.md` = the table above) |
| `qwen3_layer13_checks/` | C2 vs HBF-Sim verbatim, C2 + L2 caching, C3 with 257 slots, the C4-without-LHB ablations (link, hop, queues, per-tick rates) |
| `qwen3_preflight/` | prologue on C1–C4; C4 layer 13 with continuous vs one-kernel lookahead |
| `vectoradd_validation/` | the 23 validation runs + `VALIDATION.md` |
| `vectoradd_regression/` | the final binary reproducing the validated cycles |

The full raw `run.log` files (3.9 GB) are not in the repository.

## 11. Provenance, licenses and citation
| Component | Upstream | License |
|---|---|---|
| Accel-Sim framework | github.com/accel-sim/accel-sim-framework (dev, `d930ad6`) | BSD-2-Clause (`simulator/accel-sim-framework/LICENSE`) |
| GPGPU-Sim | github.com/accel-sim/gpgpu-sim_distribution (`91880c53`) + 8 commits (`simulator/patches/gpgpu-sim/`) | BSD-style (`.../gpgpu-sim/COPYRIGHT`) |
| Ramulator 2 | github.com/CMU-SAFARI/ramulator2 (`c6c7a5d`) + GPGPU-Sim plugins (`simulator/patches/ramulator2-gpgpusim.patch`) | MIT (`simulator/ramulator2/LICENSE`) |
| HBF-Sim (HBF tier, 16 `hbf_*` files) | github.com/jager197/accel-sim-hbf, arXiv 2609.29246 | BSD-2-Clause |
| Qwen3-1.7B trace | HBF-Sim's artifact (`artifact/inputs/qwen3`) | model: Apache-2.0 |

If you use this work, please cite:
- HBF-Sim (Li et al., arXiv 2609.29246);
- H3 (Ha et al., IEEE CAL 2026);
- Accel-Sim (Khairy et al., ISCA 2020);
- Ramulator 2 (Luo et al., IEEE CAL 2023).

Limitations of the models and of the comparison are in
[docs/STUDY_EXPLAINED.md §10–11](docs/STUDY_EXPLAINED.md). The most important: an 8-SM GPU slice,
tR = 15 µs as an assumption, and oracle prefetch hints.
