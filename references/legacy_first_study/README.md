# HBM vs HBF placement study (C1-C4)

The study compares four GPU memory organizations on LLM decode (Qwen3-1.7B, BF16). It uses one
cycle-level simulator: Accel-Sim, plus Ramulator 2.1 for HBM/DRAM, plus HBF-Sim's flash tier, extended
for this study.

> **Second study (2026-11-02): [`hbfsim_4arch/`](hbfsim_4arch/README.md).** It runs the same four
> organizations on HBF-Sim's exact GPU and HBF configuration. C2 there is HBF-Sim itself, and C4 is a
> structural H3 model (`-gpgpu_hbf_attach 3`), where the GPU does not know about HBF and the HBM base
> die routes each request. Validation is in `hbfsim_4arch/results/validation/VALIDATION.md`.

| Folder | Organization |
|---|---|
| `C1_hbm_only/` | HBM only (baseline) |
| `C2_hbf_dram/` | HBF + conventional DRAM (GDDR6); `legacy_paper/` = HBF-Sim exactly as published (validation only) |
| `C3_hbf_hbm_direct/` | HBF and HBM each on their own GPU PHYs, address split by the host: `iso_stack/`, `iso_shoreline/` |
| `C4_hbf_behind_hbm/` | HBF behind the HBM base die (H3); HBF data shares the HBM PHY |

Each folder has a `README.md` describing the model, a `config.env` plus `config.overrides` (its
parameters), and a `run.sh`.

## Documents
- `docs/01_hbfsim_paper_explained.md`: what HBF-Sim does and how its paper evaluates it
- `docs/02_design.md`: every modelling decision, with sources (configs, address mapping, parameters, metrics, revisions)
- `docs/03_answers.md`: the four questions: LLM and metrics; HBF-Sim/Ramulator inputs, outputs and integration; address mapping; the configs
- `docs/04_results.md`: results and discussion (after the study has run)
- `docs/research/`: literature and code reports (HBF/LLM papers, H3, device parameters, merge analysis, trace analysis, critique)

## Quick start
```bash
sim/build.sh                       # build sim/build/accel-sim.out (Apptainer image, ~2 min)
sim/tests/accept.sh $PWD/sim/tests/out va_builtin va_hbm3 lud qproj_ticks qproj_ns qp_a1 qp_a2_phyinf
scripts/run_suite.sh smoke         # vectorAdd on every config (~15 min, 6 parallel jobs max)
C4_hbf_behind_hbm/run.sh qwen3_layer13 --tR-us 15 --prefetch on    # one run
scripts/run_suite.sh study --jobs 6                                 # the full matrix (41 runs, hours)
scripts/make_report.sh             # results/REPORT.md, runs.csv, plots/, PROJECTION.md
```
Workloads: `vectoradd`, `qwen3_layer13`, `qwen3_prologue`, `qwen3_tail`, `qwen3_full`,
`qwen3_kernel:<kernel id>`, or a path to any `kernelslist.g`.

## Layout
```
sim/          merged simulator: accel-sim-framework copy; gpgpu-sim branch ramulator-hbf (commits on top
              of 91880c53: Ramulator bridge, HBF port, ns timers, attach modes + prefetcher, prefetch fix);
              build.sh, run_sim.sh, tests/accept.sh
common/       gpu/qwen3_slice.config (16-SM slice), hbf_base.config, memory/ (Ramulator HBM3 / HBM3-half / GDDR6)
workloads/    qwen3_1p7b/trace (bundled 1,819-kernel trace + lists + hints), vectoradd/{original,work,hbf}
tools/        parse_metrics, report, project_model, hbf_prefetch_hints, remap_ranges, place_tensors,
              tracelib, traceDsm (tracez -> text), ramtrace (standalone Ramulator replay)
scripts/      run_config.sh (one run), run_suite.sh (matrices), make_report.sh, capture/ (rented-GPU traces)
results/      one folder per run: gpgpusim.config, memory.yaml, run.log, run_meta.json, metrics.json
```

## Simulator options added for this study (on top of HBF-Sim's 48)
`-gpgpu_hbf_<timer>_ns` (clock-independent NAND and buffer timings), `-gpgpu_hbf_attach 0|1|2` (legacy /
direct with its own queue and credits / behind HBM with a shared PHY), `-gpgpu_hbf_credits`,
`-gpgpu_hbf_frontend_latency_ns`, `-gpgpu_hbf_phy_gbps`, `-gpgpu_hbf_l2_queue_size`,
`-gpgpu_hbf_prefetch_file`, `_window`, `_lookahead`, `-gpgpu_hbf_lhb_pages`,
`-gpgpu_hbf_lhb_hit_latency[_ns]`. See `docs/02_design.md` §5 and the commit messages in
`sim/accel-sim-framework/gpu-simulator/gpgpu-sim`.

## Adding a configuration
Copy a folder, then edit `config.env`:
- `MEM` is a Ramulator config in `common/memory/` (`CHANNELS` must be 8);
- `HBF`, `ATTACH`, `HBF_CHANNELS`, `HBF_LINK_GBPS`, `PHY_GBPS` and `FRONTEND_NS` set the HBF tier;
- raw simulator options go in `config.overrides`.

Media parallelism, credits and the LHB are derived from tR by `scripts/run_config.sh`.
