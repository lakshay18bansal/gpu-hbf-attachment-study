# Run commands (copy-paste)

All commands run from `<btp_work>/hbm_hbf_study` on the laptop. No GPU is needed: the
complete Qwen3-1.7B trace (all 1,819 kernels) is in `workloads/qwen3_1p7b/trace/`.

**Laptop limits:** 16 cores and 7 GB RAM, so use `--jobs 6` at most (each run takes 0.3-0.5 GB).
**Never edit `scripts/*.sh` while a run is in progress.** Bash reads scripts as it runs them, so an edit
breaks the running jobs.

## 0. Check the installation (≈ 35 min)
```bash
sim/build.sh                                  # only after changing C++ (2 min)
python3 tools/validate_models.py --jobs 6     # 31 vectorAdd/q_proj tests -> results/validation/VALIDATION.md
```

## 1. One run on one configuration
```bash
C1_hbm_only/run.sh                     qwen3_layer13
C2_hbf_dram/run.sh                     qwen3_layer13 --tR-us 15 --prefetch on
C3_hbf_hbm_direct/iso_stack/run.sh     qwen3_layer13 --tR-us 15 --prefetch on
C3_hbf_hbm_direct/iso_shoreline/run.sh qwen3_layer13 --tR-us 15 --prefetch on
C4_hbf_behind_hbm/run.sh               qwen3_layer13 --tR-us 15 --prefetch on
C2_hbf_dram/legacy_paper/run.sh        qwen3_layer13                  # HBF-Sim exactly as published (8 SMs)
```

- **Workloads:**
  - `vectoradd`;
  - `qwen3_layer13`, or `qwen3_layer:<0..27>` for any layer (64 kernels each);
  - `qwen3_prologue` (18 kernels) and `qwen3_tail` (9 kernels, includes the 622 MB lm_head);
  - `qwen3_full` (all 1,819 kernels in one run; about 20-25 h for one configuration);
  - `qwen3_kernel:<id>`;
  - or the path to any `kernelslist.g`.
- **Options:**
  - `--tR-us X` and `--prefetch on|off`;
  - `--ram-set "K=V,..."` to change HBM/DRAM parameters;
  - `--set VAR=VALUE` to change a `config.env` variable;
  - `-- -raw_option value`;
  - `--out DIR` and `--force`.
- Parameters are documented in `PARAMETERS.md`.

## 2. Suites (parallel; skip finished runs; `--dry-run` lists them)
| Command | Runs | Laptop time | What it gives |
|---|---|---|---|
| `scripts/run_suite.sh smoke --jobs 6` | 9 | 15 min | vectorAdd on every configuration |
| `scripts/run_suite.sh qproj --jobs 6` | 9 | 15 min | layer-13 q_proj GEMV |
| `scripts/run_suite.sh study --jobs 6` | 41 | ~15 h (lm_head tails dominate) | layer 13 + prologue + tail, tR sweep, hop, paper. **Already done** (`results/study`) |
| `scripts/run_suite.sh fullmodel --jobs 6` | 140 | **~7 h** | all 28 layers simulated (the paper's full-step method) for C1 and the four HBF configurations, prefetch on, tR 15. Prologue and tail are reused from `study` |
| `FULL_PF_OFF=1 scripts/run_suite.sh fullmodel --jobs 6` | 252 | ~1.5 days | the same plus prefetch off |

## 3. Paper-exact and H3-exact variants
```bash
# HBF-Sim paper setup (8 SMs, built-in DRAM, shared DRAM credits, 16 x 192 GB/s, 512 slots, no prefetch),
# whole decode step in one run (reproduces the paper's 352.4 M cycles; ~20 h):
C2_hbf_dram/legacy_paper/run.sh qwen3_full --out results/paper_full/C2_legacy

# C4 with H3's own parameters: tR = 20 us (SLC), LHB = 2 x BW x tR = 4000 pages for the slice:
C4_hbf_behind_hbm/run.sh qwen3_layer13 --tR-us 20 --prefetch on --set LHB_PAGES=4000 --out results/h3/C4_tR20_lhb4000
```

## 4. Report
```bash
scripts/make_report.sh        # results/REPORT.md (+ section 2b when fullmodel is complete), runs.csv,
                              # plots/tpot_vs_tR.png, PROJECTION.md
```
