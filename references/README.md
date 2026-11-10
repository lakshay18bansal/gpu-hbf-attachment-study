# References

Background material. None of it is needed to build or run the four models.

| Path | Content |
|---|---|
| `literature/lit_hbf_llm_partC_2026-11.md` | Latest literature sweep (to 2026-11-05) and the review of this study's modelling choices against it |
| `literature/lit_hbf_llm_partA.md`, `partB.md` | Earlier sweeps: FLINT, FlashAccel, "HBF Sucks?", DASH, TileLens, HAVEN, the OCP HBF spec, H3, Yu et al., HBFSim-SlugLab |
| `literature/h3_primary_notes.md`, `h3_and_device_params.md` | Notes on the H3 paper (IEEE CAL 2026) and the device parameters taken from it |
| `literature/*standalone*.md`, `merge_feasibility.md`, `qwen3_trace_feasibility.md`, `critic.md` | Feasibility notes made before merging HBF-Sim, Ramulator and Accel-Sim |
| `hbfsim_paper_explained.md` | HBF-Sim (arXiv 2609.29246): what it models and how its authors evaluate it |
| `hbfsim_implementation_notes.md` | How HBF-Sim is implemented, and runs of the original simulator |
| `legacy_first_study/` | The first version of this study, superseded by `models/` |

## The first study (`legacy_first_study/`)
- **Hardware:** 16 SMs, HBM3 (C1/C3/C4) and GDDR6 (C2).
- **C3:** two variants, iso-stack and iso-shoreline.
- **C4:** the approximate attach mode 2, where the GPU itself splits the HBF traffic.

Its configurations, scripts, design notes and result summaries are kept for reference:
- `docs/04_results.md`
- `results/REPORT.md`
- `results/runs.csv`

Its run scripts expect the old folder layout and are not maintained.

Attach mode 2 is still in the simulator, so its results can be reproduced.
