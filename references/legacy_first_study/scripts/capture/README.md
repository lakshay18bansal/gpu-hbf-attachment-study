# Capturing new LLM traces on a rented GPU

The bundled Qwen3-1.7B trace is batch 1, context 29, so it shows weight streaming only. These scripts
capture the same model at **batch {1, 8, 32} x context {512, 2048}**, one decoder layer per capture.
They use a `StaticCache` so the KV cache has fixed addresses, and they record the address of every
weight **and KV** tensor. Weights → HBF, KV → HBM, or KV → HBF can then all be studied from the same trace.

## Which GPU to rent

- **A100 (sm_80) is the best choice**: its SASS matches the simulated SM80 slice. An H100 (sm_90) also works.
- **Do not use an RTX 4090, L4 or L40S (sm_89).** Accel-Sim's front end rejects binary version 89 unless
  the libraries happen to contain sm_80/86 code. The script warns about this.
- One GPU with ≥ 24 GB is enough. Qwen3-1.7B BF16 needs about 4 GB, plus 0.1-2 GB of KV.

## Steps

1. Start the instance. Put its host and port in `accelsim-work/ssh_traces_build/remote_trace.conf`
   (`SSH_HOST`, `SSH_PORT`), and check that `ssh -p <port> root@<host>` works.
2. From the project root:
   ```bash
   scripts/capture/capture_suite.sh --dry-run          # list the captures
   scripts/capture/capture_suite.sh                    # all of them (or --only qwen3_b8_c2048_l13)
   ```
   This copies `remote_setup.sh` (from ssh_traces_build) and the two capture files to the box, builds
   NVBit and the tracer (cached after the first run), installs `transformers` if needed, then for each capture:
   - a native, untraced run for timing;
   - a traced run with `ACTIVE_FROM_START=0`, where only the region between `cudaProfilerStart` and
     `cudaProfilerStop` around layer 13 of decode step 2 is traced;
   - `post-traces-processing`, then a tarball that is copied to `scripts/capture/staging/`.
3. Place tensors and simulate, locally:
   ```bash
   python3 tools/place_tensors.py scripts/capture/staging/qwen3_b8_c2048_l13 \
           -o workloads/qwen3_b8_c2048_l13 --hbf weights            # or weights+kv, kv
   C4_hbf_behind_hbm/run.sh workloads/qwen3_b8_c2048_l13/kernelslist.g --tR-us 15 --prefetch on
   ```
   `place_tensors.py` converts `.tracez` to text (`tools/bin/traceDsm`). It packs the chosen tensors from
   256 GiB, 4 KiB-aligned, in model order, and writes `placement.json`. A kernel list given as a path works
   with every `run.sh` and with `scripts/run_config.sh`; hints for `--prefetch on` are generated
   automatically in `<workload>/hints/`.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `capture_suite.sh` | laptop | the capture matrix, upload, remote setup, download |
| `remote_capture.sh` | GPU box | native + traced run, post-processing, tarball |
| `capture_qwen3_layer.py` | GPU box | HF Transformers decode with StaticCache; ROI around one layer/step; `metadata/manifest.json` (weights + KV `data_ptr`) and `metadata/capture.json` |
| `../../tools/place_tensors.py` | laptop | tensor classes → HBF window (uses `tools/remap_ranges.py`) |

## Cost estimate

Tracing slows a step by roughly 10^3. A one-layer capture takes about 5-20 minutes (batch 32 × 2048 is
the largest) and the full-step capture 1-2 hours. That is about 3 GPU-hours for the whole suite: roughly
$5-10 on Vast.ai at A100 prices. Disk is 0.2-3 GB per compressed capture.

## Tested without a GPU

- `place_tensors.py`: on a synthetic capture built from the bundled q_proj kernel. All 4,194,304 lane
  addresses were moved, the page sequence was unchanged, and the tied embedding/lm_head was placed once.
- `capture_qwen3_layer.py` and the shell scripts: syntax only, because torch is not installed on the
  laptop. The StaticCache code handles both the old (`key_cache`) and the new (`layers[i].keys`)
  transformers APIs.
- On the first real capture, check that `metadata/manifest.json` has 2 × 28 KV tensors and that
  `metadata/capture.json` reports `"traced": true`.
