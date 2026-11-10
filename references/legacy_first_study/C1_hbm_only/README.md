# C1: HBM only (baseline)

```
   16 SMs ── NoC ── L2 (8 partitions x 4 slices, 8 MB)
                          │  8 memory controllers (GPU side)
                          ▼
                   HBM3 PHY x 8 channels (64 bit, 6.4 Gb/s)  = 409.6 GB/s
                          │
                   [ HBM3 stack: 8 of its 16 channels ]   weights + KV cache + activations
```

**What it models.** Today's GPU: all data in HBM. It is the reference for every slowdown and
speed-up in the study.

| Item | Value | Where it is set |
|---|---|---|
| GPU | 16-SM A100-profile slice, 1410 MHz, 8 MB L2, 5000-cycle kernel launch | `common/gpu/qwen3_slice.config` |
| Memory | Ramulator 2.1 HBM3, 8 channels x 2 pseudo-channels, 6.4 Gb/s, 409.6 GB/s peak, tCL 12.5 ns, tRCD 19.4 ns, tRP 16.3 ns, all-bank refresh, FR-FCFS (GPGPU-Sim style), front end 52.49 ns | `common/memory/hbm3_slice8.py` |
| Bytes per SM | 409.6 / 16 = 25.6 GB/s, the same ratio as an H100 (3.35 TB/s / 132 SMs) | – |
| HBF | disabled (`-gpgpu_hbf_enabled 0`) | `config.env: HBF=0` |

**Address path.** L2 miss → GPGPU-Sim partition decode (channel) → Ramulator controller of that channel
→ `GPUBgCoBaRo` mapper (pseudo-channel, bank group, bank, row, column). The workload trace is the
same file as for C2-C4: weights sit at their 256 GiB addresses, which here simply decode to HBM. This
keeps L2 sets and DRAM rows identical across configurations.

**Run.**
```bash
./run.sh vectoradd
./run.sh qwen3_layer13            # one decoder layer of Qwen3-1.7B decode
./run.sh qwen3_kernel:4667        # one kernel (layer-13 q_proj GEMV)
```
`--tR-us` and `--prefetch` are ignored (there is no HBF tier).

Checked: Ramulator serves exactly the reads the GPU sends; the same Ramulator bridge matches GPGPU-Sim's
own DRAM model within 0.1% on an OPT-125m layer (`accelsim-work/RAMULATOR_GUIDE.md`).
