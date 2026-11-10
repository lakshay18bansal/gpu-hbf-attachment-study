# Analytical projection to B200-class packages

Model and formulas: `tools/project_model.py` docstring; inputs: `tools/projection_inputs.json`. Efficiencies (sustained / peak) measured in the simulations: C1_hbm_only: hbm 0.71 (qwen3_tail), C2_hbf_dram: hbf 0.74 (qwen3_tail), C3_iso_shoreline: hbf 0.98 (qwen3_tail), C3_iso_stack: hbf 0.74 (qwen3_tail), C4_behind_hbm: hbf 0.73 (qwen3_tail).

This is a first-order model (no communication, no queuing, kernels serialized); use the ratios, not the absolute numbers.

| Model | Context | Batch | Config | GPUs | TPOT (ms) | Tokens/s/GPU | vs C1 | Memory $/GPU | Tok/s per k$ |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | 2048 | 1 | C1_hbm_only | 1 | 0.75 | 1327.1 | 1.00x | 2,880 | 460.80 |
| Qwen3-1.7B | 2048 | 1 | C2_hbf_dram | 1 | 0.91 | 1094.3 | 0.82x | 8,081 | 135.42 |
| Qwen3-1.7B | 2048 | 1 | C3_iso_stack | 1 | 0.73 | 1367.9 | 1.03x | 10,577 | 129.33 |
| Qwen3-1.7B | 2048 | 1 | C3_iso_shoreline | 1 | 1.11 | 897.7 | 0.68x | 5,288 | 169.75 |
| Qwen3-1.7B | 2048 | 1 | C4_behind_hbm | 1 | 0.73 | 1366.2 | 1.03x | 10,577 | 129.17 |
| Llama-3.1-70B | 8192 | 1 | C1_hbm_only | 1 | 25.24 | 39.6 | 1.00x | 2,880 | 13.75 |
| Llama-3.1-70B | 8192 | 1 | C2_hbf_dram | 1 | 26.56 | 37.7 | 0.95x | 8,081 | 4.66 |
| Llama-3.1-70B | 8192 | 1 | C3_iso_stack | 1 | 24.46 | 40.9 | 1.03x | 10,577 | 3.87 |
| Llama-3.1-70B | 8192 | 1 | C3_iso_shoreline | 1 | 36.82 | 27.2 | 0.69x | 5,288 | 5.14 |
| Llama-3.1-70B | 8192 | 1 | C4_behind_hbm | 1 | 24.49 | 40.8 | 1.03x | 10,577 | 3.86 |
| Llama-3.1-70B | 8192 | 8 | C1_hbm_only | 1 | 28.54 | 280.3 | 1.00x | 2,880 | 97.33 |
| Llama-3.1-70B | 8192 | 8 | C2_hbf_dram | 1 | 44.45 | 180.0 | 0.64x | 8,081 | 22.27 |
| Llama-3.1-70B | 8192 | 8 | C3_iso_stack | 1 | 27.76 | 288.2 | 1.03x | 10,577 | 27.25 |
| Llama-3.1-70B | 8192 | 8 | C3_iso_shoreline | 1 | 43.41 | 184.3 | 0.66x | 5,288 | 34.85 |
| Llama-3.1-70B | 8192 | 8 | C4_behind_hbm | 1 | 27.79 | 287.9 | 1.03x | 10,577 | 27.22 |
| Llama-3.1-70B | 8192 | 32 | C1_hbm_only | 2 | 19.92 | 803.1 | 1.00x | 2,880 | 278.87 |
| Llama-3.1-70B | 8192 | 32 | C2_hbf_dram | 2 | 52.91 | 302.4 | 0.38x | 8,081 | 37.43 |
| Llama-3.1-70B | 8192 | 32 | C3_iso_stack | 1 | 39.06 | 819.2 | 1.02x | 10,577 | 77.45 |
| Llama-3.1-70B | 8192 | 32 | C3_iso_shoreline | 1 | 66.02 | 484.7 | 0.60x | 5,288 | 91.66 |
| Llama-3.1-70B | 8192 | 32 | C4_behind_hbm | 1 | 39.09 | 818.5 | 1.02x | 10,577 | 77.39 |
| Llama-3.1-70B | 8192 | 128 | C1_hbm_only | 3 | 28.35 | 1504.9 | 1.00x | 2,880 | 522.53 |
| Llama-3.1-70B | 8192 | 128 | C2_hbf_dram | 8 | 43.90 | 364.4 | 0.24x | 8,081 | 45.10 |
| Llama-3.1-70B | 8192 | 128 | C3_iso_stack | 2 | 42.14 | 1518.9 | 1.01x | 10,577 | 143.60 |
| Llama-3.1-70B | 8192 | 128 | C3_iso_shoreline | 4 | 39.11 | 818.2 | 0.54x | 5,288 | 154.72 |
| Llama-3.1-70B | 8192 | 128 | C4_behind_hbm | 2 | 42.15 | 1518.3 | 1.01x | 10,577 | 143.55 |
| Llama-3.1-405B | 8192 | 1 | C1_hbm_only | 3 | 23.93 | 13.9 | 1.00x | 2,880 | 4.84 |
| Llama-3.1-405B | 8192 | 1 | C2_hbf_dram | 1 | 72.87 | 13.7 | 0.99x | 8,081 | 1.70 |
| Llama-3.1-405B | 8192 | 1 | C3_iso_stack | 1 | 69.56 | 14.4 | 1.03x | 10,577 | 1.36 |
| Llama-3.1-405B | 8192 | 1 | C3_iso_shoreline | 1 | 104.38 (>SLO) | 9.6 | 0.69x | 5,288 | 1.81 |
| Llama-3.1-405B | 8192 | 1 | C4_behind_hbm | 1 | 69.65 | 14.4 | 1.03x | 10,577 | 1.36 |
| Llama-3.1-405B | 8192 | 8 | C1_hbm_only | 3 | 25.66 | 103.9 | 1.00x | 2,880 | 36.08 |
| Llama-3.1-405B | 8192 | 8 | C2_hbf_dram | 1 | 101.05 (>SLO) | 79.2 | 0.76x | 8,081 | 9.80 |
| Llama-3.1-405B | 8192 | 8 | C3_iso_stack | 1 | 74.75 | 107.0 | 1.03x | 10,577 | 10.12 |
| Llama-3.1-405B | 8192 | 8 | C3_iso_shoreline | 1 | 114.77 (>SLO) | 69.7 | 0.67x | 5,288 | 13.18 |
| Llama-3.1-405B | 8192 | 8 | C4_behind_hbm | 1 | 74.84 | 106.9 | 1.03x | 10,577 | 10.11 |
| Llama-3.1-405B | 8192 | 32 | C1_hbm_only | 4 | 23.70 | 337.6 | 1.00x | 2,880 | 117.22 |
| Llama-3.1-405B | 8192 | 32 | C2_hbf_dram | 4 | 49.42 | 161.9 | 0.48x | 8,081 | 20.03 |
| Llama-3.1-405B | 8192 | 32 | C3_iso_stack | 1 | 92.55 | 345.8 | 1.02x | 10,577 | 32.69 |
| Llama-3.1-405B | 8192 | 32 | C3_iso_shoreline | 2 | 75.19 | 212.8 | 0.63x | 5,288 | 40.24 |
| Llama-3.1-405B | 8192 | 32 | C4_behind_hbm | 1 | 92.64 | 345.4 | 1.02x | 10,577 | 32.66 |
| Llama-3.1-405B | 8192 | 128 | C1_hbm_only | 6 | 27.67 | 771.1 | 1.00x | 2,880 | 267.74 |
| Llama-3.1-405B | 8192 | 128 | C2_hbf_dram | 13 | 44.94 | 219.1 | 0.28x | 8,081 | 27.11 |
| Llama-3.1-405B | 8192 | 128 | C3_iso_stack | 4 | 40.94 | 781.6 | 1.01x | 10,577 | 73.90 |
| Llama-3.1-405B | 8192 | 128 | C3_iso_shoreline | 7 | 41.83 | 437.2 | 0.57x | 5,288 | 82.67 |
| Llama-3.1-405B | 8192 | 128 | C4_behind_hbm | 4 | 40.96 | 781.2 | 1.01x | 10,577 | 73.86 |
| Llama-3.1-405B | 131072 | 1 | C1_hbm_only | 3 | 27.64 | 12.1 | 1.00x | 2,880 | 4.19 |
| Llama-3.1-405B | 131072 | 1 | C2_hbf_dram | 2 | 66.63 | 7.5 | 0.62x | 8,081 | 0.93 |
| Llama-3.1-405B | 131072 | 1 | C3_iso_stack | 1 | 80.68 | 12.4 | 1.03x | 10,577 | 1.17 |
| Llama-3.1-405B | 131072 | 1 | C3_iso_shoreline | 1 | 126.64 (>SLO) | 7.9 | 0.65x | 5,288 | 1.49 |
| Llama-3.1-405B | 131072 | 1 | C4_behind_hbm | 1 | 80.77 | 12.4 | 1.03x | 10,577 | 1.17 |
| Llama-3.1-405B | 131072 | 4 | C1_hbm_only | 4 | 29.63 | 33.7 | 1.00x | 2,880 | 11.72 |
| Llama-3.1-405B | 131072 | 4 | C2_hbf_dram | 7 | 46.65 | 12.2 | 0.36x | 8,081 | 1.52 |
| Llama-3.1-405B | 131072 | 4 | C3_iso_stack | 2 | 58.14 | 34.4 | 1.02x | 10,577 | 3.25 |
| Llama-3.1-405B | 131072 | 4 | C3_iso_shoreline | 4 | 49.46 | 20.2 | 0.60x | 5,288 | 3.82 |
| Llama-3.1-405B | 131072 | 4 | C4_behind_hbm | 2 | 58.19 | 34.4 | 1.02x | 10,577 | 3.25 |
| Llama-3.1-405B | 131072 | 16 | C1_hbm_only | 9 | 28.99 | 61.3 | 1.00x | 2,880 | 21.29 |
| Llama-3.1-405B | 131072 | 16 | C2_hbf_dram | 26 | 42.29 | 14.6 | 0.24x | 8,081 | 1.80 |
| Llama-3.1-405B | 131072 | 16 | C3_iso_stack | 7 | 36.96 | 61.8 | 1.01x | 10,577 | 5.85 |
| Llama-3.1-405B | 131072 | 16 | C3_iso_shoreline | 13 | 37.13 | 33.1 | 0.54x | 5,288 | 6.27 |
| Llama-3.1-405B | 131072 | 16 | C4_behind_hbm | 7 | 36.97 | 61.8 | 1.01x | 10,577 | 5.85 |
| Llama-3.1-405B | 131072 | 64 | C1_hbm_only | 28 | 29.66 | 77.1 | 1.00x | 2,880 | 26.75 |
| Llama-3.1-405B | 131072 | 64 | C2_hbf_dram | 101 | 41.51 | 15.3 | 0.20x | 8,081 | 1.89 |
| Llama-3.1-405B | 131072 | 64 | C3_iso_stack | 26 | 31.86 | 77.3 | 1.00x | 10,577 | 7.30 |
| Llama-3.1-405B | 131072 | 64 | C3_iso_shoreline | 51 | 31.80 | 39.5 | 0.51x | 5,288 | 7.46 |
| Llama-3.1-405B | 131072 | 64 | C4_behind_hbm | 26 | 31.86 | 77.3 | 1.00x | 10,577 | 7.30 |
| Qwen3-235B-A22B | 32768 | 1 | C1_hbm_only | 2 | 2.81 | 178.1 | 1.00x | 2,880 | 61.82 |
| Qwen3-235B-A22B | 32768 | 1 | C2_hbf_dram | 1 | 10.38 | 96.4 | 0.54x | 8,081 | 11.93 |
| Qwen3-235B-A22B | 32768 | 1 | C3_iso_stack | 1 | 5.47 | 182.7 | 1.03x | 10,577 | 17.27 |
| Qwen3-235B-A22B | 32768 | 1 | C3_iso_shoreline | 1 | 8.74 | 114.4 | 0.64x | 5,288 | 21.63 |
| Qwen3-235B-A22B | 32768 | 1 | C4_behind_hbm | 1 | 5.48 | 182.5 | 1.02x | 10,577 | 17.25 |
| Qwen3-235B-A22B | 32768 | 16 | C1_hbm_only | 2 | 22.50 | 355.6 | 1.00x | 2,880 | 123.48 |
| Qwen3-235B-A22B | 32768 | 16 | C2_hbf_dram | 3 | 40.85 | 130.5 | 0.37x | 8,081 | 16.16 |
| Qwen3-235B-A22B | 32768 | 16 | C3_iso_stack | 1 | 44.13 | 362.6 | 1.02x | 10,577 | 34.28 |
| Qwen3-235B-A22B | 32768 | 16 | C3_iso_shoreline | 2 | 37.46 | 213.5 | 0.60x | 5,288 | 40.38 |
| Qwen3-235B-A22B | 32768 | 16 | C4_behind_hbm | 1 | 44.16 | 362.3 | 1.02x | 10,577 | 34.25 |
| Qwen3-235B-A22B | 32768 | 64 | C1_hbm_only | 4 | 27.86 | 574.3 | 1.00x | 2,880 | 199.42 |
| Qwen3-235B-A22B | 32768 | 64 | C2_hbf_dram | 10 | 42.38 | 151.0 | 0.26x | 8,081 | 18.69 |
| Qwen3-235B-A22B | 32768 | 64 | C3_iso_stack | 3 | 36.72 | 581.0 | 1.01x | 10,577 | 54.93 |
| Qwen3-235B-A22B | 32768 | 64 | C3_iso_shoreline | 5 | 40.09 | 319.3 | 0.56x | 5,288 | 60.37 |
| Qwen3-235B-A22B | 32768 | 64 | C4_behind_hbm | 3 | 36.73 | 580.7 | 1.01x | 10,577 | 54.91 |
| Qwen3-235B-A22B | 32768 | 256 | C1_hbm_only | 11 | 29.51 | 788.8 | 1.00x | 2,880 | 273.88 |
| Qwen3-235B-A22B | 32768 | 256 | C2_hbf_dram | 38 | 41.53 | 162.2 | 0.21x | 8,081 | 20.08 |
| Qwen3-235B-A22B | 32768 | 256 | C3_iso_stack | 10 | 32.33 | 791.9 | 1.00x | 10,577 | 74.88 |
| Qwen3-235B-A22B | 32768 | 256 | C3_iso_shoreline | 19 | 32.97 | 408.7 | 0.52x | 5,288 | 77.28 |
| Qwen3-235B-A22B | 32768 | 256 | C4_behind_hbm | 10 | 32.33 | 791.8 | 1.00x | 10,577 | 74.86 |
| Llama-3.1-405B | 1048576 | 1 | C1_hbm_only | 6 | 27.67 | 6.0 | 1.00x | 2,880 | 2.09 |
| Llama-3.1-405B | 1048576 | 1 | C2_hbf_dram | 13 | 44.94 | 1.7 | 0.28x | 8,081 | 0.21 |
| Llama-3.1-405B | 1048576 | 1 | C3_iso_stack | 4 | 40.94 | 6.1 | 1.01x | 10,577 | 0.58 |
| Llama-3.1-405B | 1048576 | 1 | C3_iso_shoreline | 7 | 41.83 | 3.4 | 0.57x | 5,288 | 0.65 |
| Llama-3.1-405B | 1048576 | 1 | C4_behind_hbm | 4 | 40.96 | 6.1 | 1.01x | 10,577 | 0.58 |
| Llama-3.1-405B | 1048576 | 2 | C1_hbm_only | 9 | 28.99 | 7.7 | 1.00x | 2,880 | 2.66 |
| Llama-3.1-405B | 1048576 | 2 | C2_hbf_dram | 26 | 42.29 | 1.8 | 0.24x | 8,081 | 0.23 |
| Llama-3.1-405B | 1048576 | 2 | C3_iso_stack | 7 | 36.96 | 7.7 | 1.01x | 10,577 | 0.73 |
| Llama-3.1-405B | 1048576 | 2 | C3_iso_shoreline | 13 | 37.13 | 4.1 | 0.54x | 5,288 | 0.78 |
| Llama-3.1-405B | 1048576 | 2 | C4_behind_hbm | 7 | 36.97 | 7.7 | 1.01x | 10,577 | 0.73 |
| Llama-3.1-405B | 1048576 | 4 | C1_hbm_only | 15 | 30.06 | 8.9 | 1.00x | 2,880 | 3.08 |
| Llama-3.1-405B | 1048576 | 4 | C2_hbf_dram | 51 | 41.77 | 1.9 | 0.21x | 8,081 | 0.23 |
| Llama-3.1-405B | 1048576 | 4 | C3_iso_stack | 13 | 34.51 | 8.9 | 1.00x | 10,577 | 0.84 |
| Llama-3.1-405B | 1048576 | 4 | C3_iso_shoreline | 26 | 33.17 | 4.6 | 0.52x | 5,288 | 0.88 |
| Llama-3.1-405B | 1048576 | 4 | C4_behind_hbm | 13 | 34.51 | 8.9 | 1.00x | 10,577 | 0.84 |

## Best tokens/s per GPU with TPOT <= 100 ms (over the listed batch sizes)

| Model | Context | C1_hbm_only | C2_hbf_dram | C3_iso_stack | C3_iso_shoreline | C4_behind_hbm |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 2048 | 1327.1 (B=1, 1 GPU, 1.00x) | 1094.3 (B=1, 1 GPU, 0.82x) | 1367.9 (B=1, 1 GPU, 1.03x) | 897.7 (B=1, 1 GPU, 0.68x) | 1366.2 (B=1, 1 GPU, 1.03x) |
| Llama-3.1-70B | 8192 | 1504.9 (B=128, 3 GPU, 1.00x) | 364.4 (B=128, 8 GPU, 0.24x) | 1518.9 (B=128, 2 GPU, 1.01x) | 818.2 (B=128, 4 GPU, 0.54x) | 1518.3 (B=128, 2 GPU, 1.01x) |
| Llama-3.1-405B | 8192 | 771.1 (B=128, 6 GPU, 1.00x) | 219.1 (B=128, 13 GPU, 0.28x) | 781.6 (B=128, 4 GPU, 1.01x) | 437.2 (B=128, 7 GPU, 0.57x) | 781.2 (B=128, 4 GPU, 1.01x) |
| Llama-3.1-405B | 131072 | 77.1 (B=64, 28 GPU, 1.00x) | 15.3 (B=64, 101 GPU, 0.20x) | 77.3 (B=64, 26 GPU, 1.00x) | 39.5 (B=64, 51 GPU, 0.51x) | 77.3 (B=64, 26 GPU, 1.00x) |
| Qwen3-235B-A22B | 32768 | 788.8 (B=256, 11 GPU, 1.00x) | 162.2 (B=256, 38 GPU, 0.21x) | 791.9 (B=256, 10 GPU, 1.00x) | 408.7 (B=256, 19 GPU, 0.52x) | 791.8 (B=256, 10 GPU, 1.00x) |
| Llama-3.1-405B | 1048576 | 8.9 (B=4, 15 GPU, 1.00x) | 1.9 (B=4, 51 GPU, 0.21x) | 8.9 (B=4, 13 GPU, 1.00x) | 4.6 (B=4, 26 GPU, 0.52x) | 8.9 (B=4, 13 GPU, 1.00x) |

## Fixed node of 1 GPU(s): largest batch within TPOT <= 100 ms (capacity-limited)

Each cell: tokens/s/GPU (max batch, ratio vs C1). "-" = model does not fit even at batch 1.

| Model | Context | C1_hbm_only | C2_hbf_dram | C3_iso_stack | C3_iso_shoreline | C4_behind_hbm |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 2048 | 23641.7 (B=654, 1.00x) | 4395.8 (B=182, 0.19x) | 23726.2 (B=735, 1.00x) | 11731.0 (B=365, 0.50x) | 23725.5 (B=735, 1.00x) |
| Llama-3.1-70B | 8192 | 280.3 (B=8, 1.00x) | 246.5 (B=16, 0.88x) | 1182.3 (B=64, 4.22x) | 467.8 (B=30, 1.67x) | 1181.6 (B=64, 4.22x) |
| Llama-3.1-405B | 8192 | - | 47.1 (B=4, C1 does not fit) | 384.4 (B=37, C1 does not fit) | - | 384.0 (B=37, C1 does not fit) |
| Llama-3.1-405B | 131072 | - | - | 21.6 (B=2, C1 does not fit) | - | 21.6 (B=2, C1 does not fit) |
| Qwen3-235B-A22B | 32768 | - | 115.4 (B=4, C1 does not fit) | 427.4 (B=27, C1 does not fit) | 183.6 (B=8, C1 does not fit) | 427.1 (B=27, C1 does not fit) |
| Llama-3.1-405B | 1048576 | - | - | - | - | - |

## Fixed node of 8 GPU(s): largest batch within TPOT <= 100 ms (capacity-limited)

Each cell: tokens/s/GPU (max batch, ratio vs C1). "-" = model does not fit even at batch 1.

| Model | Context | C1_hbm_only | C2_hbf_dram | C3_iso_stack | C3_iso_shoreline | C4_behind_hbm |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 2048 | 24189.4 (B=5419, 1.00x) | 4459.9 (B=1321, 0.18x) | 24191.8 (B=5419, 1.00x) | 12076.8 (B=2674, 0.50x) | 24191.8 (B=5419, 1.00x) |
| Llama-3.1-70B | 8192 | 1905.9 (B=461, 1.00x) | 364.4 (B=128, 0.19x) | 1912.0 (B=461, 1.00x) | 924.6 (B=257, 0.49x) | 1911.8 (B=461, 1.00x) |
| Llama-3.1-405B | 8192 | 950.5 (B=229, 1.00x) | 205.1 (B=81, 0.22x) | 1048.8 (B=325, 1.10x) | 472.0 (B=162, 0.50x) | 1048.5 (B=325, 1.10x) |
| Llama-3.1-405B | 131072 | 48.2 (B=8, 1.00x) | 12.2 (B=4, 0.25x) | 65.3 (B=20, 1.36x) | 27.3 (B=8, 0.57x) | 65.3 (B=20, 1.36x) |
| Qwen3-235B-A22B | 32768 | 734.6 (B=162, 1.00x) | 147.9 (B=51, 0.20x) | 767.8 (B=204, 1.05x) | 357.4 (B=102, 0.49x) | 767.6 (B=204, 1.04x) |
| Llama-3.1-405B | 1048576 | 6.0 (B=1, 1.00x) | - | 7.7 (B=2, 1.28x) | 3.4 (B=1, 0.57x) | 7.7 (B=2, 1.28x) |

Published ratios for comparison:

- H3 (IEEE CAL 2026): max batch 2.6x at 1M ctx, throughput/power up to 2.69x vs HBM-only (secondary sources)
- FlashAccel (2607.10186): throughput/GPU at 100 ms SLO: CSI (C4-like) 2.49x, CLI (C3-like) 2.00x vs 8xH200
- FLINT (2608.25062): decode throughput/GPU 1.5-3.1x vs HBM-only
- DASH (2608.14333): DirectOnly (C3) ~ RelayOnly (C4) at equal links
