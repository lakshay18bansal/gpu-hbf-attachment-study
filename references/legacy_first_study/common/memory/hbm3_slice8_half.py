"""Ramulator config: HBM3 at half data rate for the 8-partition slice (C3-iso-shoreline).

Identical to hbm3_slice8.py except DATA_RATE_MBPS 6400 -> 3200. In C3-iso-shoreline
the slice's 8 shoreline sites are split 4 HBM + 4 HBF (02_design.md section 1),
so the HBM tier gets half the bandwidth. It keeps 8 channels (CHANNELS must equal
-gpgpu_n_mem = 8, and the GPU side stays identical in every config); the
bandwidth is halved through the data rate instead. timings_at_rate() keeps every
timing in ns (only nBL, nCCDS, nCCDR, nPPD stay in clock cycles), so tCK doubles
(312 -> 625 ps), DRAM latencies in ns are unchanged and the burst takes twice as
long. Note: the GPGPU-Sim DRAM clock domain follows the tick (1602.6 MHz here).

    8 x 64 x 3200 / 8 = 204.8 GB/s

Derived from accelsim-work/ramulator/configs/accelsim/hbm3_rtx2060s.py with
CHANNELS 16 -> 8, FIXED_LATENCY_NS pinned in ns, and DRAM power off.

The slice (common/gpu/qwen3_slice.config) has 8 GPU memory partitions
(-gpgpu_n_mem 8). Each partition becomes one HBM3 channel (64-bit = 2
pseudo-channels x 32-bit; 2 SIDs x 4 bank groups x 4 banks per pseudo-channel;
1 GB), i.e. half of one HBM3 stack (8 of its 16 channels, 8 GB). Timings:
Ramulator's HBM3_6400Mbps preset (labelled a "guesstimate" by Ramulator).

    peak bandwidth = CHANNELS x 64 bit x DATA_RATE_MBPS / 8

Front-end latency: 52.49 ns, the value hbm3_rtx2060s.py gets at its 1905 MHz
core clock (100 core cycles). It is given in ns here, not through
core_cycles_to_ns(), so it does not change with GPU_CORE_CLOCK_MHZ: the slice
runs at 1410 MHz, and 02_design.md fixes 52.49 ns for every tier.

DRAM power: off ("none"); power is out of scope for this study (02_design.md
section 0). RAMULATOR_SET="DRAM_POWER_MODEL=ayna" turns the Ayna HBM model back on.

Refresh: all-bank refresh (REFab, every 3.9 us per pseudo-channel), as in
hbm3_rtx2060s.py (Ramulator's HBM3 controller stalls the whole channel while a
per-bank refresh waits, so hbm_perbank overstates refresh time).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Ramulator's Python package is used in place (read-only): PYTHONPATH wins if set
sys.path.append(os.environ.get("RAMULATOR_PYTHON",
                               "<btp_work>/accelsim-work/ramulator/python"))
import ramulator  # noqa: E402
from accelsim_common import (apply_overrides, build_system,  # noqa: E402
                             ns_to_cycles, summary, timings_at_rate)

# ============================ EDIT HERE ======================================
P = {
    "CHANNELS": 8,                # must equal -gpgpu_n_mem of the GPU config (qwen3_slice: 8)
    "DATA_RATE_MBPS": 3200,       # per-pin data rate: half of 6400 (C3-iso-shoreline)
    "READ_BUFFER": 64,
    "WRITE_BUFFER": 64,
    "FIXED_LATENCY_NS": 100 / 1905.0 * 1e3,  # memory-controller front end, 52.49 ns
                                  # (= 100 core cycles at 1905 MHz, as hbm3_rtx2060s.py)
    "REFRESH": "allbank",         # allbank (REFab) | hbm_perbank (REFpb) | none
    "SCHEDULER": "GPGPUSimFRFCFS",  # GPGPUSimFRFCFS (row hits first, as Accel-Sim) | FRFCFS | FRFCFSRowHit
    "ROW_POLICY": "open",
    "ADDR_MAPPER": "GPUBgCoBaRo",
    "DRAM_POWER_MODEL": "none",   # none (power out of scope) | ayna
    "AYNA": {},                   # Ayna options, e.g. {"dq_rate": 0.5} (data toggle rates)
    "TIMING_CYCLES": {},          # clock cycles (CK)
    "TIMING_NS": {},              # nanoseconds
    "ORG": {},                    # organization of one channel on top of ORG_PRESET
}
ORG_PRESET = "HBM3_16Gb_8hi"
TIMING_PRESET = "HBM3_6400Mbps"
CLOCK_TIMINGS = {"nBL", "nCCDS", "nCCDR", "nPPD"}
# =============================================================================
P = apply_overrides(P)

T = timings_at_rate(ramulator.dram.HBM3, ORG_PRESET, TIMING_PRESET,
                    P["DATA_RATE_MBPS"], CLOCK_TIMINGS, P["ORG"])
T.update(P["TIMING_CYCLES"])
T.update({k: ns_to_cycles(v, T["tCK_ps"]) for k, v in P["TIMING_NS"].items()})


def make_dram():
    return ramulator.dram.HBM3(org_preset=ORG_PRESET, timing_preset=TIMING_PRESET, **P["ORG"], **T)


ORG, _ = make_dram().resolve()
sim = build_system(P, make_dram, ramulator.controller.HBM34)
SUMMARY = "HBM3: " + summary(P, ORG, T, 64, T["tCK_ps"])
