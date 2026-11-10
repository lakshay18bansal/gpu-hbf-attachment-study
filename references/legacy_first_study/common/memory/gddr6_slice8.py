"""Ramulator config: GDDR6 for the 8-partition GPU slice (C2's conventional DRAM tier).

Derived from accelsim-work/ramulator/configs/accelsim/gddr6_rtx2060s.py. The only
changes are CHANNELS 16 -> 8 and FIXED_LATENCY_NS pinned in ns. Same device,
timings, refresh, scheduler, row policy and mapper.

8 channels x 16-bit GDDR6 (one 8 Gb x16 device per channel, 512 MB each, 4 GB
total), one channel per GPU memory partition of common/gpu/qwen3_slice.config
(-gpgpu_n_mem 8). Timings: Ramulator's GDDR6_14000_1350mV_double preset
(Samsung K4Z80325BC datasheet).

Bandwidth knob: DATA_RATE_MBPS (the clock changes, every timing keeps its value
in ns except the ones in CLOCK_TIMINGS, which the GDDR6 standard defines in clock
cycles):
    peak bandwidth = CHANNELS x 16 bit x DATA_RATE_MBPS / 8
    8 x 16 x 14000 / 8 = 224 GB/s

Front-end latency: 52.49 ns (100 core cycles at 1905 MHz, the value of
gddr6_rtx2060s.py and of the HBM3 configs), given in ns so it does not follow
GPU_CORE_CLOCK_MHZ (02_design.md: the same front end for every tier).

DRAM power: none (Ramulator has no GDDR6 power model; power is out of scope).
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
    "DATA_RATE_MBPS": 14000,      # per-pin data rate (bandwidth knob)
    "READ_BUFFER": 64,            # controller read queue per channel
    "WRITE_BUFFER": 64,           # controller write queue per channel
    "FIXED_LATENCY_NS": 100 / 1905.0 * 1e3,  # memory-controller front end, 52.49 ns
                                  # (= 100 core cycles at 1905 MHz, as gddr6_rtx2060s.py)
    "REFRESH": "allbank",         # allbank | perbank | none
    "SCHEDULER": "GPGPUSimFRFCFS",  # GPGPUSimFRFCFS (row hits first, as Accel-Sim) | FRFCFS | FRFCFSRowHit
    "ROW_POLICY": "open",         # open | closed
    "ADDR_MAPPER": "GPUBgCoBaRo",  # GPUBgCoBaRo | RoBaRaCoCh | MOP4CLXOR | ChRaBaRoCo
    "DRAM_POWER_MODEL": "none",   # Ramulator has no GDDR6 power model
    "TIMING_CYCLES": {},          # e.g. {"nCL": 30}         (clock cycles)
    "TIMING_NS": {},              # e.g. {"nRCDRD": 20.0}    (nanoseconds)
    "ORG": {},                    # organization overrides, e.g. {"row": 1 << 15}
}
ORG_PRESET = "GDDR6_8Gb_x16"
TIMING_PRESET = "GDDR6_14000_1350mV_double"
# Defined in clock cycles by the GDDR6 standard (do not scale with the clock)
CLOCK_TIMINGS = {"nBL", "nCCDS", "nCCDL", "nPPD", "nRTP", "nCWL"}
# =============================================================================
P = apply_overrides(P)

T = timings_at_rate(ramulator.dram.GDDR6, ORG_PRESET, TIMING_PRESET,
                    P["DATA_RATE_MBPS"], CLOCK_TIMINGS, P["ORG"])
T["nRTW"] = T["nCL"] + 2 * T["nBL"] - T["nCWL"] + 3  # Ramulator's GDDR6 formula
T.update(P["TIMING_CYCLES"])
T.update({k: ns_to_cycles(v, T["tCK_ps"]) for k, v in P["TIMING_NS"].items()})


def make_dram():
    return ramulator.dram.GDDR6(org_preset=ORG_PRESET, timing_preset=TIMING_PRESET, **P["ORG"], **T)


ORG, _ = make_dram().resolve()
sim = build_system(P, make_dram, ramulator.controller.GenericDDR)
SUMMARY = "GDDR6: " + summary(P, ORG, T, 16, T["tCK_ps"])
