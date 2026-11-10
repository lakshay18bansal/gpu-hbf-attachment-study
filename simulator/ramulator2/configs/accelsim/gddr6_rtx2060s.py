"""Ramulator config: realistic GDDR6 for the RTX 2060 Super (its real memory type).

16 channels x 16-bit GDDR6 (one 8 Gb x16 device per channel, 512 MB each, 8 GB
total). Timings: Ramulator's GDDR6_14000_1350mV_double preset (Samsung
K4Z80325BC datasheet).

Bandwidth knob: DATA_RATE_MBPS. Changing it changes the clock (tCK) and keeps
every timing at the same value in nanoseconds (except the ones listed in
CLOCK_TIMINGS, which are fixed numbers of clock cycles in the GDDR6 standard),
so bandwidth scales and DRAM latency stays the same:
    peak bandwidth = CHANNELS x 16 bit x DATA_RATE_MBPS / 8
    14000 Mbps -> 448 GB/s (real card), 28000 -> 896 GB/s, 7000 -> 224 GB/s
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ramulator  # noqa: E402
from accelsim_common import (apply_overrides, build_system, core_cycles_to_ns,  # noqa: E402
                             ns_to_cycles, summary, timings_at_rate)

# ============================ EDIT HERE ======================================
P = {
    "CHANNELS": 16,               # must equal -gpgpu_n_mem of the GPU config
    "DATA_RATE_MBPS": 14000,      # per-pin data rate (bandwidth knob)
    "READ_BUFFER": 64,            # controller read queue per channel
    "WRITE_BUFFER": 64,           # controller write queue per channel
    "FIXED_LATENCY_NS": core_cycles_to_ns(100),  # memory-controller front end: 100 GPU core
                                  # cycles at the GPU's core clock (52.49 ns @ 1905 MHz)
    "REFRESH": "allbank",         # allbank | perbank | none
    "SCHEDULER": "GPGPUSimFRFCFS",  # GPGPUSimFRFCFS (row hits first, as Accel-Sim) | FRFCFS | FRFCFSRowHit
    "ROW_POLICY": "open",         # open | closed
    "ADDR_MAPPER": "GPUBgCoBaRo",  # GPUBgCoBaRo | RoBaRaCoCh | MOP4CLXOR | ChRaBaRoCo
    # DRAM power: Ramulator has no GDDR6 power model, so DRAM power comes from
    # AccelWattch (calibrated on the real RTX 2060 Super's GDDR6)
    "DRAM_POWER_MODEL": "none",
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
