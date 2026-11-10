"""Ramulator config: HBM3 for the RTX 2060 Super (hypothetical: the real card has GDDR6).

Each of the GPU's 16 memory channels becomes one HBM3 channel (64-bit = 2
pseudo-channels x 32-bit; 2 SIDs x 4 bank groups x 4 banks per pseudo-channel;
1 GB), i.e. exactly one HBM3 stack of 16 GB. Timings: Ramulator's
HBM3_6400Mbps preset (labelled a "guesstimate" by Ramulator).

Bandwidth knob: DATA_RATE_MBPS (same rule as the GDDR6 config: the clock
changes, timings keep their value in nanoseconds):
    peak bandwidth = CHANNELS x 64 bit x DATA_RATE_MBPS / 8
    6400 Mbps -> 819 GB/s, 12800 -> 1638 GB/s

DRAM power: Ramulator's Ayna HBM power model (DRAM_POWER_MODEL="ayna"); the run's
summary then shows AccelWattch (GPU chip) + Ayna (DRAM) = combined power.

Refresh: all-bank refresh (REFab, every 3.9 us per pseudo-channel) by default.
Ayna has no energy model for per-bank refresh (REFpb), and Ramulator's HBM3
controller stops scheduling reads/writes on the whole channel while a per-bank
refresh waits (HBMControllerBase::try_issue_slot), so hbm_perbank both misses
refresh energy and overstates refresh time.
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
    "DATA_RATE_MBPS": 6400,       # per-pin data rate (bandwidth knob)
    "READ_BUFFER": 64,
    "WRITE_BUFFER": 64,
    "FIXED_LATENCY_NS": core_cycles_to_ns(100),  # memory-controller front end: 100 GPU core
                                  # cycles at the GPU's core clock (52.49 ns @ 1905 MHz)
    "REFRESH": "allbank",         # allbank (REFab, modelled by Ayna) | hbm_perbank (REFpb) | none
    "SCHEDULER": "GPGPUSimFRFCFS",  # GPGPUSimFRFCFS (row hits first, as Accel-Sim) | FRFCFS | FRFCFSRowHit
    "ROW_POLICY": "open",
    "ADDR_MAPPER": "GPUBgCoBaRo",
    "DRAM_POWER_MODEL": "ayna",   # ayna (HBM DRAM power from Ramulator) | none
    "AYNA": {},                   # Ayna options, e.g. {"dq_rate": 0.5} (data toggle rates)
    "TIMING_CYCLES": {},          # clock cycles (CK)
    "TIMING_NS": {},              # nanoseconds
    # Organization of one channel, on top of ORG_PRESET (powers of two), e.g.
    #   {"pseudochannel": 4}  {"bank": 8}  {"bankgroup": 8}  {"row": 1 << 15}
    #   {"stack_height": 16, "sid": 4, "die_density": 32768, "channel_density": 32768}
    "ORG": {},
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
