"""Ramulator config: HBM2E matched to HBF-Sim's DRAM, for all four configurations.

HBF-Sim's Qwen3 config (base/hbfsim_qwen3.config) uses GPGPU-Sim's built-in DRAM
model: 8 channels, 16 B bus, dual bus, 1512 MHz DRAM clock = 387.07 GB/s, with
"HBM2-like" timings (nbk=16, nbkgrp=4) and -dram_latency 190 core cycles. This file
gives the same device in Ramulator: 8 HBM2 channels (one per GPU memory partition,
-gpgpu_n_mem 8) of 128 bit (2 pseudo-channels x 64 bit) at 3024 Mbps/pin
(HBM2E speed grade), so

    peak bandwidth = 8 x 128 bit x 3024 Mbps / 8 = 387.07 GB/s   (= HBF-Sim)
    controller clock = 3024 / 2 = 1512 MHz                       (= HBF-Sim)

Timings: Ramulator's HBM2_2400Mbps preset (labelled a "guesstimate" by Ramulator)
re-expressed at 3024 Mbps with every timing kept in ns except the burst/CCD
timings, which the standard defines in clock cycles (timings_at_rate). Organization
HBM2_8Gb (8-high, 2 SIDs, 4 bank groups x 4 banks per pseudo-channel, 2 GB/channel).

Front-end latency: 190 core cycles at 1410 MHz = 134.75 ns, HBF-Sim's -dram_latency
(Ramulator ignores -dram_latency; this is its replacement).
Read/write buffers: 64/64 (HBF-Sim: -gpgpu_frfcfs_dram_sched_queue_size 64).
Refresh: all-bank (REFab every 3.9 us per pseudo-channel). DRAM power off.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.environ.get("RAMULATOR_PYTHON", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "simulator", "ramulator2", "python")))
import ramulator  # noqa: E402
from accelsim_common import (apply_overrides, build_system,  # noqa: E402
                             ns_to_cycles, summary, timings_at_rate)

# ============================ EDIT HERE ======================================
P = {
    "CHANNELS": 8,                # must equal -gpgpu_n_mem (HBF-Sim qwen3.config: 8)
    "DATA_RATE_MBPS": 3024,       # per-pin data rate: 387.07 GB/s over 8 x 128 bit
    "READ_BUFFER": 64,
    "WRITE_BUFFER": 64,
    "FIXED_LATENCY_NS": 190 / 1410.0 * 1e3,  # HBF-Sim -dram_latency 190 @ 1410 MHz = 134.75 ns
    "REFRESH": "allbank",         # allbank (REFab) | none
    "SCHEDULER": "GPGPUSimFRFCFS",  # row hits first, as GPGPU-Sim's -gpgpu_dram_scheduler 1
    "ROW_POLICY": "open",
    "ADDR_MAPPER": "GPUBgCoBaRo",
    "DRAM_POWER_MODEL": "none",
    "AYNA": {},
    "TIMING_CYCLES": {},          # clock cycles (CK)
    "TIMING_NS": {},              # nanoseconds
    "ORG": {},                    # organization of one channel on top of ORG_PRESET
}
ORG_PRESET = "HBM2_8Gb"
TIMING_PRESET = "HBM2_2400Mbps"
CLOCK_TIMINGS = {"nBL", "nCCDS", "nCCDL", "nCCDR"}
# =============================================================================
P = apply_overrides(P)

T = timings_at_rate(ramulator.dram.HBM2, ORG_PRESET, TIMING_PRESET,
                    P["DATA_RATE_MBPS"], CLOCK_TIMINGS, P["ORG"])
T.update(P["TIMING_CYCLES"])
T.update({k: ns_to_cycles(v, T["tCK_ps"]) for k, v in P["TIMING_NS"].items()})


def make_dram():
    return ramulator.dram.HBM2(org_preset=ORG_PRESET, timing_preset=TIMING_PRESET, **P["ORG"], **T)


ORG, _ = make_dram().resolve()
sim = build_system(P, make_dram, ramulator.controller.HBM12)
SUMMARY = "HBM2E: " + summary(P, ORG, T, 128, T["tCK_ps"])
