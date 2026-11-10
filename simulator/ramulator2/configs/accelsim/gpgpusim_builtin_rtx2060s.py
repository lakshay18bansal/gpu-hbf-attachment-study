"""Ramulator config: GPGPU-Sim's OWN DRAM model of the RTX 2060 Super, rebuilt in Ramulator.

Purpose: validation. Every parameter below is copied from
accel-sim-framework/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM75_RTX2060_S/gpgpusim.config
and derived the way GPGPU-Sim does it (memory_config::init in gpu-sim.h), so a
run with this config should closely match a run with GPGPU-Sim's built-in DRAM
model. Bank/row/column come from GPGPU-Sim's own address decode
(-gpgpu_mem_addr_mapping) through PassThroughAddrMapper.

All timings are in cycles of the DRAM command clock (DRAM_CLOCK_MHZ), as in
-gpgpu_dram_timing_opt.

Scheduler: GPGPUSimFRFCFS (ramulator/src/ramulator/gpgpusim/gpgpusim_frfcfs.cpp)
implements GPGPU-Sim's policy: an open row is not closed while queued requests
hit it; then ready, row-hit, oldest. (Ramulator's own FRFCFS is "ready, then
oldest", which closes rows that still have queued hits.)

Known differences that remain:
  - Ramulator keeps reads and writes in separate 64-entry buffers and drains
    writes in batches (watermarks); GPGPU-Sim has one shared queue.
  - GPGPU-Sim hands one request per bank to the bank and schedules one bank
    per cycle; Ramulator picks the best command over all banks each cycle.
  - tCK is rounded to whole picoseconds (286 ps instead of 285.71 ps, 0.1%).
  - GPGPU-Sim's tRTPL (read to precharge in the same bank group, 4 cycles) has
    no Ramulator equivalent; tRTP (same bank, 4 cycles) is modelled.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ramulator  # noqa: E402
from accelsim_common import apply_overrides, build_system, core_cycles_to_ns, summary  # noqa: E402

# ============================ EDIT HERE ======================================
P = {
    "CHANNELS": 16,               # -gpgpu_n_mem 16 (must equal it)
    "DRAM_CLOCK_MHZ": 3500.0,     # DRAM value of -gpgpu_clock_domains
    # -gpgpu_dram_timing_opt "nbk=16:CCD=4:RRD=10:RCD=20:RAS=50:RP=20:RC=62:
    #                         CL=20:WL=8:CDLR=9:WR=20:nbkgrp=4:CCDL=4:RTPL=4"
    "CCD": 4, "RRD": 10, "RCD": 20, "RAS": 50, "RP": 20, "RC": 62,
    "CL": 20, "WL": 8, "CDLR": 9, "WR": 20, "CCDL": 4,
    "BURST_LENGTH": 16,           # -gpgpu_dram_burst_length 16
    "DATA_COMMAND_FREQ_RATIO": 4,  # -dram_data_command_freq_ratio 4
    "READ_BUFFER": 64,            # -gpgpu_frfcfs_dram_sched_queue_size 64
    "WRITE_BUFFER": 64,
    "FIXED_LATENCY_NS": core_cycles_to_ns(100),  # -dram_latency 100 core cycles (52.49 ns @ 1905 MHz)
    "REFRESH": "none",            # GPGPU-Sim models no refresh
    "SCHEDULER": "GPGPUSimFRFCFS",  # -gpgpu_dram_scheduler 1: FR-FCFS, row hits first
    "ROW_POLICY": "open",         # GPGPU-Sim keeps rows open
    "ADDR_MAPPER": "PassThroughAddrMapper",  # use GPGPU-Sim's bank/row/column decode
}
# =============================================================================
P = apply_overrides(P)

tck_ps = int(round(1e6 / P["DRAM_CLOCK_MHZ"]))
burst = P["BURST_LENGTH"] // P["DATA_COMMAND_FREQ_RATIO"]  # data cycles per 32B burst
T = {
    # same derivations as memory_config::init
    "rate": int(P["BURST_LENGTH"] / burst * P["DRAM_CLOCK_MHZ"]),  # Mbps per pin
    "tCK_ps": tck_ps,
    "nBL": burst,
    "nCL": P["CL"],
    "nCWL": P["WL"],
    "nRCDRD": P["RCD"],
    "nRCDWR": P["RCD"] - (P["WL"] + 1),            # tRCDWR
    "nRP": P["RP"],
    "nRAS": P["RAS"],
    "nRC": P["RC"],
    "nWR": P["WR"],          # WR->PRE = WL + burst + WR  (= tWTP)
    "nRTP": burst,           # RD->PRE = BL/ratio        (RTPc)
    "nCCDS": P["CCD"],
    "nCCDL": P["CCDL"],
    "nRRDS": P["RRD"],
    "nRRDL": P["RRD"],
    "nWTRS": P["CDLR"],      # WR->RD = WL + burst + CDLR (= tWTR)
    "nWTRL": P["CDLR"],
    "nRTW": P["CL"] + burst + 2 - P["WL"],          # tRTW
    "nFAW": 1,               # GPGPU-Sim has no four-activate window
    "nPPD": 1,               # nor a precharge-to-precharge delay
}
T.update(P["TIMING_CYCLES"])
T.update({k: max(1, round(v * 1000 / tck_ps)) for k, v in P["TIMING_NS"].items()})


def make_dram():
    # 16 banks (4 groups x 4), 16384 rows, 2 KB rows: same as nbk=16, nbkgrp=4
    return ramulator.dram.GDDR6(org_preset="GDDR6_8Gb_x16",
                                timing_preset="GDDR6_14000_1350mV_double", **T)


sim = build_system(P, make_dram, ramulator.controller.GenericDDR)
SUMMARY = "GPGPU-Sim built-in DRAM model in Ramulator: " + summary(P, None, T, 16, tck_ps)
