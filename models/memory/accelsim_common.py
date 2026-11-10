# Copied unchanged (apart from this header) from
#   simulator/ramulator2/configs/accelsim/accelsim_common.py
# so that the configs in this folder are importable on their own.
# Source sha256: 8942bc8a0db49f811ad1a827ededd61204a77f881b4e66405b8b32e89fa76630
"""Helpers shared by the Accel-Sim memory configs in this folder.

A config file defines a dict of parameters (its "EDIT HERE" block) and calls
build_system(). Any parameter can also be overridden for one run, without
editing the file, through the environment variable RAMULATOR_SET:

    RAMULATOR_SET="DATA_RATE_MBPS=28000,REFRESH=none"

(scripts/simulate.sh passes --ram-set "..." through as RAMULATOR_SET).
Timing parameters can be overridden the same way: "nCL=30" sets nCL in clock
cycles, "ns.nCL=15" sets it in nanoseconds; organization fields with "org.":
"org.bank=8", "org.pseudochannel=4".
"""

import math
import os

import ramulator
import ramulator.power  # Ayna DRAM power model config


def gpu_core_clock_mhz():
    """GPU core clock of the run: first value of -gpgpu_clock_domains (passed in by
    scripts/simulate.sh as GPU_CORE_CLOCK_MHZ; 1905 MHz = SM75_RTX2060_S default)."""
    return float(os.environ.get("GPU_CORE_CLOCK_MHZ", "1905"))


def core_cycles_to_ns(cycles):
    """GPU core cycles -> ns at the run's core clock (keeps a latency fixed in cycles)."""
    return cycles / gpu_core_clock_mhz() * 1e3


def apply_overrides(params):
    """Return params updated from RAMULATOR_SET (type follows the default)."""
    params = dict(params)
    # Write drain: the controller switches to writes when its write queue is more
    # than WR_HIGH_WATERMARK full (or no reads wait), and back to reads below
    # WR_LOW_WATERMARK (Ramulator defaults 0.8 / 0.2)
    params.setdefault("WR_HIGH_WATERMARK", 0.8)
    params.setdefault("WR_LOW_WATERMARK", 0.2)
    # DRAM power model inside Ramulator: "ayna" (HBM2/HBM3/HBM4 only) or "none".
    # With "ayna", Accel-Sim reports a combined power: AccelWattch for the GPU chip
    # + Ayna for the DRAM (see scripts/combined_power.py).
    params.setdefault("DRAM_POWER_MODEL", "none")
    params.setdefault("TIMING_CYCLES", {})
    params.setdefault("TIMING_NS", {})
    params.setdefault("ORG", {})
    spec = os.environ.get("RAMULATOR_SET", "").strip()
    if not spec:
        return params
    for item in spec.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"RAMULATOR_SET: expected KEY=VALUE, got '{item}'")
        key, value = (x.strip() for x in item.split("=", 1))
        if key.startswith("ns."):
            params["TIMING_NS"][key[3:]] = float(value)
        elif key.startswith("org."):
            params["ORG"][key[4:]] = int(value)
        elif key.startswith("n") and key not in params:
            params["TIMING_CYCLES"][key] = int(value)
        elif key in params:
            default = params[key]
            if isinstance(default, bool):
                params[key] = value.lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                params[key] = int(float(value))
            elif isinstance(default, float):
                params[key] = float(value)
            else:
                params[key] = value
        else:
            raise ValueError(f"RAMULATOR_SET: unknown parameter '{key}' "
                             f"(known: {', '.join(sorted(params))})")
    return params


def timings_at_rate(std_cls, org_preset, timing_preset, data_rate_mbps,
                    clock_timings, org=None):
    """Resolve a Ramulator timing preset and re-express it at another data rate.

    Timings in `clock_timings` (e.g. the burst length) keep their cycle count;
    every other timing keeps its value in nanoseconds, so changing the data rate
    changes bandwidth but not DRAM latency. Returns (timings, tCK_ps).
    """
    # organization overrides matter: refresh timings depend on density, stack height, banks
    _, base = std_cls(org_preset=org_preset, timing_preset=timing_preset, **(org or {})).resolve()
    base_tck, base_rate = base["tCK_ps"], base["rate"]
    tck = base_tck * base_rate / data_rate_mbps
    out = {}
    for name, value in base.items():
        if name in ("rate", "tCK_ps") or value is None or value < 0:
            continue
        if name in clock_timings:
            out[name] = value
        else:
            out[name] = max(1, math.ceil(value * base_tck / tck - 1e-6))
    out["rate"] = int(data_rate_mbps)
    out["tCK_ps"] = int(round(tck))
    return out


def ns_to_cycles(ns, tck_ps):
    return max(1, math.ceil(ns * 1000.0 / tck_ps - 1e-6))


def _org_text(org):
    keys = [k for k in ("stack_height", "pseudochannel", "sid", "bankgroup", "bank", "row",
                        "column", "dq", "channel_density") if k in org]
    return " ".join(f"{k}={org[k]}" for k in keys)


def build_system(p, make_dram, controller_cls):
    """Build the full DRAM system: one controller per GPU memory channel.

    p             parameter dict (after apply_overrides)
    make_dram     function returning a new DRAM spec object (one per channel)
    controller_cls Ramulator controller class (e.g. ramulator.controller.GenericDDR)
    """
    refresh = {
        "none": ramulator.refresh_manager.NoRefresh,
        "allbank": ramulator.refresh_manager.AllBank,
        "perbank": ramulator.refresh_manager.PerBank,
        "hbm_perbank": ramulator.refresh_manager.HBM34PerBankRefresh,
    }
    if not 0 < p["WR_LOW_WATERMARK"] < p["WR_HIGH_WATERMARK"] <= 1:
        # Ramulator leaves write mode when the write queue is *below* the low
        # watermark, so 0 would keep it in write mode forever (GPU deadlock)
        raise ValueError("need 0 < WR_LOW_WATERMARK < WR_HIGH_WATERMARK <= 1")
    if p["REFRESH"] not in refresh:
        raise ValueError(f"REFRESH must be one of {sorted(refresh)}")
    scheduler = {"GPGPUSimFRFCFS": ramulator.scheduler.GPGPUSimFRFCFS,
                 "FRFCFS": ramulator.scheduler.FRFCFS,
                 "FRFCFSRowHit": ramulator.scheduler.FRFCFSRowHit}[p["SCHEDULER"]]
    row_policy = {"open": ramulator.row_policy.Open,
                  "closed": ramulator.row_policy.ClosedCAP}[p["ROW_POLICY"]]
    mapper = getattr(ramulator.addr_mapper, p["ADDR_MAPPER"])
    if p["DRAM_POWER_MODEL"] not in ("none", "ayna"):
        raise ValueError("DRAM_POWER_MODEL must be 'ayna' or 'none'")
    is_hbm = type(make_dram()).__name__.startswith("HBM")
    if is_hbm and p["REFRESH"] == "perbank":
        raise ValueError("REFRESH=perbank is the DDR/GDDR per-bank refresh; for HBM use "
                         "hbm_perbank (or allbank / none)")
    if not is_hbm and p["REFRESH"] == "hbm_perbank":
        raise ValueError("REFRESH=hbm_perbank is for HBM3/HBM4 only; use perbank, allbank or none")
    if p["DRAM_POWER_MODEL"] == "ayna" and p["REFRESH"] in ("perbank", "hbm_perbank"):
        raise ValueError("DRAM_POWER_MODEL=ayna: Ayna does not model per-bank refresh (REFpb), so its "
                         "energy would be missing; use REFRESH=allbank or none")
    if p["DRAM_POWER_MODEL"] == "ayna" and not type(make_dram()).__name__.startswith("HBM"):
        raise ValueError("DRAM_POWER_MODEL=ayna: Ayna models HBM2/HBM3/HBM4 only "
                         f"(this config uses {type(make_dram()).__name__}); use 'none'")

    def plugins():
        # GPGPUSimCounter is required by Accel-Sim: per-channel command counts for
        # its stats and AccelWattch (DRAM energy = reads, writes and precharges)
        out = [ramulator.controller_plugin.GPGPUSimCounter()]
        if p["DRAM_POWER_MODEL"] == "ayna":
            # Ayna HBM power model (CMU-SAFARI HBM-Power): command- and
            # current-based DRAM energy, reported per channel in ramulator_stats
            out.append(ramulator.controller_plugin.Ayna(
                power=ramulator.power.ayna_power(**p.get("AYNA", {}))))
        return out

    controllers = []
    for _ in range(p["CHANNELS"]):
        controllers.append(controller_cls(
            dram=make_dram(),
            read_buffer_size=p["READ_BUFFER"],
            write_buffer_size=p["WRITE_BUFFER"],
            wr_high_watermark=float(p["WR_HIGH_WATERMARK"]),
            wr_low_watermark=float(p["WR_LOW_WATERMARK"]),
            scheduler=scheduler(),
            refresh_manager=refresh[p["REFRESH"]](),
            row_policy=row_policy(),
            addr_mapper=mapper(),
            controller_plugins=plugins(),
        ))

    frontend = ramulator.frontend.GPGPUSim(
        clock_ratio=1, fixed_latency_ns=float(p["FIXED_LATENCY_NS"]))
    memory = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        controllers=controllers,
        # Accel-Sim decides the channel (GPU memory partition) of each request
        channel_mapper=ramulator.channel_mapper.PassThroughChannelMapper(),
    )
    return ramulator.Simulation(frontend, memory)


def summary(p, org, timings, channel_width_bits, tck_ps, ticks_per_ck=1):
    """One-paragraph description printed by simulate.sh.

    org: resolved organization dict (or None); its data width replaces channel_width_bits
    (HBM: pseudo-channels x dq bits per channel)."""
    if org:
        channel_width_bits = (org["pseudochannel"] * org["dq"] if "pseudochannel" in org
                              else org["channel_width"])
    rate = timings.get("rate", 0)
    bw = p["CHANNELS"] * channel_width_bits * rate / 8 / 1000  # GB/s
    t = lambda n: f"{n}={timings[n]} ({timings[n] * tck_ps / 1000:.2f} ns)"
    keys = [k for k in ("nCL", "nRCDRD", "nRP", "nRAS", "nRC", "nBL", "nCCDS",
                        "nCCDL", "nRRDS", "nFAW", "nWR") if k in timings]
    return (f"{p['CHANNELS']} channels x {channel_width_bits}-bit at {rate} Mbps/pin"
            f" = {bw:.1f} GB/s peak; clock {1e6 / tck_ps:.1f} MHz"
            f" (tCK {tck_ps} ps); refresh {p['REFRESH']}; mapper {p['ADDR_MAPPER']};"
            f" buffers {p['READ_BUFFER']}/{p['WRITE_BUFFER']};"
            f" fixed latency {p['FIXED_LATENCY_NS']:.3f} ns"
            + (f"; organization per channel: {_org_text(org)}" if org else "")
            + "\n  timings (cycles): "
            + ", ".join(t(k) for k in keys))
