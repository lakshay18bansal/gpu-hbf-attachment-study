"""Standalone Ramulator 2.1 run: HBM3 (same device/controller settings as
accelsim-work/ramulator/configs/accelsim/hbm3_rtx2060s.py) driven by a trace
frontend instead of Accel-Sim.

Environment knobs (all optional):
  TRACE      path of the trace file
  FRONTEND   LoadStoreTrace (LD/ST <addr>)  | ReadWriteTrace (R/W <addr_vec>)
  CHANNELS   number of HBM3 channels (power of two for CacheLineInterleave)
  FE_RATIO   } frequency ratio frontend : memory = FE_RATIO : MEM_RATIO
  MEM_RATIO  } (the trace frontends offer one request per frontend tick)
  STATS_OUT  write sim.stats_yaml here (default: print to stdout)

Run (no build needed, uses the prebuilt python/ramulator/_ramulator*.so):
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<ramulator>/python python3 -m ramulator run hbm3_standalone.py
Export to YAML (what the C++ library / Accel-Sim reads):
  PYTHONPATH=<ramulator>/python python3 -m ramulator export hbm3_standalone.py -o out.yaml
"""
import os

import ramulator

TRACE = os.environ.get("TRACE", "traces/stream_ld.trace")
FRONTEND = os.environ.get("FRONTEND", "LoadStoreTrace")
CHANNELS = int(os.environ.get("CHANNELS", "1"))
FE_RATIO = int(os.environ.get("FE_RATIO", "1"))
MEM_RATIO = int(os.environ.get("MEM_RATIO", "1"))


def make_ctrl(mapper):
    hbm3 = ramulator.dram.HBM3(org_preset="HBM3_16Gb_8hi", timing_preset="HBM3_6400Mbps")
    return ramulator.controller.HBM34(
        dram=hbm3,
        read_buffer_size=64,
        write_buffer_size=64,
        scheduler=ramulator.scheduler.GPGPUSimFRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=mapper,
        # per-channel ACT/PRE/RD/WR/REF counters (the plugin Accel-Sim reads)
        controller_plugins=[ramulator.controller_plugin.GPGPUSimCounter()],
    )


if FRONTEND == "LoadStoreTrace":
    frontend = ramulator.frontend.LoadStoreTrace(clock_ratio=FE_RATIO, path=TRACE)
    channel_mapper = ramulator.channel_mapper.CacheLineInterleave()
else:  # ReadWriteTrace: the trace gives the whole address vector incl. channel
    frontend = ramulator.frontend.ReadWriteTrace(clock_ratio=FE_RATIO, path=TRACE)
    channel_mapper = ramulator.channel_mapper.PassThroughChannelMapper()

# LoadStoreTrace: flat address -> GPUBgCoBaRo (the mapper our Accel-Sim runs use)
# ReadWriteTrace: address vector already complete -> PassThroughAddrMapper
Mapper = (ramulator.addr_mapper.GPUBgCoBaRo if FRONTEND == "LoadStoreTrace"
          else ramulator.addr_mapper.PassThroughAddrMapper)
ctrls = [make_ctrl(Mapper()) for _ in range(CHANNELS)]

mem = ramulator.memory_system.GenericDRAM(
    clock_ratio=MEM_RATIO, controllers=ctrls, channel_mapper=channel_mapper)

sim = ramulator.Simulation(frontend, mem)
sim.run()
text = sim.stats_yaml
if text:
    out = os.environ.get("STATS_OUT")
    if out:
        with open(out, "w") as f:
            f.write(text)
    else:
        print(text)
