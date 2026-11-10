// Ramulator components for running Ramulator as the DRAM model of
// GPGPU-Sim / Accel-Sim (the bridge lives in Accel-Sim:
// gpu-simulator/gpgpu-sim/src/gpgpu-sim/ramulator_dram.cc).
//
// Components (all selectable from a Ramulator config):
//   frontend          GPGPUSim         External-style frontend plus the fixed
//                                      latency of the GPU memory controller
//                                      front end (param fixed_latency_ns)
//   controller_plugin GPGPUSimCounter  per-channel command counters read by
//                                      Accel-Sim for its stats and AccelWattch
//   addr_mapper       GPUBgCoBaRo      GPU-style mapping inside a channel
//
// This header only exposes plain data and functions; the registered
// component classes stay inside libramulator.

#ifndef RAMULATOR_GPGPUSIM_GPGPUSIM_H
#define RAMULATOR_GPGPUSIM_GPGPUSIM_H

#include <string>
#include <vector>

namespace Ramulator {

class IFrontEnd;

namespace GPGPUSim {

// Filled by one GPGPUSimCounter plugin (one per controller = channel).
struct ChannelInfo {
  int channel = -1;

  // Command counts since the start of the simulation
  unsigned long long n_act = 0;
  unsigned long long n_pre = 0;  // PREpb + PREab + auto-precharges (RDA/WRA)
  unsigned long long n_rd = 0;   // RD + RDA
  unsigned long long n_wr = 0;   // WR + WRA
  unsigned long long n_ref = 0;  // REFab + REFpb
  unsigned long long n_issued = 0;

  // Device / controller description (from the DRAM spec and controller)
  std::vector<std::string> level_names;  // "Channel", "BankGroup", ...
  std::vector<int> level_sizes;
  int internal_prefetch = 1;
  int tx_bytes = 0;
  int burst_ticks = 1;  // data-bus ticks per RD/WR (nBL)
  int data_buses = 1;   // independent data buses (HBM pseudo-channels)
  int read_buffer_size = 0;
  int write_buffer_size = 0;
  float tck_ns = 0;  // one controller tick
  std::string standard;
  std::string addr_mapper;
};

// One entry per GPGPUSimCounter created, in creation order (= channel order).
// Clear it before creating a memory system.
std::vector<ChannelInfo*>& channels();

// Fixed front-end latency of a GPGPUSim frontend; -1 for any other frontend.
float fixed_latency_ns(IFrontEnd* frontend);

}  // namespace GPGPUSim
}  // namespace Ramulator

#endif  // RAMULATOR_GPGPUSIM_GPGPUSIM_H
