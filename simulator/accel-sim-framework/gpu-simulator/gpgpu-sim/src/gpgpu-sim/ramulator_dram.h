// Bridge between GPGPU-Sim's memory partitions and Ramulator 2.1.
//
// With -gpgpu_ramulator_enabled 1, every memory partition (channel) uses a
// ramulator_dram_t instead of GPGPU-Sim's dram_t. All channels share one
// Ramulator memory system built from -gpgpu_ramulator_config; channel i of the
// GPU is Ramulator controller i. Everything about the DRAM - device, timing,
// clock, scheduler, row policy, refresh, address mapping inside the channel,
// queue sizes and the fixed controller front-end latency - comes from that
// Ramulator config. GPGPU-Sim still owns the GPU side: SMs, caches, the
// interconnect, the L2 and which channel an address belongs to.
//
// Ramulator components used here (frontend GPGPUSim, controller plugin
// GPGPUSimCounter, addr_mapper GPUBgCoBaRo) live in the Ramulator tree:
// ramulator/src/ramulator/gpgpusim/.
//
// This header includes no Ramulator headers: Ramulator needs C++20, the rest
// of GPGPU-Sim is built as C++17. Everything Ramulator-specific is in
// ramulator_dram.cc.

#ifndef RAMULATOR_DRAM_H
#define RAMULATOR_DRAM_H

#include "dram_interface.h"

class memory_config;
class ramulator_channel;

// Controller clock period (ns) of the Ramulator system in
// -gpgpu_ramulator_config; GPGPU-Sim's DRAM clock domain is set to it. The
// first call creates the shared Ramulator memory system.
double ramulator_dram_tck_ns(const memory_config *config);

class ramulator_dram_t : public dram_interface {
 public:
  ramulator_dram_t(unsigned int partition_id, const memory_config *config,
                   class memory_stats_t *stats,
                   class memory_partition_unit *mp, class gpgpu_sim *gpu);
  ~ramulator_dram_t() override;

  bool full(bool is_write) const override;
  void push(mem_fetch *data) override;
  void cycle() override;
  mem_fetch *return_queue_top() override;
  mem_fetch *return_queue_pop() override;

  void print(FILE *simFile) const override;
  void print_stat(FILE *simFile) override;
  void visualize() const override {}
  void visualizer_print(gzFile visualizer_file) override {}
  void dram_log(int task) override {}
  void set_dram_power_stats(unsigned &cmd, unsigned &activity, unsigned &nop,
                            unsigned &act, unsigned &pre, unsigned &rd,
                            unsigned &wr, unsigned &wr_WB,
                            unsigned &req) const override;

  int max_outstanding() const override;
  bool has_own_latency() const override { return true; }

 private:
  ramulator_channel *m_ch;
};

#endif  // RAMULATOR_DRAM_H
