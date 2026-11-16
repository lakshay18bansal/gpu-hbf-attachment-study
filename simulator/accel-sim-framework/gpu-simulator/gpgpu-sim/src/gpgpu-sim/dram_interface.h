// What a memory partition (memory_partition_unit) needs from its DRAM model.
// Implemented by dram_t (GPGPU-Sim's own DRAM model) and ramulator_dram_t
// (Ramulator 2.1, see ramulator_dram.h).

#ifndef DRAM_INTERFACE_H
#define DRAM_INTERFACE_H

#include <stdio.h>
#include <zlib.h>

class mem_fetch;

class dram_interface {
 public:
  virtual ~dram_interface() {}

  // Request path (called on the DRAM clock by memory_partition_unit)
  virtual bool full(bool is_write) const = 0;
  virtual void push(mem_fetch *data) = 0;
  virtual void cycle() = 0;
  virtual mem_fetch *return_queue_top() = 0;
  virtual mem_fetch *return_queue_pop() = 0;

  // Stats
  virtual void print(FILE *simFile) const = 0;
  virtual void print_stat(FILE *simFile) = 0;
  virtual void visualize() const = 0;
  virtual void visualizer_print(gzFile visualizer_file) = 0;
  virtual void dram_log(int task) = 0;
  virtual void set_dram_power_stats(unsigned &cmd, unsigned &activity,
                                    unsigned &nop, unsigned &act,
                                    unsigned &pre, unsigned &rd, unsigned &wr,
                                    unsigned &wr_WB, unsigned &req) const = 0;

  // >= 0: how many requests this channel may have outstanding (in the model
  // plus its return queue). Replaces the limit the partition otherwise derives
  // from -gpgpu_frfcfs_dram_sched_queue_size + -gpgpu_dram_return_queue_size.
  virtual int max_outstanding() const { return -1; }

  // true: the model applies its own front-end latency, so the partition does
  // not add -dram_latency.
  virtual bool has_own_latency() const { return false; }
};

#endif  // DRAM_INTERFACE_H
