// Bridge between GPGPU-Sim's memory partitions and Ramulator 2.1 (see
// ramulator_dram.h). Compiled as C++20.
//
// Per DRAM clock (= one Ramulator controller tick, see init_clock_domains):
//   memory_partition_unit::dram_cycle() of every channel calls cycle():
//     - requests whose fixed front-end latency has passed are handed to
//       Ramulator controller <channel>, split into 32B transactions
//       (L2 write-backs can be up to 128B);
//   the last channel's cycle() then ticks the Ramulator memory system once.
// When Ramulator completes every part of a mem_fetch it is retired exactly
// like dram_t does it: reads go to the return queue, write-backs are done.

// Ramulator headers first: dram.h #defines READ/WRITE.
#include <ramulator/base/base.h>
#include <ramulator/base/config.h>
#include <ramulator/base/factory.h>
#include <ramulator/base/request.h>
#include <ramulator/frontend/i_frontend.h>
#include <ramulator/gpgpusim/gpgpusim.h>
#include <ramulator/memory_system/i_memory_system.h>

#include <cmath>
#include <deque>
#include <fstream>
#include <unordered_map>

#include "ramulator_dram.h"

#include "../../libcuda/gpgpu_context.h"
#include "../cuda-sim/ptx-stats.h"
#include "gpu-sim.h"
#include "l2cache.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"

namespace RG = Ramulator::GPGPUSim;

static const unsigned SECTOR_BYTES = 32;  // GPGPU-Sim's L2 <-> DRAM unit

static void fatal(const char *msg, const std::string &detail = "") {
  fprintf(stderr, "ERROR (Ramulator DRAM): %s%s\n", msg, detail.c_str());
  abort();
}

// ---------------------------------------------------------------------------
// The Ramulator memory system shared by all channels
// ---------------------------------------------------------------------------
class ramulator_memory {
 public:
  static ramulator_memory *get(const memory_config *config) {
    static ramulator_memory *instance = nullptr;
    if (!instance) instance = new ramulator_memory(config);
    return instance;
  }

  const memory_config *m_config;
  Ramulator::IFrontEnd *m_frontend = nullptr;
  Ramulator::IMemorySystem *m_memsys = nullptr;
  std::vector<RG::ChannelInfo *> m_info;  // one per channel
  std::vector<ramulator_channel *> m_channels;

  double m_tck_ns = 0;
  double m_fixed_latency_ns = 0;
  unsigned long long m_latency_ticks = 0;
  unsigned long long m_clk = 0;  // Ramulator ticks so far
  unsigned long long m_channel_bytes = 0;
  bool m_gpgpusim_decode = false;  // PassThroughAddrMapper: use GPGPU-Sim's
                                   // bank/row/column decode

  void tick();
  void write_ramulator_stats(unsigned long long gpu_cycle);
  unsigned m_snapshots = 0;

 private:
  explicit ramulator_memory(const memory_config *config);
};

ramulator_memory::ramulator_memory(const memory_config *config)
    : m_config(config) {
  const char *path = config->ramulator_config;
  if (!path || !path[0])
    fatal("-gpgpu_ramulator_enabled needs -gpgpu_ramulator_config <file.yaml>");

  RG::channels().clear();
  auto ram_cfg = Ramulator::Config::parse_config_file(path);
  m_frontend = Ramulator::Factory::create_frontend(ram_cfg);
  m_memsys = Ramulator::Factory::create_memory_system(ram_cfg);
  m_frontend->connect_memory_system(m_memsys);
  m_memsys->connect_frontend(m_frontend);
  m_info = RG::channels();

  // --- the Ramulator config must describe exactly this GPU's channels ---
  if (m_info.empty())
    fatal("every controller in the Ramulator config needs the "
          "GPGPUSimCounter controller plugin");
  if (m_info.size() != config->m_n_mem)
    fatal("channel count mismatch: ",
          std::to_string(m_info.size()) +
              " Ramulator controllers (with GPGPUSimCounter) but -gpgpu_n_mem " +
              std::to_string(config->m_n_mem));
  for (unsigned i = 0; i < m_info.size(); i++) {
    if (m_info[i]->channel != (int)i)
      fatal("controller order does not match channel ids");
    if (m_info[i]->tck_ns != m_info[0]->tck_ns ||
        m_info[i]->standard != m_info[0]->standard)
      fatal("all channels must use the same DRAM standard and clock");
    if (m_info[i]->tx_bytes < (int)SECTOR_BYTES)
      fatal("DRAM transaction size is smaller than 32 bytes");
  }
  m_fixed_latency_ns = RG::fixed_latency_ns(m_frontend);
  if (m_fixed_latency_ns < 0)
    fatal("the Ramulator config must use frontend GPGPUSim "
          "(ramulator.frontend.GPGPUSim)");

  const RG::ChannelInfo &c = *m_info[0];
  m_tck_ns = c.tck_ns;
  m_latency_ticks = llround(m_fixed_latency_ns / m_tck_ns);

  // Capacity of one channel: transaction size x addressable transactions
  m_channel_bytes = c.tx_bytes;
  for (size_t l = 1; l < c.level_sizes.size(); l++) {
    unsigned long long n = c.level_sizes[l];
    if (c.level_names[l] == "Column") n /= c.internal_prefetch;
    m_channel_bytes *= n;
  }

  if (c.addr_mapper == "PassThroughAddrMapper") {
    std::vector<std::string> expect = {"Channel", "BankGroup", "Bank", "Row",
                                       "Column"};
    if (c.level_names != expect)
      fatal("PassThroughAddrMapper (GPGPU-Sim address decode) only supports "
            "DRAM organizations Channel/BankGroup/Bank/Row/Column, not ",
            c.standard);
    m_gpgpusim_decode = true;
  }

  m_channels.assign(m_info.size(), nullptr);

  printf("Ramulator DRAM: %s\n", path);
  printf("  %zu channels of %s, %llu MB each, controller tick %.4f ns "
         "(%.1f MHz)\n",
         m_info.size(), c.standard.c_str(), m_channel_bytes >> 20, m_tck_ns,
         1e3 / m_tck_ns);
  printf("  address mapper %s, read/write buffers %d/%d, fixed front-end "
         "latency %.3f ns (%llu ticks)\n",
         c.addr_mapper.c_str(), c.read_buffer_size, c.write_buffer_size,
         m_fixed_latency_ns, m_latency_ticks);
  printf("  GPGPU-Sim DRAM options NOT used: DRAM value of "
         "-gpgpu_clock_domains, -gpgpu_dram_timing_opt, -gpgpu_dram_buswidth, "
         "-gpgpu_dram_burst_length, -dram_data_command_freq_ratio, "
         "-gpgpu_n_mem_per_ctrlr, -gpgpu_dram_scheduler, "
         "-gpgpu_frfcfs_dram_sched_queue_size, -gpgpu_dram_return_queue_size, "
         "-dram_latency, -dram_dual_bus_interface, -dram_bnk_indexing_policy, "
         "-dram_seperate_write_queue_enable, -dram_write_queue_size, "
         "-dram_elimnate_rw_turnaround%s\n",
         m_gpgpusim_decode ? ""
                           : ", -dram_bnkgrp_indexing_policy, bank/row/column "
                             "bits of -gpgpu_mem_addr_mapping");
}

// ---------------------------------------------------------------------------
// One channel (one memory partition)
// ---------------------------------------------------------------------------
class ramulator_channel {
 public:
  ramulator_channel(unsigned id, const memory_config *config,
                    memory_stats_t *stats, memory_partition_unit *mp,
                    gpgpu_sim *gpu)
      : m_id(id), m_config(config), m_stats(stats), m_mp(mp), m_gpu(gpu) {
    m_mem = ramulator_memory::get(config);
    if (id >= m_mem->m_channels.size() || m_mem->m_channels[id])
      fatal("bad channel id");
    m_mem->m_channels[id] = this;
    m_info = m_mem->m_info[id];
  }

  void push(mem_fetch *mf) {
    m_pending.push_back({mf, m_mem->m_clk + m_mem->m_latency_ticks});
    m_n_req++;
    m_stats->total_n_access++;
    if (mf->get_type() == WRITE_REQUEST)
      m_stats->total_n_writes++;
    else if (mf->get_type() == READ_REQUEST)
      m_stats->total_n_reads++;
    mf->set_status(IN_PARTITION_MC_INPUT_QUEUE, now());
    if (mf->get_pc() != (unsigned)-1)
      m_gpu->gpgpu_ctx->stats->ptx_file_line_stats_add_dram_traffic(
          mf->get_pc(), mf->get_data_size());
  }

  bool full() const {
    if (!m_config->ramulator_frontend_pipeline)
      return m_pending.size() >= queue_limit();
    // Count only requests whose front-end latency has passed (m_pending is in
    // ready order: every request waits the same latency).
    unsigned ready = 0;
    for (const pending_t &p : m_pending) {
      if (p.ready_tick > m_mem->m_clk) break;
      if (++ready >= queue_limit()) return true;
    }
    return false;
  }

  void cycle() {
    send_ready();
    if (m_id == m_mem->m_channels.size() - 1) m_mem->tick();
  }

  // called by ramulator_memory::tick() after every Ramulator tick
  void after_tick(bool issued) {
    m_ticks++;
    if (issued) m_issue_ticks++;
    if (!m_pending.empty() || !m_outstanding.empty()) m_active_ticks++;
  }

  mem_fetch *returnq_top() const {
    return m_returnq.empty() ? nullptr : m_returnq.front();
  }
  mem_fetch *returnq_pop() {
    if (m_returnq.empty()) return nullptr;
    mem_fetch *mf = m_returnq.front();
    m_returnq.pop_front();
    return mf;
  }

  unsigned queue_limit() const {
    return m_info->read_buffer_size + m_info->write_buffer_size;
  }
  int max_outstanding() const { return 2 * queue_limit(); }

  unsigned long long issued() const { return m_info->n_issued; }

  void power_stats(unsigned &cmd, unsigned &activity, unsigned &nop,
                   unsigned &act, unsigned &pre, unsigned &rd, unsigned &wr,
                   unsigned &wr_WB, unsigned &req) const {
    unsigned long long wb = std::min(m_n_wb_parts, m_info->n_wr);
    cmd = m_ticks;
    activity = m_active_ticks;
    nop = m_ticks - m_issue_ticks;
    act = m_info->n_act;
    pre = m_info->n_pre;
    rd = m_info->n_rd;
    wr = m_info->n_wr - wb;
    wr_WB = wb;
    req = m_n_req;
  }

  // kernel_end: called from gpu_print_stat at the end of a kernel; the last
  // channel then also records a snapshot of Ramulator's own statistics.
  void print(FILE *fp, bool kernel_end) const {
    const RG::ChannelInfo &c = *m_info;
    unsigned long long col = c.n_rd + c.n_wr;
    unsigned long long wb = std::min(m_n_wb_parts, c.n_wr);
    // fraction of the channel's peak: each data bus moves one burst per
    // burst_ticks
    double busy = (double)col * c.burst_ticks / c.data_buses;
    double row_hit = col ? std::max(0.0, 1.0 - (double)c.n_act / col) : 0;
    // Same layout as dram_t::print so scripts parse both models.
    fprintf(fp,
            "DRAM[%u]: Ramulator 2.1 %s, tick=%.4fns, burst=%d ticks, %d data "
            "bus(es), mapper %s\n",
            m_id, c.standard.c_str(), c.tck_ns, c.burst_ticks, c.data_buses,
            c.addr_mapper.c_str());
    fprintf(fp,
            "n_cmd=%llu n_nop=%llu n_act=%llu n_pre=%llu n_ref_event=%llu "
            "n_req=%llu n_rd=%llu n_rd_L2_A=0 n_write=%llu n_wr_bk=%llu "
            "bw_util=%.4g\n",
            m_ticks, m_ticks - m_issue_ticks, c.n_act, c.n_pre, c.n_ref,
            m_n_req, c.n_rd, c.n_wr - wb, wb, m_ticks ? busy / m_ticks : 0);
    fprintf(fp, "n_activity=%llu dram_eff=%.4g\n", m_active_ticks,
            m_active_ticks ? busy / m_active_ticks : 0);
    fprintf(fp, "Row_Buffer_Locality = %f\n", row_hit);
    fprintf(fp, "in_queue=%zu outstanding=%zu returnq=%zu\n", m_pending.size(),
            m_outstanding.size(), m_returnq.size());
    if (kernel_end && m_id == m_mem->m_channels.size() - 1)
      m_mem->write_ramulator_stats(now());
  }

 private:
  struct pending_t {
    mem_fetch *mf;
    unsigned long long ready_tick;
  };

  unsigned long long now() const {
    return m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
  }

  // 32B parts of a mem_fetch: sector requests are one part; write-backs carry
  // a sector mask.
  unsigned n_parts(const mem_fetch *mf) const {
    unsigned n = (mf->get_data_size() + SECTOR_BYTES - 1) / SECTOR_BYTES;
    return n ? n : 1;
  }
  new_addr_type part_addr(const mem_fetch *mf, unsigned p) const {
    new_addr_type addr = mf->get_addr();
    unsigned n = n_parts(mf);
    if (n > 1) {
      const mem_access_sector_mask_t &mask = mf->get_access_sector_mask();
      if (mask.count() == n) {
        unsigned seen = 0;
        for (unsigned i = 0; i < mask.size(); i++)
          if (mask.test(i) && seen++ == p)
            return (addr & ~(new_addr_type)(MAX_MEMORY_ACCESS_SIZE - 1)) +
                   i * SECTOR_BYTES;
      }
      return addr + p * SECTOR_BYTES;
    }
    return addr & ~(new_addr_type)(SECTOR_BYTES - 1);
  }

  // Fill a Ramulator request for one 32B part.
  void make_request(Ramulator::Request &req, const mem_fetch *mf,
                    new_addr_type addr) const {
    // Address inside the channel, wrapped to the channel capacity
    new_addr_type a = m_config->m_address_mapping.channel_address(addr) %
                      m_mem->m_channel_bytes;
    req.addr = a & ~(new_addr_type)(SECTOR_BYTES - 1);
    req.size_bytes = SECTOR_BYTES;
    if (!m_mem->m_gpgpusim_decode) {
      req.addr_vec = {(int)m_id};  // channel; the addr_mapper does the rest
      return;
    }
    // PassThroughAddrMapper: bank/row/column from GPGPU-Sim's own decode
    // (-gpgpu_mem_addr_mapping, -dram_bnkgrp_indexing_policy)
    addrdec_t tlx;
    m_config->m_address_mapping.addrdec_tlx(addr, &tlx, mf->get_tpc());
    const RG::ChannelInfo &c = *m_info;
    int n_bg = c.level_sizes[1], n_bank = c.level_sizes[2];
    int n_row = c.level_sizes[3];
    int n_col = c.level_sizes[4] / c.internal_prefetch;
    int bg, bank;
    if (m_config->dram_bnkgrp_indexing_policy == LOWER_BITS) {
      bg = tlx.bk & (n_bg - 1);
      bank = tlx.bk / n_bg;
    } else {
      bg = tlx.bk >> m_config->bk_tag_length;
      bank = tlx.bk & ((1 << m_config->bk_tag_length) - 1);
    }
    int col = tlx.col / c.tx_bytes;
    if (bg >= n_bg || bank >= n_bank || (int)tlx.row >= n_row || col >= n_col)
      fatal("GPGPU-Sim address decode does not fit the Ramulator device "
            "(bank groups/banks/rows/columns); check -gpgpu_mem_addr_mapping");
    req.addr_vec = {(int)m_id, bg, bank, (int)tlx.row, col};
  }

  void send_ready() {
    while (!m_pending.empty() &&
           m_pending.front().ready_tick <= m_mem->m_clk) {
      mem_fetch *mf = m_pending.front().mf;
      unsigned n = n_parts(mf);
      bool first = (m_next_part == 0);
      bool is_wb = mf->get_access_type() == L1_WRBK_ACC ||
                   mf->get_access_type() == L2_WRBK_ACC;
      // Register before sending: Ramulator may complete a request inside
      // send() (a write merged into its write buffer).
      if (first) {
        m_outstanding[mf] = n;
        mf->set_status(IN_PARTITION_DRAM, now());
      }
      Ramulator::Request req(
          (Ramulator::Addr_t)0,
          mf->is_write() ? Ramulator::Request::Type::Write
                         : Ramulator::Request::Type::Read,
          -1,  // source_id -1: Ramulator indexes per-core stats with it
          [this, mf](Ramulator::Request &) { part_done(mf); });
      make_request(req, mf, part_addr(mf, m_next_part));
      // mf must not be touched after this call: it may already be retired.
      bool accepted = m_mem->m_memsys->send(req);
      if (!accepted) {  // controller buffer full: retry next tick
        if (first) m_outstanding.erase(mf);
        return;
      }
      if (req.addr_vec[0] != (int)m_id)
        fatal("a request went to the wrong Ramulator channel; the config "
              "must use channel_mapper PassThroughChannelMapper");
      if (is_wb) m_n_wb_parts++;
      if (++m_next_part == n) {
        m_next_part = 0;
        m_pending.pop_front();
      }
    }
  }

  void part_done(mem_fetch *mf) {
    auto it = m_outstanding.find(mf);
    assert(it != m_outstanding.end());
    if (--it->second == 0) {
      m_outstanding.erase(it);
      retire(mf);
    }
  }

  void retire(mem_fetch *mf) {
    // Same as the end of a request in dram_t::cycle()
    mf->set_status(IN_PARTITION_MC_RETURNQ, now());
    if (mf->get_access_type() != L1_WRBK_ACC &&
        mf->get_access_type() != L2_WRBK_ACC) {
      mf->set_reply();
      m_returnq.push_back(mf);
    } else {
      m_mp->set_done(mf);
      delete mf;
    }
  }

  unsigned m_id;
  const memory_config *m_config;
  memory_stats_t *m_stats;
  memory_partition_unit *m_mp;
  gpgpu_sim *m_gpu;
  ramulator_memory *m_mem;
  RG::ChannelInfo *m_info;

  std::deque<pending_t> m_pending;  // accepted from the partition, not yet
                                    // (fully) sent to Ramulator
  unsigned m_next_part = 0;         // next 32B part of m_pending.front()
  std::unordered_map<mem_fetch *, unsigned> m_outstanding;  // parts left
  std::deque<mem_fetch *> m_returnq;                        // finished reads

  unsigned long long m_ticks = 0, m_issue_ticks = 0, m_active_ticks = 0;
  unsigned long long m_n_req = 0, m_n_wb_parts = 0;
};

void ramulator_memory::tick() {
  std::vector<unsigned long long> before(m_channels.size());
  for (size_t i = 0; i < m_channels.size(); i++)
    before[i] = m_channels[i]->issued();
  m_memsys->tick();
  m_clk++;
  for (size_t i = 0; i < m_channels.size(); i++)
    m_channels[i]->after_tick(m_channels[i]->issued() != before[i]);
}

// Ramulator's own statistics (per channel: requests, row hits/misses/
// conflicts, read latency, DRAM energy if a power plugin is configured, ...),
// cumulative since the start of the simulation:
//   ramulator_stats.txt          latest (whole run so far)
//   ramulator_stats_history.txt  one snapshot per kernel end, used to split
//                                DRAM energy per kernel (scripts/combined_power.py)
void ramulator_memory::write_ramulator_stats(unsigned long long gpu_cycle) {
  m_memsys->update_stats_recursive();
  {
    std::ofstream out("ramulator_stats.txt");
    m_memsys->print_stats(out);
  }
  std::ofstream hist("ramulator_stats_history.txt",
                     m_snapshots ? std::ios::app : std::ios::trunc);
  hist << "=== snapshot " << ++m_snapshots << " gpu_cycle " << gpu_cycle
       << " dram_tick " << m_clk << " tick_ns " << m_tck_ns << "\n";
  m_memsys->print_stats(hist);
}

// ---------------------------------------------------------------------------

double ramulator_dram_tck_ns(const memory_config *config) {
  return ramulator_memory::get(config)->m_tck_ns;
}

ramulator_dram_t::ramulator_dram_t(unsigned int partition_id,
                                   const memory_config *config,
                                   memory_stats_t *stats,
                                   memory_partition_unit *mp, gpgpu_sim *gpu)
    : m_ch(new ramulator_channel(partition_id, config, stats, mp, gpu)) {}

ramulator_dram_t::~ramulator_dram_t() { delete m_ch; }

bool ramulator_dram_t::full(bool is_write) const { return m_ch->full(); }
void ramulator_dram_t::push(mem_fetch *data) { m_ch->push(data); }
void ramulator_dram_t::cycle() { m_ch->cycle(); }
mem_fetch *ramulator_dram_t::return_queue_top() { return m_ch->returnq_top(); }
mem_fetch *ramulator_dram_t::return_queue_pop() { return m_ch->returnq_pop(); }
void ramulator_dram_t::print(FILE *simFile) const { m_ch->print(simFile, true); }
void ramulator_dram_t::print_stat(FILE *simFile) { m_ch->print(simFile, false); }
int ramulator_dram_t::max_outstanding() const { return m_ch->max_outstanding(); }
void ramulator_dram_t::set_dram_power_stats(unsigned &cmd, unsigned &activity,
                                            unsigned &nop, unsigned &act,
                                            unsigned &pre, unsigned &rd,
                                            unsigned &wr, unsigned &wr_WB,
                                            unsigned &req) const {
  m_ch->power_stats(cmd, activity, nop, act, pre, rd, wr, wr_WB, req);
}
