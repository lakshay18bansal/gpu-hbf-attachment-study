// Copyright (c) 2009-2021, Tor M. Aamodt, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdint>
#include <list>
#include <set>

#include "../abstract_hardware_model.h"
#include "../option_parser.h"
#include "../statwrapper.h"
#include "basic_components.h"
#include "dram.h"
#ifdef GPGPUSIM_USE_RAMULATOR
#include "ramulator_dram.h"
#endif
#include "gpu-cache.h"
#include "gpu-sim.h"
#include "histogram.h"
#include "l2cache.h"
#include "l2cache_trace.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"

mem_fetch *partition_mf_allocator::alloc(new_addr_type addr,
                                         mem_access_type type, unsigned size,
                                         bool wr, unsigned long long cycle,
                                         unsigned long long streamID) const {
  assert(wr);
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID, WRITE_PACKET_SIZE, -1,
                                -1, -1, m_memory_config, cycle);
  return mf;
}

mem_fetch *partition_mf_allocator::alloc(
    new_addr_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf, unsigned long long streamID) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
                      m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(access, NULL, streamID,
                                wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid,
                                sid, tpc, m_memory_config, cycle, original_mf);
  return mf;
}
// Memory access trace (-gpgpu_mem_access_trace_file): one file for the whole GPU,
// opened on first use and closed at exit. Lines are written in simulation order.
static FILE *g_mem_access_trace = NULL;
static void close_mem_access_trace() {
  if (g_mem_access_trace) fclose(g_mem_access_trace);
  g_mem_access_trace = NULL;
}

void memory_partition_unit::trace_mem_access(mem_fetch *mf, bool to_hbf) {
  const char *path = m_config->mem_access_trace_file;
  if (path == NULL || path[0] == '\0') return;
  if (g_mem_access_trace == NULL) {
    g_mem_access_trace = fopen(path, "w");
    if (g_mem_access_trace == NULL) {
      fprintf(stderr, "cannot open -gpgpu_mem_access_trace_file %s\n", path);
      exit(1);
    }
    setvbuf(g_mem_access_trace, NULL, _IOFBF, 1 << 20);
    atexit(close_mem_access_trace);
    fprintf(g_mem_access_trace,
            "# cycle address R/W%s  (cycle = GPU core cycle; one line per request leaving "
            "the L2 for DRAM/HBM or HBF)\n",
            m_config->mem_access_trace_detail ? " tier partition l2_subpartition bytes" : "");
  }
  if (m_config->mem_access_trace_detail)
    fprintf(g_mem_access_trace, "%llu 0x%llx %c %s %u %u %u\n",
            m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
            (unsigned long long)mf->get_addr(), mf->is_write() ? 'W' : 'R',
            to_hbf ? "HBF" : "DRAM", m_id, mf->get_sub_partition_id(),
            mf->get_data_size());
  else
    fprintf(g_mem_access_trace, "%llu 0x%llx %c\n",
            m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle,
            (unsigned long long)mf->get_addr(), mf->is_write() ? 'W' : 'R');
}

memory_partition_unit::memory_partition_unit(
    unsigned partition_id, const memory_config *config,
    class memory_stats_t *stats, class gpgpu_sim *gpu,
    std::vector<LatencyQueue<mem_fetch *>> &request_0_to_1,
    std::vector<LatencyQueue<mem_fetch *>> &request_1_to_0,
    std::vector<LatencyQueue<mem_fetch *>> &reply_0_to_1,
    std::vector<LatencyQueue<mem_fetch *>> &reply_1_to_0,
    class hbf_system_t *hbf_system)
    : m_id(partition_id),
      m_config(config),
      m_stats(stats),
      m_arbitration_metadata(config),
      m_gpu(gpu) {
  unsigned partitions_per_chiplet =
      config->m_n_mem / config->m_address_mapping.get_n_chiplet_partition();
  m_chiplet_id = partition_id / partitions_per_chiplet;

#ifdef GPGPUSIM_USE_RAMULATOR
  if (m_config->ramulator_enabled)
    m_dram = new ramulator_dram_t(m_id, m_config, m_stats, this, gpu);
  else
#endif
    m_dram = new dram_t(m_id, m_config, m_stats, this, gpu);
  if (config->dram_max_outstanding > 0)
    m_arbitration_metadata.set_max_outstanding(config->dram_max_outstanding);
  else if (m_dram->max_outstanding() >= 0)
    m_arbitration_metadata.set_max_outstanding(m_dram->max_outstanding());

  // One logical HBF system is shared by all memory partitions.  Partitions
  // only retain the pointer for local injection and completion routing.
  m_hbf_system = hbf_system;
  m_hbf_credits_used = 0;
  m_hbf_frontend_cycles = (unsigned long long)llround(
      config->hbf_frontend_latency_ns * 1e-9 * gpu->get_config().get_core_freq());
  m_phy_rate = ((config->hbf_attach == 2 || config->hbf_attach == 3) &&
                config->hbf_phy_gbps > 0 &&
                config->hbf_dram_freq_hz > 0)
                   ? config->hbf_phy_gbps * 1e9 / config->hbf_dram_freq_hz
                   : 0.0;
  // Burst: one 128 B transaction on top of a tick's worth of bytes.
  m_phy_cap = m_phy_rate + 128.0;
  m_phy_tokens = m_phy_cap;
  m_phy_hbm_turn = false;
  m_phy_hbm_waiting = false;
  m_h3_up_first = false;
  m_h3_hbf_rr = 0;
  if (config->hbf_attach == 3 && config->simple_dram_model) {
    fprintf(stderr, "HBF: -gpgpu_hbf_attach 3 does not support the simple DRAM model\n");
    exit(1);
  }

  m_sub_partition = new memory_sub_partition
      *[m_config->m_n_sub_partition_per_memory_channel];
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    unsigned sub_partition_id =
        m_id * m_config->m_n_sub_partition_per_memory_channel + p;
    m_sub_partition[p] = new memory_sub_partition(
        sub_partition_id, m_config, stats, gpu, request_0_to_1, request_1_to_0,
        reply_0_to_1, reply_1_to_0);
    assert(m_sub_partition[p]->get_chiplet_id() == m_chiplet_id);
  }
}

void memory_partition_unit::handle_memcpy_to_gpu(
    size_t addr, unsigned global_subpart_id, mem_access_sector_mask_t mask) {
  unsigned p = global_sub_partition_id_to_local_id(global_subpart_id);
  std::string mystring = mask.to_string<char, std::string::traits_type,
                                        std::string::allocator_type>();
  MEMPART_DPRINTF(
      "Copy Engine Request Received For Address=%zx, local_subpart=%u, "
      "global_subpart=%u, sector_mask=%s \n",
      addr, p, global_subpart_id, mystring.c_str());
  m_sub_partition[p]->force_l2_tag_update(
      addr, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle, mask);
}

memory_partition_unit::~memory_partition_unit() {
  delete m_dram;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    delete m_sub_partition[p];
  }
  delete[] m_sub_partition;
}

memory_partition_unit::arbitration_metadata::arbitration_metadata(
    const memory_config *config)
    : m_last_borrower(config->m_n_sub_partition_per_memory_channel - 1),
      m_private_credit(config->m_n_sub_partition_per_memory_channel, 0),
      m_shared_credit(0) {
  // each sub partition get at least 1 credit for forward progress
  // the rest is shared among with other partitions
  m_private_credit_limit = 1;
  m_shared_credit_limit = config->gpgpu_frfcfs_dram_sched_queue_size +
                          config->gpgpu_dram_return_queue_size -
                          (config->m_n_sub_partition_per_memory_channel - 1);
  if (config->seperate_write_queue_enabled)
    m_shared_credit_limit += config->gpgpu_frfcfs_dram_write_queue_size;
  if (config->gpgpu_frfcfs_dram_sched_queue_size == 0 or
      config->gpgpu_dram_return_queue_size == 0) {
    m_shared_credit_limit =
        0;  // no limit if either of the queue has no limit in size
  }
  assert(m_shared_credit_limit >= 0);
}

void memory_partition_unit::arbitration_metadata::set_max_outstanding(
    int max_outstanding) {
  // each sub partition keeps its private credit; the rest is shared
  m_shared_credit_limit =
      max_outstanding - (int)m_private_credit.size() * m_private_credit_limit;
  assert(m_shared_credit_limit > 0);
}

bool memory_partition_unit::arbitration_metadata::has_credits(
    int inner_sub_partition_id) const {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    return true;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    return true;
  } else {
    return false;
  }
}

void memory_partition_unit::arbitration_metadata::borrow_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] < m_private_credit_limit) {
    m_private_credit[spid] += 1;
  } else if (m_shared_credit_limit == 0 ||
             m_shared_credit < m_shared_credit_limit) {
    m_shared_credit += 1;
  } else {
    assert(0 && "DRAM arbitration error: Borrowing from depleted credit!");
  }
  m_last_borrower = spid;
}

void memory_partition_unit::arbitration_metadata::return_credit(
    int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  if (m_private_credit[spid] > 0) {
    m_private_credit[spid] -= 1;
  } else {
    m_shared_credit -= 1;
  }
  assert((m_shared_credit >= 0) &&
         "DRAM arbitration error: Returning more than available credits!");
}

int memory_partition_unit::arbitration_metadata::used() const {
  int n = m_shared_credit;
  for (int c : m_private_credit) n += c;
  return n;
}

int memory_partition_unit::arbitration_metadata::limit() const {
  if (m_shared_credit_limit == 0) return 0;
  return m_shared_credit_limit +
         (int)m_private_credit.size() * m_private_credit_limit;
}

void memory_partition_unit::arbitration_metadata::print(FILE *fp) const {
  fprintf(fp, "private_credit = ");
  for (unsigned p = 0; p < m_private_credit.size(); p++) {
    fprintf(fp, "%d ", m_private_credit[p]);
  }
  fprintf(fp, "(limit = %d)\n", m_private_credit_limit);
  fprintf(fp, "shared_credit = %d (limit = %d)\n", m_shared_credit,
          m_shared_credit_limit);
}

bool memory_partition_unit::busy() const {
  bool busy = false;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    if (m_sub_partition[p]->busy()) {
      busy = true;
    }
  }
  if (m_id == 0 && m_hbf_system && m_hbf_system->busy_demand()) busy = true;
  return busy;
}

void memory_partition_unit::cache_cycle(unsigned cycle) {
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->cache_cycle(cycle);
  }
}

void memory_partition_unit::visualizer_print(gzFile visualizer_file) const {
  m_dram->visualizer_print(visualizer_file);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->visualizer_print(visualizer_file);
  }
}

// determine whether a given subpartition can issue to DRAM
bool memory_partition_unit::can_issue_to_dram(int inner_sub_partition_id) {
  int spid = inner_sub_partition_id;
  bool sub_partition_contention = m_sub_partition[spid]->dram_L2_queue_full();
  bool has_dram_resource = m_arbitration_metadata.has_credits(spid);

  MEMPART_DPRINTF(
      "sub partition %d sub_partition_contention=%c has_dram_resource=%c\n",
      spid, (sub_partition_contention) ? 'T' : 'F',
      (has_dram_resource) ? 'T' : 'F');

  return (has_dram_resource && !sub_partition_contention);
}

int memory_partition_unit::global_sub_partition_id_to_local_id(
    int global_sub_partition_id) const {
  return (global_sub_partition_id -
          m_id * m_config->m_n_sub_partition_per_memory_channel);
}

// Shared multi-stack integration.  The GPU advances the system once before
// visiting partitions in a rotating order.  This method only drains this
// partition's completions and attempts one local admission.
bool memory_partition_unit::hbf_cycle() {
  if (!m_hbf_system) return false;
  if (m_config->hbf_attach == 3) return h3_hbf_cycle();
  if (m_config->hbf_attach >= 1) return hbf_direct_cycle();

  for (unsigned p = 0;
       p < m_config->m_n_sub_partition_per_memory_channel; ++p) {
    unsigned global_spid =
        m_id * m_config->m_n_sub_partition_per_memory_channel + p;
    if (m_sub_partition[p]->dram_L2_queue_full()) continue;
    mem_fetch *mf_return = m_hbf_system->pop_return_for(global_spid);
    if (!mf_return) continue;
    if (mf_return->get_access_type() == L1_WRBK_ACC ||
        mf_return->get_access_type() == L2_WRBK_ACC) {
      set_done(mf_return);
      delete mf_return;
    } else {
      m_sub_partition[p]->dram_L2_queue_push(mf_return);
      mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                            m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.return_credit(p);
    }
  }

  // With -gpgpu_hbf_frontend_latency_ns > 0 an HBF request first waits out
  // the GPU front end (MC + PHY, as DRAM requests do) in an in-order pipeline
  // bounded only by the DRAM credits it holds; with 0 it goes straight to the
  // HBF system, as in HBF-Sim.
  const unsigned long long now =
      m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
  while (!m_hbf_frontend_queue.empty() &&
         m_hbf_frontend_queue.front().ready_cycle <= now) {
    mem_fetch *mf = m_hbf_frontend_queue.front().req;
    if (m_hbf_system->full(mf->get_addr(), mf->is_write())) break;
    m_hbf_frontend_queue.pop_front();
    m_hbf_system->push(mf);
  }

  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0;
       p < m_config->m_n_sub_partition_per_memory_channel; ++p) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (m_sub_partition[spid]->L2_dram_queue_empty() ||
        !can_issue_to_dram(spid))
      continue;
    mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
    if (!m_config->is_hbf_addr(mf->get_addr())) continue;
    if (m_hbf_frontend_cycles > 0) {
      m_sub_partition[spid]->L2_dram_queue_pop();
      trace_mem_access(mf, true);
      hbf_delay_t d;
      d.ready_cycle = now + m_hbf_frontend_cycles;
      d.req = mf;
      m_hbf_frontend_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, now);
      m_arbitration_metadata.borrow_credit(spid);
      return true;
    }
    // Backpressure belongs to the addressed stack.  Continue checking other
    // local subpartitions because a different stack may still accept work.
    if (m_hbf_system->full(mf->get_addr(), mf->is_write())) continue;
    m_sub_partition[spid]->L2_dram_queue_pop();
    trace_mem_access(mf, true);
    m_hbf_system->push(mf);
    mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    m_arbitration_metadata.borrow_credit(spid);
    return true;
  }
  return false;
}

// -gpgpu_hbf_attach 1/2.  The partition owns an HBF port that is separate
// from its DRAM/HBM controller: HBF misses arrive through the sub-partitions'
// L2->HBF queues, take one of hbf_credits credits, wait the front-end latency
// and enter the (GPU-wide) HBF system.  Replies go back through the normal
// DRAM->L2 queue.  With attach 2 every HBF reply byte and every HBM request
// byte of this partition is charged to one PHY token bucket (the HBM stack's
// GPU-facing link, which in H3 also carries the HBF data).
// Called once per DRAM tick.  Returns true if a request was admitted.
bool memory_partition_unit::hbf_direct_cycle() {
  const unsigned n_sub = m_config->m_n_sub_partition_per_memory_channel;
  const unsigned long long now = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
  hbf_attach_stats_t &st = m_hbf_system->attach_stats();
  const bool phy = m_phy_rate > 0;
  bool yield = false;
  if (phy) {
    m_phy_tokens = std::min(m_phy_cap, m_phy_tokens + m_phy_rate);
    // When both tiers want the PHY they alternate ticks; on HBM's tick HBF
    // replies hold back so a stream of replies cannot starve HBM requests.
    m_phy_hbm_turn = !m_phy_hbm_turn;
    yield = m_phy_hbm_turn && m_phy_hbm_waiting;
    m_phy_hbm_waiting = false;
  }

  // 1. Replies (one per sub-partition per tick).
  for (unsigned p = 0; p < n_sub; ++p) {
    unsigned global_spid = m_id * n_sub + p;
    if (m_sub_partition[p]->dram_L2_queue_full()) continue;
    if (!m_hbf_system->has_return_for(global_spid)) continue;
    mem_fetch *mf_return = NULL;
    if (phy && (yield || m_phy_tokens <= 0.0)) {
      // Only read replies carry data over the PHY; write acks pass.
      mf_return = m_hbf_system->peek_return_for(global_spid);
      if (mf_return && !mf_return->is_write()) {
        st.phy_hbf_stall_ticks++;
        continue;
      }
    }
    mf_return = m_hbf_system->pop_return_for(global_spid);
    assert(mf_return);
    bool is_wb = mf_return->get_access_type() == L1_WRBK_ACC ||
                 mf_return->get_access_type() == L2_WRBK_ACC;
    if (phy && !mf_return->is_write()) {
      m_phy_tokens -= mf_return->get_data_size();
      st.phy_hbf_bytes += mf_return->get_data_size();
    }
    auto issued = m_hbf_issue_cycle.find(mf_return);
    assert(issued != m_hbf_issue_cycle.end());
    if (!mf_return->is_write()) st.record_latency(now - issued->second);
    m_hbf_issue_cycle.erase(issued);
    assert(m_hbf_credits_used > 0);
    m_hbf_credits_used--;
    st.returned++;
    if (is_wb) {
      // Write-backs only held an HBF credit, not a DRAM one.
      m_sub_partition[p]->set_done(mf_return);
      delete mf_return;
    } else {
      m_sub_partition[p]->dram_L2_queue_push(mf_return);
      mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, now);
    }
  }

  // 2. Admission from the L2->HBF queues (one per sub-partition per tick).
  bool admitted = false;
  bool waiting = false;
  for (unsigned p = 0; p < n_sub; ++p) {
    if (m_sub_partition[p]->L2_hbf_queue_empty()) continue;
    if (m_hbf_credits_used >= m_config->hbf_credits) {
      waiting = true;
      break;
    }
    mem_fetch *mf = m_sub_partition[p]->L2_hbf_queue_top();
    // Write payload crosses the PHY on the way in: wait for tokens.
    if (phy && mf->is_write() && m_phy_tokens <= 0.0) continue;
    m_sub_partition[p]->L2_hbf_queue_pop();
    if (phy && mf->is_write()) {
      // write payload crosses the PHY on the way in
      m_phy_tokens -= mf->get_data_size();
      st.phy_hbf_bytes += mf->get_data_size();
    }
    m_hbf_credits_used++;
    st.admitted++;
    m_hbf_issue_cycle[mf] = mf->get_status_change_cycle();
    // Logged when it leaves the L2 for the HBF port (before the front-end
    // latency), like DRAM requests are logged when they enter the DRAM path.
    trace_mem_access(mf, true);
    mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, now);
    hbf_delay_t d;
    d.ready_cycle = now + m_hbf_frontend_cycles;
    d.req = mf;
    m_hbf_frontend_queue.push_back(d);
    admitted = true;
  }
  // 3. Front-end latency queue -> HBF system (in order).
  while (!m_hbf_frontend_queue.empty() &&
         m_hbf_frontend_queue.front().ready_cycle <= now) {
    mem_fetch *mf = m_hbf_frontend_queue.front().req;
    if (m_hbf_system->full(mf->get_addr(), mf->is_write())) break;
    m_hbf_frontend_queue.pop_front();
    m_hbf_system->push(mf);
  }

  if (waiting) st.credit_block_ticks++;
  if (m_hbf_credits_used > st.max_credits_used)
    st.max_credits_used = m_hbf_credits_used;
  st.credit_used_sum += m_hbf_credits_used;
  st.credit_samples++;
  return admitted;
}

bool memory_partition_unit::phy_take_hbm(unsigned bytes) {
  if (m_phy_rate <= 0) return true;
  if (m_phy_tokens <= 0.0) {
    m_phy_hbm_waiting = true;
    m_hbf_system->attach_stats().phy_hbm_stall_ticks++;
    return false;
  }
  m_phy_tokens -= bytes;
  m_hbf_system->attach_stats().phy_hbm_bytes += bytes;
  return true;
}

// -gpgpu_hbf_attach 3: H3 (Ha et al., IEEE CAL 2026, Fig. 3(b)).  Replaces
// dram_cycle() and runs once per DRAM tick, in the same order: replies, DRAM
// tick, GPU issue, DRAM latency queue.  The GPU memory controller part (L2->
// DRAM queues, arbitration credits, one issue per tick) is dram_cycle()'s and
// never looks at the HBF window.  Requests cross the GPU<->HBM link (token
// bucket m_phy_*: write payload down, read data up, both tiers) into the HBM
// base die, whose address decoder & router sends each one to the HBM
// controller (m_dram) or to the D2D queue of the HBF stack.  HBF replies come
// back through the base die and the same link.  With no HBF traffic and an
// unlimited link the HBM path differs from dram_cycle() only by the ingress
// FIFO (requests may leave the L2 queue while the HBM controller is full).
void memory_partition_unit::h3_dram_cycle() {
  const unsigned long long now =
      m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
  hbf_attach_stats_t &st = m_hbf_system->attach_stats();
  if (m_phy_rate > 0)
    m_phy_tokens = std::min(m_phy_cap, m_phy_tokens + m_phy_rate);

  // 1. Replies, HBM base die -> link -> GPU.  The tier that goes first
  //    alternates, so neither can starve the other on the link.
  m_h3_up_first = !m_h3_up_first;
  if (m_h3_up_first) {
    h3_return_hbm();
    h3_return_hbf();
  } else {
    h3_return_hbf();
    h3_return_hbm();
  }

  m_dram->cycle();
  m_dram->dram_log(SAMPLELOG);

  // 2. GPU memory controller: one request per tick from the L2->DRAM queues,
  //    on a DRAM credit, onto the link (flow control: base-die ingress space).
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  unsigned issued_now = 0;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (m_sub_partition[spid]->L2_dram_queue_empty() ||
        !can_issue_to_dram(spid))
      continue;
    if (m_h3_ingress.size() >= m_config->hbf_h3_queue_size) break;
    mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
    // Write payload crosses the link on the way down.
    if (mf->is_write()) {
      if (!h3_link_take(mf->get_data_size())) {
        st.phy_down_stall_ticks++;
        break;
      }
      st.phy_down_bytes += mf->get_data_size();
    }
    m_sub_partition[spid]->L2_dram_queue_pop();
    h3_req_t r;
    r.req = mf;
    r.l2_exit_cycle = mf->get_status_change_cycle();
    mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE, now);
    m_arbitration_metadata.borrow_credit(spid);
    m_h3_ingress.push_back(r);
    // One request per tick by default, as dram_cycle(); at most one per
    // sub-partition per tick otherwise.
    if (++issued_now >= m_config->hbf_h3_issues_per_tick) break;
  }
  unsigned long long used = (unsigned long long)m_arbitration_metadata.used();
  if (used > st.h3_max_dram_credits) st.h3_max_dram_credits = used;
  st.h3_dram_credit_limit = (unsigned long long)m_arbitration_metadata.limit();

  // 3. HBM base die: address decoder & router, one request per tick (default),
  //    in arrival order (a full destination blocks the ingress FIFO).
  for (unsigned n = 0;
       n < m_config->hbf_h3_issues_per_tick && !m_h3_ingress.empty(); ++n) {
    mem_fetch *mf = m_h3_ingress.front().req;
    if (m_config->is_hbf_addr(mf->get_addr())) {
      m_hbf_issue_cycle[mf] = m_h3_ingress.front().l2_exit_cycle;
      m_h3_ingress.pop_front();
      trace_mem_access(mf, true);
      // The HBM path pays the GPU front end (MC + PHY) inside the HBM
      // controller model (Ramulator's fixed latency); HBF requests cross the
      // same GPU MC and link, so they pay hbf_frontend_latency_ns here.
      hbf_delay_t d;
      d.ready_cycle = now + m_hbf_frontend_cycles;
      d.req = mf;
      m_h3_frontend.push_back(d);
      st.h3_routed_hbf++;
      st.admitted++;
    } else if (!m_dram->full(mf->is_write())) {
      m_h3_ingress.pop_front();
      trace_mem_access(mf, false);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle =
          now + (m_dram->has_own_latency() ? 0 : m_config->dram_latency);
      m_dram_latency_queue.push_back(d);
      st.h3_routed_hbm++;
    } else {
      st.h3_router_stall_ticks++;
      break;
    }
  }

  // 4. DRAM latency queue -> HBM controller (as dram_cycle()).
  if (!m_dram_latency_queue.empty() &&
      (now >= m_dram_latency_queue.front().ready_cycle) &&
      !m_dram->full(m_dram_latency_queue.front().req->is_write())) {
    mem_fetch *mf = m_dram_latency_queue.front().req;
    m_dram_latency_queue.pop_front();
    m_dram->push(mf);
  }
}

// attach 3: the GPU<->HBM link.  May go negative by one transfer (burst).
bool memory_partition_unit::h3_link_take(unsigned bytes) {
  if (m_phy_rate <= 0) return true;
  if (m_phy_tokens <= 0.0) return false;
  m_phy_tokens -= bytes;
  return true;
}

// attach 3: one HBM controller reply per tick over the link (dram_cycle()'s
// return step plus the link).
void memory_partition_unit::h3_return_hbm() {
  mem_fetch *mf_return = m_dram->return_queue_top();
  if (!mf_return) {
    m_dram->return_queue_pop();
    return;
  }
  hbf_attach_stats_t &st = m_hbf_system->attach_stats();
  unsigned dest_global_spid = mf_return->get_sub_partition_id();
  int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
  assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
  if (m_sub_partition[dest_spid]->dram_L2_queue_full()) return;
  if (mf_return->get_access_type() == L1_WRBK_ACC) {
    m_sub_partition[dest_spid]->set_done(mf_return);
    delete mf_return;
  } else {
    if (!mf_return->is_write()) {
      if (!h3_link_take(mf_return->get_data_size())) {
        st.phy_hbm_stall_ticks++;
        return;
      }
      st.phy_hbm_bytes += mf_return->get_data_size();
    }
    m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
    mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    m_arbitration_metadata.return_credit(dest_spid);
  }
  m_dram->return_queue_pop();
}

// attach 3: one HBF reply per tick, HBF stack -> D2D -> HBM base die -> link.
// Write-backs give the DRAM credit back through set_done(), as in hbf_cycle().
void memory_partition_unit::h3_return_hbf() {
  const unsigned n_sub = m_config->m_n_sub_partition_per_memory_channel;
  const unsigned long long now =
      m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
  hbf_attach_stats_t &st = m_hbf_system->attach_stats();
  const unsigned start = m_h3_hbf_rr;
  unsigned returned_now = 0;
  for (unsigned k = 0; k < n_sub; ++k) {
    unsigned p = (start + k) % n_sub;
    unsigned global_spid = m_id * n_sub + p;
    if (!m_hbf_system->has_return_for(global_spid)) continue;
    if (m_sub_partition[p]->dram_L2_queue_full()) continue;
    mem_fetch *mf_return = m_hbf_system->peek_return_for(global_spid);
    assert(mf_return);
    if (!mf_return->is_write()) {
      if (!h3_link_take(mf_return->get_data_size())) {
        st.phy_hbf_stall_ticks++;
        return;
      }
      st.phy_hbf_bytes += mf_return->get_data_size();
    }
    mf_return = m_hbf_system->pop_return_for(global_spid);
    assert(mf_return);
    auto issued = m_hbf_issue_cycle.find(mf_return);
    assert(issued != m_hbf_issue_cycle.end());
    if (!mf_return->is_write()) st.record_latency(now - issued->second);
    m_hbf_issue_cycle.erase(issued);
    st.returned++;
    if (mf_return->get_access_type() == L1_WRBK_ACC ||
        mf_return->get_access_type() == L2_WRBK_ACC) {
      set_done(mf_return);
      delete mf_return;
    } else {
      m_sub_partition[p]->dram_L2_queue_push(mf_return);
      mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE, now);
      m_arbitration_metadata.return_credit(p);
    }
    m_h3_hbf_rr = (p + 1) % n_sub;
    if (++returned_now >= m_config->hbf_h3_returns_per_tick) return;
  }
}

// attach 3: front end -> D2D queue -> HBF stack, one request per partition per
// tick, in the GPU-wide rotating order of gpgpu_sim::cycle().
bool memory_partition_unit::h3_hbf_cycle() {
  const unsigned long long now =
      m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
  while (!m_h3_frontend.empty() && m_h3_frontend.front().ready_cycle <= now &&
         m_h3_d2d.size() < m_config->hbf_h3_queue_size) {
    m_h3_d2d.push_back(m_h3_frontend.front().req);
    m_h3_frontend.pop_front();
  }
  bool admitted = false;
  for (unsigned i = 0; i < m_config->hbf_h3_admits_per_tick; ++i) {
    if (m_h3_d2d.empty()) break;
    mem_fetch *mf = m_h3_d2d.front();
    if (m_hbf_system->full(mf->get_addr(), mf->is_write())) break;
    m_h3_d2d.pop_front();
    m_hbf_system->push(mf);
    admitted = true;
  }
  return admitted;
}

void memory_partition_unit::simple_dram_model_cycle() {
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition

  unsigned dram_cycles = m_config->simple_dram_clock_multiplier;
  for (unsigned c = 0; c < dram_cycles; c++) {
    if (m_dram_latency_queue.empty()) break;
    if (((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
         m_dram_latency_queue.front().ready_cycle)) {
      mem_fetch *mf_return = m_dram_latency_queue.front().req;
      if (mf_return->get_access_type() != L1_WRBK_ACC &&
          mf_return->get_access_type() != L2_WRBK_ACC) {
        mf_return->set_reply();
        unsigned dest_global_spid = mf_return->get_sub_partition_id();
        int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
        assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
        if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
          // Update stats for simple dram when request is actually dequeued
          m_stats->memlatstat_dram_access(mf_return);
          if (mf_return->get_access_type() == L1_WRBK_ACC) {
            m_sub_partition[dest_spid]->set_done(mf_return);
            delete mf_return;
          } else {
            m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
            mf_return->set_status(
                IN_PARTITION_DRAM_TO_L2_QUEUE,
                m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
            m_arbitration_metadata.return_credit(dest_spid);
            MEMPART_DPRINTF(
                "mem_fetch request %p return from dram to sub partition %d\n",
                mf_return, dest_spid);
          }
          m_dram_latency_queue.pop_front();
        } else {
          break;
        }
      } else {
        // Update stats for simple dram when request is actually dequeued
        m_stats->memlatstat_dram_access(mf_return);
        this->set_done(mf_return);
        delete mf_return;
        m_dram_latency_queue.pop_front();
      }
    } else {
      // The front of the DRAM latency queue is not ready yet, so we cannot
      // process any more requests in this cycle
      break;
    }
  }

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  unsigned processed = 0;
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) {
        continue;
      }

      // HBF-addressed requests are owned by hbf_cycle().  Leave them in the
      // L2-to-memory queue so the HBF path can admit the original request.
      // Popping before this check silently dropped requests in the simple
      // DRAM model.
      if (m_config->gpu_routes_hbf(mf->get_addr())) continue;

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);

      trace_mem_access(mf, false);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      m_config->dram_latency;
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.borrow_credit(spid);
      processed++;
      if (processed >= dram_cycles) {
        break;  // the DRAM should only accept these request per cycle
      }
    }
  }
  //}
}

void memory_partition_unit::dram_cycle() {
  if (m_config->hbf_attach == 3 && m_hbf_system) {
    h3_dram_cycle();
    return;
  }
  // pop completed memory request from dram and push it to dram-to-L2 queue
  // of the original sub partition
  mem_fetch *mf_return = m_dram->return_queue_top();
  if (mf_return) {
    unsigned dest_global_spid = mf_return->get_sub_partition_id();
    int dest_spid = global_sub_partition_id_to_local_id(dest_global_spid);
    assert(m_sub_partition[dest_spid]->get_id() == dest_global_spid);
    if (!m_sub_partition[dest_spid]->dram_L2_queue_full()) {
      if (mf_return->get_access_type() == L1_WRBK_ACC) {
        m_sub_partition[dest_spid]->set_done(mf_return);
        delete mf_return;
      } else {
        m_sub_partition[dest_spid]->dram_L2_queue_push(mf_return);
        mf_return->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_arbitration_metadata.return_credit(dest_spid);
        MEMPART_DPRINTF(
            "mem_fetch request %p return from dram to sub partition %d\n",
            mf_return, dest_spid);
      }
      m_dram->return_queue_pop();
    }
  } else {
    m_dram->return_queue_pop();
  }

  m_dram->cycle();
  m_dram->dram_log(SAMPLELOG);

  // mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
  // if( !m_dram->full(mf->is_write()) ) {
  // L2->DRAM queue to DRAM latency queue
  // Arbitrate among multiple L2 subpartitions
  int last_issued_partition = m_arbitration_metadata.last_borrower();
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    int spid = (p + last_issued_partition + 1) %
               m_config->m_n_sub_partition_per_memory_channel;
    if (!m_sub_partition[spid]->L2_dram_queue_empty() &&
        can_issue_to_dram(spid)) {
      mem_fetch *mf = m_sub_partition[spid]->L2_dram_queue_top();
      if (m_dram->full(mf->is_write())) break;

      // Skip HBF-addressed requests — handled by hbf_cycle()
      if (m_config->gpu_routes_hbf(mf->get_addr())) continue;

      // attach 2: HBM requests share the PHY with HBF replies.
      if (m_config->hbf_attach == 2 && m_hbf_system &&
          !phy_take_hbm(mf->get_data_size()))
        break;

      m_sub_partition[spid]->L2_dram_queue_pop();
      MEMPART_DPRINTF(
          "Issue mem_fetch request %p from sub partition %d to dram\n", mf,
          spid);
      trace_mem_access(mf, false);
      dram_delay_t d;
      d.req = mf;
      d.ready_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                      (m_dram->has_own_latency() ? 0 : m_config->dram_latency);
      m_dram_latency_queue.push_back(d);
      mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_arbitration_metadata.borrow_credit(spid);
      break;  // the DRAM should only accept one request per cycle
    }
  }
  //}

  // DRAM latency queue
  if (!m_dram_latency_queue.empty() &&
      ((m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle) >=
       m_dram_latency_queue.front().ready_cycle) &&
      !m_dram->full(m_dram_latency_queue.front().req->is_write())) {
    mem_fetch *mf = m_dram_latency_queue.front().req;
    m_dram_latency_queue.pop_front();
    m_dram->push(mf);
  }
}

void memory_partition_unit::set_done(mem_fetch *mf) {
  unsigned global_spid = mf->get_sub_partition_id();
  int spid = global_sub_partition_id_to_local_id(global_spid);
  assert(m_sub_partition[spid]->get_id() == global_spid);
  if (mf->get_access_type() == L1_WRBK_ACC ||
      mf->get_access_type() == L2_WRBK_ACC) {
    m_arbitration_metadata.return_credit(spid);
    MEMPART_DPRINTF(
        "mem_fetch request %p return from dram to sub partition %d\n", mf,
        spid);
  }
  m_sub_partition[spid]->set_done(mf);
}

void memory_partition_unit::set_dram_power_stats(
    unsigned &n_cmd, unsigned &n_activity, unsigned &n_nop, unsigned &n_act,
    unsigned &n_pre, unsigned &n_rd, unsigned &n_wr, unsigned &n_wr_WB,
    unsigned &n_req) const {
  m_dram->set_dram_power_stats(n_cmd, n_activity, n_nop, n_act, n_pre, n_rd,
                               n_wr, n_wr_WB, n_req);
}

void memory_partition_unit::print(FILE *fp) const {
  fprintf(fp, "Memory Partition %u: \n", m_id);
  for (unsigned p = 0; p < m_config->m_n_sub_partition_per_memory_channel;
       p++) {
    m_sub_partition[p]->print(fp);
  }
  fprintf(fp, "In Dram Latency Queue (total = %zd): \n",
          m_dram_latency_queue.size());
  for (std::list<dram_delay_t>::const_iterator mf_dlq =
           m_dram_latency_queue.begin();
       mf_dlq != m_dram_latency_queue.end(); ++mf_dlq) {
    mem_fetch *mf = mf_dlq->req;
    fprintf(fp, "Ready @ %llu - ", mf_dlq->ready_cycle);
    if (mf)
      mf->print(fp);
    else
      fprintf(fp, " <NULL mem_fetch?>\n");
  }
  m_dram->print(fp);
  if (m_hbf_system && m_id == 0) m_hbf_system->print_stat(fp);
}

memory_sub_partition::memory_sub_partition(
    unsigned sub_partition_id, const memory_config *config,
    class memory_stats_t *stats, class gpgpu_sim *gpu,
    std::vector<LatencyQueue<mem_fetch *>> &request_0_to_1,
    std::vector<LatencyQueue<mem_fetch *>> &request_1_to_0,
    std::vector<LatencyQueue<mem_fetch *>> &reply_0_to_1,
    std::vector<LatencyQueue<mem_fetch *>> &reply_1_to_0) {
  m_id = sub_partition_id;
  m_config = config;
  m_stats = stats;
  m_gpu = gpu;
  m_memcpy_cycle_offset = 0;
  m_chiplet_id = sub_partition_id / config->m_n_sub_partition_per_chiplet;
  uint32_t local_sub_partition_id =
      sub_partition_id % config->m_n_sub_partition_per_chiplet;
  m_chiplet_icnt.init(m_chiplet_id, local_sub_partition_id, request_0_to_1,
                      request_1_to_0, reply_0_to_1, reply_1_to_0);

  if (gpu->getShaderCoreConfig()->n_chiplet == 1) {
    m_chiplet_disabled = true;
  } else {
    m_chiplet_disabled = false;
  }

  assert(m_id < m_config->m_n_mem_sub_partition);

  char L2c_name[32];
  snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
  m_L2interface = new L2interface(this);
  m_chiplet_interface = new Chipletinterface(this);
  m_mf_allocator = new partition_mf_allocator(config);

  if (!m_config->m_L2_config.disabled())
    m_L2cache = new l2_cache(L2c_name, m_config->m_L2_config, -1, -1,
                             m_L2interface, m_chiplet_interface, m_mf_allocator,
                             IN_PARTITION_L2_MISS_QUEUE, gpu, L2_GPU_CACHE,
                             m_id, m_chiplet_id, stats);

  unsigned int icnt_L2;
  unsigned int L2_dram;
  unsigned int dram_L2;
  unsigned int L2_icnt;
  sscanf(m_config->gpgpu_L2_queue_config, "%u:%u:%u:%u", &icnt_L2, &L2_dram,
         &dram_L2, &L2_icnt);
  m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2", 0, icnt_L2);
  m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram", 0, L2_dram);
  m_L2_hbf_queue = new fifo_pipeline<mem_fetch>(
      "L2-to-hbf", 0, m_config->hbf_l2_queue_size ? m_config->hbf_l2_queue_size : 1);
  m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2", 0, dram_L2);
  m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt", 0, L2_icnt);
  wb_addr = -1;

  // Initialize the LRC if enabled
  if (m_config->lrc_enabled) {
    m_lrc = new L2RequestCoalescer(m_config->lrc_max_entries,
                                   m_config->lrc_max_merged);
  } else {
    m_lrc = nullptr;
  }
}

memory_sub_partition::~memory_sub_partition() {
  delete m_lrc;
  delete m_icnt_L2_queue;
  delete m_L2_dram_queue;
  delete m_L2_hbf_queue;
  delete m_dram_L2_queue;
  delete m_L2_icnt_queue;
  delete m_L2cache;
  delete m_L2interface;
}

void memory_sub_partition::cache_cycle(unsigned cycle) {
  // L2 fill responses
  if (!m_config->m_L2_config.disabled()) {
    if (m_L2cache->access_ready() && !m_L2_icnt_queue->full()) {
      mem_fetch *mf = m_L2cache->next_access();
      if (mf->get_access_type() !=
          L2_WR_ALLOC_R) {  // Don't pass write allocate read request back to
                            // upper level cache
        mf->set_reply();
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2_icnt_queue->push(mf);
      } else {
        if (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE) {
          mem_fetch *original_wr_mf = mf->get_original_wr_mf();
          assert(original_wr_mf);
          original_wr_mf->set_reply();
          original_wr_mf->set_status(
              IN_PARTITION_L2_TO_ICNT_QUEUE,
              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          m_L2_icnt_queue->push(original_wr_mf);
        }
        m_request_tracker.erase(mf);
        delete mf;
      }
    }
  }

  // DRAM to L2 (texture) and icnt (not texture)
  if (!m_dram_L2_queue->empty()) {
    mem_fetch *mf = m_dram_L2_queue->top();
    if (!m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf)) {
      if (m_L2cache->fill_port_free()) {
        mf->set_status(IN_PARTITION_L2_FILL_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        m_L2cache->fill(mf, m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                m_memcpy_cycle_offset);
        m_dram_L2_queue->pop();
      }
    } else if (!m_L2_icnt_queue->full()) {
      if (mf->is_write() && mf->get_type() == WRITE_ACK)
        mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                       m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      m_L2_icnt_queue->push(mf);
      m_dram_L2_queue->pop();
    }
  }

  // prior L2 misses inserted into m_L2_dram_queue here
  if (!m_config->m_L2_config.disabled()) m_L2cache->cycle();

  // new L2 texture accesses and/or non-texture accesses
  mem_fetch *mf = nullptr;

  // Alternate priority each call to balance latency
  bool prioritize_chiplet = (cycle % 2 == 0);
  bool from_icnt = false;

  if (prioritize_chiplet) {
    mf = get_chiplet_req();
    if (!mf && !m_icnt_L2_queue->empty()) {
      mf = m_icnt_L2_queue->top();
      from_icnt = true;
    }
  } else {
    if (!m_icnt_L2_queue->empty()) {
      mf = m_icnt_L2_queue->top();
      from_icnt = true;
    }
    if (!mf) mf = get_chiplet_req();
  }

  if (mf) {
    if (mf->get_type() == WRITE_FORWARD) {
      // pseudo-coherence update from other chiplet, directly update L2 tag.
      assert(mf->is_write());
      assert(from_icnt == false);  // can only come from chiplet
      new_addr_type addr = mf->get_addr();
      mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
      force_l2_tag_update(addr, cycle, sector_mask);
      m_chiplet_icnt.from_peer_request()->pop();
      delete mf;
    } else if (!m_config->m_L2_config.disabled() &&
               ((m_config->m_L2_texure_only && mf->istexture()) ||
                (!m_config->m_L2_texure_only))) {
      // L2 is enabled and access is for L2
      bool output_full = m_L2_icnt_queue->full();
      bool port_free = m_L2cache->data_port_free();
      // can only accept write if request queue has space for write forward
      bool interchip_free =
          !mf->is_write() || !m_chiplet_icnt.to_peer_request()->full();
      if (!interchip_free) {
        ++m_stats->chiplet_write_fail[m_id];
      }
      // HBF (merge): the default HBF policy bypasses L2 admission. Pop the
      // queue the request came from (icnt or peer chiplet) and respect the
      // L2->DRAM queue capacity (the original patch pushed unconditionally).
      if (m_config->gpu_routes_hbf(mf->get_addr()) &&
          m_config->hbf_l2_policy == 0) {
        if (!memory_queue_full(mf)) {
          mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                         m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
          memory_queue_push(mf);
          if (from_icnt) {
            m_icnt_L2_queue->pop();
          } else {
            m_chiplet_icnt.from_peer_request()->pop();
          }
          return;  // as in HBF-Sim: one request per cycle, skips ROP stage
        }
      } else if (!output_full && port_free && interchip_free) {
        bool accepted = false;
        std::list<cache_event> events;
        enum cache_request_status status =
            m_L2cache->access(mf->get_addr(), mf,
                              m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle +
                                  m_memcpy_cycle_offset,
                              events);
        bool write_sent = was_write_sent(events);
        bool read_sent = was_read_sent(events);
        MEM_SUBPART_DPRINTF("Probing L2 cache Address=%llx, status=%u\n",
                            mf->get_addr(), status);

        if (status == HIT) {
          if (!write_sent) {
            // L2 cache replies
            assert(!read_sent);
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else {
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
            accepted = true;
          } else {
            assert(write_sent);
            accepted = true;
          }
        } else if (status != RESERVATION_FAIL) {
          if (!m_chiplet_disabled && mf->is_write()) {
            forward_write_to_peer_chiplet(mf);
          }
          if (mf->is_write() &&
              (m_config->m_L2_config.m_write_alloc_policy == FETCH_ON_WRITE ||
               m_config->m_L2_config.m_write_alloc_policy ==
                   LAZY_FETCH_ON_READ) &&
              !was_writeallocate_sent(events)) {
            if (mf->get_access_type() == L1_WRBK_ACC) {
              m_request_tracker.erase(mf);
              delete mf;
            } else if (m_config->m_L2_config.get_write_policy() == WRITE_BACK) {
              mf->set_reply();
              mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,
                             m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
              m_L2_icnt_queue->push(mf);
            }
          }
          // L2 cache accepted request
          accepted = true;
        } else {
          assert(!write_sent);
          assert(!read_sent);
          // L2 cache lock-up: will try again next cycle
        }
        if (accepted) {
          if (from_icnt) {
            m_icnt_L2_queue->pop();
          } else {
            m_chiplet_icnt.from_peer_request()->pop();
          }
        }
      }
    } else if (!memory_queue_full(mf)) {
      // L2 is disabled or non-texture access to texture-only L2
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,
                     m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
      memory_queue_push(mf);
      if (from_icnt) {
        m_icnt_L2_queue->pop();
      } else {
        m_chiplet_icnt.from_peer_request()->pop();
      }
    }
  }

  // ROP delay queue
  if (!m_rop.empty() && (cycle >= m_rop.front().ready_cycle) &&
      !m_icnt_L2_queue->full()) {
    mem_fetch *mf = m_rop.front().req;
    m_rop.pop();
    m_icnt_L2_queue->push(mf);
    mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  }
}

bool memory_sub_partition::full() const { return m_icnt_L2_queue->full(); }

bool memory_sub_partition::lrc_full() const {
  return m_lrc != nullptr && m_lrc->full();
}

bool memory_sub_partition::full(unsigned size) const {
  return m_icnt_L2_queue->is_avilable_size(size);
}

bool memory_sub_partition::lrc_full(unsigned size) const {
  return m_lrc != nullptr && m_lrc->full(size);
}

bool memory_sub_partition::L2_dram_queue_empty() const {
  return m_L2_dram_queue->empty();
}

class mem_fetch *memory_sub_partition::L2_dram_queue_top() const {
  return m_L2_dram_queue->top();
}

void memory_sub_partition::L2_dram_queue_pop() { m_L2_dram_queue->pop(); }

bool memory_sub_partition::uses_hbf_queue(const mem_fetch *mf) const {
  return (m_config->hbf_attach == 1 || m_config->hbf_attach == 2) &&
         m_config->is_hbf_addr(mf->get_addr());
}

bool memory_sub_partition::memory_queue_full(const mem_fetch *mf) const {
  return uses_hbf_queue(mf) ? m_L2_hbf_queue->full() : m_L2_dram_queue->full();
}

void memory_sub_partition::memory_queue_push(mem_fetch *mf) {
  if (uses_hbf_queue(mf))
    m_L2_hbf_queue->push(mf);
  else
    m_L2_dram_queue->push(mf);
}

bool memory_sub_partition::dram_L2_queue_full() const {
  return m_dram_L2_queue->full();
}

void memory_sub_partition::dram_L2_queue_push(class mem_fetch *mf) {
  m_dram_L2_queue->push(mf);
}

void memory_sub_partition::print_cache_stat(unsigned &accesses,
                                            unsigned &misses) const {
  FILE *fp = stdout;
  if (!m_config->m_L2_config.disabled()) m_L2cache->print(fp, accesses, misses);
}

void memory_sub_partition::print(FILE *fp) const {
  if (!m_request_tracker.empty()) {
    fprintf(fp, "Memory Sub Parition %u: pending memory requests:\n", m_id);
    for (auto r = m_request_tracker.begin(); r != m_request_tracker.end();
         ++r) {
      mem_fetch *mf = *r;
      if (mf)
        mf->print(fp);
      else
        fprintf(fp, " <NULL mem_fetch?>\n");
    }
  }
  if (!m_config->m_L2_config.disabled()) m_L2cache->display_state(fp);
}

void memory_stats_t::visualizer_print(gzFile visualizer_file) {
  gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
  gzprintf(visualizer_file, "Ltwowritehit: %d\n", L2_write_hit);
  gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
  gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_hit);
  clear_L2_stats_pw();

  if (num_mfs)
    gzprintf(visualizer_file, "averagemflatency: %lld\n",
             mf_total_lat / num_mfs);
}

void memory_stats_t::clear_L2_stats_pw() {
  L2_write_miss = 0;
  L2_write_hit = 0;
  L2_read_miss = 0;
  L2_read_hit = 0;
}

void gpgpu_sim::print_dram_stats(FILE *fout) const {
  unsigned cmd = 0;
  unsigned activity = 0;
  unsigned nop = 0;
  unsigned act = 0;
  unsigned pre = 0;
  unsigned rd = 0;
  unsigned wr = 0;
  unsigned wr_WB = 0;
  unsigned req = 0;
  unsigned tot_cmd = 0;
  unsigned tot_nop = 0;
  unsigned tot_act = 0;
  unsigned tot_pre = 0;
  unsigned tot_rd = 0;
  unsigned tot_wr = 0;
  unsigned tot_req = 0;

  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i]->set_dram_power_stats(cmd, activity, nop, act,
                                                     pre, rd, wr, wr_WB, req);
    tot_cmd += cmd;
    tot_nop += nop;
    tot_act += act;
    tot_pre += pre;
    tot_rd += rd;
    tot_wr += wr + wr_WB;
    tot_req += req;
  }
  fprintf(fout, "gpgpu_n_dram_reads = %d\n", tot_rd);
  fprintf(fout, "gpgpu_n_dram_writes = %d\n", tot_wr);
  fprintf(fout, "gpgpu_n_dram_activate = %d\n", tot_act);
  fprintf(fout, "gpgpu_n_dram_commands = %d\n", tot_cmd);
  fprintf(fout, "gpgpu_n_dram_noops = %d\n", tot_nop);
  fprintf(fout, "gpgpu_n_dram_precharges = %d\n", tot_pre);
  fprintf(fout, "gpgpu_n_dram_requests = %d\n", tot_req);
}

unsigned memory_sub_partition::flushL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->flush();
  }
  return 0;  // TODO: write the flushed data to the main memory
}

unsigned memory_sub_partition::invalidateL2() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->invalidate();
  }
  return 0;
}

bool memory_sub_partition::busy() const { return !m_request_tracker.empty(); }

std::vector<mem_fetch *>
memory_sub_partition::breakdown_request_to_sector_requests(mem_fetch *mf) {
  std::vector<mem_fetch *> result;
  mem_access_sector_mask_t sector_mask = mf->get_access_sector_mask();
  if (mf->get_data_size() == SECTOR_SIZE &&
      mf->get_access_sector_mask().count() == 1) {
    result.push_back(mf);
  } else if (mf->get_data_size() == MAX_MEMORY_ACCESS_SIZE) {
    // break down every sector
    mem_access_byte_mask_t mask;
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
          mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());

      result.push_back(n_mf);
    }
    // This is for constant cache
  } else if (mf->get_data_size() == 64 &&
             (mf->get_access_sector_mask().all() ||
              mf->get_access_sector_mask().none())) {
    unsigned start;
    if (mf->get_addr() % MAX_MEMORY_ACCESS_SIZE == 0)
      start = 0;
    else
      start = 2;
    mem_access_byte_mask_t mask;
    for (unsigned i = start; i < start + 2; i++) {
      for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
        mask.set(k);
      }
      mem_fetch *n_mf = m_mf_allocator->alloc(
          mf->get_addr(), mf->get_access_type(), mf->get_access_warp_mask(),
          mf->get_access_byte_mask() & mask,
          std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE, mf->is_write(),
          m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, mf->get_wid(),
          mf->get_sid(), mf->get_tpc(), mf, mf->get_streamID());

      result.push_back(n_mf);
    }
  } else {
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
      if (sector_mask.test(i)) {
        mem_access_byte_mask_t mask;
        for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
          mask.set(k);
        }
        mem_fetch *n_mf = m_mf_allocator->alloc(
            mf->get_addr() + SECTOR_SIZE * i, mf->get_access_type(),
            mf->get_access_warp_mask(), mf->get_access_byte_mask() & mask,
            std::bitset<SECTOR_CHUNCK_SIZE>().set(i), SECTOR_SIZE,
            mf->is_write(), m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle,
            mf->get_wid(), mf->get_sid(), mf->get_tpc(), mf,
            mf->get_streamID());

        result.push_back(n_mf);
      }
    }
  }
  if (result.size() == 0) assert(0 && "no mf sent");
  return result;
}

void memory_sub_partition::push(mem_fetch *m_req, unsigned long long cycle) {
  if (m_req) {
    m_stats->memlatstat_icnt2mem_pop(m_req);
    std::vector<mem_fetch *> reqs;
    if (m_config->m_L2_config.m_cache_type == SECTOR)
      reqs = breakdown_request_to_sector_requests(m_req);
    else
      reqs.push_back(m_req);

    for (unsigned i = 0; i < reqs.size(); ++i) {
      mem_fetch *req = reqs[i];
      // Here we insert the read request into the LRC queue
      // And we only send request down if a new entry is allocated
      // For write request, it is not coalesced with LRC
      bool allocated = true;
      if (m_lrc && !(req->is_write())) {
        // insert into LRC queue
        // if new entry is allocated, we need to send request down
        allocated = m_lrc->insert(req->get_addr(), req);
        // Increment LRC stats for ICNT to LRC
        m_stats->add_icnt_to_lrc_sectors(get_id(), req);
      }

      // Either a new entry is allocated in LRC or we disable it
      // either way, this mem_fetch needs to be sent down to L2
      if (allocated) {
        m_request_tracker.insert(req);
        if (req->istexture()) {
          m_icnt_L2_queue->push(req);
          req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,
                          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        } else {
          rop_delay_t r;
          r.req = req;
          r.ready_cycle = cycle + m_config->rop_latency;
          m_rop.push(r);
          req->set_status(IN_PARTITION_ROP_DELAY,
                          m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
        }

        // Increment LRC stats for LRC to L2 if LRC is enabled
        if (m_lrc && !(req->is_write())) {
          // Increment LRC stats for LRC to L2
          m_stats->add_lrc_to_l2_sectors(get_id(), req);
          // Update the size of the LRC queue
          m_stats->update_lrc_queue_size(get_id(), m_lrc->size());
          // Update the maximum coalescing size for this sub-partition
          // m_stats->update_current_max_coalesced_count(
          //     get_id(), m_lrc->max_coalescing_count());
          // Update the average coalescing size for this sub-partition
          m_stats->update_current_avg_coalesced_count(
              get_id(), m_lrc->avg_coalescing_count());
        }
      }
    }
  }
}

void memory_sub_partition::push_direct(mem_fetch *m_req,
                                       unsigned long long cycle) {
  assert(m_req);
  assert(!m_icnt_L2_queue->full());
  if (m_config->m_L2_config.m_cache_type == SECTOR) {
    // for debug. remove this after verification
    assert(m_req->get_data_size() ==
           SECTOR_SIZE);  // should be broken down to sector access already
  }
  m_icnt_L2_queue->push(m_req);
  m_req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE, cycle);
}

mem_fetch *memory_sub_partition::pop() {
  mem_fetch *mf = m_L2_icnt_queue->pop();
  m_request_tracker.erase(mf);
  if (mf && mf->isatomic()) mf->do_atomic();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    delete mf;
    mf = NULL;
  }
  return mf;
}

mem_fetch *memory_sub_partition::top() {
  mem_fetch *mf = m_L2_icnt_queue->top();
  if (mf && (mf->get_access_type() == L2_WRBK_ACC ||
             mf->get_access_type() == L1_WRBK_ACC)) {
    m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    delete mf;
    mf = NULL;
  }
  return mf;
}

void memory_sub_partition::set_done(mem_fetch *mf) {
  m_request_tracker.erase(mf);
}

void memory_sub_partition::accumulate_L2cache_stats(
    class cache_stats &l2_stats) const {
  if (!m_config->m_L2_config.disabled()) {
    l2_stats += m_L2cache->get_stats();
  }
}

void memory_sub_partition::get_L2cache_sub_stats(
    struct cache_sub_stats &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats(css);
  }
}

void memory_sub_partition::get_L2cache_sub_stats_pw(
    struct cache_sub_stats_pw &css) const {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->get_sub_stats_pw(css);
  }
}

void memory_sub_partition::clear_L2cache_stats_pw() {
  if (!m_config->m_L2_config.disabled()) {
    m_L2cache->clear_pw();
  }
}

void memory_sub_partition::visualizer_print(gzFile visualizer_file) {
  // Support for L2 AerialVision stats
  // Per-sub-partition stats would be trivial to extend from this
  cache_sub_stats_pw temp_sub_stats;
  get_L2cache_sub_stats_pw(temp_sub_stats);

  m_stats->L2_read_miss += temp_sub_stats.read_misses;
  m_stats->L2_write_miss += temp_sub_stats.write_misses;
  m_stats->L2_read_hit += temp_sub_stats.read_hits;
  m_stats->L2_write_hit += temp_sub_stats.write_hits;

  clear_L2cache_stats_pw();
}

bool L2RequestCoalescer::insert(new_addr_type sector_addr, mem_fetch *mf) {
  // This assumes there are space left in the queue
  // First we search the queue for mergeable entries
  auto entries = m_lrc_queue.equal_range(sector_addr);
  // Then we iterate through existing entries to find mergeable entry
  for (auto it = entries.first; it != entries.second; ++it) {
    if (it->first == sector_addr && it->second.size() < m_max_merged) {
      // Found the sector address in the queue and the entry still have space
      // left to merge
      it->second.push_back(std::make_pair(mf, false));
      m_total_coalesced_count++;
      return false;
    }
  }

  // Now we allocate a new entry
  LRCEntry new_entry;
  new_entry.push_back(std::make_pair(mf, false));
  m_lrc_queue.insert(std::make_pair(sector_addr, new_entry));
  m_total_coalesced_count++;
  assert(m_lrc_queue.size() <= m_max_entries &&
         "LRC queue is full in insert()");
  return true;
}

LRCEntry &L2RequestCoalescer::get_entry(new_addr_type sector_addr,
                                        unsigned uid) {
  auto entries = m_lrc_queue.equal_range(sector_addr);
  for (auto it = entries.first; it != entries.second; ++it) {
    auto &entry = it->second;
    if (entry.front().first->get_request_uid() == uid) {
      return entry;
    }
  }
  assert(false && "No matched entry found");
}

void L2RequestCoalescer::remove_entry(new_addr_type sector_addr, unsigned uid) {
  auto entries = m_lrc_queue.equal_range(sector_addr);
  for (auto it = entries.first; it != entries.second; ++it) {
    if (it->second.front().first->get_request_uid() == uid) {
      m_total_coalesced_count -= it->second.size();
      m_lrc_queue.erase(it);
      return;
    }
  }
  assert(false && "No matched entry found");
}
void L2interface::push(mem_fetch *mf) {
  mf->set_status(
      IN_PARTITION_L2_TO_DRAM_QUEUE,
      m_unit->m_gpu->gpu_sim_cycle + m_unit->m_gpu->gpu_tot_sim_cycle);
  m_unit->memory_queue_push(mf);
}

void Chipletinterface::push(mem_fetch *mf) {
  // Reclassify for separate statistics
  if (mf->get_access_type() == GLOBAL_ACC_R) {
    mf->set_access_type(CHIPLET_ACC_R);
  }
  mf->set_status(
      IN_PARTITION_L2_TO_CHIPLET_REQUEST_QUEUE,
      m_unit->m_gpu->gpu_sim_cycle + m_unit->m_gpu->gpu_tot_sim_cycle);
  m_unit->m_chiplet_icnt.to_peer_request()->push(mf);
  ++m_unit->m_stats->interchip_read_requests;
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status l2_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
  return data_cache::access(addr, mf, time, events);
}

void l2_cache::cycle() {
  if (!m_miss_queue.empty()) {
    mem_fetch *mf = m_miss_queue.front();
    uint32_t chiplet_idx = mf->get_dest_chiplet();
    if (chiplet_idx == m_chiplet_id || mf->get_type() == WRITE_REQUEST) {
      // This request belongs to the same chiplet, send it to DRAM
      // for writes, a seperate WRITE_FORWARD is generated
      // always write to local
      // The L2's memory port is an L2interface; ask it about this request's
      // tier (attach 1/2).  The miss queue is still one FIFO, so an HBF miss
      // at its head waits for room in the L2->HBF queue; that only happens
      // when the partition runs out of HBF credits ('HBF Credit Block Ticks').
      const L2interface *l2if = dynamic_cast<const L2interface *>(m_memport);
      bool port_full = l2if ? l2if->full(mf)
                            : m_memport->full(mf->size(), mf->get_is_write());
      if (!port_full) {
        m_miss_queue.pop_front();
        m_memport->push(mf);
      } else {
        ++m_mem_stats->L2_dram_queue_full[m_sub_partition_id];
      }
    } else {
      // This request belongs to a different chiplet, send it to the chiplet
      // interface
      if (!m_chiplet_port->full(mf->size(), mf->get_is_write())) {
        m_miss_queue.pop_front();
        unsigned sub_partition_per_chiplet =
            m_gpu->getMemoryConfig()->m_n_sub_partition_per_chiplet;
        unsigned peer_l2 = m_sub_partition_id % sub_partition_per_chiplet +
                           (chiplet_idx * sub_partition_per_chiplet);
        mf->set_partition(peer_l2);
        mf->set_chip(
            peer_l2 /
            m_gpu->getMemoryConfig()->m_n_sub_partition_per_memory_channel);
        m_chiplet_port->push(mf);
      } else {
        ++m_mem_stats->chiplet_queue_full[m_sub_partition_id];
      }
    }
  }
  bool data_port_busy = !m_bandwidth_management.data_port_free();
  bool fill_port_busy = !m_bandwidth_management.fill_port_free();
  m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
  m_bandwidth_management.replenish_port_bandwidth();
}
mem_fetch *memory_sub_partition::get_chiplet_req() const {
  if (!m_chiplet_icnt.from_peer_request()->front_ready()) {
    return nullptr;
  }

  mem_fetch *mf = m_chiplet_icnt.from_peer_request()->front().data;
  assert(mf->get_type() == READ_REQUEST || mf->get_type() == WRITE_FORWARD);
  assert(mf->get_dest_chiplet() == m_chiplet_id);
  return mf;
}
bool memory_sub_partition::push_chiplet_reply(mem_fetch *mf) {
  // note that here is still request queue
  // request is for sending, reply is for receiving
  if (m_chiplet_icnt.to_peer_reply()->full()) {
    return false;
  }
  m_chiplet_icnt.to_peer_reply()->push(mf);
  return true;
}
bool memory_sub_partition::handle_chiplet_reply() {
  if (!m_chiplet_icnt.from_peer_reply()->front_ready()) {
    return false;
  }
  if (m_dram_L2_queue->full()) {
    return false;
  }

  mem_fetch *mf = m_chiplet_icnt.from_peer_reply()->front().data;
  assert(mf->get_type() == READ_REPLY);
  assert(mf->get_src_chiplet() == m_chiplet_id);
  m_request_tracker.erase(mf);
  m_chiplet_icnt.from_peer_reply()->pop();
  m_dram_L2_queue->push(mf);
  return true;
}

void memory_sub_partition::forward_write_to_peer_chiplet(mem_fetch *mf) {
  uint64_t cycle = m_gpu->global_cycle();
  mem_fetch *new_mf = new mem_fetch(
      mf->get_mem_access(), NULL, mf->get_streamID(), mf->get_data_size(),
      mf->get_wid(), mf->get_sid(), mf->get_tpc(), m_config, cycle, mf);
  // Reclassify for separate statistics
  if (new_mf->get_access_type() == GLOBAL_ACC_W) {
    new_mf->set_access_type(CHIPLET_ACC_W);
  }

  // Assuming 2 chiplets, flip it
  assert(m_gpu->getShaderCoreConfig()->n_chiplet <= 2);
  uint32_t peer_chiplet_id = m_chiplet_id ^ 1;
  new_mf->set_write_interchip(peer_chiplet_id);
  new_mf->set_status(IN_PARTITION_L2_TO_CHIPLET_REQUEST_QUEUE, cycle);
  if (m_chiplet_icnt.to_peer_request()->full()) {
    delete new_mf;
    return;
  }
  m_chiplet_icnt.to_peer_request()->push(new_mf);
  ++m_stats->interchip_write_requests;
}
