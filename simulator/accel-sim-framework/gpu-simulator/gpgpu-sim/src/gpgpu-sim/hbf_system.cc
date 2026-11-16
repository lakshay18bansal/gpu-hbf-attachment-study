#include "hbf_system.h"

#include <assert.h>
#include <fstream>
#include <sstream>
#include <string>
#include <limits.h>
#include <stdlib.h>

#include "gpu-sim.h"
#include "hbf_cube.h"
#include "mem_fetch.h"

hbf_system_t::hbf_system_t(const memory_config *config,
                           memory_stats_t *stats, gpgpu_sim *gpu)
    : m_config(config),
      m_num_stacks(config->hbf_num_stacks > 0 ? config->hbf_num_stacks : 1),
      m_stack_map(config->hbf_stack_map == 1 ? 1 : 0),
      m_rewrite_addresses(m_num_stacks > 1 || config->hbf_route_all),
      m_total_capacity(config->hbf_size),
      m_stack_capacity(config->hbf_size / m_num_stacks) {
  if (!capacity_has_equal_whole_blocks(
          m_total_capacity, config->hbf_page_size,
          config->hbf_pages_per_block, m_num_stacks)) {
    fprintf(stderr,
            "HBF: total capacity (%llu bytes) must contain an equal, "
            "erase-block-aligned capacity for each of %u stacks "
            "(page=%u bytes, pages/block=%u)\n",
            m_total_capacity, m_num_stacks, config->hbf_page_size,
            config->hbf_pages_per_block);
    exit(1);
  }
  if (m_num_stacks > 1 && config->hbf_placement_mode == 2) {
    fprintf(stderr,
            "HBF: explicit page,channel placement tables do not encode a "
            "stack; use stack_map 0/1 for multi-stack runs\n");
    exit(1);
  }

  m_prefetch_streamed_upto = 0;
  m_prefetch_hint_pages = 0;
  if (config->hbf_prefetch_file && config->hbf_prefetch_file[0] != '\0') {
    if (m_num_stacks != 1 || config->hbf_route_all) {
      fprintf(stderr, "HBF: the prefetcher supports one stack without route_all\n");
      exit(1);
    }
    load_prefetch_hints(config->hbf_prefetch_file);
  }

  m_stacks.reserve(m_num_stacks);
  m_routed_requests.assign(m_num_stacks, 0);
  m_completed_requests.assign(m_num_stacks, 0);
  for (unsigned stack = 0; stack < m_num_stacks; ++stack) {
    m_stacks.push_back(new hbf_cube_t(config, stats, gpu, stack,
                                      m_num_stacks, m_stack_capacity));
  }
}

hbf_system_t::~hbf_system_t() {
  // Outstanding requests are still simulator-owned.  Restore their externally
  // visible addresses before releasing device metadata.
  for (auto &entry : m_original_addresses)
    entry.first->set_addr(entry.second);
  m_original_addresses.clear();
  for (hbf_cube_t *stack : m_stacks) delete stack;
}

hbf_stack_location_t hbf_system_t::locate(
    unsigned long long address) const {
  const unsigned long long logical = logical_address_for_mapping(
      address, m_config->hbf_base_addr, m_config->hbf_route_all);
  return map_relative_address(logical, m_total_capacity,
                              m_config->hbf_page_size, m_num_stacks,
                              m_stack_map);
}

bool hbf_system_t::full(unsigned long long address, bool is_write) const {
  hbf_stack_location_t location = locate(address);
  assert(location.stack < m_stacks.size());
  return m_stacks[location.stack]->full(is_write);
}

void hbf_system_t::push(mem_fetch *request) {
  assert(request != NULL);
  hbf_stack_location_t location = locate(request->get_addr());
  assert(location.stack < m_stacks.size());

  if (m_rewrite_addresses) {
    const unsigned long long original_address = request->get_addr();
    bool inserted =
        m_original_addresses.insert(std::make_pair(request, original_address))
            .second;
    assert(inserted && "HBF request was pushed more than once");
    if (location.local_offset > ULLONG_MAX - m_config->hbf_base_addr) {
      fprintf(stderr, "HBF: stack-local address overflows GPU address space\n");
      exit(1);
    }
    request->set_addr(m_config->hbf_base_addr + location.local_offset);
  }

  m_routed_requests[location.stack]++;
  m_stacks[location.stack]->push(request);
}

void hbf_system_t::cycle() {
  for (hbf_cube_t *stack : m_stacks) stack->cycle();
}

void hbf_system_t::restore_address(mem_fetch *request) {
  if (!m_rewrite_addresses || request == NULL) return;
  auto original = m_original_addresses.find(request);
  assert(original != m_original_addresses.end() &&
         "HBF completion has no saved GPU address");
  request->set_addr(original->second);
  m_original_addresses.erase(original);
}

mem_fetch *hbf_system_t::pop_return_for(unsigned global_subpartition_id) {
  unsigned &next = m_return_rr_next[global_subpartition_id];
  if (next >= m_num_stacks) next = 0;
  for (unsigned checked = 0; checked < m_num_stacks; ++checked) {
    unsigned stack = (next + checked) % m_num_stacks;
    mem_fetch *request =
        m_stacks[stack]->pop_return_for(global_subpartition_id);
    if (request == NULL) continue;
    next = (stack + 1) % m_num_stacks;
    m_completed_requests[stack]++;
    restore_address(request);
    return request;
  }
  return NULL;
}

// hbf-prefetch-v1: '#' comment lines, then per line
//   <kernel uid> <n_pages> <page_0> ... <page_n-1>
// pages are HBF-relative page numbers in first-touch order.
void hbf_system_t::load_prefetch_hints(const char *path) {
  std::ifstream in(path);
  if (!in) {
    fprintf(stderr, "HBF: cannot open prefetch hint file %s\n", path);
    exit(1);
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    unsigned uid = 0;
    unsigned long long n = 0;
    if (!(ss >> uid >> n)) continue;
    std::vector<unsigned long long> &pages = m_prefetch_hints[uid];
    pages.reserve(pages.size() + n);
    unsigned long long page;
    for (unsigned long long i = 0; i < n && (ss >> page); ++i)
      pages.push_back(page);
    if (pages.size() < n) {
      fprintf(stderr, "HBF: prefetch hint line for kernel %u is short\n", uid);
      exit(1);
    }
    m_prefetch_hint_pages += n;
  }
  printf("GPGPU-Sim uArch: HBF prefetch hints: %zu kernels, %llu pages from %s "
         "(lookahead %u, window %u, LHB %u pages)\n",
         m_prefetch_hints.size(), m_prefetch_hint_pages, path,
         m_config->hbf_prefetch_lookahead, m_config->hbf_prefetch_window,
         m_config->hbf_lhb_pages);
}

void hbf_system_t::on_kernel_launch(unsigned kernel_uid) {
  if (m_prefetch_hints.empty()) return;
  unsigned last = kernel_uid + m_config->hbf_prefetch_lookahead;
  unsigned first = kernel_uid > m_prefetch_streamed_upto
                       ? kernel_uid
                       : m_prefetch_streamed_upto + 1;
  for (unsigned uid = first; uid <= last; ++uid) {
    auto it = m_prefetch_hints.find(uid);
    if (it != m_prefetch_hints.end()) m_stacks[0]->prefetch_enqueue(it->second);
  }
  if (last > m_prefetch_streamed_upto) m_prefetch_streamed_upto = last;
}

bool hbf_system_t::busy_demand() const {
  for (const hbf_cube_t *stack : m_stacks)
    if (stack->has_demand_work()) return true;
  return !m_original_addresses.empty();
}

mem_fetch *hbf_system_t::peek_return_for(unsigned global_subpartition_id) const {
  auto rr = m_return_rr_next.find(global_subpartition_id);
  unsigned next = rr == m_return_rr_next.end() || rr->second >= m_num_stacks ? 0 : rr->second;
  for (unsigned checked = 0; checked < m_num_stacks; ++checked) {
    mem_fetch *r = m_stacks[(next + checked) % m_num_stacks]->peek_return_for(global_subpartition_id);
    if (r) return r;
  }
  return NULL;
}

bool hbf_system_t::has_return_for(unsigned global_subpartition_id) const {
  for (const hbf_cube_t *stack : m_stacks)
    if (stack->has_return_for(global_subpartition_id)) return true;
  return false;
}

bool hbf_system_t::busy() const {
  for (const hbf_cube_t *stack : m_stacks)
    if (stack->busy()) return true;
  return !m_original_addresses.empty();
}

unsigned hbf_system_t::channels_per_stack() const {
  return m_stacks.empty() ? 0 : m_stacks.front()->num_channels();
}

void hbf_system_t::print_stat(FILE *fp) {
  fprintf(fp, "\n========= HBF System Statistics =========\n");
  fprintf(fp, "HBF Stacks:             %u\n", m_num_stacks);
  fprintf(fp, "HBF Stack Map:          %s\n",
          m_stack_map == 1 ? "contiguous" : "page-interleave");
  fprintf(fp, "HBF Total Capacity:     %llu bytes\n", m_total_capacity);
  fprintf(fp, "HBF Per-Stack Capacity: %llu bytes\n", m_stack_capacity);
  fprintf(fp, "HBF Channels Per Stack: %u\n", channels_per_stack());
  if (m_config->hbf_attach >= 1) {
    const hbf_attach_stats_t &a = m_attach_stats;
    fprintf(fp, "HBF Attach Mode:        %d (%s)\n", m_config->hbf_attach,
            m_config->hbf_attach == 1   ? "direct"
            : m_config->hbf_attach == 2 ? "behind-HBM"
                                        : "H3 (HBM base die router)");
    fprintf(fp, "HBF Attach Admitted:    %llu\n", a.admitted);
    fprintf(fp, "HBF Attach Returned:    %llu\n", a.returned);
    if (m_config->hbf_attach == 3) {
      fprintf(fp, "H3 Base Die Routed:     hbm=%llu hbf=%llu router_stall_ticks=%llu\n",
              a.h3_routed_hbm, a.h3_routed_hbf, a.h3_router_stall_ticks);
      fprintf(fp, "H3 GPU DRAM Credits:    max_used=%llu limit=%llu per partition (shared by HBM and HBF)\n",
              a.h3_max_dram_credits, a.h3_dram_credit_limit);
      fprintf(fp, "H3 Link:                %.3f GB/s/partition down_bytes=%llu up_hbm_bytes=%llu "
              "up_hbf_bytes=%llu down_stall_ticks=%llu hbm_stall_ticks=%llu hbf_stall_ticks=%llu\n",
              m_config->hbf_phy_gbps, a.phy_down_bytes, a.phy_hbm_bytes,
              a.phy_hbf_bytes, a.phy_down_stall_ticks, a.phy_hbm_stall_ticks,
              a.phy_hbf_stall_ticks);
      fprintf(fp, "H3 D2D Hop:             %u ticks (%.2f ns) per NAND operation\n",
              m_config->hbf_d2d_hop,
              m_config->hbf_dram_freq_hz > 0
                  ? m_config->hbf_d2d_hop * 1e9 / m_config->hbf_dram_freq_hz
                  : 0.0);
    } else {
      fprintf(fp, "HBF Credits Max Used:   %llu (limit %u per partition)\n",
              a.max_credits_used, m_config->hbf_credits);
      fprintf(fp, "HBF Credits Avg Used:   %.2f\n",
              a.credit_samples ? a.credit_used_sum / a.credit_samples : 0.0);
      fprintf(fp, "HBF Credit Block Ticks: %llu\n", a.credit_block_ticks);
    }
    fprintf(fp, "HBF E2E Read Latency:   avg=%.1f p50=%llu p95=%llu p99=%llu max=%llu core cycles (%llu reqs)\n",
            a.lat_count ? a.lat_sum / a.lat_count : 0.0, a.percentile(0.50),
            a.percentile(0.95), a.percentile(0.99), a.lat_max, a.lat_count);
    if (m_config->hbf_attach == 2)
      fprintf(fp, "HBF PHY Shared:         %.2f GB/s/partition hbm_bytes=%llu hbf_bytes=%llu "
              "hbm_stall_ticks=%llu hbf_stall_ticks=%llu\n",
              m_config->hbf_phy_gbps, a.phy_hbm_bytes, a.phy_hbf_bytes,
              a.phy_hbm_stall_ticks, a.phy_hbf_stall_ticks);
  }
  for (unsigned stack = 0; stack < m_num_stacks; ++stack) {
    fprintf(fp,
            "\n--------- HBF Stack %u (routed=%llu completed=%llu) ---------\n",
            stack, m_routed_requests[stack], m_completed_requests[stack]);
    m_stacks[stack]->print_stat(fp);
  }
}
