// GPUBgCoBaRo address mapper: GPU-style mapping inside one channel.
//
// From the least significant bit of the (transaction-aligned) address:
//   [PseudoChannel] [BankGroup] | Column | other levels (e.g. Sid, Bank) | Row
// then every bank-level index is XORed with the low row bits.
//
// Consecutive 32B transactions rotate over pseudo-channels and bank groups, so
// back-to-back column commands are tCCD_S apart instead of tCCD_L; the XOR
// stops arrays aligned to large powers of two (e.g. a[], b[], c[]) from
// fighting over one bank. Ramulator's built-in mappers put the column lowest,
// which costs GPU streaming traffic 1.5-2.7x bandwidth on GDDR6.

#include <algorithm>

#include "ramulator/controller/addr_mapper/addr_mapper_base.h"
#include "ramulator/controller/addr_mapper/i_addr_mapper.h"

namespace Ramulator {

class GPUBgCoBaRo : public IAddrMapper, public AddrMapperBase {
  RAMULATOR_REGISTER_IMPLEMENTATION_DERIVED(IAddrMapper, GPUBgCoBaRo, AddrMapperBase, "GPUBgCoBaRo")

  std::vector<int> m_low;  // m_addr_bits indices mapped below the column
  std::vector<int> m_mid;  // between column and row

 public:
  void init() override {
    AddrMapperBase::init();
    const DRAMSpec& spec = *m_ctrl->m_device.m_spec;
    for (const char* name : {"PseudoChannel", "BankGroup"}) {
      if (spec.has_level(name)) m_low.push_back(spec.get_level_id(name) - 1);
    }
    for (int i = 0; i < m_row_idx; i++) {
      if (std::find(m_low.begin(), m_low.end(), i) == m_low.end()) m_mid.push_back(i);
    }
  }

  void apply(Request& req) override {
    req.addr_vec.resize(m_num_mapped_levels + 1, -1);
    Addr_t addr = req.intra_channel_addr >> m_tx_offset;
    for (int i : m_low) {
      req.addr_vec[i + 1] = slice_lower_bits(addr, m_addr_bits[i]);
    }
    req.addr_vec[m_col_idx + 1] = slice_lower_bits(addr, m_addr_bits[m_col_idx]);
    for (int i : m_mid) {
      req.addr_vec[i + 1] = slice_lower_bits(addr, m_addr_bits[i]);
    }
    req.addr_vec[m_row_idx + 1] = slice_lower_bits(addr, m_addr_bits[m_row_idx]);

    // Permutation interleaving: XOR bank-level indices with the low row bits
    int row = req.addr_vec[m_row_idx + 1], shift = 0;
    for (int i = 0; i < m_row_idx; i++) {
      if (m_addr_bits[i] == 0) continue;
      req.addr_vec[i + 1] ^= (row >> shift) & ((1 << m_addr_bits[i]) - 1);
      shift += m_addr_bits[i];
    }
  }
};

}  // namespace Ramulator
