// GPGPUSimFRFCFS scheduler: classic FR-FCFS (Rixner et al., ISCA 2000), the
// policy of GPGPU-Sim's built-in DRAM scheduler (dram_sched.cc):
//   1. an open row is not closed while requests that hit it are queued
//      (GPGPU-Sim keeps serving a row until its queue "bin" is empty),
//   2. among the rest: commands that can issue now first, row hits first,
//      then the oldest request.
// Ramulator's FRFCFS ranks only by "can issue now" then age, so an older
// request to another row can close a row that still has queued hits;
// FRFCFSRowHit only protects hits older than the closing request.

#include <vector>

#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/scheduler/i_scheduler.h"

namespace Ramulator {

class GPGPUSimFRFCFS : public IScheduler, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IScheduler, GPGPUSimFRFCFS, "GPGPUSimFRFCFS")

  ControllerBase* m_ctrl = nullptr;
  std::vector<char> m_bank_has_hit;  // per flat bank: a queued request hits the open row

 public:
  void init() override {
    m_ctrl = cast_parent<ControllerBase>();
    m_bank_has_hit.assign(m_ctrl->m_device.m_bank_nodes.size(), 0);
  }

  ReqBuffer::iterator get_best_request(ReqBuffer& buffer, RequestFilterRef filter) override {
    if (buffer.size() == 0) return buffer.end();

    // Pass 1: next command of every request, and which banks have queued row hits
    std::fill(m_bank_has_hit.begin(), m_bank_has_hit.end(), 0);
    for (auto it = buffer.begin(); it != buffer.end(); it++) {
      it->command = m_ctrl->get_preq_command(it->final_command, it->addr_vec);
      if (is_row_hit(*it)) {
        int bank = m_ctrl->m_device.get_flat_bank_id(it->addr_vec);
        if (bank >= 0 && bank < (int)m_bank_has_hit.size()) m_bank_has_hit[bank] = 1;
      }
    }

    // Pass 2: best candidate = (ready, row hit, oldest)
    auto best = buffer.end();
    bool best_ready = false, best_hit = false;
    for (auto it = buffer.begin(); it != buffer.end(); it++) {
      if (filter && !filter(*it)) continue;
      if (closes_row_with_hits(*it)) continue;
      bool ready = m_ctrl->check_timing(it->command, it->addr_vec);
      bool hit = is_row_hit(*it);
      if (best == buffer.end() || ready > best_ready ||
          (ready == best_ready && (hit > best_hit ||
                                   (hit == best_hit && it->arrive < best->arrive)))) {
        best = it;
        best_ready = ready;
        best_hit = hit;
      }
    }
    return best;
  }

 private:
  // The request's next command is its final column command on an open row
  bool is_row_hit(const Request& req) const {
    return req.command == req.final_command &&
           m_ctrl->m_device.check_rowbuffer_hit(req.final_command, req.addr_vec, m_ctrl->m_clk);
  }

  // The request's next command would close a row that queued requests still hit
  bool closes_row_with_hits(const Request& req) const {
    const auto& spec = *m_ctrl->m_device.m_spec;
    if (!spec.command_meta[req.command].is_closing || req.command == req.final_command) return false;
    if (spec.bank_targets[req.command] != BankTarget::Single) return false;
    int bank = m_ctrl->m_device.get_flat_bank_id(req.addr_vec);
    return bank >= 0 && bank < (int)m_bank_has_hit.size() && m_bank_has_hit[bank];
  }
};

}  // namespace Ramulator
