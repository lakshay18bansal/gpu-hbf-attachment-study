// GPGPUSimCounter controller plugin: counts the DRAM commands one controller
// (= one channel) issues and describes the device, so Accel-Sim can print
// per-channel DRAM stats and feed AccelWattch (which charges energy per read,
// write and precharge).

#include "ramulator/base/base.h"
#include "ramulator/controller/controller_base.h"
#include "ramulator/controller/plugin/i_controller_plugin.h"
#include "ramulator/dram/dram_spec.h"
#include "ramulator/gpgpusim/gpgpusim.h"

namespace Ramulator {

namespace GPGPUSim {

std::vector<ChannelInfo*>& channels() {
  static std::vector<ChannelInfo*> registry;
  return registry;
}

}  // namespace GPGPUSim

class GPGPUSimCounter : public IControllerPlugin, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IControllerPlugin, GPGPUSimCounter, "GPGPUSimCounter")

 private:
  enum Kind { OTHER, ACT, PRE, RD, RDA, WR, WRA, REF };
  std::vector<Kind> m_kind;  // indexed by command id
  GPGPUSim::ChannelInfo m_info;

 public:
  void init() override { GPGPUSim::channels().push_back(&m_info); }

  void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
    auto* ctrl = cast_parent<ControllerBase>();
    const DRAMSpec& spec = *ctrl->m_device.m_spec;

    m_kind.assign(spec.command_names.size(), OTHER);
    for (size_t i = 0; i < spec.command_names.size(); i++) {
      const std::string& c = spec.command_names[i];
      if (c == "ACT")
        m_kind[i] = ACT;
      else if (c.rfind("PRE", 0) == 0)
        m_kind[i] = PRE;
      else if (c == "RD")
        m_kind[i] = RD;
      else if (c == "RDA")
        m_kind[i] = RDA;
      else if (c == "WR")
        m_kind[i] = WR;
      else if (c == "WRA")
        m_kind[i] = WRA;
      else if (c.rfind("REF", 0) == 0)
        m_kind[i] = REF;
    }

    m_info.channel = ctrl->m_channel_id;
    m_info.level_names = spec.level_names;
    m_info.level_sizes.assign(spec.organization.level_sizes.begin(),
                              spec.organization.level_sizes.end());
    m_info.internal_prefetch = spec.internal_prefetch_size;
    m_info.tx_bytes = ctrl->get_tx_bytes();
    m_info.burst_ticks = spec.has_timing("nBL") ? spec.get_timing_value("nBL") : 1;
    m_info.data_buses = spec.has_level("PseudoChannel") ? spec.get_level_size("PseudoChannel") : 1;
    m_info.read_buffer_size = ctrl->get_read_buffer_size();
    m_info.write_buffer_size = ctrl->get_write_buffer_size();
    m_info.tck_ns = ctrl->get_tCK();
    m_info.standard = spec.standard_name;
    m_info.addr_mapper = ctrl->m_addr_mapper->m_impl->get_name();
  }

  void on_issue(const Request& req) override {
    m_info.n_issued++;
    if (req.command < 0 || req.command >= (int)m_kind.size()) return;
    switch (m_kind[req.command]) {
      case ACT: m_info.n_act++; break;
      case PRE: m_info.n_pre++; break;
      case RD: m_info.n_rd++; break;
      case RDA: m_info.n_rd++; m_info.n_pre++; break;  // auto-precharge closes the row
      case WR: m_info.n_wr++; break;
      case WRA: m_info.n_wr++; m_info.n_pre++; break;
      case REF: m_info.n_ref++; break;
      default: break;
    }
  }
};

}  // namespace Ramulator
