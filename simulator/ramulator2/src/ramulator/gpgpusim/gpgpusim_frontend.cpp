// GPGPUSim frontend: like the External frontend (the host simulator injects
// requests and ticks the memory system), plus the fixed latency of the GPU
// memory controller front end, which Accel-Sim applies before a request
// reaches the Ramulator controller.

#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/gpgpusim/gpgpusim.h"

namespace Ramulator {

class GPGPUSimFrontEnd : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, GPGPUSimFrontEnd, "GPGPUSim")

 public:
  float m_fixed_latency_ns = 0;

  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").default_val(1);
    RAMULATOR_PARSE_PARAM(m_fixed_latency_ns, float, "fixed_latency_ns").default_val(0.0f);
    if (m_fixed_latency_ns < 0) {
      throw std::runtime_error("GPGPUSim frontend: fixed_latency_ns must be >= 0");
    }
  }

  void tick() override {}

  bool is_finished() override { return false; }

  bool receive_external_requests(int req_type_id, Addr_t addr, int source_id,
                                 std::function<void(Request&)> callback,
                                 int size_bytes) override {
    return receive_external_requests(req_type_id, addr, source_id, -1,
                                     std::move(callback), size_bytes);
  }

  bool receive_external_requests(int req_type_id, Addr_t addr, int source_id,
                                 int ingress_id,
                                 std::function<void(Request&)> callback,
                                 int size_bytes) override {
    Request req(addr, req_type_id, source_id, std::move(callback));
    req.ingress_id = ingress_id;
    req.size_bytes = size_bytes;
    return m_memory_system->send(req);
  }
};

namespace GPGPUSim {

float fixed_latency_ns(IFrontEnd* frontend) {
  auto* fe = dynamic_cast<GPGPUSimFrontEnd*>(frontend);
  return fe ? fe->m_fixed_latency_ns : -1.0f;
}

}  // namespace GPGPUSim
}  // namespace Ramulator
