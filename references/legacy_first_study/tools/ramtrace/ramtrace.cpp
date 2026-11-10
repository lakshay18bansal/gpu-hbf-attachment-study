// ramtrace: a minimal standalone replay driver for Ramulator 2.1 that talks to
// libramulator.so exactly the way the Accel-Sim bridge
// (gpgpu-sim/src/gpgpu-sim/ramulator_dram.cc) does:
//   - loads a YAML config exported with `python3 -m ramulator export`
//     (any config whose memory_system uses PassThroughChannelMapper, e.g. the
//     Accel-Sim configs in ramulator/configs/accelsim/*.py),
//   - builds Request{addr_vec={channel}, addr=intra-channel byte address,
//     size_bytes=32, callback}, calls IMemorySystem::send() directly,
//   - retries rejected requests on the next tick (backpressure),
//   - ticks the memory system once per controller tick (tCK of the config),
//   - delays each request by the GPGPUSim frontend's fixed_latency_ns,
//   - keeps ticking until every request has completed (the Python
//     Simulation.run() loop stops as soon as the trace frontend has *sent*
//     the last request, so in-flight requests are never served there).
//
// Trace format (one request per line, '#' comments allowed):
//   <tick> <R|W> <channel> <intra-channel byte address, dec or 0x hex>
// <tick> is the controller tick at which the request leaves the GPU side
// (requests must be sorted by tick).
//
// Build (no rebuild of the Ramulator tree; links the existing library):
//   g++ -std=c++20 -O2 -I<ram>/src ramtrace.cpp -o ramtrace \
//       <ram>/libramulator.so -Wl,-rpath,<ram>
// Run:
//   ./ramtrace config.yaml trace.txt [latency.csv] [stats.yaml]

#include <ramulator/base/base.h>
#include <ramulator/base/config.h>
#include <ramulator/base/factory.h>
#include <ramulator/base/request.h>
#include <ramulator/frontend/i_frontend.h>
#include <ramulator/gpgpusim/gpgpusim.h>
#include <ramulator/memory_system/i_memory_system.h>

#include <cmath>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct TraceReq {
  unsigned long long tick;
  bool is_write;
  int channel;
  long long addr;
  // filled during the run
  long long sent = -1, done = -1;
};

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s config.yaml trace.txt [latency.csv] [stats.yaml]\n", argv[0]);
    return 1;
  }
  std::vector<TraceReq> reqs;
  {
    std::ifstream in(argv[2]);
    if (!in) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      TraceReq r{};
      std::string op, a;
      ss >> r.tick >> op >> r.channel >> a;
      if (op != "R" && op != "W") { fprintf(stderr, "bad op in: %s\n", line.c_str()); return 1; }
      r.is_write = (op == "W");
      r.addr = std::stoll(a, nullptr, 0);
      reqs.push_back(r);
    }
  }

  Ramulator::GPGPUSim::channels().clear();
  auto cfg = Ramulator::Config::parse_config_file(argv[1]);
  Ramulator::IFrontEnd* fe = Ramulator::Factory::create_frontend(cfg);
  Ramulator::IMemorySystem* mem = Ramulator::Factory::create_memory_system(cfg);
  fe->connect_memory_system(mem);
  mem->connect_frontend(fe);

  auto& info = Ramulator::GPGPUSim::channels();
  double tck_ns = mem->get_tCK();
  double fixed_ns = Ramulator::GPGPUSim::fixed_latency_ns(fe);
  unsigned long long lat_ticks = fixed_ns > 0 ? llround(fixed_ns / tck_ns) : 0;
  int n_ch = info.empty() ? 1 : (int)info.size();
  printf("config %s: %d channel(s), %s, tCK %.4f ns, tx %d B, front-end latency %.3f ns = %llu ticks\n",
         argv[1], n_ch, info.empty() ? "?" : info[0]->standard.c_str(), tck_ns,
         mem->get_tx_bytes(), fixed_ns, lat_ticks);

  size_t next = 0, n_done = 0;
  std::vector<std::deque<size_t>> waiting(n_ch);  // per channel, after front-end delay
  unsigned long long clk = 0, n_reject = 0;
  const unsigned long long MAX_TICKS = 2000000000ULL;
  while (n_done < reqs.size() && clk < MAX_TICKS) {
    // requests whose GPU-side time + front-end latency has come
    while (next < reqs.size() && reqs[next].tick + lat_ticks <= clk) {
      if (reqs[next].channel < 0 || reqs[next].channel >= n_ch) {
        fprintf(stderr, "channel %d out of range\n", reqs[next].channel); return 1;
      }
      waiting[reqs[next].channel].push_back(next);
      next++;
    }
    for (int c = 0; c < n_ch; c++) {
      while (!waiting[c].empty()) {
        size_t i = waiting[c].front();
        TraceReq& r = reqs[i];
        Ramulator::Request req((Ramulator::Addr_t)0,
                               r.is_write ? Ramulator::Request::Type::Write
                                          : Ramulator::Request::Type::Read,
                               -1, [&reqs, &n_done, &clk, i](Ramulator::Request&) {
                                 reqs[i].done = (long long)clk;
                                 n_done++;
                               });
        req.addr = r.addr & ~31LL;
        req.size_bytes = 32;
        req.addr_vec = {c};
        r.sent = (long long)clk;
        if (!mem->send(req)) { r.sent = -1; n_reject++; break; }  // retry next tick
        waiting[c].pop_front();
      }
    }
    mem->tick();
    clk++;
  }

  // Report
  double sum_r = 0, sum_w = 0; size_t nr = 0, nw = 0; long long max_r = 0;
  for (auto& r : reqs) {
    long long l = r.done - (long long)r.tick;  // GPU-side issue -> completion
    if (r.is_write) { sum_w += l; nw++; } else { sum_r += l; nr++; if (l > max_r) max_r = l; }
  }
  printf("ticks simulated %llu (%.1f ns); requests %zu (R %zu, W %zu), completed %zu, send() rejections %llu\n",
         clk, clk * tck_ns, reqs.size(), nr, nw, n_done, n_reject);
  if (nr) printf("read latency incl. front-end: avg %.1f ticks (%.2f ns), max %lld ticks\n",
                 sum_r / nr, sum_r / nr * tck_ns, max_r);
  if (nw) printf("write 'latency' incl. front-end (callback fires when WR issues): avg %.1f ticks (%.2f ns)\n",
                 sum_w / nw, sum_w / nw * tck_ns);
  printf("achieved bandwidth %.2f GB/s over the run\n",
         reqs.size() * 32.0 / (clk * tck_ns));
  for (int c = 0; c < (int)info.size(); c++) {
    auto* ci = info[c];
    if (ci->n_rd + ci->n_wr == 0) continue;
    printf("  ch%-2d ACT %llu PRE %llu RD %llu WR %llu REF %llu\n", c, ci->n_act, ci->n_pre,
           ci->n_rd, ci->n_wr, ci->n_ref);
  }
  if (argc > 3) {
    std::ofstream csv(argv[3]);
    csv << "idx,tick,op,channel,addr,sent_tick,done_tick,latency_ticks\n";
    for (size_t i = 0; i < reqs.size(); i++) {
      auto& r = reqs[i];
      csv << i << "," << r.tick << "," << (r.is_write ? "W" : "R") << "," << r.channel << ",0x"
          << std::hex << r.addr << std::dec << "," << r.sent << "," << r.done << ","
          << (r.done - (long long)r.tick) << "\n";
    }
  }
  mem->update_stats_recursive();
  if (argc > 4) {
    std::ofstream st(argv[4]);
    mem->print_stats(st);
  }
  fe->finalize();
  mem->finalize();
  return 0;
}
