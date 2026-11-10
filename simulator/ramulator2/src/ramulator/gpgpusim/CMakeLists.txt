# Components for running Ramulator as the DRAM model of GPGPU-Sim / Accel-Sim
add_library(ramulator-gpgpusim OBJECT)

target_sources(
  ramulator-gpgpusim PRIVATE
  gpgpusim.h
  gpgpusim_frontend.cpp
  gpgpusim_counter.cpp
  gpgpusim_frfcfs.cpp
  gpu_bg_co_ba_ro.cpp
)

target_link_libraries(
  ramulator
  PRIVATE
  ramulator-gpgpusim
)
