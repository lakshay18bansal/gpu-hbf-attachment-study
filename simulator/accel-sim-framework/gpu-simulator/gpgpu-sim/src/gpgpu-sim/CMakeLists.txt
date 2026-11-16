# Exclude power_interface.cc if no power model
list(APPEND gpgpusim_SRC addrdec.cc
    dram.cc
    dram_sched.cc
    gpu-cache.cc
    gpu-misc.cc
    gpu-sim.cc
    hbf_controller.cc
    hbf_cube.cc
    hbf_ftl.cc
    hbf_page_cache.cc
    hbf_subarray.cc
    hbf_system.cc
    hbf_trace.cc
    hashing.cc
    histogram.cc
    icnt_wrapper.cc
    l2cache.cc
    local_interconnect.cc
    mem_fetch.cc
    mem_latency_stat.cc
    power_interface.cc
    power_stat.cc
    scoreboard.cc
    shader.cc
    stack.cc
    stat-tool.cc
    traffic_breakdown.cc
    visualizer.cc)
if(NOT GPGPUSIM_USE_POWER_MODEL)
    list(REMOVE_ITEM ${gpgpusim_SRC} power_interface.cc)
endif()

# Create libgpgpusim.a
add_library(gpgpusim STATIC ${gpgpusim_SRC})
target_include_directories(gpgpusim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpgpusim PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

# Optional: Ramulator 2.1 as the DRAM model (ramulator_dram.cc).
# Configure with -DGPGPUSIM_USE_RAMULATOR=ON -DRAMULATOR_ROOT=<ramulator2 dir>
option(GPGPUSIM_USE_RAMULATOR "Build with Ramulator 2.1 DRAM model support" OFF)
if(GPGPUSIM_USE_RAMULATOR)
    if(NOT RAMULATOR_ROOT)
        message(FATAL_ERROR "GPGPUSIM_USE_RAMULATOR=ON needs -DRAMULATOR_ROOT=<path to ramulator2>")
    endif()
    target_sources(gpgpusim PRIVATE ramulator_dram.cc)
    # Ramulator headers need C++20; only this file is built with it.
    set_source_files_properties(ramulator_dram.cc PROPERTIES COMPILE_OPTIONS "-std=c++20")
    # The flag must be visible to every file that includes gpu-sim.h/l2cache.h
    target_compile_definitions(gpgpusim PUBLIC GPGPUSIM_USE_RAMULATOR)
    target_include_directories(gpgpusim PRIVATE
        ${RAMULATOR_ROOT}/src
        ${RAMULATOR_ROOT}/build/_deps/yaml-cpp-src/include
        ${RAMULATOR_ROOT}/build/_deps/fmt-src/include)
    target_link_libraries(gpgpusim PUBLIC ${RAMULATOR_ROOT}/libramulator.so)
    target_link_options(gpgpusim PUBLIC "-Wl,-rpath,${RAMULATOR_ROOT}")
endif()

if(GPGPUSIM_USE_POWER_MODEL)
target_compile_definitions(gpgpusim PRIVATE GPGPUSIM_POWER_MODEL)
target_include_directories(gpgpusim PRIVATE ${GPGPUSIM_POWER_MODEL})
endif()
