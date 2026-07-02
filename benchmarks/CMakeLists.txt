if(NOT TARGET NEPAdapters::cpu_nep3)
  return()
endif()

if(NOT EXISTS "${NEP_ADAPTERS_NEP89_MODEL_PATH}" OR
   NOT EXISTS "${NEP_ADAPTERS_NEP89_XYZ_PATH}")
  return()
endif()

add_executable(nep_adapters_bench_cpu_nep89 cpp/bench_cpu_nep89.cpp)
target_link_libraries(nep_adapters_bench_cpu_nep89 PRIVATE NEPAdapters::cpu_nep3)
if(TARGET NEPAdapters::cpu_opt)
  target_link_libraries(nep_adapters_bench_cpu_nep89 PRIVATE NEPAdapters::cpu_opt)
  target_compile_definitions(nep_adapters_bench_cpu_nep89 PRIVATE
    NEP_ADAPTERS_BENCH_HAS_CPU_OPT=1)
else()
  target_compile_definitions(nep_adapters_bench_cpu_nep89 PRIVATE
    NEP_ADAPTERS_BENCH_HAS_CPU_OPT=0)
endif()
target_include_directories(
  nep_adapters_bench_cpu_nep89
  PRIVATE
    "${PROJECT_SOURCE_DIR}/tests"
)
if(NEP_ADAPTERS_CPU_NEP3_ENABLE_OPENMP OR NEP_ADAPTERS_CPU_OPT_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(nep_adapters_bench_cpu_nep89 PRIVATE OpenMP::OpenMP_CXX)
  endif()
  target_compile_definitions(nep_adapters_bench_cpu_nep89 PRIVATE
    NEP_ADAPTERS_BENCH_OPENMP_ENABLED=1)
else()
  target_compile_definitions(nep_adapters_bench_cpu_nep89 PRIVATE
    NEP_ADAPTERS_BENCH_OPENMP_ENABLED=0)
endif()
target_compile_definitions(
  nep_adapters_bench_cpu_nep89
  PRIVATE
    NEP_ADAPTERS_NEP89_MODEL_PATH="${NEP_ADAPTERS_NEP89_MODEL_PATH}"
    NEP_ADAPTERS_NEP89_XYZ_PATH="${NEP_ADAPTERS_NEP89_XYZ_PATH}"
)

add_test(NAME nep_adapters_bench_cpu_nep89 COMMAND nep_adapters_bench_cpu_nep89)
set_tests_properties(nep_adapters_bench_cpu_nep89 PROPERTIES
  LABELS "bench;performance;engine;cpu_nep3;nep89"
  TIMEOUT 120)

add_test(
  NAME nep_adapters_bench_cpu_nep89_lammps
  COMMAND nep_adapters_bench_cpu_nep89
    --mode lammps
    --engine cpu_nep3
    --warmup 0
    --iterations 1
    --replicate 1x1x1
    --rank-grid 2x1x1
    --rank-id 0)
set_tests_properties(nep_adapters_bench_cpu_nep89_lammps PROPERTIES
  LABELS "bench;performance;engine;cpu_nep3;nep89;lammps"
  TIMEOUT 120)

if(TARGET NEPAdapters::cpu_opt)
  add_test(
    NAME nep_adapters_bench_cpu_nep89_lammps_cpu_opt
    COMMAND nep_adapters_bench_cpu_nep89
      --mode lammps
      --engine cpu_opt
      --warmup 0
      --iterations 1
      --replicate 1x1x1
      --rank-grid 2x1x1
      --rank-id 0)
  set_tests_properties(nep_adapters_bench_cpu_nep89_lammps_cpu_opt PROPERTIES
    LABELS "bench;performance;engine;cpu_opt;nep89;lammps"
    TIMEOUT 120)
endif()
