if(TARGET NEPAdapters::cpu AND
   EXISTS "${NEP_ADAPTERS_NEP89_MODEL_PATH}" AND
   EXISTS "${NEP_ADAPTERS_NEP89_XYZ_PATH}")
  add_executable(nep_adapters_bench_cpu_nep89 cpp/bench_cpu_nep89.cpp)
  target_link_libraries(nep_adapters_bench_cpu_nep89 PRIVATE NEPAdapters::cpu)
  target_include_directories(
    nep_adapters_bench_cpu_nep89
    PRIVATE
      "${PROJECT_SOURCE_DIR}/tests"
  )
  if(NEP_ADAPTERS_CPU_ENABLE_OPENMP)
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
    LABELS "bench;performance;engine;cpu;nep89"
    TIMEOUT 120)

  add_test(
    NAME nep_adapters_bench_cpu_nep89_lammps
    COMMAND nep_adapters_bench_cpu_nep89
      --mode lammps
      --engine cpu
      --warmup 0
      --iterations 1
      --replicate 1x1x1
      --rank-grid 2x1x1
      --rank-id 0)
  set_tests_properties(nep_adapters_bench_cpu_nep89_lammps PROPERTIES
    LABELS "bench;performance;engine;cpu;nep89;lammps"
    TIMEOUT 120)
endif()

if(TARGET NEPAdapters::cpu)
  add_executable(nep_adapters_bench_cpu_spin_lammps
    cpp/bench_cpu_spin_lammps.cpp)
  target_link_libraries(nep_adapters_bench_cpu_spin_lammps PRIVATE NEPAdapters::cpu)
  if(NEP_ADAPTERS_CPU_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    if(TARGET OpenMP::OpenMP_CXX)
      target_link_libraries(nep_adapters_bench_cpu_spin_lammps PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_compile_definitions(nep_adapters_bench_cpu_spin_lammps PRIVATE
      NEP_ADAPTERS_BENCH_OPENMP_ENABLED=1)
  else()
    target_compile_definitions(nep_adapters_bench_cpu_spin_lammps PRIVATE
      NEP_ADAPTERS_BENCH_OPENMP_ENABLED=0)
  endif()
  target_compile_definitions(
    nep_adapters_bench_cpu_spin_lammps
    PRIVATE
      NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE="${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/nep.txt"
      NEP_ADAPTERS_NONMAG_FIXTURE="${PROJECT_SOURCE_DIR}/tests/fixtures/cpu_baseline/nep.txt"
  )
  add_test(
    NAME nep_adapters_bench_cpu_spin_lammps
    COMMAND nep_adapters_bench_cpu_spin_lammps --warmup 1 --iterations 3)
  set_tests_properties(nep_adapters_bench_cpu_spin_lammps PROPERTIES
    LABELS "bench;performance;engine;cpu;spin;lammps"
    TIMEOUT 120)
endif()

if(TARGET NEPAdapters::cuda)
  add_executable(nep_adapters_bench_cuda_batch cuda/bench_cuda_batch.cpp)
  target_link_libraries(nep_adapters_bench_cuda_batch PRIVATE NEPAdapters::cuda)
  target_compile_definitions(nep_adapters_bench_cuda_batch PRIVATE
    NEP_ADAPTERS_BENCH_DEVICE_PIPELINE=1)
  target_include_directories(nep_adapters_bench_cuda_batch PRIVATE
    "${PROJECT_SOURCE_DIR}/engines/cuda"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  add_test(
    NAME nep_adapters_bench_cuda_batch
    COMMAND nep_adapters_bench_cuda_batch --warmup 1 --iterations 3)
  set_tests_properties(nep_adapters_bench_cuda_batch PROPERTIES
    LABELS "bench;performance;engine;cuda;batch"
    TIMEOUT 120)

  find_package(CUDAToolkit REQUIRED)

    add_executable(nep_adapters_bench_cuda_lammps_device_layout
      cuda/bench_cuda_lammps_device_layout.cu)
    target_link_libraries(
      nep_adapters_bench_cuda_lammps_device_layout
      PRIVATE
        NEPAdapters::cuda)
    target_include_directories(
      nep_adapters_bench_cuda_lammps_device_layout
      PRIVATE
        "${PROJECT_SOURCE_DIR}/engines/cuda"
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(nep_adapters_bench_cuda_lammps_device_layout PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)

    add_executable(nep_adapters_bench_cuda_nve_compare
      cuda/bench_cuda_nve_compare.cu)
    target_link_libraries(
      nep_adapters_bench_cuda_nve_compare
      PRIVATE
        NEPAdapters::cuda)
    target_include_directories(
      nep_adapters_bench_cuda_nve_compare
      PRIVATE
        "${PROJECT_SOURCE_DIR}/engines/cuda"
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(nep_adapters_bench_cuda_nve_compare PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON)
endif()

if(TARGET NEPAdapters::cuda AND TARGET NEPAdapters::cpu AND
   EXISTS "${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/nep.txt")
  add_executable(nep_adapters_bench_spin_gpu cuda/bench_spin_gpu.cpp)
  target_link_libraries(
    nep_adapters_bench_spin_gpu
    PRIVATE
      NEPAdapters::cuda
      NEPAdapters::cpu)
  target_compile_definitions(
    nep_adapters_bench_spin_gpu
    PRIVATE
      NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE="${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/nep.txt")
  add_test(
    NAME nep_adapters_bench_spin_gpu
    COMMAND nep_adapters_bench_spin_gpu --warmup 1 --iterations 2 --replicate 2)
  set_tests_properties(nep_adapters_bench_spin_gpu PROPERTIES
    LABELS "bench;performance;engine;cuda;spin"
    TIMEOUT 120)
endif()
