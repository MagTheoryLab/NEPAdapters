set(NEP_ADAPTERS_LAMMPS_SOURCE_DIR "" CACHE PATH
  "Path to a LAMMPS source tree containing src/pair.h")
set(NEP_ADAPTERS_LAMMPS_EXECUTABLE "" CACHE FILEPATH
  "Optional LAMMPS executable used for runtime plugin smoke tests")
set(NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR "" CACHE PATH
  "LAMMPS Kokkos build tree required by the CUDA pair style")

if(NOT NEP_ADAPTERS_LAMMPS_SOURCE_DIR)
  foreach(_candidate IN ITEMS
      "${PROJECT_SOURCE_DIR}/../lammps")
    if(EXISTS "${_candidate}/src/pair.h")
      set(NEP_ADAPTERS_LAMMPS_SOURCE_DIR "${_candidate}" CACHE PATH
        "Path to a LAMMPS source tree containing src/pair.h" FORCE)
      break()
    endif()
  endforeach()
endif()

if(NOT NEP_ADAPTERS_LAMMPS_EXECUTABLE)
  foreach(_candidate IN ITEMS
      "${PROJECT_SOURCE_DIR}/build-local-lammps-mpi/lmp"
      "${PROJECT_SOURCE_DIR}/../lammps/build/lmp")
    if(EXISTS "${_candidate}")
      set(NEP_ADAPTERS_LAMMPS_EXECUTABLE "${_candidate}" CACHE FILEPATH
        "Optional LAMMPS executable used for runtime plugin smoke tests" FORCE)
      break()
    endif()
  endforeach()
endif()

if(NOT EXISTS "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/src/pair.h")
  message(FATAL_ERROR
    "NEP_ADAPTERS_ENABLE_LAMMPS=ON requires NEP_ADAPTERS_LAMMPS_SOURCE_DIR "
    "to point to a LAMMPS source tree containing src/pair.h.")
endif()

if(NOT TARGET NEPAdapters::cpu AND NOT TARGET NEPAdapters::cuda)
  message(FATAL_ERROR
    "NEP_ADAPTERS_ENABLE_LAMMPS=ON requires at least one enabled engine.")
endif()

find_package(MPI QUIET COMPONENTS CXX)

if(TARGET NEPAdapters::cuda AND NOT NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR)
  message(FATAL_ERROR
    "The NEPAdapters CUDA LAMMPS pair is Kokkos device-only. "
    "Set NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR to a CUDA-enabled "
    "LAMMPS Kokkos build tree.")
endif()

if(NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR)
  list(APPEND CMAKE_PREFIX_PATH
    "${NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR}/cmake_packages/Kokkos")
  set(_nep_adapters_kokkos_launcher
    "${NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR}/lib/kokkos/temp/kokkos_launch_compiler")
  if(NOT Kokkos_COMPILE_LAUNCHER AND
     EXISTS "${_nep_adapters_kokkos_launcher}")
    set(Kokkos_COMPILE_LAUNCHER "${_nep_adapters_kokkos_launcher}"
      CACHE FILEPATH "Kokkos compiler launcher" FORCE)
  endif()
  set(_nep_adapters_nvcc_wrapper
    "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/lib/kokkos/bin/nvcc_wrapper")
  if(NOT Kokkos_NVCC_WRAPPER AND EXISTS "${_nep_adapters_nvcc_wrapper}")
    set(Kokkos_NVCC_WRAPPER "${_nep_adapters_nvcc_wrapper}"
      CACHE FILEPATH "Kokkos nvcc wrapper" FORCE)
  endif()
  find_package(Kokkos CONFIG REQUIRED)
  if(TARGET NEPAdapters::cuda AND NOT Kokkos_ENABLE_CUDA)
    message(FATAL_ERROR
      "The NEPAdapters CUDA LAMMPS pair requires Kokkos_ENABLE_CUDA=ON.")
  endif()
  set_source_files_properties(
    pair_nep_adapters_cuda.cpp
    PROPERTIES
      LANGUAGE CUDA
      COMPILE_OPTIONS "--extended-lambda")
endif()

function(nep_adapters_lammps_add_kokkos_interface target)
  foreach(_kokkos_target IN ITEMS
      Kokkos::kokkoscore
      Kokkos::kokkoscontainers
      Kokkos::kokkosalgorithms
      Kokkos::kokkossimd)
    foreach(_include_prop IN ITEMS
        INTERFACE_INCLUDE_DIRECTORIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
      get_target_property(_kokkos_includes ${_kokkos_target} ${_include_prop})
      if(_kokkos_includes)
        target_include_directories(${target} SYSTEM PRIVATE ${_kokkos_includes})
      endif()
    endforeach()

    get_target_property(_kokkos_definitions
      ${_kokkos_target} INTERFACE_COMPILE_DEFINITIONS)
    if(_kokkos_definitions)
      target_compile_definitions(${target} PRIVATE ${_kokkos_definitions})
    endif()

    get_target_property(_kokkos_options
      ${_kokkos_target} INTERFACE_COMPILE_OPTIONS)
    if(_kokkos_options)
      target_compile_options(${target} PRIVATE ${_kokkos_options})
    endif()
  endforeach()
endfunction()

function(nep_adapters_lammps_enable_kokkos target)
  if(NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR)
    set(_link_kokkos TRUE)
    if(ARGC GREATER 1 AND "${ARGV1}" STREQUAL "NO_LINK")
      set(_link_kokkos FALSE)
    endif()
    target_compile_definitions(${target} PRIVATE LMP_KOKKOS)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_include_directories(${target} PRIVATE
      "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/src/KOKKOS")
    if(_link_kokkos)
      target_link_libraries(${target} PRIVATE Kokkos::kokkos)
    else()
      nep_adapters_lammps_add_kokkos_interface(${target})
    endif()
  endif()
endfunction()

add_library(nep_adapters_lammps_frontend STATIC
  pair_nep_adapters_common.cpp
)
add_library(NEPAdapters::lammps_frontend ALIAS nep_adapters_lammps_frontend)

if(TARGET NEPAdapters::cpu)
  target_sources(nep_adapters_lammps_frontend PRIVATE
    pair_nep_adapters_cpu.cpp)
  target_link_libraries(
    nep_adapters_lammps_frontend PUBLIC NEPAdapters::cpu)
  target_compile_definitions(
    nep_adapters_lammps_frontend PUBLIC NEP_ADAPTERS_LAMMPS_ENABLE_CPU=1)
endif()
if(TARGET NEPAdapters::cuda)
  target_sources(nep_adapters_lammps_frontend PRIVATE
    pair_nep_adapters_cuda.cpp
  )
  target_link_libraries(nep_adapters_lammps_frontend PUBLIC NEPAdapters::cuda)
  target_compile_definitions(
    nep_adapters_lammps_frontend
    PUBLIC
      NEP_ADAPTERS_LAMMPS_ENABLE_CUDA=1)
endif()
nep_adapters_lammps_enable_kokkos(nep_adapters_lammps_frontend)
if(MPI_CXX_FOUND)
  target_link_libraries(nep_adapters_lammps_frontend PRIVATE MPI::MPI_CXX)
else()
  message(WARNING
    "MPI CXX was not found; using LAMMPS src/STUBS for the LAMMPS compile-smoke target.")
  target_include_directories(
    nep_adapters_lammps_frontend
    PUBLIC
      "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/src/STUBS"
  )
endif()

target_include_directories(
  nep_adapters_lammps_frontend
  PUBLIC
    "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/src"
)

target_compile_features(nep_adapters_lammps_frontend PUBLIC cxx_std_17)
set_target_properties(
  nep_adapters_lammps_frontend
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

function(nep_adapters_lammps_configure_plugin target)
  if(MPI_CXX_FOUND)
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
  else()
    target_include_directories(
      ${target}
      PRIVATE
        "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/src/STUBS"
    )
  endif()
  target_include_directories(
    ${target}
    PRIVATE
      "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/src"
  )
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(
    ${target}
    PROPERTIES
      PREFIX ""
      SUFFIX ".so"
      POSITION_INDEPENDENT_CODE ON
      BUILD_RPATH "$ORIGIN"
      INSTALL_RPATH "$ORIGIN"
  )
  if(APPLE)
    target_link_options(${target} PRIVATE "-undefined" "dynamic_lookup")
  endif()
  install(
    TARGETS ${target}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endfunction()

if(EXISTS "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/src/lammpsplugin.h")
  if(TARGET NEPAdapters::cpu)
    add_library(nepadapterscpuplugin MODULE
      nepadaptersplugin.cpp
      pair_nep_adapters_common.cpp
      pair_nep_adapters_cpu.cpp
    )
    target_link_libraries(nepadapterscpuplugin PRIVATE NEPAdapters::cpu)
    target_compile_definitions(nepadapterscpuplugin PRIVATE
      NEP_ADAPTERS_LAMMPS_ENABLE_CPU=1)
    nep_adapters_lammps_configure_plugin(nepadapterscpuplugin)
  endif()

  if(TARGET NEPAdapters::cuda)
    add_library(nepadaptersgpuplugin MODULE
      nepadaptersplugin.cpp
      pair_nep_adapters_common.cpp
      pair_nep_adapters_cuda.cpp
    )
    target_link_libraries(nepadaptersgpuplugin PRIVATE NEPAdapters::cuda)
    target_compile_definitions(nepadaptersgpuplugin PRIVATE
      NEP_ADAPTERS_LAMMPS_ENABLE_CUDA=1)
    nep_adapters_lammps_enable_kokkos(nepadaptersgpuplugin NO_LINK)
    nep_adapters_lammps_configure_plugin(nepadaptersgpuplugin)
  endif()
endif()

if(NEP_ADAPTERS_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter QUIET)

  add_test(
    NAME nep_adapters_lammps_frontend_compile_test
    COMMAND
      "${CMAKE_COMMAND}"
      --build "${CMAKE_BINARY_DIR}"
      --target nep_adapters_lammps_frontend
  )
  set_tests_properties(nep_adapters_lammps_frontend_compile_test PROPERTIES
    LABELS "smoke;frontend;lammps")

  if(TARGET nepadapterscpuplugin)
    add_test(
      NAME nep_adapters_lammps_cpu_plugin_compile_test
      COMMAND
        "${CMAKE_COMMAND}"
        --build "${CMAKE_BINARY_DIR}"
        --target nepadapterscpuplugin
    )
    set_tests_properties(nep_adapters_lammps_cpu_plugin_compile_test PROPERTIES
      LABELS "smoke;frontend;lammps;plugin;cpu")
  endif()

  if(TARGET nepadaptersgpuplugin)
    add_test(
      NAME nep_adapters_lammps_gpu_plugin_compile_test
      COMMAND
        "${CMAKE_COMMAND}"
        --build "${CMAKE_BINARY_DIR}"
        --target nepadaptersgpuplugin
    )
    set_tests_properties(nep_adapters_lammps_gpu_plugin_compile_test PROPERTIES
      LABELS "smoke;frontend;lammps;plugin;cuda;kokkos")
  endif()

  if(Python3_Interpreter_FOUND AND
     EXISTS "${NEP_ADAPTERS_LAMMPS_EXECUTABLE}" AND
     EXISTS "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt" AND
     EXISTS "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/train.xyz")
    if(TARGET nepadapterscpuplugin)
      add_test(
        NAME nep_adapters_lammps_plugin_baseline_test
        COMMAND
          "${Python3_EXECUTABLE}"
          "${PROJECT_SOURCE_DIR}/tools/run_lammps_baseline_smoke.py"
          --lmp "${NEP_ADAPTERS_LAMMPS_EXECUTABLE}"
          --plugin "$<TARGET_FILE:nepadapterscpuplugin>"
          --model "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt"
          --fixture "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/train.xyz"
          --work-dir "${CMAKE_BINARY_DIR}/lammps_baseline_smoke"
      )
      set_tests_properties(
        nep_adapters_lammps_plugin_baseline_test PROPERTIES
        LABELS "smoke;frontend;lammps;plugin;baseline;cpu"
        DEPENDS "nep_adapters_lammps_cpu_plugin_compile_test")
    endif()

    if(TARGET nepadaptersgpuplugin AND Kokkos_ENABLE_CUDA)
      add_test(
        NAME nep_adapters_lammps_cuda_plugin_baseline_test
        COMMAND
          "${Python3_EXECUTABLE}"
          "${PROJECT_SOURCE_DIR}/tools/run_lammps_baseline_smoke.py"
          --lmp "${NEP_ADAPTERS_LAMMPS_EXECUTABLE}"
          --plugin "$<TARGET_FILE:nepadaptersgpuplugin>"
          --model "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt"
          --fixture "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/train.xyz"
          --pair-style nep/gpu
          --energy-tolerance 1.0e-6
          --force-tolerance 1.0e-6
          --virial-tolerance 5.0e-6
          --work-dir "${CMAKE_BINARY_DIR}/lammps_cuda_baseline_smoke"
      )
      set_tests_properties(
        nep_adapters_lammps_cuda_plugin_baseline_test PROPERTIES
        LABELS "smoke;frontend;lammps;plugin;baseline;cuda;kokkos;device"
        DEPENDS "nep_adapters_lammps_gpu_plugin_compile_test")
    endif()
  endif()

  set(_nep_adapters_spin_fixture_dir
    "${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol")
  if(Python3_Interpreter_FOUND AND
     EXISTS "${NEP_ADAPTERS_LAMMPS_EXECUTABLE}" AND
     EXISTS "${_nep_adapters_spin_fixture_dir}/nep.txt" AND
     EXISTS "${_nep_adapters_spin_fixture_dir}/lammps_structure.json" AND
     EXISTS "${_nep_adapters_spin_fixture_dir}/reference_small_pbc.txt")
    if(TARGET nepadapterscpuplugin)
      add_test(
        NAME nep_adapters_lammps_spin_cpu_plugin_test
        COMMAND
          "${Python3_EXECUTABLE}"
          "${PROJECT_SOURCE_DIR}/tools/run_lammps_spin_smoke.py"
          --lmp "${NEP_ADAPTERS_LAMMPS_EXECUTABLE}"
          --plugin "$<TARGET_FILE:nepadapterscpuplugin>"
          --model "${_nep_adapters_spin_fixture_dir}/nep.txt"
          --structure "${_nep_adapters_spin_fixture_dir}/lammps_structure.json"
          --reference "${_nep_adapters_spin_fixture_dir}/reference_small_pbc.txt"
          --pair-style nep/cpu
          --work-dir "${CMAKE_BINARY_DIR}/lammps_spin_cpu_smoke"
      )
      set_tests_properties(
        nep_adapters_lammps_spin_cpu_plugin_test PROPERTIES
        LABELS "smoke;frontend;lammps;plugin;spin;cpu;real_lammps"
        DEPENDS "nep_adapters_lammps_cpu_plugin_compile_test")

      if(MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
        set(_nep_adapters_mpiexec_numproc_flag "${MPIEXEC_NUMPROC_FLAG}")
        if(NOT _nep_adapters_mpiexec_numproc_flag)
          set(_nep_adapters_mpiexec_numproc_flag "-np")
        endif()
        foreach(_mpi_ranks IN ITEMS 2 4 8)
          if(MPIEXEC_MAX_NUMPROCS AND
             _mpi_ranks GREATER MPIEXEC_MAX_NUMPROCS)
            continue()
          endif()
          set(_mpi_test_name
            "nep_adapters_lammps_spin_cpu_mpi_${_mpi_ranks}_test")
          add_test(
            NAME "${_mpi_test_name}"
            COMMAND
              "${Python3_EXECUTABLE}"
              "${PROJECT_SOURCE_DIR}/tools/run_lammps_spin_smoke.py"
              --lmp "${NEP_ADAPTERS_LAMMPS_EXECUTABLE}"
              --plugin "$<TARGET_FILE:nepadapterscpuplugin>"
              --model "${_nep_adapters_spin_fixture_dir}/nep.txt"
              --structure
                "${_nep_adapters_spin_fixture_dir}/lammps_structure.json"
              --reference
                "${_nep_adapters_spin_fixture_dir}/reference_small_pbc.txt"
              --pair-style nep/cpu
              --mpiexec "${MPIEXEC_EXECUTABLE}"
              "--mpiexec-numproc-flag=${_nep_adapters_mpiexec_numproc_flag}"
              --mpi-ranks "${_mpi_ranks}"
              --work-dir
                "${CMAKE_BINARY_DIR}/lammps_spin_cpu_mpi_${_mpi_ranks}_smoke"
          )
          set_tests_properties(
            "${_mpi_test_name}" PROPERTIES
            LABELS
              "smoke;frontend;lammps;plugin;spin;cpu;mpi;mpi_${_mpi_ranks};real_lammps"
            PROCESSORS "${_mpi_ranks}"
            ENVIRONMENT "OMP_NUM_THREADS=1"
            DEPENDS "nep_adapters_lammps_cpu_plugin_compile_test")
        endforeach()
      endif()
    endif()

    if(TARGET nepadaptersgpuplugin AND Kokkos_ENABLE_CUDA)
      add_test(
        NAME nep_adapters_lammps_spin_cuda_plugin_test
        COMMAND
          "${Python3_EXECUTABLE}"
          "${PROJECT_SOURCE_DIR}/tools/run_lammps_spin_smoke.py"
          --lmp "${NEP_ADAPTERS_LAMMPS_EXECUTABLE}"
          --plugin "$<TARGET_FILE:nepadaptersgpuplugin>"
          --model "${_nep_adapters_spin_fixture_dir}/nep.txt"
          --structure "${_nep_adapters_spin_fixture_dir}/lammps_structure.json"
          --reference "${_nep_adapters_spin_fixture_dir}/reference_small_pbc.txt"
          --pair-style nep/gpu/kk
          --work-dir "${CMAKE_BINARY_DIR}/lammps_spin_cuda_smoke"
          --energy-tolerance 1.0e-6
          --force-tolerance 2.0e-4
          --mforce-tolerance 2.0e-4
          --virial-tolerance 2.0e-4
      )
      set_tests_properties(
        nep_adapters_lammps_spin_cuda_plugin_test PROPERTIES
        LABELS "smoke;frontend;lammps;plugin;spin;cuda;kokkos;device;real_lammps"
        DEPENDS "nep_adapters_lammps_gpu_plugin_compile_test")
    endif()
  endif()
endif()
