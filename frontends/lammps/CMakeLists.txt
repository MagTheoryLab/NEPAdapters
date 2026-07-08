set(NEP_ADAPTERS_LAMMPS_SOURCE_DIR "" CACHE PATH
  "Path to a LAMMPS source tree containing src/pair.h")
set(NEP_ADAPTERS_LAMMPS_EXECUTABLE "" CACHE FILEPATH
  "Optional LAMMPS executable used for runtime plugin smoke tests")
set(NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR "" CACHE PATH
  "Optional LAMMPS Kokkos build tree used to compile Kokkos-aware plugin styles")

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

if(NOT TARGET NEPAdapters::cpu_nep3)
  message(FATAL_ERROR "NEP_ADAPTERS_ENABLE_LAMMPS=ON currently requires cpu_nep3.")
endif()

find_package(MPI QUIET COMPONENTS CXX)

if(NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR)
  list(APPEND CMAKE_PREFIX_PATH
    "${NEP_ADAPTERS_LAMMPS_KOKKOS_BUILD_DIR}/cmake_packages/Kokkos")
  find_package(Kokkos CONFIG REQUIRED)
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
  pair_nep_adapters_cpu.cpp
)
add_library(NEPAdapters::lammps_frontend ALIAS nep_adapters_lammps_frontend)

target_link_libraries(
  nep_adapters_lammps_frontend
  PUBLIC
    NEPAdapters::cpu_nep3
)
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

if(EXISTS "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/src/lammpsplugin.h")
  add_library(nepadaptersplugin MODULE
    nepadaptersplugin.cpp
    pair_nep_adapters_common.cpp
    pair_nep_adapters_cpu.cpp
  )
  target_link_libraries(nepadaptersplugin PRIVATE NEPAdapters::cpu_nep3)
  if(TARGET NEPAdapters::cuda)
    target_sources(nepadaptersplugin PRIVATE pair_nep_adapters_cuda.cpp)
    target_link_libraries(nepadaptersplugin PRIVATE NEPAdapters::cuda)
    target_compile_definitions(nepadaptersplugin PRIVATE
      NEP_ADAPTERS_LAMMPS_ENABLE_CUDA=1)
  endif()
  nep_adapters_lammps_enable_kokkos(nepadaptersplugin NO_LINK)
  if(MPI_CXX_FOUND)
    target_link_libraries(nepadaptersplugin PRIVATE MPI::MPI_CXX)
  else()
    target_include_directories(
      nepadaptersplugin
      PRIVATE
        "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/src/STUBS"
    )
  endif()
  target_include_directories(
    nepadaptersplugin
    PRIVATE
      "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/src"
  )
  target_compile_features(nepadaptersplugin PRIVATE cxx_std_17)
  set_target_properties(
    nepadaptersplugin
    PROPERTIES
      PREFIX ""
      SUFFIX ".so"
      POSITION_INDEPENDENT_CODE ON
      BUILD_RPATH "$ORIGIN"
      INSTALL_RPATH "$ORIGIN"
  )
  if(APPLE)
    target_link_options(nepadaptersplugin PRIVATE "-undefined" "dynamic_lookup")
  endif()
  install(
    TARGETS nepadaptersplugin
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
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

  if(TARGET nepadaptersplugin)
    add_test(
      NAME nep_adapters_lammps_plugin_compile_test
      COMMAND
        "${CMAKE_COMMAND}"
        --build "${CMAKE_BINARY_DIR}"
        --target nepadaptersplugin
    )
    set_tests_properties(nep_adapters_lammps_plugin_compile_test PROPERTIES
      LABELS "smoke;frontend;lammps;plugin")

    if(Python3_Interpreter_FOUND AND
       EXISTS "${NEP_ADAPTERS_LAMMPS_EXECUTABLE}" AND
       EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt" AND
       EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz")
      add_test(
        NAME nep_adapters_lammps_plugin_baseline_test
        COMMAND
          "${Python3_EXECUTABLE}"
          "${PROJECT_SOURCE_DIR}/tools/run_lammps_baseline_smoke.py"
          --lmp "${NEP_ADAPTERS_LAMMPS_EXECUTABLE}"
          --plugin "$<TARGET_FILE:nepadaptersplugin>"
          --model "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt"
          --fixture "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz"
          --work-dir "${CMAKE_BINARY_DIR}/lammps_baseline_smoke"
      )
      set_tests_properties(nep_adapters_lammps_plugin_baseline_test PROPERTIES
        LABELS "smoke;frontend;lammps;plugin;baseline"
        DEPENDS "nep_adapters_lammps_plugin_compile_test")
    endif()
  endif()
endif()
