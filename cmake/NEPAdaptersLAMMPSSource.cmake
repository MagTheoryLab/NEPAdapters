# Compile the copied NEPAdapters pair styles directly into a LAMMPS target.
#
# This file is loaded through CMAKE_PROJECT_INCLUDE while configuring LAMMPS.
# The pair sources themselves are installed by tools/install_lammps_source.py.

if(NOT PROJECT_NAME STREQUAL "lammps")
  return()
endif()

get_property(_nep_adapters_source_configured GLOBAL PROPERTY
  NEP_ADAPTERS_LAMMPS_SOURCE_CONFIGURED)
if(_nep_adapters_source_configured)
  return()
endif()
set_property(GLOBAL PROPERTY NEP_ADAPTERS_LAMMPS_SOURCE_CONFIGURED ON)

set(NEP_ADAPTERS_SOURCE_DIR "" CACHE PATH
  "Path to the NEPAdapters source tree")
set(NEP_ADAPTERS_LAMMPS_SOURCE_BACKEND "cpu" CACHE STRING
  "NEPAdapters source-tree backend: cpu, cuda, or both")
set_property(CACHE NEP_ADAPTERS_LAMMPS_SOURCE_BACKEND PROPERTY STRINGS
  cpu cuda both)

if(NOT EXISTS "${NEP_ADAPTERS_SOURCE_DIR}/CMakeLists.txt" OR
   NOT EXISTS "${NEP_ADAPTERS_SOURCE_DIR}/include/nep_adapters/api.h")
  message(FATAL_ERROR
    "NEP_ADAPTERS_SOURCE_DIR must point to the NEPAdapters source tree")
endif()

string(TOLOWER "${NEP_ADAPTERS_LAMMPS_SOURCE_BACKEND}"
  _nep_adapters_source_backend)
if(NOT _nep_adapters_source_backend MATCHES "^(cpu|cuda|both)$")
  message(FATAL_ERROR
    "NEP_ADAPTERS_LAMMPS_SOURCE_BACKEND must be cpu, cuda, or both")
endif()

get_filename_component(_nep_adapters_lammps_dir
  "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(_nep_adapters_lammps_src "${_nep_adapters_lammps_dir}/src")
set(_nep_adapters_required_pair_files
  pair_nep_adapters_common.cpp
  pair_nep_adapters_common.h)

set(_nep_adapters_source_enable_cpu OFF)
set(_nep_adapters_source_enable_cuda OFF)
if(_nep_adapters_source_backend STREQUAL "cpu" OR
   _nep_adapters_source_backend STREQUAL "both")
  set(_nep_adapters_source_enable_cpu ON)
  list(APPEND _nep_adapters_required_pair_files
    pair_nep_adapters_cpu.cpp
    pair_nep_adapters_cpu.h)
endif()
if(_nep_adapters_source_backend STREQUAL "cuda" OR
   _nep_adapters_source_backend STREQUAL "both")
  set(_nep_adapters_source_enable_cuda ON)
  list(APPEND _nep_adapters_required_pair_files
    pair_nep_adapters_cuda.cpp
    pair_nep_adapters_cuda.h)
endif()

foreach(_pair_file IN LISTS _nep_adapters_required_pair_files)
  if(NOT EXISTS "${_nep_adapters_lammps_src}/${_pair_file}")
    message(FATAL_ERROR
      "Missing ${_nep_adapters_lammps_src}/${_pair_file}. "
      "Run tools/install_lammps_source.py before configuring LAMMPS.")
  endif()
endforeach()

# Keep the embedded runtime narrow. LAMMPS supplies the frontend; the
# NEPAdapters sub-build only supplies core and the selected engines.
set(NEP_ADAPTERS_ENABLE_CPU ${_nep_adapters_source_enable_cpu}
  CACHE BOOL "Enable the CPU engine" FORCE)
set(NEP_ADAPTERS_ENABLE_CUDA ${_nep_adapters_source_enable_cuda}
  CACHE BOOL "Enable the CUDA engine" FORCE)
set(NEP_ADAPTERS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
set(NEP_ADAPTERS_ENABLE_LAMMPS OFF CACHE BOOL "" FORCE)
set(NEP_ADAPTERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEP_ADAPTERS_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(NEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES OFF CACHE BOOL "" FORCE)

if(_nep_adapters_source_enable_cuda)
  # pair_nep_adapters_cuda.cpp belongs to the parent LAMMPS target. CUDA must
  # therefore be enabled in this directory, not only by the NEPAdapters
  # subproject, so CMake creates the parent target's CUDA compile rules.
  enable_language(CUDA)
endif()

# The embedded engines are implementation details of the LAMMPS pair. Keep
# them static even when the caller builds a shared liblammps, so installing
# lmp does not create a second NEPAdapters runtime-library deployment problem.
set(_nep_adapters_build_shared_libs_was_defined OFF)
if(DEFINED BUILD_SHARED_LIBS)
  set(_nep_adapters_build_shared_libs_was_defined ON)
  set(_nep_adapters_saved_build_shared_libs "${BUILD_SHARED_LIBS}")
endif()
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(
  "${NEP_ADAPTERS_SOURCE_DIR}"
  "${CMAKE_BINARY_DIR}/_deps/nep_adapters")
if(_nep_adapters_build_shared_libs_was_defined)
  set(BUILD_SHARED_LIBS "${_nep_adapters_saved_build_shared_libs}")
else()
  unset(BUILD_SHARED_LIBS)
endif()

function(_nep_adapters_finish_lammps_source_integration)
  if(NOT TARGET lammps)
    message(FATAL_ERROR
      "NEPAdapters source-tree integration expected the LAMMPS target 'lammps'")
  endif()

  target_compile_features(lammps PRIVATE cxx_std_17)
  # LAMMPS discovers copied pair sources before this deferred hook runs. Add
  # the public adapter headers explicitly so those sources do not depend on
  # deferred transitive usage requirements from the engine link target.
  target_include_directories(lammps PRIVATE
    "${NEP_ADAPTERS_SOURCE_DIR}/include")
  if(_nep_adapters_source_enable_cpu)
    target_link_libraries(lammps PRIVATE NEPAdapters::cpu)
  endif()

  if(_nep_adapters_source_enable_cuda)
    if(NOT PKG_KOKKOS OR NOT Kokkos_ENABLE_CUDA)
      message(FATAL_ERROR
        "NEPAdapters CUDA source-tree integration requires "
        "PKG_KOKKOS=ON and Kokkos_ENABLE_CUDA=ON")
    endif()
    if(NOT TARGET NEPAdapters::cuda)
      message(FATAL_ERROR "NEPAdapters CUDA target was not configured")
    endif()

    set(_nep_adapters_cuda_pair
      "${_nep_adapters_lammps_src}/pair_nep_adapters_cuda.cpp")
    set_source_files_properties(
      "${_nep_adapters_cuda_pair}"
      PROPERTIES
        LANGUAGE CUDA
        COMPILE_OPTIONS "--extended-lambda")
    target_include_directories(lammps PRIVATE
      "${_nep_adapters_lammps_src}/KOKKOS")
    target_compile_definitions(lammps PRIVATE LMP_KOKKOS)
    target_link_libraries(lammps PRIVATE NEPAdapters::cuda)
  endif()

  message(STATUS
    "NEPAdapters pair styles are compiled into LAMMPS "
    "(${_nep_adapters_source_backend})")
endfunction()

# The project hook runs immediately after LAMMPS project(). Defer the target
# wiring until LAMMPS has created its library and processed Kokkos packages.
cmake_language(DEFER CALL _nep_adapters_finish_lammps_source_integration)
