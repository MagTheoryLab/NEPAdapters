cmake_minimum_required(VERSION 3.24)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION"
  NEP_ADAPTERS_VERSION
  LIMIT_COUNT 1)
string(STRIP "${NEP_ADAPTERS_VERSION}" NEP_ADAPTERS_VERSION)
if(NOT NEP_ADAPTERS_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR
    "VERSION must contain exactly one semantic version in MAJOR.MINOR.PATCH form")
endif()

include(cmake/NEPAdaptersCUDAAutoDetect.cmake)
nep_adapters_resolve_cuda_default(NEP_ADAPTERS_ENABLE_CUDA_DEFAULT)
option(NEP_ADAPTERS_ENABLE_CUDA
  "Enable CUDA; auto-detects nvcc unless NEP_CUDA or this option is set"
  ${NEP_ADAPTERS_ENABLE_CUDA_DEFAULT})
unset(NEP_ADAPTERS_ENABLE_CUDA_DEFAULT)
option(
  NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM
  "Build the optional qNEP PPPM implementation and link cuFFT"
  OFF)

if(NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM AND NOT NEP_ADAPTERS_ENABLE_CUDA)
  message(FATAL_ERROR
    "NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM=ON requires NEP_ADAPTERS_ENABLE_CUDA=ON.")
endif()

if(NEP_ADAPTERS_ENABLE_CUDA AND
   NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
if(NEP_ADAPTERS_ENABLE_CUDA)
  message(STATUS
    "NEPAdapters CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if(NEP_ADAPTERS_ENABLE_CUDA)
  project(NEPAdapters VERSION "${NEP_ADAPTERS_VERSION}" LANGUAGES CXX CUDA)
else()
  project(NEPAdapters VERSION "${NEP_ADAPTERS_VERSION}" LANGUAGES CXX)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(NEP_ADAPTERS_GENERATED_INCLUDE_DIR
  "${CMAKE_CURRENT_BINARY_DIR}/generated/include")
file(MAKE_DIRECTORY
  "${NEP_ADAPTERS_GENERATED_INCLUDE_DIR}/nep_adapters")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/NEPAdaptersVersion.h.in"
  "${NEP_ADAPTERS_GENERATED_INCLUDE_DIR}/nep_adapters/version.h"
  @ONLY)
set_property(GLOBAL PROPERTY
  NEP_ADAPTERS_GENERATED_INCLUDE_DIR
  "${NEP_ADAPTERS_GENERATED_INCLUDE_DIR}")

option(NEP_ADAPTERS_BUILD_TESTS "Build NEPAdapters tests" ON)
option(NEP_ADAPTERS_BUILD_BENCHMARKS "Build NEPAdapters performance benchmarks" OFF)
option(NEP_ADAPTERS_ENABLE_CPU "Enable the CPU engine" ON)
option(NEP_ADAPTERS_ENABLE_PYTHON "Enable the Python frontend" OFF)
option(NEP_ADAPTERS_ENABLE_LAMMPS "Enable the LAMMPS frontend" OFF)
option(
  NEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES
  "Install C/C++ libraries, headers, and CMake package metadata"
  ON)

set(NEP_ADAPTERS_PACKAGE_NEEDS_OPENMP OFF)
set(NEP_ADAPTERS_PACKAGE_NEEDS_BLAS OFF)
set(NEP_ADAPTERS_PACKAGE_NEEDS_CUDATOOLKIT OFF)

if(NEP_ADAPTERS_BUILD_TESTS OR NEP_ADAPTERS_BUILD_BENCHMARKS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/NEPAdaptersTestData.cmake")
  enable_testing()
endif()

add_subdirectory(src)

if(NEP_ADAPTERS_ENABLE_CPU)
  add_subdirectory(engines/cpu)
endif()

if(NEP_ADAPTERS_ENABLE_CUDA)
  add_subdirectory(engines/cuda)
endif()

if(NEP_ADAPTERS_ENABLE_PYTHON)
  add_subdirectory(frontends/python)
endif()

if(NEP_ADAPTERS_ENABLE_LAMMPS)
  add_subdirectory(frontends/lammps)
endif()

if(NEP_ADAPTERS_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(NEP_ADAPTERS_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(NEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES)
  install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
  install(
    FILES
      "${NEP_ADAPTERS_GENERATED_INCLUDE_DIR}/nep_adapters/version.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nep_adapters
  )

  install(
    TARGETS nep_adapters
    EXPORT NEPAdaptersTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  install(
    EXPORT NEPAdaptersTargets
    NAMESPACE NEPAdapters::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NEPAdapters
  )
endif()

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NEPAdaptersConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/NEPAdaptersConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NEPAdapters
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/NEPAdaptersConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

if(NEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES)
  install(
    FILES
      ${CMAKE_CURRENT_BINARY_DIR}/NEPAdaptersConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/NEPAdaptersConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NEPAdapters
  )
endif()
