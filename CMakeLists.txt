cmake_minimum_required(VERSION 3.24)

set(NEP_ADAPTERS_ENABLE_CUDA_DEFAULT OFF)
if(DEFINED ENV{NEP_CUDA})
  string(TOUPPER "$ENV{NEP_CUDA}" NEP_ADAPTERS_CUDA_ENV)
  if(NEP_ADAPTERS_CUDA_ENV MATCHES "^(1|ON|TRUE|YES)$")
    set(NEP_ADAPTERS_ENABLE_CUDA_DEFAULT ON)
  elseif(NOT NEP_ADAPTERS_CUDA_ENV MATCHES "^(0|OFF|FALSE|NO)$")
    message(FATAL_ERROR
      "NEP_CUDA must be one of 1/0, ON/OFF, TRUE/FALSE, or YES/NO.")
  endif()
  unset(NEP_ADAPTERS_CUDA_ENV)
endif()
option(NEP_ADAPTERS_ENABLE_CUDA
  "Enable the CUDA engine; NEP_CUDA=1 provides a source-install shortcut"
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
  project(NEPAdapters VERSION 0.1.0 LANGUAGES CXX CUDA)
else()
  project(NEPAdapters VERSION 0.1.0 LANGUAGES CXX)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

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
