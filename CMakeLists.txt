cmake_minimum_required(VERSION 3.20)

if(NEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME AND
   NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

if(NEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME)
  project(NEPAdapters VERSION 0.1.0 LANGUAGES CXX CUDA)
else()
  project(NEPAdapters VERSION 0.1.0 LANGUAGES CXX)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(NEP_ADAPTERS_BUILD_TESTS "Build NEPAdapters tests" ON)
option(NEP_ADAPTERS_BUILD_BENCHMARKS "Build NEPAdapters performance benchmarks" OFF)
option(NEP_ADAPTERS_ENABLE_CPU_NEP3 "Enable the NEP3 CPU reference engine shim" ON)
option(NEP_ADAPTERS_ENABLE_CPU_OPT "Enable the optimized CPU engine" OFF)
option(NEP_ADAPTERS_ENABLE_CUDA "Enable the CUDA engine" OFF)
option(NEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME "Build CUDA runtime upload/device code" OFF)
option(NEP_ADAPTERS_ENABLE_PYTHON "Enable the Python frontend" OFF)
option(NEP_ADAPTERS_ENABLE_LAMMPS "Enable the LAMMPS frontend" OFF)

if(NEP_ADAPTERS_BUILD_TESTS OR NEP_ADAPTERS_BUILD_BENCHMARKS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/NEPAdaptersTestData.cmake")
  enable_testing()
endif()

add_subdirectory(src)

if(NEP_ADAPTERS_ENABLE_CPU_NEP3)
  add_subdirectory(engines/cpu_nep3)
endif()

if(NEP_ADAPTERS_ENABLE_CPU_OPT)
  add_subdirectory(engines/cpu_opt)
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

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NEPAdaptersConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NEPAdaptersConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NEPAdapters
)
