set(NEP_ADAPTERS_CPU_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/native" CACHE PATH
  "Path to the optimized NEP CPU source directory containing nep.cpp and nep.h")

if(NOT EXISTS "${NEP_ADAPTERS_CPU_SOURCE_DIR}/nep.h" OR
   NOT EXISTS "${NEP_ADAPTERS_CPU_SOURCE_DIR}/nep.cpp" OR
   NOT EXISTS "${NEP_ADAPTERS_CPU_SOURCE_DIR}/neighbor_nep.cpp" OR
   NOT EXISTS "${NEP_ADAPTERS_CPU_SOURCE_DIR}/ewald_nep.cpp")
  message(FATAL_ERROR
    "NEP_ADAPTERS_ENABLE_CPU=ON requires NEP_ADAPTERS_CPU_SOURCE_DIR "
    "to point to a NEP CPU source directory containing nep.cpp, neighbor_nep.cpp, "
    "ewald_nep.cpp, and nep.h.")
endif()

option(NEP_ADAPTERS_CPU_ENABLE_OPENMP "Build cpu with OpenMP when available" ON)
option(NEP_ADAPTERS_CPU_ENABLE_NATIVE_ARCH
  "Build cpu for the host CPU architecture when supported"
  OFF)
set(NEP_ADAPTERS_CPU_USE_RADIAL_TABLE_DEFAULT ON)
option(NEP_ADAPTERS_CPU_USE_RADIAL_TABLE
  "Enable tabulated radial functions in cpu"
  ${NEP_ADAPTERS_CPU_USE_RADIAL_TABLE_DEFAULT})

add_library(nep_adapters_cpu
  cpu_engine.cpp
  "${NEP_ADAPTERS_CPU_SOURCE_DIR}/nep.cpp"
  "${NEP_ADAPTERS_CPU_SOURCE_DIR}/neighbor_nep.cpp"
  "${NEP_ADAPTERS_CPU_SOURCE_DIR}/ewald_nep.cpp"
)
add_library(NEPAdapters::cpu ALIAS nep_adapters_cpu)

target_link_libraries(nep_adapters_cpu PUBLIC NEPAdapters::nep_adapters)

target_include_directories(
  nep_adapters_cpu
  PRIVATE
    "${NEP_ADAPTERS_CPU_SOURCE_DIR}"
)

target_compile_features(nep_adapters_cpu PUBLIC cxx_std_17)
set_target_properties(nep_adapters_cpu PROPERTIES
  EXPORT_NAME cpu
  POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  target_compile_options(nep_adapters_cpu PRIVATE -Mnovect)
endif()

include(CheckCXXCompilerFlag)
if(NEP_ADAPTERS_CPU_ENABLE_NATIVE_ARCH)
  check_cxx_compiler_flag("-march=native" NEP_ADAPTERS_CPU_HAS_MARCH_NATIVE)
  if(NEP_ADAPTERS_CPU_HAS_MARCH_NATIVE)
    target_compile_options(nep_adapters_cpu PRIVATE -march=native)
  endif()
endif()

check_cxx_compiler_flag("-fno-math-errno" NEP_ADAPTERS_CPU_HAS_NO_MATH_ERRNO)
if(NEP_ADAPTERS_CPU_HAS_NO_MATH_ERRNO)
  target_compile_options(nep_adapters_cpu PRIVATE -fno-math-errno)
endif()
check_cxx_compiler_flag("-ffast-math" NEP_ADAPTERS_CPU_HAS_FAST_MATH)
if(NEP_ADAPTERS_CPU_HAS_FAST_MATH)
  target_compile_options(nep_adapters_cpu PRIVATE -ffast-math)
endif()

target_compile_definitions(
  nep_adapters_cpu
  PRIVATE
    NEP=NEPAdaptersCpuNativeNEP
    EwaldNep=NEPAdaptersCpuNativeEwaldNep
    find_neighbor_list_small_box=nep_adapters_cpu_find_neighbor_list_small_box
)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(nep_adapters_cpu PRIVATE NEP_ADAPTERS_BUILDING)
endif()

if(NEP_ADAPTERS_CPU_USE_RADIAL_TABLE)
  target_compile_definitions(nep_adapters_cpu PRIVATE USE_TABLE_FOR_RADIAL_FUNCTIONS)
endif()

if(NEP_ADAPTERS_CPU_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(nep_adapters_cpu PRIVATE OpenMP::OpenMP_CXX)
    set(NEP_ADAPTERS_PACKAGE_NEEDS_OPENMP ON PARENT_SCOPE)
  else()
    message(FATAL_ERROR "NEP_ADAPTERS_CPU_ENABLE_OPENMP=ON but OpenMP was not found.")
  endif()
endif()

if(APPLE)
  find_library(NEP_ADAPTERS_ACCELERATE_FRAMEWORK Accelerate)
  if(NOT NEP_ADAPTERS_ACCELERATE_FRAMEWORK)
    message(FATAL_ERROR "cpu requires Apple's Accelerate CBLAS provider.")
  endif()
  target_link_libraries(nep_adapters_cpu PRIVATE ${NEP_ADAPTERS_ACCELERATE_FRAMEWORK})
  target_compile_definitions(
    nep_adapters_cpu
    PRIVATE
      ACCELERATE_NEW_LAPACK=1
      NEP_ADAPTERS_CPU_USE_CBLAS=1
      NEP_ADAPTERS_CPU_USE_ACCELERATE=1)
else()
  include(CheckIncludeFileCXX)
  check_include_file_cxx(cblas.h NEP_ADAPTERS_CPU_HAS_CBLAS_H)
  find_package(BLAS QUIET)
  if(BLAS_FOUND AND NEP_ADAPTERS_CPU_HAS_CBLAS_H)
    target_link_libraries(nep_adapters_cpu PRIVATE BLAS::BLAS)
    set(NEP_ADAPTERS_PACKAGE_NEEDS_BLAS ON PARENT_SCOPE)
    target_compile_definitions(
      nep_adapters_cpu
      PRIVATE
        NEP_ADAPTERS_CPU_USE_CBLAS=1)
  else()
    message(STATUS
      "CBLAS is unavailable; building the portable scalar ANN path in cpu.")
  endif()
endif()

if(NEP_ADAPTERS_INSTALL_DEVELOPMENT_FILES)
  install(
    TARGETS nep_adapters_cpu
    EXPORT NEPAdaptersTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
