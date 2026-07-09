set(NEP_ADAPTERS_CPU_OPT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/native" CACHE PATH
  "Path to the optimized NEP CPU source directory containing nep.cpp and nep.h")

if(NOT EXISTS "${NEP_ADAPTERS_CPU_OPT_SOURCE_DIR}/nep.h" OR
   NOT EXISTS "${NEP_ADAPTERS_CPU_OPT_SOURCE_DIR}/nep.cpp" OR
   NOT EXISTS "${NEP_ADAPTERS_CPU_OPT_SOURCE_DIR}/neighbor_nep.cpp" OR
   NOT EXISTS "${NEP_ADAPTERS_CPU_OPT_SOURCE_DIR}/ewald_nep.cpp")
  message(FATAL_ERROR
    "NEP_ADAPTERS_ENABLE_CPU_OPT=ON requires NEP_ADAPTERS_CPU_OPT_SOURCE_DIR "
    "to point to a NEP CPU source directory containing nep.cpp, neighbor_nep.cpp, "
    "ewald_nep.cpp, and nep.h.")
endif()

option(NEP_ADAPTERS_CPU_OPT_ENABLE_OPENMP "Build cpu_opt with OpenMP when available" ON)
option(NEP_ADAPTERS_CPU_OPT_ENABLE_NATIVE_ARCH
  "Build cpu_opt for the host CPU architecture when supported"
  ON)
option(NEP_ADAPTERS_CPU_OPT_ENABLE_BATCH_ANN
  "Enable type-grouped CBLAS batch ANN kernels in cpu_opt when a CBLAS provider is available"
  ON)
set(NEP_ADAPTERS_CPU_OPT_USE_RADIAL_TABLE_DEFAULT ON)
option(NEP_ADAPTERS_CPU_OPT_USE_RADIAL_TABLE
  "Enable tabulated radial functions in cpu_opt"
  ${NEP_ADAPTERS_CPU_OPT_USE_RADIAL_TABLE_DEFAULT})

add_library(nep_adapters_cpu_opt
  cpu_opt_engine.cpp
  "${NEP_ADAPTERS_CPU_OPT_SOURCE_DIR}/nep.cpp"
  "${NEP_ADAPTERS_CPU_OPT_SOURCE_DIR}/neighbor_nep.cpp"
  "${NEP_ADAPTERS_CPU_OPT_SOURCE_DIR}/ewald_nep.cpp"
)
add_library(NEPAdapters::cpu_opt ALIAS nep_adapters_cpu_opt)

target_link_libraries(nep_adapters_cpu_opt PUBLIC NEPAdapters::nep_adapters)

target_include_directories(
  nep_adapters_cpu_opt
  PRIVATE
    "${NEP_ADAPTERS_CPU_OPT_SOURCE_DIR}"
)

target_compile_features(nep_adapters_cpu_opt PUBLIC cxx_std_17)
set_target_properties(nep_adapters_cpu_opt PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  target_compile_options(nep_adapters_cpu_opt PRIVATE -Mnovect)
endif()

include(CheckCXXCompilerFlag)
if(NEP_ADAPTERS_CPU_OPT_ENABLE_NATIVE_ARCH)
  check_cxx_compiler_flag("-march=native" NEP_ADAPTERS_CPU_OPT_HAS_MARCH_NATIVE)
  if(NEP_ADAPTERS_CPU_OPT_HAS_MARCH_NATIVE)
    target_compile_options(nep_adapters_cpu_opt PRIVATE -march=native)
  endif()
endif()

check_cxx_compiler_flag("-fno-math-errno" NEP_ADAPTERS_CPU_OPT_HAS_NO_MATH_ERRNO)
if(NEP_ADAPTERS_CPU_OPT_HAS_NO_MATH_ERRNO)
  target_compile_options(nep_adapters_cpu_opt PRIVATE -fno-math-errno)
endif()
check_cxx_compiler_flag("-ffast-math" NEP_ADAPTERS_CPU_OPT_HAS_FAST_MATH)
if(NEP_ADAPTERS_CPU_OPT_HAS_FAST_MATH)
  target_compile_options(nep_adapters_cpu_opt PRIVATE -ffast-math)
endif()

target_compile_definitions(
  nep_adapters_cpu_opt
  PRIVATE
    NEP=NEPAdaptersCpuOptNativeNEP
    EwaldNep=NEPAdaptersCpuOptNativeEwaldNep
    find_neighbor_list_small_box=nep_adapters_cpu_opt_find_neighbor_list_small_box
)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(nep_adapters_cpu_opt PRIVATE NEP_ADAPTERS_BUILDING)
endif()

if(NEP_ADAPTERS_CPU_OPT_USE_RADIAL_TABLE)
  target_compile_definitions(nep_adapters_cpu_opt PRIVATE USE_TABLE_FOR_RADIAL_FUNCTIONS)
endif()

if(NEP_ADAPTERS_CPU_OPT_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(nep_adapters_cpu_opt PRIVATE OpenMP::OpenMP_CXX)
  else()
    message(FATAL_ERROR "NEP_ADAPTERS_CPU_OPT_ENABLE_OPENMP=ON but OpenMP was not found.")
  endif()
endif()

if(NEP_ADAPTERS_CPU_OPT_ENABLE_BATCH_ANN AND APPLE)
  find_library(NEP_ADAPTERS_ACCELERATE_FRAMEWORK Accelerate)
  if(NEP_ADAPTERS_ACCELERATE_FRAMEWORK)
    target_link_libraries(nep_adapters_cpu_opt PRIVATE ${NEP_ADAPTERS_ACCELERATE_FRAMEWORK})
    target_compile_definitions(
      nep_adapters_cpu_opt
      PRIVATE
        ACCELERATE_NEW_LAPACK=1
        NEP_ADAPTERS_CPU_OPT_USE_CBLAS=1
        NEP_ADAPTERS_CPU_OPT_USE_ACCELERATE=1)
  else()
    message(STATUS "Accelerate not found; cpu_opt batch ANN will use scalar ANN fallback.")
  endif()
elseif(NEP_ADAPTERS_CPU_OPT_ENABLE_BATCH_ANN)
  include(CheckIncludeFileCXX)
  check_include_file_cxx(cblas.h NEP_ADAPTERS_CPU_OPT_HAS_CBLAS_H)
  find_package(BLAS)
  if(BLAS_FOUND AND NEP_ADAPTERS_CPU_OPT_HAS_CBLAS_H)
    target_link_libraries(nep_adapters_cpu_opt PRIVATE BLAS::BLAS)
    target_compile_definitions(
      nep_adapters_cpu_opt
      PRIVATE
        NEP_ADAPTERS_CPU_OPT_USE_CBLAS=1)
  else()
    message(STATUS "BLAS with cblas.h not found; cpu_opt batch ANN will use scalar ANN fallback.")
  endif()
endif()

install(
  TARGETS nep_adapters_cpu_opt
  EXPORT NEPAdaptersTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
