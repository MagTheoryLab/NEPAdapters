set(NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR "" CACHE PATH
  "Path to the existing NEP CPU source directory containing nep.cpp and nep.h")

if(NOT NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR)
  foreach(_candidate IN ITEMS
      "${PROJECT_SOURCE_DIR}/../NepTrainKit/src/nep_cpu"
      "${PROJECT_SOURCE_DIR}/../NEP_CPU/src"
      "${PROJECT_SOURCE_DIR}/../NEP_CPU")
    if(EXISTS "${_candidate}/nep.h" AND EXISTS "${_candidate}/nep.cpp")
      set(NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR "${_candidate}" CACHE PATH
        "Path to the existing NEP CPU source directory containing nep.cpp and nep.h" FORCE)
      break()
    endif()
  endforeach()
endif()

if(NOT EXISTS "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}/nep.h" OR
   NOT EXISTS "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}/nep.cpp" OR
   NOT EXISTS "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}/neighbor_nep.cpp" OR
   NOT EXISTS "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}/ewald_nep.cpp")
  message(FATAL_ERROR
    "NEP_ADAPTERS_ENABLE_CPU_NEP3=ON requires NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR "
    "to point to a NEP CPU source directory containing nep.cpp, neighbor_nep.cpp, "
    "ewald_nep.cpp, and nep.h.")
endif()

option(NEP_ADAPTERS_CPU_NEP3_ENABLE_OPENMP "Build cpu_nep3 with OpenMP when available" OFF)

add_library(nep_adapters_cpu_nep3
  cpu_nep3_engine.cpp
  "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}/nep.cpp"
  "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}/neighbor_nep.cpp"
  "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}/ewald_nep.cpp"
)
add_library(NEPAdapters::cpu_nep3 ALIAS nep_adapters_cpu_nep3)

target_link_libraries(nep_adapters_cpu_nep3 PUBLIC NEPAdapters::nep_adapters)

target_include_directories(
  nep_adapters_cpu_nep3
  PRIVATE
    "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}"
)

target_compile_features(nep_adapters_cpu_nep3 PUBLIC cxx_std_17)
set_target_properties(nep_adapters_cpu_nep3 PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(nep_adapters_cpu_nep3 PRIVATE NEP_ADAPTERS_BUILDING)
endif()

if(NEP_ADAPTERS_CPU_NEP3_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(nep_adapters_cpu_nep3 PRIVATE OpenMP::OpenMP_CXX)
  else()
    message(FATAL_ERROR "NEP_ADAPTERS_CPU_NEP3_ENABLE_OPENMP=ON but OpenMP was not found.")
  endif()
else()
  target_include_directories(nep_adapters_cpu_nep3 PRIVATE "${CMAKE_CURRENT_LIST_DIR}/compat")
endif()

install(
  TARGETS nep_adapters_cpu_nep3
  EXPORT NEPAdaptersTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
