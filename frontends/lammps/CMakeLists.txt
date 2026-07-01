set(NEP_ADAPTERS_LAMMPS_SOURCE_DIR "" CACHE PATH
  "Path to a LAMMPS source tree containing src/pair.h")

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

if(NOT EXISTS "${NEP_ADAPTERS_LAMMPS_SOURCE_DIR}/src/pair.h")
  message(FATAL_ERROR
    "NEP_ADAPTERS_ENABLE_LAMMPS=ON requires NEP_ADAPTERS_LAMMPS_SOURCE_DIR "
    "to point to a LAMMPS source tree containing src/pair.h.")
endif()

if(NOT TARGET NEPAdapters::cpu_nep3)
  message(FATAL_ERROR "NEP_ADAPTERS_ENABLE_LAMMPS=ON currently requires cpu_nep3.")
endif()

find_package(MPI QUIET COMPONENTS CXX)

add_library(nep_adapters_lammps_frontend STATIC
  pair_nep_adapters_cpu.cpp
)
add_library(NEPAdapters::lammps_frontend ALIAS nep_adapters_lammps_frontend)

target_link_libraries(
  nep_adapters_lammps_frontend
  PUBLIC
    NEPAdapters::cpu_nep3
)
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
    pair_nep_adapters_cpu.cpp
  )
  target_link_libraries(nepadaptersplugin PRIVATE NEPAdapters::cpu_nep3)
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
  endif()
endif()
