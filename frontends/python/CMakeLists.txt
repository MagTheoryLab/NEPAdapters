add_library(nep_adapters_python_frontend INTERFACE)
add_library(NEPAdapters::python_frontend ALIAS nep_adapters_python_frontend)

target_link_libraries(nep_adapters_python_frontend INTERFACE NEPAdapters::nep_adapters)

if(NOT TARGET NEPAdapters::cpu_nep3)
  message(FATAL_ERROR "NEP_ADAPTERS_ENABLE_PYTHON=ON currently requires cpu_nep3.")
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE NEP_ADAPTERS_PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE NEP_ADAPTERS_PYBIND11_CMAKE_DIR_STATUS
)
if(NOT NEP_ADAPTERS_PYBIND11_CMAKE_DIR_STATUS EQUAL 0)
  message(FATAL_ERROR
    "NEP_ADAPTERS_ENABLE_PYTHON=ON requires pybind11 in the selected Python environment.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${NEP_ADAPTERS_PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(nep_adapters_python_native MODULE
  src/python_module.cpp
)
target_link_libraries(
  nep_adapters_python_native
  PRIVATE
    NEPAdapters::cpu_nep3
)
set_target_properties(
  nep_adapters_python_native
  PROPERTIES
    OUTPUT_NAME "_native"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nep_adapters"
)
add_custom_command(
  TARGET nep_adapters_python_native
  POST_BUILD
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory
    "${CMAKE_CURRENT_LIST_DIR}/nep_adapters"
    "${CMAKE_CURRENT_BINARY_DIR}/nep_adapters"
)

install(
  TARGETS nep_adapters_python_native
  LIBRARY DESTINATION nep_adapters
)

if(NEP_ADAPTERS_BUILD_TESTS AND BUILD_SHARED_LIBS AND TARGET NEPAdapters::cpu_nep3)
  add_test(
    NAME nep_adapters_python_cpu_nep3_smoke_test
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_LIST_DIR}/tests/test_cpu_nep3_smoke.py"
  )
  set_tests_properties(
    nep_adapters_python_cpu_nep3_smoke_test
    PROPERTIES
      LABELS "smoke;frontend;python;cpu_nep3"
      ENVIRONMENT
        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_PYTHON_TEST_MODEL=${NEP_ADAPTERS_NEP89_MODEL_PATH};NEP_ADAPTERS_PYTHON_TEST_XYZ=${NEP_ADAPTERS_NEP89_XYZ_PATH}"
  )
endif()
