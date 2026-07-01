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
add_custom_target(
  nep_adapters_python_package ALL
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory
    "${CMAKE_CURRENT_LIST_DIR}/nep_adapters"
    "${CMAKE_CURRENT_BINARY_DIR}/nep_adapters"
)
add_dependencies(nep_adapters_python_package nep_adapters_python_native)

install(
  TARGETS nep_adapters_python_native
  LIBRARY DESTINATION nep_adapters
)

if(NEP_ADAPTERS_BUILD_TESTS AND BUILD_SHARED_LIBS AND TARGET NEPAdapters::cpu_nep3
   AND EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt"
   AND EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz")
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
        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_PYTHON_TEST_MODEL=${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt;NEP_ADAPTERS_PYTHON_TEST_XYZ=${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz"
  )

  add_test(
    NAME nep_adapters_python_calculator_test
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_LIST_DIR}/tests/test_calculator.py"
  )
  set_tests_properties(
    nep_adapters_python_calculator_test
    PROPERTIES
      LABELS "smoke;frontend;python;calculator;cpu_nep3"
      ENVIRONMENT
        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_PYTHON_TEST_MODEL=${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt;NEP_ADAPTERS_PYTHON_TEST_XYZ=${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz"
  )

  add_test(
    NAME nep_adapters_python_ase_calculator_test
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_LIST_DIR}/tests/test_ase_calculator.py"
  )
  set_tests_properties(
    nep_adapters_python_ase_calculator_test
    PROPERTIES
      LABELS "smoke;frontend;python;ase;calculator;cpu_nep3"
      ENVIRONMENT
        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_PYTHON_TEST_MODEL=${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt;NEP_ADAPTERS_PYTHON_TEST_XYZ=${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz"
  )
endif()
