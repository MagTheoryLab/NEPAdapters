add_library(nep_adapters_python_frontend INTERFACE)
add_library(NEPAdapters::python_frontend ALIAS nep_adapters_python_frontend)

target_link_libraries(nep_adapters_python_frontend INTERFACE NEPAdapters::nep_adapters)

if(NOT TARGET NEPAdapters::cpu)
  message(FATAL_ERROR "NEP_ADAPTERS_ENABLE_PYTHON=ON requires cpu.")
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

pybind11_add_module(nep_adapters_python_cpu MODULE
  src/python_module.cpp
)
target_link_libraries(
  nep_adapters_python_cpu
  PRIVATE
    NEPAdapters::cpu
)
set_target_properties(
  nep_adapters_python_cpu
  PROPERTIES
    OUTPUT_NAME "nep_cpu"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nep_adapters"
)

if(TARGET NEPAdapters::cuda)
  pybind11_add_module(nep_adapters_python_gpu MODULE
    src/python_module.cpp
  )
  target_link_libraries(
    nep_adapters_python_gpu
    PRIVATE
      NEPAdapters::cuda
  )
  target_compile_definitions(
    nep_adapters_python_gpu
    PRIVATE
      NEP_ADAPTERS_PYTHON_GPU_MODULE=1
  )
  set_target_properties(
    nep_adapters_python_gpu
    PROPERTIES
      OUTPUT_NAME "nep_gpu"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nep_adapters"
  )
endif()

add_custom_target(
  nep_adapters_python_package ALL
  COMMAND
    "${CMAKE_COMMAND}" -E copy_directory
    "${CMAKE_CURRENT_LIST_DIR}/nep_adapters"
    "${CMAKE_CURRENT_BINARY_DIR}/nep_adapters"
)
add_dependencies(nep_adapters_python_package nep_adapters_python_cpu)
if(TARGET nep_adapters_python_gpu)
  add_dependencies(nep_adapters_python_package nep_adapters_python_gpu)
endif()

install(
  TARGETS nep_adapters_python_cpu
  LIBRARY DESTINATION nep_adapters
  RUNTIME DESTINATION nep_adapters
)
if(TARGET nep_adapters_python_gpu)
  install(
    TARGETS nep_adapters_python_gpu
    LIBRARY DESTINATION nep_adapters
    RUNTIME DESTINATION nep_adapters
  )
endif()

if(NEP_ADAPTERS_BUILD_TESTS AND BUILD_SHARED_LIBS AND TARGET NEPAdapters::cpu
   AND EXISTS "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt"
   AND EXISTS "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/train.xyz"
   AND EXISTS "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/descriptor.txt")
  add_test(
    NAME nep_adapters_python_cpu_smoke_test
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_LIST_DIR}/tests/test_cpu_smoke.py"
  )
  set_tests_properties(
    nep_adapters_python_cpu_smoke_test
    PROPERTIES
      LABELS "smoke;frontend;python;cpu"
      ENVIRONMENT
        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_PYTHON_TEST_MODEL=${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt;NEP_ADAPTERS_PYTHON_TEST_XYZ=${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/train.xyz"
  )

  add_test(
    NAME nep_adapters_python_backend_isolation_test
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_LIST_DIR}/tests/test_backend_isolation.py"
  )
  set_tests_properties(
    nep_adapters_python_backend_isolation_test
    PROPERTIES
      LABELS "contract;frontend;python;cpu;isolation"
      ENVIRONMENT
        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_PYTHON_TEST_MODEL=${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt"
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
      LABELS "smoke;frontend;python;calculator;cpu"
      ENVIRONMENT
        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_PYTHON_TEST_MODEL=${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt;NEP_ADAPTERS_PYTHON_TEST_XYZ=${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/train.xyz"
  )

  add_test(
    NAME nep_adapters_python_production_api_test
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_LIST_DIR}/tests/test_production_api.py"
  )
  set_tests_properties(
    nep_adapters_python_production_api_test
    PROPERTIES
      LABELS "production;contract;oracle;frontend;python;cpu;qnep;dftd3;dipole;polarizability;cancellation"
      ENVIRONMENT
        "OMP_NUM_THREADS=4;PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_PYTHON_TEST_MODEL=${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt;NEP_ADAPTERS_QNEP_TEST_DATA_DIR=${NEP_ADAPTERS_QNEP_TEST_DATA_DIR};NEP_ADAPTERS_PRODUCTION_FIXTURE_DIR=${PROJECT_SOURCE_DIR}/tests/fixtures/production_api"
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
      LABELS "smoke;frontend;python;ase;calculator;cpu"
      ENVIRONMENT
        "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_PYTHON_TEST_MODEL=${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt;NEP_ADAPTERS_PYTHON_TEST_XYZ=${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/train.xyz"
  )

  if(EXISTS "${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/nep.txt" AND
     EXISTS "${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/reference_small_pbc.txt")
    add_test(
      NAME nep_adapters_python_spin_calculator_test
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_spin_calculator.py"
    )
    set_tests_properties(
      nep_adapters_python_spin_calculator_test
      PROPERTIES
        LABELS "parity;frontend;python;calculator;cpu;spin"
        ENVIRONMENT
          "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_PYTHON_SPIN_MODEL=${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/nep.txt;NEP_ADAPTERS_PYTHON_SPIN_REFERENCE=${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/reference_small_pbc.txt"
    )

    if(TARGET NEPAdapters::cuda)
      add_test(
        NAME nep_adapters_python_cuda_smoke_test
        COMMAND
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_LIST_DIR}/tests/test_cuda_smoke.py"
      )
      set_tests_properties(
        nep_adapters_python_cuda_smoke_test
        PROPERTIES
          LABELS "parity;frontend;python;cuda;spin"
          ENVIRONMENT
            "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_PYTHON_TEST_MODEL=${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt;NEP_ADAPTERS_PYTHON_TEST_XYZ=${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/train.xyz;NEP_ADAPTERS_PYTHON_SPIN_MODEL=${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/nep.txt"
      )
    endif()
  endif()

  if(TARGET NEPAdapters::cuda
     AND EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/nep.txt"
     AND EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/xyz.in"
     AND EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/force_analytical_ref.out"
     AND EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/virial_ref.out")
    add_test(
      NAME nep_adapters_python_cuda_qnep_test
      COMMAND
        "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/tests/test_cuda_qnep.py"
    )
    set_tests_properties(
      nep_adapters_python_cuda_qnep_test
      PROPERTIES
        LABELS "parity;frontend;python;cuda;qnep"
        ENVIRONMENT
          "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};NEP_ADAPTERS_QNEP_TEST_DATA_DIR=${NEP_ADAPTERS_QNEP_TEST_DATA_DIR};NEP_ADAPTERS_PRODUCTION_FIXTURE_DIR=${PROJECT_SOURCE_DIR}/tests/fixtures/production_api"
    )
  endif()
endif()
