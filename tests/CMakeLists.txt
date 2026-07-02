add_executable(nep_adapters_registry_test test_registry.cpp)
target_link_libraries(nep_adapters_registry_test PRIVATE NEPAdapters::nep_adapters)

add_test(NAME nep_adapters_registry_test COMMAND nep_adapters_registry_test)
set_tests_properties(nep_adapters_registry_test PROPERTIES
  LABELS "contract;core")

add_executable(nep_adapters_domain_decomp_contract_test test_domain_decomp_contract.cpp)
target_link_libraries(
  nep_adapters_domain_decomp_contract_test
  PRIVATE
    NEPAdapters::nep_adapters
)

add_test(
  NAME nep_adapters_domain_decomp_contract_test
  COMMAND nep_adapters_domain_decomp_contract_test)
set_tests_properties(nep_adapters_domain_decomp_contract_test PROPERTIES
  LABELS "contract;domain_decomp;lammps")

add_executable(nep_adapters_virial_order_test test_virial_order.cpp)
target_link_libraries(
  nep_adapters_virial_order_test
  PRIVATE
    NEPAdapters::nep_adapters
)

add_test(
  NAME nep_adapters_virial_order_test
  COMMAND nep_adapters_virial_order_test)
set_tests_properties(nep_adapters_virial_order_test PROPERTIES
  LABELS "contract;virial")

if(TARGET NEPAdapters::cpu_nep3 AND
   EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt" AND
   EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz" AND
   EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/descriptor.txt")
  add_executable(nep_adapters_cpu_nep3_smoke_test test_cpu_nep3.cpp)
  target_link_libraries(
    nep_adapters_cpu_nep3_smoke_test
    PRIVATE
      NEPAdapters::cpu_nep3
  )
  target_compile_definitions(
    nep_adapters_cpu_nep3_smoke_test
    PRIVATE
      NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}"
  )
  add_test(NAME nep_adapters_cpu_nep3_smoke_test COMMAND nep_adapters_cpu_nep3_smoke_test)
  set_tests_properties(nep_adapters_cpu_nep3_smoke_test PROPERTIES
    LABELS "smoke;engine;cpu_nep3")

  add_executable(nep_adapters_cpu_nep3_parity_test test_cpu_nep3_parity.cpp)
  target_link_libraries(
    nep_adapters_cpu_nep3_parity_test
    PRIVATE
      NEPAdapters::cpu_nep3
  )
  target_include_directories(
    nep_adapters_cpu_nep3_parity_test
    PRIVATE
      "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}"
  )
  target_compile_definitions(
    nep_adapters_cpu_nep3_parity_test
    PRIVATE
      NEP_ADAPTERS_PARITY_MODEL_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt"
      NEP_ADAPTERS_PARITY_XYZ_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz"
  )
  add_test(NAME nep_adapters_cpu_nep3_parity_test COMMAND nep_adapters_cpu_nep3_parity_test)
  set_tests_properties(nep_adapters_cpu_nep3_parity_test PROPERTIES
    LABELS "parity;engine;cpu_nep3")

  add_executable(nep_adapters_cpu_nep3_lammps_neighbors_test
    test_cpu_nep3_lammps_neighbors.cpp)
  target_link_libraries(
    nep_adapters_cpu_nep3_lammps_neighbors_test
    PRIVATE
      NEPAdapters::cpu_nep3
  )
  target_include_directories(
    nep_adapters_cpu_nep3_lammps_neighbors_test
    PRIVATE
      "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}"
  )
  target_compile_definitions(
    nep_adapters_cpu_nep3_lammps_neighbors_test
    PRIVATE
      NEP_ADAPTERS_PARITY_MODEL_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt"
      NEP_ADAPTERS_PARITY_XYZ_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz"
  )
  add_test(
    NAME nep_adapters_cpu_nep3_lammps_neighbors_test
    COMMAND nep_adapters_cpu_nep3_lammps_neighbors_test)
  set_tests_properties(nep_adapters_cpu_nep3_lammps_neighbors_test PROPERTIES
    LABELS "parity;engine;cpu_nep3;lammps")
endif()

if(TARGET NEPAdapters::cpu_nep3 AND
   TARGET NEPAdapters::cpu_opt AND
   EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt" AND
   EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz" AND
   EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/descriptor.txt")
  add_executable(nep_adapters_cpu_opt_parity_test test_cpu_opt_parity.cpp)
  target_link_libraries(
    nep_adapters_cpu_opt_parity_test
    PRIVATE
      NEPAdapters::cpu_nep3
      NEPAdapters::cpu_opt
  )
  target_include_directories(
    nep_adapters_cpu_opt_parity_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/tests"
  )
  target_compile_definitions(
    nep_adapters_cpu_opt_parity_test
    PRIVATE
      NEP_ADAPTERS_PARITY_MODEL_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt"
      NEP_ADAPTERS_PARITY_XYZ_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz"
      NEP_ADAPTERS_EXPECTED_DESCRIPTOR_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/descriptor.txt"
  )
  add_test(NAME nep_adapters_cpu_opt_parity_test COMMAND nep_adapters_cpu_opt_parity_test)
  set_tests_properties(nep_adapters_cpu_opt_parity_test PROPERTIES
    LABELS "parity;engine;cpu_opt;cpu_nep3")

  add_executable(nep_adapters_cpu_opt_lammps_neighbors_test
    test_cpu_nep3_lammps_neighbors.cpp)
  target_link_libraries(
    nep_adapters_cpu_opt_lammps_neighbors_test
    PRIVATE
      NEPAdapters::cpu_nep3
      NEPAdapters::cpu_opt
  )
  target_include_directories(
    nep_adapters_cpu_opt_lammps_neighbors_test
    PRIVATE
      "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}"
  )
  target_compile_definitions(
    nep_adapters_cpu_opt_lammps_neighbors_test
    PRIVATE
      NEP_ADAPTERS_LAMMPS_ENABLE_CPU_OPT=1
      NEP_ADAPTERS_LAMMPS_ENGINE_NAME="cpu_opt"
      NEP_ADAPTERS_PARITY_MODEL_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt"
      NEP_ADAPTERS_PARITY_XYZ_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz"
  )
  add_test(
    NAME nep_adapters_cpu_opt_lammps_neighbors_test
    COMMAND nep_adapters_cpu_opt_lammps_neighbors_test)
  set_tests_properties(nep_adapters_cpu_opt_lammps_neighbors_test PROPERTIES
    LABELS "parity;engine;cpu_opt;cpu_nep3;lammps")
endif()

if(TARGET NEPAdapters::cpu_nep3 AND
   EXISTS "${NEP_ADAPTERS_NEP89_MODEL_PATH}" AND
   EXISTS "${NEP_ADAPTERS_NEP89_XYZ_PATH}")
  add_executable(nep_adapters_cpu_nep89_smoke_test test_cpu_nep89.cpp)
  target_link_libraries(
    nep_adapters_cpu_nep89_smoke_test
    PRIVATE
      NEPAdapters::cpu_nep3
  )
  target_compile_definitions(
    nep_adapters_cpu_nep89_smoke_test
    PRIVATE
      NEP_ADAPTERS_NEP89_MODEL_PATH="${NEP_ADAPTERS_NEP89_MODEL_PATH}"
      NEP_ADAPTERS_NEP89_XYZ_PATH="${NEP_ADAPTERS_NEP89_XYZ_PATH}"
  )
  add_test(NAME nep_adapters_cpu_nep89_smoke_test COMMAND nep_adapters_cpu_nep89_smoke_test)
  set_tests_properties(nep_adapters_cpu_nep89_smoke_test PROPERTIES
    LABELS "smoke;engine;cpu_nep3;large_model;nep89")

  add_executable(nep_adapters_cpu_nep89_parity_test test_cpu_nep3_parity.cpp)
  target_link_libraries(
    nep_adapters_cpu_nep89_parity_test
    PRIVATE
      NEPAdapters::cpu_nep3
  )
  target_include_directories(
    nep_adapters_cpu_nep89_parity_test
    PRIVATE
      "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}"
  )
  target_compile_definitions(
    nep_adapters_cpu_nep89_parity_test
    PRIVATE
      NEP_ADAPTERS_PARITY_MODEL_PATH="${NEP_ADAPTERS_NEP89_MODEL_PATH}"
      NEP_ADAPTERS_PARITY_XYZ_PATH="${NEP_ADAPTERS_NEP89_XYZ_PATH}"
  )
  add_test(NAME nep_adapters_cpu_nep89_parity_test COMMAND nep_adapters_cpu_nep89_parity_test)
  set_tests_properties(nep_adapters_cpu_nep89_parity_test PROPERTIES
    LABELS "parity;engine;cpu_nep3;large_model;nep89")
endif()

if(TARGET NEPAdapters::cpu_nep3 AND
   TARGET NEPAdapters::cpu_opt AND
   EXISTS "${NEP_ADAPTERS_NEP89_MODEL_PATH}" AND
   EXISTS "${NEP_ADAPTERS_NEP89_XYZ_PATH}")
  add_executable(nep_adapters_cpu_opt_nep89_parity_test test_cpu_opt_parity.cpp)
  target_link_libraries(
    nep_adapters_cpu_opt_nep89_parity_test
    PRIVATE
      NEPAdapters::cpu_nep3
      NEPAdapters::cpu_opt
  )
  target_include_directories(
    nep_adapters_cpu_opt_nep89_parity_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/tests"
  )
  target_compile_definitions(
    nep_adapters_cpu_opt_nep89_parity_test
    PRIVATE
      NEP_ADAPTERS_PARITY_MODEL_PATH="${NEP_ADAPTERS_NEP89_MODEL_PATH}"
      NEP_ADAPTERS_PARITY_XYZ_PATH="${NEP_ADAPTERS_NEP89_XYZ_PATH}"
  )
  add_test(NAME nep_adapters_cpu_opt_nep89_parity_test COMMAND nep_adapters_cpu_opt_nep89_parity_test)
  set_tests_properties(nep_adapters_cpu_opt_nep89_parity_test PROPERTIES
    LABELS "parity;engine;cpu_opt;cpu_nep3;large_model;nep89")
endif()
