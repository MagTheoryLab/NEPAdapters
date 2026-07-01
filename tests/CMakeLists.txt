add_executable(nep_adapters_registry_test test_registry.cpp)
target_link_libraries(nep_adapters_registry_test PRIVATE NEPAdapters::nep_adapters)

add_test(NAME nep_adapters_registry_test COMMAND nep_adapters_registry_test)
set_tests_properties(nep_adapters_registry_test PROPERTIES
  LABELS "contract;core")

if(TARGET NEPAdapters::cpu_nep3 AND
   EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt" AND
   EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/train.xyz")
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
