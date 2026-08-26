add_executable(nep_adapters_registry_test test_registry.cpp)
target_link_libraries(nep_adapters_registry_test PRIVATE NEPAdapters::nep_adapters)

add_test(NAME nep_adapters_registry_test COMMAND nep_adapters_registry_test)
set_tests_properties(nep_adapters_registry_test PROPERTIES
  LABELS "contract;core")

if(TARGET NEPAdapters::cpu)
  add_executable(
    nep_adapters_cpu_spin2_shape_matrix_test
    test_cpu_spin2_shape_matrix.cpp)
  target_include_directories(
    nep_adapters_cpu_spin2_shape_matrix_test
    PRIVATE "${PROJECT_SOURCE_DIR}/engines/cpu/native")
  target_compile_features(
    nep_adapters_cpu_spin2_shape_matrix_test
    PRIVATE cxx_std_17)
  add_test(
    NAME nep_adapters_cpu_spin2_shape_matrix_test
    COMMAND nep_adapters_cpu_spin2_shape_matrix_test)
  set_tests_properties(
    nep_adapters_cpu_spin2_shape_matrix_test
    PROPERTIES LABELS "oracle;finite_difference;engine;cpu;spin2;shape_matrix")

  add_executable(
    nep_adapters_cpu_spin2_oc_oracle_test
    test_cuda_spin2_oc_oracle.cpp)
  target_compile_definitions(
    nep_adapters_cpu_spin2_oc_oracle_test
    PRIVATE NEP_ADAPTERS_TEST_CPU_ENGINE=1)
  target_link_libraries(
    nep_adapters_cpu_spin2_oc_oracle_test
    PRIVATE NEPAdapters::cpu)
  foreach(_case IN ITEMS
      o1c1_l0_soc0
      o2c2
      o3c2
      o3c2_soc0
      o3c3
      o2c9_l1_soc1
      zbl_o3c2)
    add_test(
      NAME nep_adapters_cpu_spin2_${_case}_oracle_test
      COMMAND
        nep_adapters_cpu_spin2_oc_oracle_test
        "${PROJECT_SOURCE_DIR}/tests/fixtures/nep4_spin2_${_case}.nep"
        "${PROJECT_SOURCE_DIR}/tests/fixtures/spin2_${_case}_oracle.txt")
    set_tests_properties(
      nep_adapters_cpu_spin2_${_case}_oracle_test
      PROPERTIES LABELS "oracle;parity;engine;cpu;spin2")
  endforeach()
endif()

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

add_executable(
  nep_adapters_kokkos_view_stride_contract_test
  test_kokkos_view_strides.cpp)
target_include_directories(
  nep_adapters_kokkos_view_stride_contract_test
  PRIVATE
    "${PROJECT_SOURCE_DIR}/frontends/lammps")
target_compile_features(
  nep_adapters_kokkos_view_stride_contract_test
  PRIVATE
    cxx_std_17)
add_test(
  NAME nep_adapters_kokkos_view_stride_contract_test
  COMMAND nep_adapters_kokkos_view_stride_contract_test)
set_tests_properties(
  nep_adapters_kokkos_view_stride_contract_test PROPERTIES
  LABELS "contract;frontend;lammps;kokkos;stride")

set(NEP_ADAPTERS_INSTALL_CONSUMER_TARGETS NEPAdapters::nep_adapters)
foreach(_engine IN ITEMS cpu cuda)
  if(TARGET NEPAdapters::${_engine})
    list(APPEND NEP_ADAPTERS_INSTALL_CONSUMER_TARGETS NEPAdapters::${_engine})
  endif()
endforeach()
string(REPLACE ";" " " NEP_ADAPTERS_INSTALL_CONSUMER_LINK_TARGETS
  "${NEP_ADAPTERS_INSTALL_CONSUMER_TARGETS}")
set(NEP_ADAPTERS_INSTALL_CONSUMER_DEPENDENCY_ARGS)
if(NEP_ADAPTERS_PACKAGE_NEEDS_OPENMP)
  list(APPEND NEP_ADAPTERS_INSTALL_CONSUMER_DEPENDENCY_ARGS
    "-DOpenMP_CXX_FLAGS=${OpenMP_CXX_FLAGS}"
    "-DOpenMP_CXX_LIB_NAMES=${OpenMP_CXX_LIB_NAMES}")
  foreach(_openmp_library IN LISTS OpenMP_CXX_LIB_NAMES)
    list(APPEND NEP_ADAPTERS_INSTALL_CONSUMER_DEPENDENCY_ARGS
      "-DOpenMP_${_openmp_library}_LIBRARY=${OpenMP_${_openmp_library}_LIBRARY}")
  endforeach()
endif()
configure_file(
  test_install_consumer.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/test_install_consumer.cmake"
  @ONLY)
add_test(
  NAME nep_adapters_install_consumer_test
  COMMAND "${CMAKE_COMMAND}" -P
          "${CMAKE_CURRENT_BINARY_DIR}/test_install_consumer.cmake")
set_tests_properties(nep_adapters_install_consumer_test PROPERTIES
  LABELS "install;package;consumer")

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(
    NAME nep_adapters_profile_tool_unit_test
    COMMAND "${Python3_EXECUTABLE}" -m unittest -v
            tests.test_run_4090_spin_lmp_profile)
  set_tests_properties(nep_adapters_profile_tool_unit_test PROPERTIES
    LABELS "tooling;python;unit"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

  add_test(
    NAME nep_adapters_lammps_baseline_tool_unit_test
    COMMAND "${Python3_EXECUTABLE}" -m unittest -v
            tests.test_run_lammps_baseline_smoke)
  set_tests_properties(nep_adapters_lammps_baseline_tool_unit_test PROPERTIES
    LABELS "tooling;python;unit;lammps"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

  add_test(
    NAME nep_adapters_lammps_nep89_tool_unit_test
    COMMAND "${Python3_EXECUTABLE}" -m unittest -v
            tests.test_run_lammps_nep89_smoke)
  set_tests_properties(nep_adapters_lammps_nep89_tool_unit_test PROPERTIES
    LABELS "tooling;python;unit;lammps;nonspin;nep89"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

  add_test(
    NAME nep_adapters_lammps_spin_tool_unit_test
    COMMAND "${Python3_EXECUTABLE}" -m unittest -v
            tests.test_run_lammps_spin_smoke)
  set_tests_properties(nep_adapters_lammps_spin_tool_unit_test PROPERTIES
    LABELS "tooling;python;unit;lammps;spin"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

  add_test(
    NAME nep_adapters_lammps_source_install_unit_test
    COMMAND "${Python3_EXECUTABLE}" -m unittest -v
            tests.test_install_lammps_source)
  set_tests_properties(nep_adapters_lammps_source_install_unit_test PROPERTIES
    LABELS "tooling;python;unit;lammps;install;source_tree"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

if(TARGET NEPAdapters::cpu AND
   EXISTS "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt" AND
   EXISTS "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/train.xyz" AND
   EXISTS "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/descriptor.txt")
  add_executable(nep_adapters_cpu_smoke_test test_cpu.cpp)
  target_link_libraries(
    nep_adapters_cpu_smoke_test
    PRIVATE
      NEPAdapters::cpu
  )
  target_compile_definitions(
    nep_adapters_cpu_smoke_test
    PRIVATE
      NEP_ADAPTERS_CPU_TEST_DATA_DIR="${NEP_ADAPTERS_CPU_TEST_DATA_DIR}"
  )
  add_test(NAME nep_adapters_cpu_smoke_test COMMAND nep_adapters_cpu_smoke_test)
  set_tests_properties(nep_adapters_cpu_smoke_test PROPERTIES
    LABELS "smoke;engine;cpu")

  add_executable(
    nep_adapters_cpu_input_validation_test
    test_cpu_input_validation.cpp)
  target_link_libraries(
    nep_adapters_cpu_input_validation_test
    PRIVATE
      NEPAdapters::cpu)
  target_include_directories(
    nep_adapters_cpu_input_validation_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/engines/cpu")
  target_compile_definitions(
    nep_adapters_cpu_input_validation_test
    PRIVATE
      NEP_ADAPTERS_CPU_TEST_DATA_DIR="${NEP_ADAPTERS_CPU_TEST_DATA_DIR}")
  add_test(
    NAME nep_adapters_cpu_input_validation_test
    COMMAND nep_adapters_cpu_input_validation_test)
  set_tests_properties(
    nep_adapters_cpu_input_validation_test PROPERTIES
    LABELS "contract;engine;cpu;validation")

  add_executable(nep_adapters_cpu_lammps_neighbors_test
    test_cpu_lammps_neighbors.cpp)
  target_link_libraries(
    nep_adapters_cpu_lammps_neighbors_test
    PRIVATE
      NEPAdapters::cpu
  )
  target_compile_definitions(
    nep_adapters_cpu_lammps_neighbors_test
    PRIVATE
      NEP_ADAPTERS_PARITY_MODEL_PATH="${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt"
      NEP_ADAPTERS_PARITY_XYZ_PATH="${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/train.xyz"
  )
  add_test(
    NAME nep_adapters_cpu_lammps_neighbors_test
    COMMAND nep_adapters_cpu_lammps_neighbors_test)
  set_tests_properties(nep_adapters_cpu_lammps_neighbors_test PROPERTIES
    LABELS "baseline;engine;cpu;lammps")
endif()

if(TARGET NEPAdapters::cpu AND
   EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/nep.txt" AND
   EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/xyz.in")
  add_executable(nep_adapters_cpu_qnep_test test_cpu_qnep.cpp)
  target_link_libraries(
    nep_adapters_cpu_qnep_test
    PRIVATE
      NEPAdapters::cpu
  )
  target_compile_definitions(
    nep_adapters_cpu_qnep_test
    PRIVATE
      NEP_ADAPTERS_QNEP_TEST_DATA_DIR="${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}"
  )
  add_test(NAME nep_adapters_cpu_qnep_test COMMAND nep_adapters_cpu_qnep_test)
  set_tests_properties(nep_adapters_cpu_qnep_test PROPERTIES
    LABELS "smoke;engine;cpu;qnep")
endif()

if(TARGET NEPAdapters::cpu)
  add_executable(nep_adapters_cpu_production_api_test
    test_cpu_production_api.cpp)
  target_link_libraries(
    nep_adapters_cpu_production_api_test
    PRIVATE
      NEPAdapters::cpu)
  target_compile_definitions(
    nep_adapters_cpu_production_api_test
    PRIVATE
      NEP_ADAPTERS_PRODUCTION_FIXTURE_DIR="${PROJECT_SOURCE_DIR}/tests/fixtures/production_api")
  add_test(
    NAME nep_adapters_cpu_production_api_test
    COMMAND nep_adapters_cpu_production_api_test)
  set_tests_properties(
    nep_adapters_cpu_production_api_test
    PROPERTIES LABELS "production;contract;oracle;engine;cpu;dftd3;dipole;polarizability;cancellation")

  set(_nep_cpu_reference_target_added OFF)
  foreach(_reference_dir IN ITEMS
      "${NEP_ADAPTERS_NEP_CPU_NEP_TEST_DATA_DIR}"
      "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}")
    if(EXISTS "${_reference_dir}/nep.txt" AND
       EXISTS "${_reference_dir}/xyz.in" AND
       EXISTS "${_reference_dir}/force_analytical_ref.out" AND
       EXISTS "${_reference_dir}/virial_ref.out" AND
       EXISTS "${_reference_dir}/descriptor_ref.out")
      set(_nep_cpu_reference_target_added ON)
    endif()
  endforeach()

  if(_nep_cpu_reference_target_added)
    add_executable(nep_adapters_cpu_nep_cpu_reference_test
      test_nep_cpu_reference.cpp)
    target_link_libraries(
      nep_adapters_cpu_nep_cpu_reference_test
      PRIVATE
        NEPAdapters::cpu)
  endif()

  if(TARGET nep_adapters_cpu_nep_cpu_reference_test AND
     EXISTS "${NEP_ADAPTERS_NEP_CPU_NEP_TEST_DATA_DIR}/descriptor_ref.out")
    add_test(
      NAME nep_adapters_cpu_nep3_unsupported_test
      COMMAND nep_adapters_cpu_nep_cpu_reference_test
              "${NEP_ADAPTERS_NEP_CPU_NEP_TEST_DATA_DIR}"
              --expect-unsupported)
    set_tests_properties(
      nep_adapters_cpu_nep3_unsupported_test
      PROPERTIES LABELS "contract;engine;cpu;nep3;unsupported")
  endif()

  if(TARGET nep_adapters_cpu_nep_cpu_reference_test AND
     EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/descriptor_ref.out")
    add_test(
      NAME nep_adapters_cpu_nep_cpu_qnep_reference_test
      COMMAND nep_adapters_cpu_nep_cpu_reference_test
              "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}")
    set_tests_properties(
      nep_adapters_cpu_nep_cpu_qnep_reference_test
      PROPERTIES LABELS "parity;oracle;engine;cpu;nep_cpu;qnep")
  endif()
endif()

if(TARGET NEPAdapters::cuda AND
   EXISTS "${NEP_ADAPTERS_NEP_CPU_NEP_TEST_DATA_DIR}/descriptor_ref.out" AND
   EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/descriptor_ref.out")
  add_executable(nep_adapters_cuda_nep_cpu_reference_test
    test_nep_cpu_reference.cpp)
  target_link_libraries(
    nep_adapters_cuda_nep_cpu_reference_test
    PRIVATE
      NEPAdapters::cuda)
  target_compile_definitions(
    nep_adapters_cuda_nep_cpu_reference_test
    PRIVATE
      NEP_ADAPTERS_REFERENCE_CUDA=1)

  add_test(
    NAME nep_adapters_cuda_nep_cpu_nep3_unsupported_test
    COMMAND nep_adapters_cuda_nep_cpu_reference_test
            "${NEP_ADAPTERS_NEP_CPU_NEP_TEST_DATA_DIR}"
            --expect-unsupported)
  set_tests_properties(
    nep_adapters_cuda_nep_cpu_nep3_unsupported_test
    PROPERTIES LABELS "contract;engine;cuda;device;nep_cpu;nep3;unsupported")

  add_test(
    NAME nep_adapters_cuda_nep_cpu_qnep_reference_test
    COMMAND nep_adapters_cuda_nep_cpu_reference_test
            "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}")
  set_tests_properties(
    nep_adapters_cuda_nep_cpu_qnep_reference_test
    PROPERTIES LABELS "parity;oracle;engine;cuda;device;nep_cpu;qnep")
endif()

if(TARGET NEPAdapters::cuda AND
   EXISTS "${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt")
  add_executable(nep_adapters_cuda_contract_test test_cuda_engine_contract.cpp)
  target_link_libraries(
    nep_adapters_cuda_contract_test
    PRIVATE
      NEPAdapters::cuda
  )
  target_include_directories(
    nep_adapters_cuda_contract_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/engines/cuda"
  )
  target_compile_definitions(
    nep_adapters_cuda_contract_test
    PRIVATE
      NEP_ADAPTERS_CUDA_TEST_MODEL_PATH="${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt"
  )
  add_test(NAME nep_adapters_cuda_contract_test COMMAND nep_adapters_cuda_contract_test)
  set_tests_properties(nep_adapters_cuda_contract_test PROPERTIES
    LABELS "contract;engine;cuda")

  add_executable(nep_adapters_cuda_batch_interface_test test_cuda_batch_interface.cpp)
  target_link_libraries(
    nep_adapters_cuda_batch_interface_test
    PRIVATE
      NEPAdapters::cuda
  )
  target_compile_definitions(
    nep_adapters_cuda_batch_interface_test
    PRIVATE
      NEP_ADAPTERS_CUDA_TEST_MODEL_PATH="${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt"
  )
  add_test(
    NAME nep_adapters_cuda_batch_interface_test
    COMMAND nep_adapters_cuda_batch_interface_test)
  set_tests_properties(nep_adapters_cuda_batch_interface_test PROPERTIES
    LABELS "contract;engine;cuda;batch")

  add_executable(nep_adapters_cuda_lammps_host_sim_test test_cuda_lammps_host_sim.cpp)
  target_link_libraries(
    nep_adapters_cuda_lammps_host_sim_test
    PRIVATE
      NEPAdapters::cuda
  )
  target_compile_definitions(
    nep_adapters_cuda_lammps_host_sim_test
    PRIVATE
      NEP_ADAPTERS_CUDA_TEST_MODEL_PATH="${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt"
  )
  add_test(
    NAME nep_adapters_cuda_lammps_host_sim_test
    COMMAND nep_adapters_cuda_lammps_host_sim_test)
  set_tests_properties(nep_adapters_cuda_lammps_host_sim_test PROPERTIES
    LABELS "contract;engine;cuda;lammps")

  add_executable(nep_adapters_cuda_spin_protocol_test test_cuda_spin_protocol.cpp)
  target_link_libraries(
    nep_adapters_cuda_spin_protocol_test
    PRIVATE
      NEPAdapters::cuda
  )
  target_compile_definitions(
    nep_adapters_cuda_spin_protocol_test
    PRIVATE
      NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE="${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/nep.txt"
  )
  add_test(
    NAME nep_adapters_cuda_spin_protocol_test
    COMMAND nep_adapters_cuda_spin_protocol_test)
  set_tests_properties(nep_adapters_cuda_spin_protocol_test PROPERTIES
    LABELS "contract;engine;cuda;spin")

  add_executable(
    nep_adapters_cuda_spin2_protocol_test
    test_cuda_spin2_protocol.cpp)
  target_include_directories(
    nep_adapters_cuda_spin2_protocol_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/engines/cuda")
  target_link_libraries(
    nep_adapters_cuda_spin2_protocol_test
    PRIVATE
      NEPAdapters::cuda
      NEPAdapters::cpu)
  add_test(
    NAME nep_adapters_cuda_spin2_protocol_test
    COMMAND nep_adapters_cuda_spin2_protocol_test)
  set_tests_properties(
    nep_adapters_cuda_spin2_protocol_test
    PROPERTIES LABELS "contract;protocol;engine;cpu;cuda;spin2")

  if(EXISTS "${NEP_ADAPTERS_SPIN2_MODEL_PATH}")
    add_test(
      NAME nep_adapters_cuda_spin2_external_model_test
      COMMAND
        nep_adapters_cuda_spin2_protocol_test
        "${NEP_ADAPTERS_SPIN2_MODEL_PATH}")
    set_tests_properties(
      nep_adapters_cuda_spin2_external_model_test
      PROPERTIES LABELS "contract;protocol;engine;cuda;spin2")
  endif()

  add_executable(
    nep_adapters_cuda_spin2_oc_oracle_test
    test_cuda_spin2_oc_oracle.cpp)
  target_link_libraries(
    nep_adapters_cuda_spin2_oc_oracle_test
    PRIVATE NEPAdapters::cuda)
  add_test(
    NAME nep_adapters_cuda_spin2_o3c2_oracle_test
    COMMAND
      nep_adapters_cuda_spin2_oc_oracle_test
      "${PROJECT_SOURCE_DIR}/tests/fixtures/nep4_spin2_o3c2.nep"
      "${PROJECT_SOURCE_DIR}/tests/fixtures/spin2_o3c2_oracle.txt")
  set_tests_properties(
    nep_adapters_cuda_spin2_o3c2_oracle_test
    PROPERTIES LABELS "oracle;parity;engine;cuda;spin2;o3c2")
  add_test(
    NAME nep_adapters_cuda_spin2_zbl_o3c2_oracle_test
    COMMAND
      nep_adapters_cuda_spin2_oc_oracle_test
      "${PROJECT_SOURCE_DIR}/tests/fixtures/nep4_spin2_zbl_o3c2.nep"
      "${PROJECT_SOURCE_DIR}/tests/fixtures/spin2_zbl_o3c2_oracle.txt")
  set_tests_properties(
    nep_adapters_cuda_spin2_zbl_o3c2_oracle_test
    PROPERTIES LABELS "oracle;parity;engine;cuda;spin2;zbl;o3c2")
  add_test(
    NAME nep_adapters_cuda_spin2_o2c2_oracle_test
    COMMAND
      nep_adapters_cuda_spin2_oc_oracle_test
      "${PROJECT_SOURCE_DIR}/tests/fixtures/nep4_spin2_o2c2.nep"
      "${PROJECT_SOURCE_DIR}/tests/fixtures/spin2_o2c2_oracle.txt")
  set_tests_properties(
    nep_adapters_cuda_spin2_o2c2_oracle_test
    PROPERTIES LABELS "oracle;parity;engine;cuda;spin2;o2c2")
  add_test(
    NAME nep_adapters_cuda_spin2_o1c1_l0_soc0_oracle_test
    COMMAND
      nep_adapters_cuda_spin2_oc_oracle_test
      "${PROJECT_SOURCE_DIR}/tests/fixtures/nep4_spin2_o1c1_l0_soc0.nep"
      "${PROJECT_SOURCE_DIR}/tests/fixtures/spin2_o1c1_l0_soc0_oracle.txt")
  set_tests_properties(
    nep_adapters_cuda_spin2_o1c1_l0_soc0_oracle_test
    PROPERTIES LABELS "oracle;parity;engine;cuda;spin2;o1;c1;l0;soc0")
  add_test(
    NAME nep_adapters_cuda_spin2_o3c3_oracle_test
    COMMAND
      nep_adapters_cuda_spin2_oc_oracle_test
      "${PROJECT_SOURCE_DIR}/tests/fixtures/nep4_spin2_o3c3.nep"
      "${PROJECT_SOURCE_DIR}/tests/fixtures/spin2_o3c3_oracle.txt")
  set_tests_properties(
    nep_adapters_cuda_spin2_o3c3_oracle_test
    PROPERTIES LABELS "oracle;parity;engine;cuda;spin2;o3;c3;bulk;soc1")
  add_test(
    NAME nep_adapters_cuda_spin2_o3c2_soc0_oracle_test
    COMMAND
      nep_adapters_cuda_spin2_oc_oracle_test
      "${PROJECT_SOURCE_DIR}/tests/fixtures/nep4_spin2_o3c2_soc0.nep"
      "${PROJECT_SOURCE_DIR}/tests/fixtures/spin2_o3c2_soc0_oracle.txt")
  set_tests_properties(
    nep_adapters_cuda_spin2_o3c2_soc0_oracle_test
    PROPERTIES LABELS "oracle;parity;engine;cuda;spin2;o3;c2;l2;soc0")
  add_test(
    NAME nep_adapters_cuda_spin2_o2c9_l1_soc1_oracle_test
    COMMAND
      nep_adapters_cuda_spin2_oc_oracle_test
      "${PROJECT_SOURCE_DIR}/tests/fixtures/nep4_spin2_o2c9_l1_soc1.nep"
      "${PROJECT_SOURCE_DIR}/tests/fixtures/spin2_o2c9_l1_soc1_oracle.txt")
  set_tests_properties(
    nep_adapters_cuda_spin2_o2c9_l1_soc1_oracle_test
    PROPERTIES LABELS "oracle;parity;engine;cuda;spin2;o2;c9;l1;soc1")

  if(EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/nep.txt" AND
     EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/xyz.in" AND
     EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/force_analytical_ref.out" AND
     EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/virial_ref.out")
    add_executable(nep_adapters_cuda_qnep_reference_test
      test_cuda_qnep_reference.cpp)
    target_link_libraries(
      nep_adapters_cuda_qnep_reference_test
      PRIVATE
        NEPAdapters::cuda
        NEPAdapters::cpu
    )
    target_compile_definitions(
      nep_adapters_cuda_qnep_reference_test
      PRIVATE
        NEP_ADAPTERS_QNEP_TEST_DATA_DIR="${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}"
    )
    add_test(
      NAME nep_adapters_cuda_qnep_reference_test
      COMMAND nep_adapters_cuda_qnep_reference_test)
    set_tests_properties(nep_adapters_cuda_qnep_reference_test PROPERTIES
      LABELS "parity;engine;cuda;device;qnep")
    if(NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM)
      add_test(
        NAME nep_adapters_cuda_qnep_pppm_reference_test
        COMMAND nep_adapters_cuda_qnep_reference_test)
      set_tests_properties(nep_adapters_cuda_qnep_pppm_reference_test PROPERTIES
        LABELS "parity;engine;cuda;device;qnep;pppm"
        ENVIRONMENT "NEP_ADAPTERS_QNEP_KSPACE=pppm")
    else()
      add_test(
        NAME nep_adapters_cuda_qnep_pppm_disabled_test
        COMMAND nep_adapters_cuda_qnep_reference_test)
      set_tests_properties(nep_adapters_cuda_qnep_pppm_disabled_test PROPERTIES
        LABELS "contract;engine;cuda;device;qnep;pppm"
        ENVIRONMENT
          "NEP_ADAPTERS_QNEP_KSPACE=pppm;NEP_ADAPTERS_TEST_EXPECT_PPPM_DISABLED=1")
    endif()
  endif()

  add_executable(nep_adapters_cuda_lammps_kk_sim_test test_cuda_lammps_kk_sim.cu)
  target_link_libraries(
    nep_adapters_cuda_lammps_kk_sim_test
    PRIVATE
      NEPAdapters::cuda
  )
  target_include_directories(
    nep_adapters_cuda_lammps_kk_sim_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/engines/cuda"
  )
  set_target_properties(nep_adapters_cuda_lammps_kk_sim_test PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  add_test(
    NAME nep_adapters_cuda_lammps_kk_sim_test
    COMMAND nep_adapters_cuda_lammps_kk_sim_test)
  set_tests_properties(nep_adapters_cuda_lammps_kk_sim_test PROPERTIES
    LABELS "contract;engine;cuda;device;lammps;kokkos")

  add_executable(
    nep_adapters_cuda_lammps_reuse_overflow_test
    test_cuda_lammps_reuse_overflow.cu)
  target_link_libraries(
    nep_adapters_cuda_lammps_reuse_overflow_test
    PRIVATE
      NEPAdapters::cuda)
  set_target_properties(
    nep_adapters_cuda_lammps_reuse_overflow_test PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  add_test(
    NAME nep_adapters_cuda_lammps_reuse_overflow_test
    COMMAND nep_adapters_cuda_lammps_reuse_overflow_test)
  set_tests_properties(
    nep_adapters_cuda_lammps_reuse_overflow_test PROPERTIES
    LABELS "contract;engine;cuda;device;lammps;neighbors;reuse")

  if(TARGET NEPAdapters::cpu)
    add_executable(nep_adapters_cuda_spin_onsite_test
      test_cuda_spin_onsite.cpp)
    target_link_libraries(
      nep_adapters_cuda_spin_onsite_test
      PRIVATE
        NEPAdapters::cuda
        NEPAdapters::cpu
    )
    add_test(
      NAME nep_adapters_cuda_spin_onsite_test
      COMMAND nep_adapters_cuda_spin_onsite_test)
    set_tests_properties(nep_adapters_cuda_spin_onsite_test PROPERTIES
      LABELS "parity;engine;cuda;device;spin")

    add_executable(nep_adapters_cuda_spin_scalar_test
      test_cuda_spin_scalar.cpp)
    target_link_libraries(
      nep_adapters_cuda_spin_scalar_test
      PRIVATE
        NEPAdapters::cuda
        NEPAdapters::cpu
    )
    add_test(
      NAME nep_adapters_cuda_spin_scalar_test
      COMMAND nep_adapters_cuda_spin_scalar_test)
    set_tests_properties(nep_adapters_cuda_spin_scalar_test PROPERTIES
      LABELS "parity;engine;cuda;device;spin")

    add_executable(nep_adapters_cuda_spin_density_test
      test_cuda_spin_density.cpp)
    target_link_libraries(
      nep_adapters_cuda_spin_density_test
      PRIVATE
        NEPAdapters::cuda
        NEPAdapters::cpu
    )
    add_test(
      NAME nep_adapters_cuda_spin_density_test
      COMMAND nep_adapters_cuda_spin_density_test)
    set_tests_properties(nep_adapters_cuda_spin_density_test PROPERTIES
      LABELS "parity;engine;cuda;device;spin")

    add_executable(nep_adapters_cuda_spin_chiral_polar_test
      test_cuda_spin_chiral_polar.cpp)
    target_link_libraries(
      nep_adapters_cuda_spin_chiral_polar_test
      PRIVATE
        NEPAdapters::cuda
        NEPAdapters::cpu
    )
    add_test(
      NAME nep_adapters_cuda_spin_chiral_polar_test
      COMMAND nep_adapters_cuda_spin_chiral_polar_test)
    set_tests_properties(nep_adapters_cuda_spin_chiral_polar_test PROPERTIES
      LABELS "parity;engine;cuda;device;spin")

    add_executable(nep_adapters_cuda_spin_fixture_test
      test_cuda_spin_fixture.cpp)
    target_link_libraries(
      nep_adapters_cuda_spin_fixture_test
      PRIVATE
        NEPAdapters::cuda
        NEPAdapters::cpu
    )
    target_compile_definitions(
      nep_adapters_cuda_spin_fixture_test
      PRIVATE
        NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE="${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/nep.txt"
    )
    add_test(
      NAME nep_adapters_cuda_spin_fixture_test
      COMMAND nep_adapters_cuda_spin_fixture_test)
    set_tests_properties(nep_adapters_cuda_spin_fixture_test PROPERTIES
      LABELS "parity;engine;cuda;device;spin")
  endif()

  add_executable(nep_adapters_cuda_device_model_test test_cuda_device_model.cu)
  target_link_libraries(
    nep_adapters_cuda_device_model_test
    PRIVATE
      NEPAdapters::cuda
  )
  target_include_directories(
    nep_adapters_cuda_device_model_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/engines/cuda"
  )
  set_target_properties(nep_adapters_cuda_device_model_test PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  add_test(
    NAME nep_adapters_cuda_device_model_test
    COMMAND nep_adapters_cuda_device_model_test)
  set_tests_properties(nep_adapters_cuda_device_model_test PROPERTIES
    LABELS "contract;engine;cuda;device")

  add_executable(nep_adapters_cuda_radial_batch_force_test
    test_cuda_radial_batch_force.cpp)
  target_link_libraries(
    nep_adapters_cuda_radial_batch_force_test
    PRIVATE
      NEPAdapters::cuda
  )
  add_test(
    NAME nep_adapters_cuda_radial_batch_force_test
    COMMAND nep_adapters_cuda_radial_batch_force_test)
  set_tests_properties(nep_adapters_cuda_radial_batch_force_test PROPERTIES
    LABELS "parity;engine;cuda;device;force")

  add_executable(nep_adapters_cuda_nep5_batch_force_test
    test_cuda_nep5_batch_force.cpp)
  target_link_libraries(
    nep_adapters_cuda_nep5_batch_force_test
    PRIVATE
      NEPAdapters::cuda
  )
  add_test(
    NAME nep_adapters_cuda_nep5_batch_force_test
    COMMAND nep_adapters_cuda_nep5_batch_force_test)
  set_tests_properties(nep_adapters_cuda_nep5_batch_force_test PROPERTIES
    LABELS "parity;engine;cuda;device;force")

  add_executable(nep_adapters_cuda_zbl_batch_force_test
    test_cuda_zbl_batch_force.cpp)
  target_link_libraries(
    nep_adapters_cuda_zbl_batch_force_test
    PRIVATE
      NEPAdapters::cuda
  )
  add_test(
    NAME nep_adapters_cuda_zbl_batch_force_test
    COMMAND nep_adapters_cuda_zbl_batch_force_test)
  set_tests_properties(nep_adapters_cuda_zbl_batch_force_test PROPERTIES
    LABELS "parity;engine;cuda;device;force")

  add_executable(nep_adapters_cuda_angular_batch_force_test
    test_cuda_angular_batch_force.cpp)
  target_link_libraries(
    nep_adapters_cuda_angular_batch_force_test
    PRIVATE
      NEPAdapters::cuda
  )
  add_test(
    NAME nep_adapters_cuda_angular_batch_force_test
    COMMAND nep_adapters_cuda_angular_batch_force_test)
  set_tests_properties(nep_adapters_cuda_angular_batch_force_test PROPERTIES
    LABELS "parity;engine;cuda;device;force")

  add_executable(nep_adapters_cuda_q222_batch_force_test
    test_cuda_q222_batch_force.cpp)
  target_link_libraries(
    nep_adapters_cuda_q222_batch_force_test
    PRIVATE
      NEPAdapters::cuda
  )
  add_test(
    NAME nep_adapters_cuda_q222_batch_force_test
    COMMAND nep_adapters_cuda_q222_batch_force_test)
  set_tests_properties(nep_adapters_cuda_q222_batch_force_test PROPERTIES
    LABELS "parity;engine;cuda;device;force")

  add_executable(nep_adapters_cuda_q1111_batch_force_test
    test_cuda_q1111_batch_force.cpp)
  target_link_libraries(
    nep_adapters_cuda_q1111_batch_force_test
    PRIVATE
      NEPAdapters::cuda
  )
  add_test(
    NAME nep_adapters_cuda_q1111_batch_force_test
    COMMAND nep_adapters_cuda_q1111_batch_force_test)
  set_tests_properties(nep_adapters_cuda_q1111_batch_force_test PROPERTIES
    LABELS "parity;engine;cuda;device;force")

  add_executable(nep_adapters_cuda_q112_batch_force_test
    test_cuda_q112_batch_force.cpp)
  target_link_libraries(
    nep_adapters_cuda_q112_batch_force_test
    PRIVATE
      NEPAdapters::cuda
  )
  add_test(
    NAME nep_adapters_cuda_q112_batch_force_test
    COMMAND nep_adapters_cuda_q112_batch_force_test)
  set_tests_properties(nep_adapters_cuda_q112_batch_force_test PROPERTIES
    LABELS "parity;engine;cuda;device;force")

  add_executable(nep_adapters_cuda_q123_batch_force_test
    test_cuda_q123_batch_force.cpp)
  target_link_libraries(
    nep_adapters_cuda_q123_batch_force_test
    PRIVATE
      NEPAdapters::cuda
  )
  add_test(
    NAME nep_adapters_cuda_q123_batch_force_test
    COMMAND nep_adapters_cuda_q123_batch_force_test)
  set_tests_properties(nep_adapters_cuda_q123_batch_force_test PROPERTIES
    LABELS "parity;engine;cuda;device;force")

  add_executable(nep_adapters_cuda_q233_batch_force_test
    test_cuda_q233_batch_force.cpp)
  target_link_libraries(
    nep_adapters_cuda_q233_batch_force_test
    PRIVATE
      NEPAdapters::cuda
  )
  add_test(
    NAME nep_adapters_cuda_q233_batch_force_test
    COMMAND nep_adapters_cuda_q233_batch_force_test)
  set_tests_properties(nep_adapters_cuda_q233_batch_force_test PROPERTIES
    LABELS "parity;engine;cuda;device;force")

  add_executable(nep_adapters_cuda_q134_batch_force_test
    test_cuda_q134_batch_force.cpp)
  target_link_libraries(
    nep_adapters_cuda_q134_batch_force_test
    PRIVATE
      NEPAdapters::cuda
  )
  add_test(
    NAME nep_adapters_cuda_q134_batch_force_test
    COMMAND nep_adapters_cuda_q134_batch_force_test)
  set_tests_properties(nep_adapters_cuda_q134_batch_force_test PROPERTIES
    LABELS "parity;engine;cuda;device;force")
endif()

if(TARGET NEPAdapters::cuda AND TARGET NEPAdapters::cpu)
  add_executable(
    nep_adapters_cuda_protocol_extensions_test
    test_cuda_protocol_extensions.cpp)
  target_link_libraries(
    nep_adapters_cuda_protocol_extensions_test
    PRIVATE
      NEPAdapters::cuda
      NEPAdapters::cpu)
  add_test(
    NAME nep_adapters_cuda_protocol_extensions_test
    COMMAND nep_adapters_cuda_protocol_extensions_test)
  set_tests_properties(
    nep_adapters_cuda_protocol_extensions_test
    PROPERTIES LABELS "parity;protocol;engine;cpu;cuda;device")
endif()

if(TARGET NEPAdapters::cuda AND
   TARGET NEPAdapters::cpu)
  add_executable(nep_adapters_cuda_cpu_triclinic_parity_test
    test_cuda_cpu_triclinic_parity.cpp)
  target_link_libraries(
    nep_adapters_cuda_cpu_triclinic_parity_test
    PRIVATE
      NEPAdapters::cuda
      NEPAdapters::cpu
  )
  add_test(
    NAME nep_adapters_cuda_cpu_triclinic_parity_test
    COMMAND nep_adapters_cuda_cpu_triclinic_parity_test)
  set_tests_properties(nep_adapters_cuda_cpu_triclinic_parity_test PROPERTIES
    LABELS "parity;engine;cuda;device;cpu;triclinic;batch;multi_structure")
endif()

if(TARGET NEPAdapters::cpu)
  add_executable(nep_adapters_fp64_oracle_test
    test_fp64_oracle.cpp
    "${NEP_ADAPTERS_CPU_SOURCE_DIR}/nep.cpp"
    "${NEP_ADAPTERS_CPU_SOURCE_DIR}/neighbor_nep.cpp"
    "${NEP_ADAPTERS_CPU_SOURCE_DIR}/ewald_nep.cpp"
  )
  target_link_libraries(
    nep_adapters_fp64_oracle_test
    PRIVATE
      NEPAdapters::cpu
  )
  target_include_directories(
    nep_adapters_fp64_oracle_test
    PRIVATE
      "${PROJECT_SOURCE_DIR}/engines/cpu_common"
      "${NEP_ADAPTERS_CPU_SOURCE_DIR}"
      "${PROJECT_SOURCE_DIR}/tests"
  )
  target_compile_definitions(
    nep_adapters_fp64_oracle_test
    PRIVATE
      NEP=NEPAdaptersFp64OracleNativeNEP
      EwaldNep=NEPAdaptersFp64OracleNativeEwaldNep
      find_neighbor_list_small_box=nep_adapters_fp64_oracle_find_neighbor_list_small_box
      NEP_ADAPTERS_FP64_MODEL_PATH="${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/nep.txt"
      NEP_ADAPTERS_FP64_XYZ_PATH="${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/train.xyz"
      NEP_ADAPTERS_FP64_DESCRIPTOR_PATH="${NEP_ADAPTERS_CPU_TEST_DATA_DIR}/descriptor.txt"
      NEP_ADAPTERS_FP64_QNEP_MODEL_PATH="${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/nep.txt"
      NEP_ADAPTERS_FP64_QNEP_XYZ_PATH="${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/xyz.in"
      NEP_ADAPTERS_FP64_QNEP_FORCE_PATH="${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/force_analytical_ref.out"
      NEP_ADAPTERS_FP64_QNEP_VIRIAL_PATH="${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/virial_ref.out"
      NEP_ADAPTERS_FP64_QNEP_DESCRIPTOR_PATH="${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/descriptor_ref.out"
      NEP_ADAPTERS_FP64_SPIN_MODEL_PATH="${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/nep.txt"
      NEP_ADAPTERS_FP64_SPIN_REFERENCE_PATH="${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/reference_small_pbc.txt"
  )
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
      nep_adapters_fp64_oracle_test
      PRIVATE
        -fno-fast-math
        -ffp-contract=off
    )
  elseif(MSVC)
    target_compile_options(nep_adapters_fp64_oracle_test PRIVATE /fp:strict)
  endif()
  set(NEP_ADAPTERS_FP64_ORACLE_CUDA_LABEL "")
  if(TARGET NEPAdapters::cuda)
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(
      nep_adapters_fp64_oracle_test
      PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(
      nep_adapters_fp64_oracle_test
      PRIVATE
        NEPAdapters::cuda
        CUDA::cudart
    )
    target_compile_definitions(
      nep_adapters_fp64_oracle_test
      PRIVATE
        NEP_ADAPTERS_FP64_COMPARE_CUDA=1
    )
    set(NEP_ADAPTERS_FP64_ORACLE_CUDA_LABEL ";cuda")
  endif()
  add_test(
    NAME nep_adapters_fp64_oracle_test
    COMMAND nep_adapters_fp64_oracle_test)
  set_tests_properties(nep_adapters_fp64_oracle_test PROPERTIES
    LABELS "parity;oracle;fp64;cpu;qnep;spin${NEP_ADAPTERS_FP64_ORACLE_CUDA_LABEL}")

  add_executable(nep_adapters_cpu_spin_test
    test_cpu_spin.cpp)
  target_link_libraries(
    nep_adapters_cpu_spin_test
    PRIVATE
      NEPAdapters::cpu
  )
  target_compile_definitions(
    nep_adapters_cpu_spin_test
    PRIVATE
      NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE="${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/nep.txt"
      NEP_ADAPTERS_SPIN_CHIRAL_REFERENCE="${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/reference_small_pbc.txt"
  )
  add_test(
    NAME nep_adapters_cpu_spin_test
    COMMAND nep_adapters_cpu_spin_test)
  set_tests_properties(nep_adapters_cpu_spin_test PROPERTIES
    LABELS "smoke;engine;cpu;spin")
endif()

if(TARGET NEPAdapters::cpu AND
   EXISTS "${NEP_ADAPTERS_NEP89_MODEL_PATH}" AND
   EXISTS "${NEP_ADAPTERS_NEP89_XYZ_PATH}")
  add_executable(nep_adapters_cpu_nep89_smoke_test test_cpu_nep89.cpp)
  target_link_libraries(
    nep_adapters_cpu_nep89_smoke_test
    PRIVATE
      NEPAdapters::cpu
  )
  target_compile_definitions(
    nep_adapters_cpu_nep89_smoke_test
    PRIVATE
      NEP_ADAPTERS_NEP89_MODEL_PATH="${NEP_ADAPTERS_NEP89_MODEL_PATH}"
      NEP_ADAPTERS_NEP89_XYZ_PATH="${NEP_ADAPTERS_NEP89_XYZ_PATH}"
  )
  add_test(NAME nep_adapters_cpu_nep89_smoke_test COMMAND nep_adapters_cpu_nep89_smoke_test)
  set_tests_properties(nep_adapters_cpu_nep89_smoke_test PROPERTIES
    LABELS "smoke;engine;cpu;large_model;nep89")
endif()
