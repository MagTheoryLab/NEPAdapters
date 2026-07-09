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
   EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/nep.txt" AND
   EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/xyz.in")
  add_executable(nep_adapters_cpu_nep3_qnep_test test_cpu_nep3_qnep.cpp)
  target_link_libraries(
    nep_adapters_cpu_nep3_qnep_test
    PRIVATE
      NEPAdapters::cpu_nep3
  )
  target_include_directories(
    nep_adapters_cpu_nep3_qnep_test
    PRIVATE
      "${NEP_ADAPTERS_CPU_NEP3_SOURCE_DIR}"
  )
  target_compile_definitions(
    nep_adapters_cpu_nep3_qnep_test
    PRIVATE
      NEP_ADAPTERS_QNEP_TEST_DATA_DIR="${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}"
  )
  add_test(NAME nep_adapters_cpu_nep3_qnep_test COMMAND nep_adapters_cpu_nep3_qnep_test)
  set_tests_properties(nep_adapters_cpu_nep3_qnep_test PROPERTIES
    LABELS "smoke;engine;cpu_nep3;qnep")
endif()

if(TARGET NEPAdapters::cuda AND
   EXISTS "${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt")
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
      NEP_ADAPTERS_CUDA_TEST_MODEL_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt"
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
      NEP_ADAPTERS_CUDA_TEST_MODEL_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt"
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
      NEP_ADAPTERS_CUDA_TEST_MODEL_PATH="${NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR}/nep.txt"
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

  if(NEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME AND
     EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/nep.txt" AND
     EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/xyz.in" AND
     EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/force_analytical_ref.out" AND
     EXISTS "${NEP_ADAPTERS_QNEP_TEST_DATA_DIR}/virial_ref.out")
    add_executable(nep_adapters_cuda_qnep_reference_test
      test_cuda_qnep_reference.cpp)
    target_link_libraries(
      nep_adapters_cuda_qnep_reference_test
      PRIVATE
        NEPAdapters::cuda
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
  endif()

  if(NEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME)
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

    if(TARGET NEPAdapters::cpu_opt)
      add_executable(nep_adapters_cuda_spin_onsite_test
        test_cuda_spin_onsite.cpp)
      target_link_libraries(
        nep_adapters_cuda_spin_onsite_test
        PRIVATE
          NEPAdapters::cuda
          NEPAdapters::cpu_opt
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
          NEPAdapters::cpu_opt
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
          NEPAdapters::cpu_opt
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
          NEPAdapters::cpu_opt
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
          NEPAdapters::cpu_opt
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
endif()

if(TARGET NEPAdapters::cuda AND
   TARGET NEPAdapters::cpu_nep3 AND
   NEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME)
  add_executable(nep_adapters_cuda_cpu_triclinic_parity_test
    test_cuda_cpu_triclinic_parity.cpp)
  target_link_libraries(
    nep_adapters_cuda_cpu_triclinic_parity_test
    PRIVATE
      NEPAdapters::cuda
      NEPAdapters::cpu_nep3
  )
  add_test(
    NAME nep_adapters_cuda_cpu_triclinic_parity_test
    COMMAND nep_adapters_cuda_cpu_triclinic_parity_test)
  set_tests_properties(nep_adapters_cuda_cpu_triclinic_parity_test PROPERTIES
    LABELS "parity;engine;cuda;device;cpu_nep3;triclinic")
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

if(TARGET NEPAdapters::cpu_opt)
  add_executable(nep_adapters_cpu_opt_spin_test
    test_cpu_opt_spin.cpp)
  target_link_libraries(
    nep_adapters_cpu_opt_spin_test
    PRIVATE
      NEPAdapters::cpu_opt
  )
  target_compile_definitions(
    nep_adapters_cpu_opt_spin_test
    PRIVATE
      NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE="${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/nep.txt"
      NEP_ADAPTERS_SPIN_CHIRAL_REFERENCE="${PROJECT_SOURCE_DIR}/tests/fixtures/spin_chiral_protocol/reference_small_pbc.txt"
  )
  add_test(
    NAME nep_adapters_cpu_opt_spin_test
    COMMAND nep_adapters_cpu_opt_spin_test)
  set_tests_properties(nep_adapters_cpu_opt_spin_test PROPERTIES
    LABELS "smoke;engine;cpu_opt;spin")
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
