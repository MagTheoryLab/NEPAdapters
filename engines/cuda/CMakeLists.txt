add_library(nep_adapters_cuda
  cuda_engine.cpp
  host_staging.cpp
  model_parameters.cpp
  model_protocol.cpp
  workspace_plan.cpp
)
add_library(NEPAdapters::cuda ALIAS nep_adapters_cuda)

if(NEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME)
  find_package(CUDAToolkit REQUIRED)
  target_sources(nep_adapters_cuda PRIVATE
    ann_energy.cu
    angular_basis_cache.cu
    angular_descriptor.cu
    angular_force.cu
    batch_output.cu
    device_model.cu
    device_staging.cu
    device_workspace.cu
    internal_neighbor_builder.cu
    lammps_device_output.cu
    pair_geometry_cache.cu
    qnep_charge.cu
    radial_force.cu
    radial_descriptor.cu
    radial_basis_cache.cu
    zbl_force.cu
  )
  set_source_files_properties(
    ann_energy.cu
    angular_basis_cache.cu
    angular_descriptor.cu
    angular_force.cu
    batch_output.cu
    device_model.cu
    device_staging.cu
    device_workspace.cu
    internal_neighbor_builder.cu
    lammps_device_output.cu
    pair_geometry_cache.cu
    qnep_charge.cu
    radial_force.cu
    radial_descriptor.cu
    radial_basis_cache.cu
    zbl_force.cu
    PROPERTIES LANGUAGE CUDA)
  target_include_directories(
    nep_adapters_cuda
    PRIVATE
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_compile_definitions(nep_adapters_cuda PRIVATE NEP_ADAPTERS_CUDA_DEVICE_RUNTIME=1)
  target_link_libraries(nep_adapters_cuda PRIVATE CUDA::cufft)
endif()

target_link_libraries(nep_adapters_cuda PUBLIC NEPAdapters::nep_adapters)
target_compile_features(nep_adapters_cuda PUBLIC cxx_std_17)
set_target_properties(nep_adapters_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME)
  set_target_properties(nep_adapters_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

if(BUILD_SHARED_LIBS)
  target_compile_definitions(nep_adapters_cuda PRIVATE NEP_ADAPTERS_BUILDING)
endif()

install(
  TARGETS nep_adapters_cuda
  EXPORT NEPAdaptersTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
