add_library(nep_adapters registry.cpp)
add_library(NEPAdapters::nep_adapters ALIAS nep_adapters)

target_include_directories(
  nep_adapters
  PUBLIC
    $<BUILD_INTERFACE:${NEP_ADAPTERS_GENERATED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(nep_adapters PUBLIC cxx_std_17)

target_compile_definitions(nep_adapters PRIVATE NEP_ADAPTERS_BUILDING)
set_target_properties(nep_adapters PROPERTIES
  EXPORT_NAME nep_adapters
  POSITION_INDEPENDENT_CODE ON)

if(BUILD_SHARED_LIBS)
  target_compile_definitions(nep_adapters PUBLIC NEP_ADAPTERS_SHARED)
endif()
