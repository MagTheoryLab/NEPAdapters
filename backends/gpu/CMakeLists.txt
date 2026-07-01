add_library(nep_adapters_gpu INTERFACE)
add_library(NEPAdapters::gpu ALIAS nep_adapters_gpu)

target_link_libraries(nep_adapters_gpu INTERFACE NEPAdapters::nep_adapters)
