add_library(nep_adapters_cuda INTERFACE)
add_library(NEPAdapters::cuda ALIAS nep_adapters_cuda)

target_link_libraries(nep_adapters_cuda INTERFACE NEPAdapters::nep_adapters)
