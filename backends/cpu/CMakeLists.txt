add_library(nep_adapters_cpu INTERFACE)
add_library(NEPAdapters::cpu ALIAS nep_adapters_cpu)

target_link_libraries(nep_adapters_cpu INTERFACE NEPAdapters::nep_adapters)
