add_library(nep_adapters_cpu_opt INTERFACE)
add_library(NEPAdapters::cpu_opt ALIAS nep_adapters_cpu_opt)

target_link_libraries(nep_adapters_cpu_opt INTERFACE NEPAdapters::nep_adapters)
