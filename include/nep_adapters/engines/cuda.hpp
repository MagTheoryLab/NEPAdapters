#pragma once

#include <string>

namespace nep_adapters {

bool register_cuda_engine();
bool probe_cuda_device(int device_index, std::string& detail);

}  // namespace nep_adapters

extern "C" int nepa_register_cuda_engine(void);
