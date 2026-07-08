#pragma once

namespace nep_adapters {

bool register_cuda_engine();

}  // namespace nep_adapters

extern "C" int nepa_register_cuda_engine(void);
