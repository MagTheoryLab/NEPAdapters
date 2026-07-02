#pragma once

#include "nep_adapters/api.h"

#ifdef __cplusplus
extern "C" {
#endif

NEP_ADAPTERS_API int nepa_register_cpu_opt_engine(void);

#ifdef __cplusplus
}

namespace nep_adapters {

bool register_cpu_opt_engine();

}  // namespace nep_adapters
#endif
