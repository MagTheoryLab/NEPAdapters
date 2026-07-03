#include "nep_adapters/engines/cpu_opt.hpp"

#include "nep.h"
#include "../cpu_common/cpu_engine_adapter.hpp"

namespace nep_adapters {

bool register_cpu_opt_engine() {
  static CpuEngine<NEP> engine("cpu_opt");
  static const bool registered = register_engine(&engine);
  return registered;
}

}  // namespace nep_adapters

extern "C" int nepa_register_cpu_opt_engine(void) {
  return nep_adapters::register_cpu_opt_engine() ? 1 : 0;
}
