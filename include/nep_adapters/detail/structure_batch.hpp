#pragma once

#include "nep_adapters/api.h"

namespace nep_adapters::detail {

// Keep numerical input validation in core, outside fast-math engine code.
NEP_ADAPTERS_API bool valid_structure_batch(
    const NepaStructureBatch& batch, int num_types);

}  // namespace nep_adapters::detail
