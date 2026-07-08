#pragma once

#include "device_workspace.hpp"

namespace nep_adapters::cuda_backend {

void prepare_batched_outputs(
    int structure_count,
    int atom_count,
    DeviceWorkspace& workspace);

}  // namespace nep_adapters::cuda_backend
