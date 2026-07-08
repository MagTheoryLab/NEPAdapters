#pragma once

#include "device_workspace.hpp"
#include "model_protocol.hpp"

namespace nep_adapters::cuda_backend {

void build_angular_basis_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace);

}  // namespace nep_adapters::cuda_backend
