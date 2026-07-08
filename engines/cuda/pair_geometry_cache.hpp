#pragma once

#include "device_workspace.hpp"
#include "internal_neighbor_builder.hpp"
#include "model_protocol.hpp"

namespace nep_adapters::cuda_backend {

void build_pair_geometry_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace);

void build_angular_geometry_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace);

void build_pair_geometry_cache_batched(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace);

}  // namespace nep_adapters::cuda_backend
