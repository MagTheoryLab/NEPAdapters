#pragma once

#include "device_workspace.hpp"
#include "model_protocol.hpp"
#include "simulation_box.hpp"

namespace nep_adapters::cuda_backend {

void build_internal_neighbors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace);

void build_internal_neighbors_batched(
    const ModelProtocol& protocol,
    int structure_count,
    int atom_count,
    DeviceWorkspace& workspace,
    bool orthorhombic_fast_path = false);

}  // namespace nep_adapters::cuda_backend
