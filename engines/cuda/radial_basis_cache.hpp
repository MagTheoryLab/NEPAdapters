#pragma once

#include "device_workspace.hpp"
#include "model_protocol.hpp"
#include "simulation_box.hpp"

namespace nep_adapters::cuda_backend {

void build_radial_basis_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace);

void build_radial_geometry_basis_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace);

void build_radial_geometry_basis_cache_batched(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace);

}  // namespace nep_adapters::cuda_backend
