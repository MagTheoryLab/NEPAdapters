#pragma once

#include "device_workspace.hpp"
#include "internal_neighbor_builder.hpp"
#include "model_protocol.hpp"

namespace nep_adapters::cuda_backend {

void apply_qnep_charge_terms_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace,
    bool need_per_atom_kspace_potential = true,
    bool need_per_atom_kspace_virial = true);

}  // namespace nep_adapters::cuda_backend
