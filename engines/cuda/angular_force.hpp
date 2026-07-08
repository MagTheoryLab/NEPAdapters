#pragma once

#include "device_model.hpp"
#include "device_workspace.hpp"
#include "internal_neighbor_builder.hpp"
#include "model_protocol.hpp"

namespace nep_adapters::cuda_backend {

void accumulate_l2_angular_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial = true,
    bool virial_to_neighbor = false);

void accumulate_l2_angular_forces_batched(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

}  // namespace nep_adapters::cuda_backend
