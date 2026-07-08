#pragma once

#include "device_model.hpp"
#include "device_workspace.hpp"
#include "model_protocol.hpp"

namespace nep_adapters::cuda_backend {

void evaluate_ann_energy_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void evaluate_qnep_ann_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void zero_total_charge_on_device(
    int atom_count,
    DeviceWorkspace& workspace);

void add_charge_chain_to_fp_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace);

}  // namespace nep_adapters::cuda_backend
