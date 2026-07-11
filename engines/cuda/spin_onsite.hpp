#pragma once

#include "device_model.hpp"
#include "device_workspace.hpp"
#include "model_protocol.hpp"
#include "simulation_box.hpp"

namespace nep_adapters::cuda_backend {

void build_spin_descriptors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

struct SpinForceTimings {
  float onsite_ms = 0.0f;
  float scalar_ms = 0.0f;
  float density_ms = 0.0f;
  float chiral_ms = 0.0f;
};

void accumulate_spin_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial,
    SpinForceTimings* timings = nullptr);

}  // namespace nep_adapters::cuda_backend
