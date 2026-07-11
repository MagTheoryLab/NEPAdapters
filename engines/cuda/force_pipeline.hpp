#pragma once

#include "device_model.hpp"
#include "device_workspace.hpp"
#include "model_protocol.hpp"
#include "simulation_box.hpp"

namespace nep_adapters::cuda_backend {

enum class ForceNeighborTopology {
  batched_multi_box,
  single_box_symmetric,
  external_full,
};

struct ForcePipelineOptions {
  ForceNeighborTopology topology =
      ForceNeighborTopology::single_box_symmetric;
  bool orthorhombic_batched = false;
  bool store_potential = true;
  bool accumulate_virial = true;
  bool zbl_outputs = true;
};

struct ForcePipelineTimings {
  float descriptor_ann_ms = 0.0f;
  float radial_force_ms = 0.0f;
  float angular_force_ms = 0.0f;
  float zbl_force_ms = 0.0f;
  float spin_onsite_ms = 0.0f;
  float spin_scalar_ms = 0.0f;
  float spin_density_ms = 0.0f;
  float spin_chiral_ms = 0.0f;
};

// Runs a complete CUDA force dataflow. Kernel selection stays private so
// callers only choose the neighbor topology and requested outputs.
void run_nonspin_pipeline(
    const ModelProtocol& protocol,
    const ForcePipelineOptions& options,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    ForcePipelineTimings* timings = nullptr);

void run_spin_pipeline(
    const ModelProtocol& protocol,
    const ForcePipelineOptions& options,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    ForcePipelineTimings* timings = nullptr);

}  // namespace nep_adapters::cuda_backend
