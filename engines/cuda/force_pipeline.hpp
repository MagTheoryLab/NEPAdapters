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

enum class VirialOutputMode {
  none,
  total_only,
  per_atom_n2,
};

struct ForceEvaluationRequest {
  ForceNeighborTopology topology =
      ForceNeighborTopology::single_box_symmetric;
  VirialOutputMode virial = VirialOutputMode::total_only;
  bool orthorhombic_batched = false;
  bool store_potential = true;
};

constexpr bool requests_virial(const ForceEvaluationRequest& request) {
  return request.virial != VirialOutputMode::none;
}

constexpr bool requests_per_atom_virial(
    const ForceEvaluationRequest& request) {
  return request.virial == VirialOutputMode::per_atom_n2;
}

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
    const ForceEvaluationRequest& request,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    ForcePipelineTimings* timings = nullptr);

void run_spin_pipeline(
    const ModelProtocol& protocol,
    const ForceEvaluationRequest& request,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    ForcePipelineTimings* timings = nullptr);

}  // namespace nep_adapters::cuda_backend
