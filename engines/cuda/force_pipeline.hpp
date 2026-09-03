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
  bool store_potential = true;
  bool spin_transfer_per_atom = false;
};

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
  float spin_density_ms = 0.0f;
  float spin_density_pull_ms = 0.0f;
  float spin_density_edge_ms = 0.0f;
  float spin_chiral_ms = 0.0f;
};

// Runs the complete CUDA force dataflow for the model encoded by protocol.
// Callers specify topology and observable outputs; model-family dispatch,
// ZBL composition, virial placement, and kernel selection stay private.
// Charge models remain on their existing dedicated path until that extension
// can satisfy this same interface without widening it.
void run_force_pipeline(
    const ModelProtocol& protocol,
    const ForceEvaluationRequest& request,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    ForcePipelineTimings* timings = nullptr);

}  // namespace nep_adapters::cuda_backend
