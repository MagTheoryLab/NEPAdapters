#pragma once

#include "device_model.hpp"
#include "device_workspace.hpp"
#include "model_protocol.hpp"
#include "simulation_box.hpp"

namespace nep_adapters::cuda_backend {

enum class NonSpinNeighborTopology {
  batched_multi_box,
  single_box_symmetric,
  external_full,
};

struct NonSpinPipelineOptions {
  NonSpinNeighborTopology topology =
      NonSpinNeighborTopology::single_box_symmetric;
  bool orthorhombic_batched = false;
  bool store_potential = true;
  bool accumulate_virial = true;
  bool zbl_outputs = true;
};

struct NonSpinPipelineTimings {
  float descriptor_ann_ms = 0.0f;
  float radial_force_ms = 0.0f;
  float angular_force_ms = 0.0f;
  float zbl_force_ms = 0.0f;
};

// Runs the complete ordinary-NEP CUDA dataflow. Kernel selection and fallback
// stay private so callers only choose the neighbor topology and requested
// outputs. Timings are optional and intended for the LAMMPS phase profiler.
void run_nonspin_pipeline(
    const ModelProtocol& protocol,
    const NonSpinPipelineOptions& options,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    NonSpinPipelineTimings* timings = nullptr);

}  // namespace nep_adapters::cuda_backend
