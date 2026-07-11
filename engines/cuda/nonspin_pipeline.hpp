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

enum class NonSpinDescriptorMode {
  batched_cached,
  fused_positions,
  staged_positions,
};

enum class NonSpinRadialForceMode {
  batched,
  symmetric,
  external_full,
};

struct NonSpinPipelineRequest {
  NonSpinNeighborTopology topology =
      NonSpinNeighborTopology::single_box_symmetric;
  bool has_angular = false;
  bool orthorhombic_batched = false;
  bool store_potential = true;
  bool accumulate_virial = true;
  bool zbl_outputs = true;
};

struct NonSpinExecutionPlan {
  NonSpinDescriptorMode descriptor_mode =
      NonSpinDescriptorMode::staged_positions;
  NonSpinRadialForceMode radial_force_mode =
      NonSpinRadialForceMode::symmetric;
  bool has_angular = false;
  bool orthorhombic_batched = false;
  bool store_potential = true;
  bool accumulate_virial = true;
  bool zbl_outputs = true;
};

NonSpinExecutionPlan make_nonspin_execution_plan(
    const ModelProtocol& protocol,
    const NonSpinPipelineRequest& request);

void prepare_nonspin_descriptors(
    const ModelProtocol& protocol,
    const NonSpinExecutionPlan& plan,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void accumulate_nonspin_radial_forces(
    const ModelProtocol& protocol,
    const NonSpinExecutionPlan& plan,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void accumulate_nonspin_angular_forces(
    const ModelProtocol& protocol,
    const NonSpinExecutionPlan& plan,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void accumulate_nonspin_zbl_forces(
    const ModelProtocol& protocol,
    const NonSpinExecutionPlan& plan,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void run_nonspin_pipeline(
    const ModelProtocol& protocol,
    const NonSpinExecutionPlan& plan,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

}  // namespace nep_adapters::cuda_backend
