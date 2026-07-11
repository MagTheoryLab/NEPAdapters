#include "nonspin_pipeline.hpp"

#include "ann_energy.hpp"
#include "angular_basis_cache.hpp"
#include "angular_descriptor.hpp"
#include "angular_force.hpp"
#include "pair_geometry_cache.hpp"
#include "radial_basis_cache.hpp"
#include "radial_descriptor.hpp"
#include "radial_force.hpp"
#include "zbl_force.hpp"

#include <stdexcept>

namespace nep_adapters::cuda_backend {

NonSpinExecutionPlan make_nonspin_execution_plan(
    const ModelProtocol& protocol,
    const NonSpinPipelineRequest& request) {
  if (protocol.spin_mode != 0 || protocol.charge_mode != 0) {
    throw std::invalid_argument(
        "non-spin execution pipeline accepts ordinary models only");
  }

  NonSpinExecutionPlan plan;
  plan.has_angular = request.has_angular;
  plan.orthorhombic_batched = request.orthorhombic_batched;
  plan.store_potential = request.store_potential;
  plan.accumulate_virial = request.accumulate_virial;
  plan.zbl_outputs = request.zbl_outputs;

  switch (request.topology) {
    case NonSpinNeighborTopology::batched_multi_box:
      plan.descriptor_mode = NonSpinDescriptorMode::batched_cached;
      plan.radial_force_mode = NonSpinRadialForceMode::batched;
      break;
    case NonSpinNeighborTopology::single_box_symmetric:
      plan.descriptor_mode =
          request.has_angular && protocol.charge_mode == 0 &&
              protocol.descriptor_dim <= 64
          ? NonSpinDescriptorMode::fused_positions
          : NonSpinDescriptorMode::staged_positions;
      plan.radial_force_mode = NonSpinRadialForceMode::symmetric;
      break;
    case NonSpinNeighborTopology::external_full:
      plan.descriptor_mode =
          request.has_angular && protocol.charge_mode == 0 &&
              protocol.descriptor_dim <= 64
          ? NonSpinDescriptorMode::fused_positions
          : NonSpinDescriptorMode::staged_positions;
      plan.radial_force_mode = NonSpinRadialForceMode::external_full;
      break;
  }
  return plan;
}

namespace {

void require_ordinary_model(const ModelProtocol& protocol) {
  if (protocol.spin_mode != 0 || protocol.charge_mode != 0) {
    throw std::invalid_argument(
        "non-spin execution pipeline accepts ordinary models only");
  }
}

void build_staged_position_descriptors(
    const ModelProtocol& protocol,
    const NonSpinExecutionPlan& plan,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  if (plan.has_angular) {
    build_angular_geometry_cache_on_device(protocol, atom_count, box, workspace);
  }
  if (!build_radial_descriptors_from_geometry_on_device(
          protocol, atom_count, box, model, workspace)) {
    if (plan.has_angular) {
      build_pair_geometry_cache_on_device(protocol, atom_count, box, workspace);
      build_radial_basis_cache_on_device(protocol, atom_count, workspace);
    } else {
      build_radial_geometry_basis_cache_on_device(
          protocol, atom_count, box, workspace);
    }
    build_radial_descriptors_on_device(protocol, atom_count, model, workspace);
  }
  if (plan.has_angular) {
    if (try_build_angular_descriptors_and_ann_from_geometry_on_device(
            protocol, atom_count, model, workspace)) {
      return;
    }
    build_angular_descriptors_from_geometry_on_device(
        protocol, atom_count, model, workspace);
  }
  evaluate_ann_energy_on_device(protocol, atom_count, model, workspace);
}

}  // namespace

void prepare_nonspin_descriptors(
    const ModelProtocol& protocol,
    const NonSpinExecutionPlan& plan,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require_ordinary_model(protocol);
  switch (plan.descriptor_mode) {
    case NonSpinDescriptorMode::fused_positions:
      if (!try_build_descriptors_and_ann_from_positions_on_device(
              protocol,
              atom_count,
              box,
              model,
              workspace,
              plan.store_potential)) {
        throw std::runtime_error(
            "execution plan selected unsupported fused descriptor path");
      }
      return;
    case NonSpinDescriptorMode::staged_positions:
      build_staged_position_descriptors(
          protocol, plan, atom_count, box, model, workspace);
      return;
    case NonSpinDescriptorMode::batched_cached:
      if (!plan.orthorhombic_batched && plan.has_angular) {
        build_pair_geometry_cache_batched(protocol, atom_count, workspace);
        build_radial_basis_cache_on_device(protocol, atom_count, workspace);
      } else if (!plan.orthorhombic_batched) {
        build_radial_geometry_basis_cache_batched(
            protocol, atom_count, workspace);
      } else {
        build_radial_basis_cache_on_device(protocol, atom_count, workspace);
      }
      build_radial_descriptors_on_device(protocol, atom_count, model, workspace);
      if (plan.has_angular &&
          try_build_angular_descriptors_and_ann_from_geometry_on_device(
              protocol, atom_count, model, workspace)) {
        return;
      }
      if (plan.has_angular) {
        build_angular_descriptors_from_geometry_on_device(
            protocol, atom_count, model, workspace);
      }
      evaluate_ann_energy_on_device(protocol, atom_count, model, workspace);
      return;
  }
}

void accumulate_nonspin_radial_forces(
    const ModelProtocol& protocol,
    const NonSpinExecutionPlan& plan,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require_ordinary_model(protocol);
  switch (plan.radial_force_mode) {
    case NonSpinRadialForceMode::batched:
      accumulate_radial_forces_batched(protocol, atom_count, model, workspace);
      return;
    case NonSpinRadialForceMode::symmetric:
      accumulate_radial_forces_on_device(
          protocol,
          atom_count,
          box,
          model,
          workspace,
          plan.accumulate_virial);
      return;
    case NonSpinRadialForceMode::external_full:
      accumulate_lammps_radial_forces_on_device(
          protocol,
          atom_count,
          box,
          model,
          workspace,
          plan.accumulate_virial);
      return;
  }
}

void accumulate_nonspin_angular_forces(
    const ModelProtocol& protocol,
    const NonSpinExecutionPlan& plan,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require_ordinary_model(protocol);
  if (!plan.has_angular) {
    return;
  }
  if (plan.radial_force_mode == NonSpinRadialForceMode::batched) {
    accumulate_l2_angular_forces_batched(protocol, atom_count, model, workspace);
    return;
  }
  accumulate_l2_angular_forces_on_device(
      protocol,
      atom_count,
      box,
      model,
      workspace,
      plan.accumulate_virial);
}

void accumulate_nonspin_zbl_forces(
    const ModelProtocol& protocol,
    const NonSpinExecutionPlan& plan,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require_ordinary_model(protocol);
  if (!protocol.has_zbl) {
    return;
  }
  if (plan.radial_force_mode == NonSpinRadialForceMode::batched) {
    accumulate_zbl_forces_batched(protocol, atom_count, model, workspace);
    return;
  }
  accumulate_zbl_forces_on_device(
      protocol,
      atom_count,
      box,
      model,
      workspace,
      plan.zbl_outputs);
}

void run_nonspin_pipeline(
    const ModelProtocol& protocol,
    const NonSpinExecutionPlan& plan,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  prepare_nonspin_descriptors(
      protocol, plan, atom_count, box, model, workspace);
  accumulate_nonspin_radial_forces(
      protocol, plan, atom_count, box, model, workspace);
  accumulate_nonspin_angular_forces(
      protocol, plan, atom_count, box, model, workspace);
  accumulate_nonspin_zbl_forces(
      protocol, plan, atom_count, box, model, workspace);
}

}  // namespace nep_adapters::cuda_backend
