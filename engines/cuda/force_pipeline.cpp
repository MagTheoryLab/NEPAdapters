#include "force_pipeline.hpp"

#include "device_operations.hpp"

#include <stdexcept>

#include <cuda_runtime.h>

namespace nep_adapters::cuda_backend {
namespace {

void require_supported_model(const ModelProtocol& protocol) {
  if (protocol.charge_mode != 0) {
    throw std::invalid_argument(
        "CUDA force pipeline charge integration is not implemented");
  }
}

bool has_angular_terms(const ModelProtocol& protocol) {
  return protocol.body_channels.channel_count() > 0;
}

class PhaseTimer {
 public:
  explicit PhaseTimer(bool enabled) : enabled_(enabled) {
    if (enabled_) {
      cudaEventCreate(&mark_);
      cudaEventCreate(&now_);
      cudaEventRecord(mark_);
    }
  }

  ~PhaseTimer() {
    if (mark_ != nullptr) {
      cudaEventDestroy(mark_);
    }
    if (now_ != nullptr) {
      cudaEventDestroy(now_);
    }
  }

  void split(float& target_ms) {
    if (!enabled_) {
      return;
    }
    cudaEventRecord(now_);
    cudaEventSynchronize(now_);
    cudaEventElapsedTime(&target_ms, mark_, now_);
    cudaEventRecord(mark_);
  }

 private:
  bool enabled_ = false;
  cudaEvent_t mark_ = nullptr;
  cudaEvent_t now_ = nullptr;
};

VirialTarget select_virial_target(const ForceEvaluationRequest& request) {
  switch (request.virial) {
    case VirialOutputMode::none:
      return VirialTarget::none;
    case VirialOutputMode::total_only:
      return VirialTarget::center_atom;
    case VirialOutputMode::per_atom_n2:
      if (request.topology == ForceNeighborTopology::external_full) {
        return VirialTarget::center_and_neighbor_float_sink;
      }
      return VirialTarget::neighbor_atom;
  }
  throw std::invalid_argument("invalid virial output mode");
}

VirialTarget select_zbl_virial_target(
    VirialTarget target,
    bool accumulate_energy_virial) {
  if (!accumulate_energy_virial) {
    return VirialTarget::none;
  }
  return virial_targets_neighbor(target)
      ? VirialTarget::neighbor_atom
      : VirialTarget::center_atom;
}

void execute_force_pipeline(
    const ModelProtocol& protocol,
    const ForceEvaluationRequest& request,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    ForcePipelineTimings* timings) {
  const bool spin_model = protocol.spin_mode != 0;
  const bool has_angular = has_angular_terms(protocol);
  ForcePipelineTimings ignored_timings;
  ForcePipelineTimings& measured =
      timings == nullptr ? ignored_timings : *timings;
  PhaseTimer timer(timings != nullptr);

  if (request.topology == ForceNeighborTopology::batched_multi_box && spin_model) {
    throw std::invalid_argument(
        "spin execution pipeline does not support batched neighbor topology");
  }

  const DescriptorCoreTopology descriptor_topology =
      request.topology == ForceNeighborTopology::batched_multi_box
          ? DescriptorCoreTopology::batched_multi_box
          : DescriptorCoreTopology::single_box;
  build_descriptor_core_from_positions_on_device(
      protocol, atom_count, box, model, workspace, descriptor_topology);
  if (spin_model) {
    build_spin_descriptors_on_device(
        protocol, atom_count, box, model, workspace);
  }
  evaluate_ann_energy_on_device(protocol, atom_count, model, workspace);
  timer.split(measured.descriptor_ann_ms);

  const VirialTarget virial_target = select_virial_target(request);
  const bool use_per_atom_sink =
      virial_target == VirialTarget::center_and_neighbor_float_sink;
  const bool zbl_outputs =
      request.store_potential || accumulates_virial(virial_target);
  const bool fuse_structural_radial_into_spin =
      request.topology == ForceNeighborTopology::external_full &&
      spin_model && protocol.spin_mode == 2 &&
      protocol.spin_compress == 2 && protocol.spin_order == 3 &&
      protocol.spin_l_max == 2 && protocol.spin_soc == 1 &&
      protocol.n_max_radial == 4 && protocol.basis_size_radial == 8 &&
      protocol.spin_basis_size == 8 && protocol.num_types <= 2 &&
      !protocol.use_typewise_cutoff && !protocol.has_zbl &&
      protocol.cutoff_radial == protocol.spin_cutoff_radial &&
      !request.spin_transfer_per_atom &&
      (virial_target == VirialTarget::none ||
       virial_target == VirialTarget::center_atom);
  const bool can_fuse_zbl =
      protocol.has_zbl &&
      !protocol.flexible_zbl &&
      !protocol.use_typewise_cutoff_zbl &&
      protocol.zbl_outer <= protocol.cutoff_radial;
  const bool fuse_external_zbl =
      request.topology == ForceNeighborTopology::external_full &&
      can_fuse_zbl;
  const bool fuse_radial_zbl =
      request.topology == ForceNeighborTopology::single_box_symmetric &&
      can_fuse_zbl && !zbl_outputs;
  if (fuse_radial_zbl) {
    accumulate_radial_and_zbl_forces_on_device(
        protocol, atom_count, box, model, workspace);
  } else {
    if (use_per_atom_sink) {
      prepare_lammps_per_atom_virial_sink(atom_count, workspace);
    }
    switch (request.topology) {
      case ForceNeighborTopology::batched_multi_box:
        accumulate_radial_forces_batched(
            protocol,
            atom_count,
            model,
            workspace,
            virial_target);
        break;
      case ForceNeighborTopology::single_box_symmetric:
        accumulate_radial_forces_on_device(
            protocol,
            atom_count,
            box,
            model,
            workspace,
            virial_target,
            true);
        break;
      case ForceNeighborTopology::external_full: {
        ModelProtocol radial_protocol = protocol;
        radial_protocol.has_zbl = fuse_external_zbl;
        accumulate_lammps_radial_forces_on_device(
            radial_protocol,
            atom_count,
            box,
            model,
            workspace,
            virial_target,
            request.store_potential,
            fuse_structural_radial_into_spin);
        break;
      }
    }
  }
  timer.split(measured.radial_force_ms);

  if (has_angular) {
    if (request.topology == ForceNeighborTopology::batched_multi_box) {
      accumulate_l2_angular_forces_batched(
          protocol,
          atom_count,
          model,
          workspace,
          virial_target);
    } else {
      accumulate_l2_angular_forces_on_device(
          protocol,
          atom_count,
          model,
          workspace,
          virial_target);
    }
  }
  timer.split(measured.angular_force_ms);

  if (protocol.has_zbl && !fuse_radial_zbl && !fuse_external_zbl) {
    if (request.topology == ForceNeighborTopology::batched_multi_box) {
      accumulate_zbl_forces_batched(
          protocol,
          atom_count,
          model,
          workspace,
          select_zbl_virial_target(virial_target, zbl_outputs));
    } else {
      accumulate_zbl_forces_on_device(
          protocol,
          atom_count,
          box,
          model,
          workspace,
          zbl_outputs,
          select_zbl_virial_target(virial_target, zbl_outputs));
    }
  }
  timer.split(measured.zbl_force_ms);

  if (spin_model) {
    SpinForceTimings spin_timings;
    accumulate_spin_forces_on_device(
        protocol,
        atom_count,
        box,
        model,
        workspace,
        virial_target,
        request.spin_transfer_per_atom,
        fuse_structural_radial_into_spin,
        timings == nullptr ? nullptr : &spin_timings);
    measured.spin_onsite_ms = spin_timings.onsite_ms;
    measured.spin_density_ms = spin_timings.density_ms;
    measured.spin_chiral_ms = spin_timings.chiral_ms;
  }
}

}  // namespace

void run_force_pipeline(
    const ModelProtocol& protocol,
    const ForceEvaluationRequest& request,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    ForcePipelineTimings* timings) {
  require_supported_model(protocol);
  execute_force_pipeline(
      protocol, request, atom_count, box, model, workspace, timings);
}

}  // namespace nep_adapters::cuda_backend
