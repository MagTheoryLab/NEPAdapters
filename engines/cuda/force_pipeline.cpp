#include "force_pipeline.hpp"

#include "device_operations.hpp"

#include <stdexcept>

#if defined(NEP_ADAPTERS_CUDA_DEVICE_RUNTIME)
#include <cuda_runtime.h>
#endif

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

#if defined(NEP_ADAPTERS_CUDA_DEVICE_RUNTIME)
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
#else
class PhaseTimer {
 public:
  explicit PhaseTimer(bool) {}
  void split(float&) {}
};
#endif

VirialTarget select_virial_target(
    const ForceEvaluationRequest& request,
    bool allow_float_sink) {
  switch (request.virial) {
    case VirialOutputMode::none:
      return VirialTarget::none;
    case VirialOutputMode::total_only:
      return VirialTarget::center_atom;
    case VirialOutputMode::per_atom_n2:
      if (allow_float_sink &&
          request.topology == ForceNeighborTopology::external_full) {
        return VirialTarget::neighbor_float_sink;
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

  const VirialTarget virial_target =
      select_virial_target(request, !spin_model);
  const bool use_per_atom_sink =
      virial_target == VirialTarget::neighbor_float_sink;
  const bool zbl_outputs =
      request.store_potential || accumulates_virial(virial_target);
  const bool fuse_radial_zbl =
      request.topology == ForceNeighborTopology::single_box_symmetric &&
      protocol.has_zbl && !zbl_outputs;
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
      case ForceNeighborTopology::external_full:
        accumulate_lammps_radial_forces_on_device(
            protocol,
            atom_count,
            box,
            model,
            workspace,
            virial_target);
        break;
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
  if (use_per_atom_sink) {
    finalize_lammps_per_atom_virial_sink(atom_count, workspace);
  }
  timer.split(measured.angular_force_ms);

  if (protocol.has_zbl && !fuse_radial_zbl) {
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
        accumulates_virial(virial_target),
        virial_targets_neighbor(virial_target),
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
