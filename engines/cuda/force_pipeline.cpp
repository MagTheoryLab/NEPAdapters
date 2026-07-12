#include "force_pipeline.hpp"

#include "device_operations.hpp"

#include <stdexcept>

#if defined(NEP_ADAPTERS_CUDA_DEVICE_RUNTIME)
#include <cuda_runtime.h>
#endif

namespace nep_adapters::cuda_backend {
namespace {

void require_ordinary_model(const ModelProtocol& protocol) {
  if (protocol.spin_mode != 0 || protocol.charge_mode != 0) {
    throw std::invalid_argument(
        "non-spin execution pipeline accepts ordinary models only");
  }
}

bool has_angular_terms(const ModelProtocol& protocol) {
  return protocol.body_channels.channel_count() > 0;
}

void build_staged_position_descriptors(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool has_angular,
    bool spin_model) {
  if (has_angular) {
    build_angular_geometry_cache_on_device(protocol, atom_count, box, workspace);
  }
  if (!build_radial_descriptors_from_geometry_on_device(
          protocol, atom_count, box, model, workspace)) {
    if (has_angular) {
      build_pair_geometry_cache_on_device(protocol, atom_count, box, workspace);
      build_radial_basis_cache_on_device(protocol, atom_count, workspace);
    } else {
      build_radial_geometry_basis_cache_on_device(
          protocol, atom_count, box, workspace);
    }
    build_radial_descriptors_on_device(protocol, atom_count, model, workspace);
  }
  if (spin_model) {
    build_spin_descriptors_on_device(
        protocol, atom_count, box, model, workspace);
  }
  if (has_angular) {
    if (try_build_angular_descriptors_and_ann_from_geometry_on_device(
            protocol, atom_count, model, workspace)) {
      return;
    }
    build_angular_descriptors_from_geometry_on_device(
        protocol, atom_count, model, workspace);
  }
  evaluate_ann_energy_on_device(protocol, atom_count, model, workspace);
}

void build_batched_descriptors(
    const ModelProtocol& protocol,
    const ForceEvaluationRequest& request,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool has_angular) {
  if (!request.orthorhombic_batched && has_angular) {
    build_pair_geometry_cache_batched(protocol, atom_count, workspace);
    build_radial_basis_cache_on_device(protocol, atom_count, workspace);
  } else if (!request.orthorhombic_batched) {
    build_radial_geometry_basis_cache_batched(protocol, atom_count, workspace);
  } else {
    build_radial_basis_cache_on_device(protocol, atom_count, workspace);
  }
  build_radial_descriptors_on_device(protocol, atom_count, model, workspace);
  if (has_angular &&
      try_build_angular_descriptors_and_ann_from_geometry_on_device(
          protocol, atom_count, model, workspace)) {
    return;
  }
  if (has_angular) {
    build_angular_descriptors_from_geometry_on_device(
        protocol, atom_count, model, workspace);
  }
  evaluate_ann_energy_on_device(protocol, atom_count, model, workspace);
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

void run_force_pipeline(
    const ModelProtocol& protocol,
    const ForceEvaluationRequest& request,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool spin_model,
    ForcePipelineTimings* timings) {
  const bool has_angular = has_angular_terms(protocol);
  ForcePipelineTimings ignored_timings;
  ForcePipelineTimings& measured =
      timings == nullptr ? ignored_timings : *timings;
  PhaseTimer timer(timings != nullptr);

  if (request.topology == ForceNeighborTopology::batched_multi_box && spin_model) {
    throw std::invalid_argument(
        "spin execution pipeline does not support batched neighbor topology");
  }

  DescriptorCoreOptions descriptor_options;
  descriptor_options.topology =
      request.topology == ForceNeighborTopology::batched_multi_box
          ? DescriptorCoreTopology::batched_multi_box
          : DescriptorCoreTopology::single_box;
  descriptor_options.has_angular = has_angular;
  descriptor_options.store_potential = request.store_potential;

  bool descriptor_phase_done = false;
  if (!spin_model) {
    descriptor_options.output =
        DescriptorCoreOutput::ann_energy_and_derivatives;
    descriptor_phase_done = try_build_descriptor_core_from_positions_on_device(
        protocol, atom_count, box, model, workspace, descriptor_options);
  }
  if (!descriptor_phase_done) {
    descriptor_options.output = DescriptorCoreOutput::structural_descriptors;
    if (try_build_descriptor_core_from_positions_on_device(
            protocol, atom_count, box, model, workspace, descriptor_options)) {
      if (spin_model) {
        build_spin_descriptors_on_device(
            protocol, atom_count, box, model, workspace);
      }
      evaluate_ann_energy_on_device(protocol, atom_count, model, workspace);
    } else if (request.topology == ForceNeighborTopology::batched_multi_box) {
      build_batched_descriptors(
          protocol, request, atom_count, model, workspace, has_angular);
    } else {
      build_staged_position_descriptors(
          protocol, atom_count, box, model, workspace, has_angular, spin_model);
    }
  }
  timer.split(measured.descriptor_ann_ms);

  const bool accumulate_virial = requests_virial(request);
  const bool virial_to_neighbor = requests_per_atom_virial(request);
  const bool use_per_atom_sink = !spin_model && virial_to_neighbor &&
      request.topology == ForceNeighborTopology::external_full;
  const bool zbl_outputs = request.store_potential || accumulate_virial;
  const bool fuse_radial_zbl =
      request.topology == ForceNeighborTopology::single_box_symmetric &&
      protocol.has_zbl && !zbl_outputs && !virial_to_neighbor;
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
            virial_to_neighbor);
        break;
      case ForceNeighborTopology::single_box_symmetric:
        accumulate_radial_forces_on_device(
            protocol,
            atom_count,
            box,
            model,
            workspace,
            accumulate_virial,
            true,
            virial_to_neighbor);
        break;
      case ForceNeighborTopology::external_full:
        if (use_per_atom_sink) {
          accumulate_lammps_radial_forces_to_per_atom_sink(
              protocol, atom_count, box, model, workspace);
        } else {
          accumulate_lammps_radial_forces_on_device(
              protocol,
              atom_count,
              box,
              model,
              workspace,
              accumulate_virial,
              virial_to_neighbor);
        }
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
          virial_to_neighbor);
    } else if (use_per_atom_sink) {
      accumulate_l2_angular_forces_to_per_atom_sink(
          protocol, atom_count, box, model, workspace);
    } else {
      accumulate_l2_angular_forces_on_device(
          protocol,
          atom_count,
          box,
          model,
          workspace,
          accumulate_virial,
          virial_to_neighbor);
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
          virial_to_neighbor);
    } else {
      accumulate_zbl_forces_on_device(
          protocol,
          atom_count,
          box,
          model,
          workspace,
          zbl_outputs,
          virial_to_neighbor);
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
        accumulate_virial,
        virial_to_neighbor,
        timings == nullptr ? nullptr : &spin_timings);
    measured.spin_onsite_ms = spin_timings.onsite_ms;
    measured.spin_scalar_ms = spin_timings.scalar_ms;
    measured.spin_density_ms = spin_timings.density_ms;
    measured.spin_chiral_ms = spin_timings.chiral_ms;
  }
}

}  // namespace

void run_nonspin_pipeline(
    const ModelProtocol& protocol,
    const ForceEvaluationRequest& request,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    ForcePipelineTimings* timings) {
  require_ordinary_model(protocol);
  run_force_pipeline(
      protocol, request, atom_count, box, model, workspace, false, timings);
}

void run_spin_pipeline(
    const ModelProtocol& protocol,
    const ForceEvaluationRequest& request,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    ForcePipelineTimings* timings) {
  if (protocol.spin_mode == 0 || protocol.charge_mode != 0) {
    throw std::invalid_argument(
        "spin execution pipeline accepts spin models without charge only");
  }
  run_force_pipeline(
      protocol, request, atom_count, box, model, workspace, true, timings);
}

}  // namespace nep_adapters::cuda_backend
