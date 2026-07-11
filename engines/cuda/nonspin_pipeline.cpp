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
    bool has_angular) {
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
    const NonSpinPipelineOptions& options,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool has_angular) {
  if (!options.orthorhombic_batched && has_angular) {
    build_pair_geometry_cache_batched(protocol, atom_count, workspace);
    build_radial_basis_cache_on_device(protocol, atom_count, workspace);
  } else if (!options.orthorhombic_batched) {
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

}  // namespace

void run_nonspin_pipeline(
    const ModelProtocol& protocol,
    const NonSpinPipelineOptions& options,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    NonSpinPipelineTimings* timings) {
  require_ordinary_model(protocol);
  const bool has_angular = has_angular_terms(protocol);
  NonSpinPipelineTimings ignored_timings;
  NonSpinPipelineTimings& measured =
      timings == nullptr ? ignored_timings : *timings;
  PhaseTimer timer(timings != nullptr);

  if (options.topology == NonSpinNeighborTopology::batched_multi_box) {
    build_batched_descriptors(
        protocol, options, atom_count, model, workspace, has_angular);
  } else if (
      has_angular && protocol.descriptor_dim <= 64 &&
      try_build_descriptors_and_ann_from_positions_on_device(
          protocol,
          atom_count,
          box,
          model,
          workspace,
          options.store_potential)) {
    // Fused descriptor/ANN path completed the phase.
  } else {
    build_staged_position_descriptors(
        protocol, atom_count, box, model, workspace, has_angular);
  }
  timer.split(measured.descriptor_ann_ms);

  switch (options.topology) {
    case NonSpinNeighborTopology::batched_multi_box:
      accumulate_radial_forces_batched(protocol, atom_count, model, workspace);
      break;
    case NonSpinNeighborTopology::single_box_symmetric:
      accumulate_radial_forces_on_device(
          protocol,
          atom_count,
          box,
          model,
          workspace,
          options.accumulate_virial);
      break;
    case NonSpinNeighborTopology::external_full:
      accumulate_lammps_radial_forces_on_device(
          protocol,
          atom_count,
          box,
          model,
          workspace,
          options.accumulate_virial);
      break;
  }
  timer.split(measured.radial_force_ms);

  if (has_angular) {
    if (options.topology == NonSpinNeighborTopology::batched_multi_box) {
      accumulate_l2_angular_forces_batched(protocol, atom_count, model, workspace);
    } else {
      accumulate_l2_angular_forces_on_device(
          protocol,
          atom_count,
          box,
          model,
          workspace,
          options.accumulate_virial);
    }
  }
  timer.split(measured.angular_force_ms);

  if (protocol.has_zbl) {
    if (options.topology == NonSpinNeighborTopology::batched_multi_box) {
      accumulate_zbl_forces_batched(protocol, atom_count, model, workspace);
    } else {
      accumulate_zbl_forces_on_device(
          protocol,
          atom_count,
          box,
          model,
          workspace,
          options.zbl_outputs);
    }
  }
  timer.split(measured.zbl_force_ms);
}

}  // namespace nep_adapters::cuda_backend
