#pragma once

#include "device_model.hpp"
#include "device_workspace.hpp"
#include "model_protocol.hpp"
#include "simulation_box.hpp"

#include "nep_adapters/api.h"

namespace nep_adapters::cuda_backend {

// Internal device-operation interface used by CUDA orchestration and focused
// kernel tests. The public engine interface remains outside engines/cuda.

// Internal virial placement used by force launchers. Model/output decisions
// are made by force_pipeline; kernels only receive the placement they need.
enum class VirialTarget {
  none,
  center_atom,
  neighbor_atom,
  center_and_neighbor_float_sink,
};

constexpr bool accumulates_virial(VirialTarget target) {
  return target != VirialTarget::none;
}

constexpr bool virial_targets_neighbor(VirialTarget target) {
  return target == VirialTarget::neighbor_atom ||
         target == VirialTarget::center_and_neighbor_float_sink;
}

#if defined(__CUDACC__)
__device__ __forceinline__ int scheduled_atom_index(
    int grouped_atom,
    const int* scheduled_atoms,
    const int* schedule_active_type_counts,
    int max_active_types_without_grouping) {
  const bool use_schedule =
      schedule_active_type_counts != nullptr &&
      schedule_active_type_counts[
          grouped_atom / kTypeScheduleWindowAtoms] >
          max_active_types_without_grouping;
  return use_schedule ? scheduled_atoms[grouped_atom] : grouped_atom;
}

template <typename T>
__device__ __forceinline__ T warp_sum_equal_atom(
    unsigned peer_mask,
    T value) {
  const int lane = threadIdx.x & 31;
  unsigned remaining = peer_mask & ~(1u << lane);
  T sum = value;
  while (remaining != 0u) {
    const int source_lane = __ffs(static_cast<int>(remaining)) - 1;
    sum += __shfl_sync(peer_mask, value, source_lane);
    remaining &= remaining - 1u;
  }
  return sum;
}

__device__ __forceinline__ void atomic_add_per_atom_virial_float(
    int atom_stride,
    int atom,
    float x12,
    float y12,
    float z12,
    float fx,
    float fy,
    float fz,
    float* virial_soa9) {
  atomicAdd(&virial_soa9[atom], -x12 * fx);
  atomicAdd(&virial_soa9[atom_stride + atom], -y12 * fy);
  atomicAdd(&virial_soa9[2 * atom_stride + atom], -z12 * fz);
  atomicAdd(&virial_soa9[3 * atom_stride + atom], -x12 * fy);
  atomicAdd(&virial_soa9[4 * atom_stride + atom], -x12 * fz);
  atomicAdd(&virial_soa9[5 * atom_stride + atom], -y12 * fz);
  atomicAdd(&virial_soa9[6 * atom_stride + atom], -y12 * fx);
  atomicAdd(&virial_soa9[7 * atom_stride + atom], -z12 * fx);
  atomicAdd(&virial_soa9[8 * atom_stride + atom], -z12 * fy);
}

__device__ __forceinline__ void atomic_add_per_atom_virial_double(
    int atom_stride,
    int atom,
    float x12,
    float y12,
    float z12,
    float fx,
    float fy,
    float fz,
    double* virial_soa9) {
  atomicAdd(&virial_soa9[atom], -static_cast<double>(x12 * fx));
  atomicAdd(
      &virial_soa9[atom_stride + atom], -static_cast<double>(y12 * fy));
  atomicAdd(
      &virial_soa9[2 * atom_stride + atom], -static_cast<double>(z12 * fz));
  atomicAdd(
      &virial_soa9[3 * atom_stride + atom], -static_cast<double>(x12 * fy));
  atomicAdd(
      &virial_soa9[4 * atom_stride + atom], -static_cast<double>(x12 * fz));
  atomicAdd(
      &virial_soa9[5 * atom_stride + atom], -static_cast<double>(y12 * fz));
  atomicAdd(
      &virial_soa9[6 * atom_stride + atom], -static_cast<double>(y12 * fx));
  atomicAdd(
      &virial_soa9[7 * atom_stride + atom], -static_cast<double>(z12 * fx));
  atomicAdd(
      &virial_soa9[8 * atom_stride + atom], -static_cast<double>(z12 * fy));
}

__device__ __forceinline__ void atomic_add_spin_transfer_float(
    int atom_stride,
    int atom,
    float x12,
    float y12,
    float z12,
    const float* grad_sj,
    float* spin_transfer_soa9) {
  const float rij[3] = {x12, y12, z12};
#pragma unroll
  for (int row = 0; row < 3; ++row) {
#pragma unroll
    for (int spin_component = 0; spin_component < 3; ++spin_component) {
      atomicAdd(
          spin_transfer_soa9 +
              (3 * row + spin_component) * atom_stride + atom,
          -rij[row] * grad_sj[spin_component]);
    }
  }
}

__device__ __forceinline__ void
atomic_add_force_and_per_atom_virial_float_warp_aggregated(
    unsigned active_mask,
    int atom_stride,
    int atom,
    float x12,
    float y12,
    float z12,
    float fx,
    float fy,
    float fz,
    double* force_soa3,
    float* virial_soa9) {
  const unsigned peer_mask = __match_any_sync(active_mask, atom);
  const int lane = threadIdx.x & 31;
  const int leader = __ffs(static_cast<int>(peer_mask)) - 1;
  if ((peer_mask & (peer_mask - 1u)) == 0u) {
    atomicAdd(&force_soa3[atom], -static_cast<double>(fx));
    atomicAdd(
        &force_soa3[atom_stride + atom], -static_cast<double>(fy));
    atomicAdd(
        &force_soa3[2 * atom_stride + atom], -static_cast<double>(fz));
    atomic_add_per_atom_virial_float(
        atom_stride,
        atom,
        x12,
        y12,
        z12,
        fx,
        fy,
        fz,
        virial_soa9);
    return;
  }

  const double force_x = warp_sum_equal_atom(
      peer_mask, -static_cast<double>(fx));
  const double force_y = warp_sum_equal_atom(
      peer_mask, -static_cast<double>(fy));
  const double force_z = warp_sum_equal_atom(
      peer_mask, -static_cast<double>(fz));
  const float virial_xx = warp_sum_equal_atom(peer_mask, -x12 * fx);
  const float virial_yy = warp_sum_equal_atom(peer_mask, -y12 * fy);
  const float virial_zz = warp_sum_equal_atom(peer_mask, -z12 * fz);
  const float virial_xy = warp_sum_equal_atom(peer_mask, -x12 * fy);
  const float virial_xz = warp_sum_equal_atom(peer_mask, -x12 * fz);
  const float virial_yz = warp_sum_equal_atom(peer_mask, -y12 * fz);
  const float virial_yx = warp_sum_equal_atom(peer_mask, -y12 * fx);
  const float virial_zx = warp_sum_equal_atom(peer_mask, -z12 * fx);
  const float virial_zy = warp_sum_equal_atom(peer_mask, -z12 * fy);
  if (lane != leader) {
    return;
  }

  atomicAdd(&force_soa3[atom], force_x);
  atomicAdd(&force_soa3[atom_stride + atom], force_y);
  atomicAdd(&force_soa3[2 * atom_stride + atom], force_z);
  atomicAdd(&virial_soa9[atom], virial_xx);
  atomicAdd(&virial_soa9[atom_stride + atom], virial_yy);
  atomicAdd(&virial_soa9[2 * atom_stride + atom], virial_zz);
  atomicAdd(&virial_soa9[3 * atom_stride + atom], virial_xy);
  atomicAdd(&virial_soa9[4 * atom_stride + atom], virial_xz);
  atomicAdd(&virial_soa9[5 * atom_stride + atom], virial_yz);
  atomicAdd(&virial_soa9[6 * atom_stride + atom], virial_yx);
  atomicAdd(&virial_soa9[7 * atom_stride + atom], virial_zx);
  atomicAdd(&virial_soa9[8 * atom_stride + atom], virial_zy);
}
#endif

void stage_batch_on_device(
    const NepaStructureBatch& batch,
    DeviceWorkspace& workspace);

void stage_lammps_external_neighbors_on_device(
    const NepaLammpsNeighborInput& input,
    const ModelProtocol& protocol,
    DeviceWorkspace& workspace);

void stage_lammps_device_neighbors_on_device(
    const NepaLammpsDeviceNeighborInput& input,
    const ModelProtocol& protocol,
    DeviceWorkspace& workspace,
    bool check_overflow = true);

void build_internal_neighbors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace);

void build_internal_neighbors_batched(
    const ModelProtocol& protocol,
    int structure_count,
    int atom_count,
    DeviceWorkspace& workspace,
    bool orthorhombic_fast_path = false);

void prepare_batched_outputs(
    int structure_count,
    int atom_count,
    DeviceWorkspace& workspace);

void write_lammps_device_outputs(
    const NepaLammpsDeviceNeighborInput& input,
    const NepaLammpsDeviceNeighborResult& result,
    DeviceWorkspace& workspace);

enum class DescriptorCoreTopology {
  single_box,
  batched_multi_box,
};

// Builds only the shared structural descriptor. ANN and model-family-specific
// descriptor stages remain explicit in the force-pipeline implementation.
void build_descriptor_core_from_positions_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    DescriptorCoreTopology topology);

void evaluate_ann_energy_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void build_atom_type_schedule_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace);

void evaluate_qnep_ann_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void zero_total_charge_on_device(
    int atom_count,
    DeviceWorkspace& workspace);

void add_charge_chain_to_fp_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace);

void accumulate_radial_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    VirialTarget virial_target = VirialTarget::center_atom,
    bool clear_outputs = true);

void accumulate_lammps_radial_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    VirialTarget virial_target,
    bool accumulate_zbl_energy);

void accumulate_radial_and_zbl_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void accumulate_radial_forces_batched(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    VirialTarget virial_target = VirialTarget::center_atom);

void accumulate_l2_angular_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    VirialTarget virial_target = VirialTarget::center_atom);

void accumulate_l2_angular_forces_batched(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    VirialTarget virial_target = VirialTarget::center_atom);

void accumulate_zbl_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_energy_virial = true,
    VirialTarget virial_target = VirialTarget::center_atom);

void accumulate_zbl_forces_batched(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    VirialTarget virial_target = VirialTarget::center_atom);

void prepare_lammps_per_atom_virial_sink(
    int atom_count,
    DeviceWorkspace& workspace);

void apply_qnep_charge_terms_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace,
    bool need_per_atom_kspace_potential = true,
    bool need_per_atom_kspace_virial = true);

void build_spin_descriptors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

struct SpinForceTimings {
  float onsite_ms = 0.0f;
  float density_ms = 0.0f;
  float chiral_ms = 0.0f;
};

void accumulate_spin_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    VirialTarget virial_target,
    bool accumulate_spin_transfer,
    SpinForceTimings* timings = nullptr);

}  // namespace nep_adapters::cuda_backend
