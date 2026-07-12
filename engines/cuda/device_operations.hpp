#pragma once

#include "device_model.hpp"
#include "device_workspace.hpp"
#include "model_protocol.hpp"
#include "simulation_box.hpp"

#include "nep_adapters/api.h"

namespace nep_adapters::cuda_backend {

// Internal device-operation interface used by CUDA orchestration and focused
// kernel tests. The public engine interface remains outside engines/cuda.

struct LammpsDeviceNeighborCounts {
  int max_radial = 0;
  int max_angular = 0;
};

void stage_batch_on_device(
    const NepaStructureBatch& batch,
    DeviceWorkspace& workspace);

void stage_lammps_external_neighbors_on_device(
    const NepaLammpsNeighborInput& input,
    const ModelProtocol& protocol,
    DeviceWorkspace& workspace);

LammpsDeviceNeighborCounts stage_lammps_device_neighbors_on_device(
    const NepaLammpsDeviceNeighborInput& input,
    const ModelProtocol& protocol,
    DeviceWorkspace& workspace,
    bool check_overflow = true);

LammpsDeviceNeighborCounts count_lammps_device_neighbors_on_device(
    const NepaLammpsDeviceNeighborInput& input,
    const ModelProtocol& protocol);

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

void build_pair_geometry_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace);

void build_angular_geometry_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace);

void build_pair_geometry_cache_batched(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace);

void build_radial_basis_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace);

void build_radial_geometry_basis_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace);

void build_radial_geometry_basis_cache_batched(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace);

void build_angular_basis_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace);

void build_radial_descriptors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

bool build_radial_descriptors_from_geometry_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void build_angular_descriptors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void build_angular_descriptors_from_geometry_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

bool try_build_angular_descriptors_and_ann_from_geometry_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

bool try_build_descriptors_and_ann_from_positions_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool store_potential = true);

void evaluate_ann_energy_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
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
    bool accumulate_virial = true,
    bool clear_outputs = true,
    bool virial_to_neighbor = false);

void accumulate_lammps_radial_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial = true);

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
    DeviceWorkspace& workspace);

void accumulate_l2_angular_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial = true,
    bool virial_to_neighbor = false);

void accumulate_l2_angular_forces_batched(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace);

void accumulate_zbl_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_energy_virial = true);

void accumulate_zbl_forces_batched(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
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
  float scalar_ms = 0.0f;
  float density_ms = 0.0f;
  float chiral_ms = 0.0f;
};

void accumulate_spin_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial,
    SpinForceTimings* timings = nullptr);

}  // namespace nep_adapters::cuda_backend
