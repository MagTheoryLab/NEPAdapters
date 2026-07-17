#pragma once

#include "model_protocol.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace nep_adapters::cuda_backend {

inline constexpr int kTypeScheduleWindowAtoms = 1024;

enum class ScalarType {
  int32,
  float32,
  float64,
};

enum class NeighborSource {
  internal,
  external,
};

struct DeviceArrayPlan {
  std::string name;
  ScalarType type = ScalarType::float32;
  std::size_t element_count = 0;

  std::size_t bytes() const;
};

struct WorkspacePlan {
  NeighborSource neighbor_source = NeighborSource::internal;
  std::size_t atom_capacity = 0;
  std::size_t active_atom_capacity = 0;
  std::size_t structure_capacity = 0;
  std::vector<DeviceArrayPlan> arrays;

  std::size_t total_bytes() const;
  const DeviceArrayPlan* find_array(const std::string& name) const;
};

WorkspacePlan make_internal_neighbor_workspace_plan(
    const ModelProtocol& protocol,
    std::size_t atom_capacity,
    std::size_t structure_capacity,
    bool include_spin_transfer = false);

WorkspacePlan make_external_neighbor_workspace_plan(
    const ModelProtocol& protocol,
    std::size_t atom_capacity,
    std::size_t active_atom_capacity,
    bool include_per_atom_virial_sink = false,
    bool include_spin_transfer = false);

struct DeviceWorkspaceSummary {
  std::size_t total_bytes = 0;
  std::size_t array_count = 0;
};

struct DeviceWorkspaceView {
  int neighbor_source = 0;
  std::size_t atom_capacity = 0;
  std::size_t active_atom_capacity = 0;
  std::size_t structure_capacity = 0;

  int* types = nullptr;
  double* positions_soa3 = nullptr;
  double* spins_soa3 = nullptr;
  double* potential = nullptr;
  double* force_soa3 = nullptr;
  double* mforce_soa3 = nullptr;
  double* virial_soa9 = nullptr;
  float* spin_transfer_soa9 = nullptr;
  double* charge = nullptr;
  double* bec_soa9 = nullptr;
  float* d_real = nullptr;
  double* lammps_partial_sums = nullptr;

  int* atom_to_structure = nullptr;
  int* structure_atom_counts = nullptr;
  int* structure_atom_offsets = nullptr;
  double* boxes_row_major9 = nullptr;
  double* box_inverse_row_major9 = nullptr;
  int* pbc_flags3 = nullptr;
  double* output_forces_aos3 = nullptr;
  double* output_mforces_aos3 = nullptr;
  double* output_virials_per_atom_row_major9 = nullptr;
  double* structure_energy = nullptr;
  double* structure_virial_row_major9 = nullptr;
  int* active_atom_indices = nullptr;

  int* type_scheduled_atoms = nullptr;
  int* type_schedule_active_type_counts = nullptr;

  int* nn_radial = nullptr;
  int* nl_radial_slot_major = nullptr;
  int* nn_angular = nullptr;
  int* nl_angular_slot_major = nullptr;

  int* cell_counts = nullptr;
  int* cell_offsets = nullptr;
  int* cell_fill = nullptr;
  int* cell_atoms = nullptr;
  int* atom_cell = nullptr;
  int* cell_dims = nullptr;
  int* structure_cell_offsets = nullptr;
  int* structure_cell_dims4 = nullptr;
  int* neighbor_overflow = nullptr;

  float* fp = nullptr;
  float* charge_derivative = nullptr;
  float* descriptors = nullptr;
  float* sum_fxyz = nullptr;
  float* r12_angular = nullptr;
  float* f12x = nullptr;
  float* f12y = nullptr;
  float* f12z = nullptr;
  float* per_atom_virial_float_soa9 = nullptr;
  float* spin_density_rho0 = nullptr;
  float* spin_density_raw1 = nullptr;
  float* spin_density_angular2 = nullptr;
  float* spin_density_angular3 = nullptr;
  float* spin_density_angular4 = nullptr;
  float* spin_density_geom = nullptr;
  float* spin_density_rho0_dot = nullptr;
  float* spin_density_raw1_dot = nullptr;
  float* spin_chiral_polar = nullptr;
  float* spin_chiral_octupoles_raw = nullptr;
  float* spin_chiral_hexadecapoles_raw = nullptr;
  float* spin_chiral_chirals = nullptr;
};

class DeviceWorkspace {
 public:
  explicit DeviceWorkspace(const WorkspacePlan& plan);
  ~DeviceWorkspace();

  DeviceWorkspace(const DeviceWorkspace&) = delete;
  DeviceWorkspace& operator=(const DeviceWorkspace&) = delete;
  DeviceWorkspace(DeviceWorkspace&& other) noexcept;
  DeviceWorkspace& operator=(DeviceWorkspace&& other) noexcept;

  DeviceWorkspaceView view() const;
  DeviceWorkspaceSummary summary() const;

  void reset_runtime_overrides();
  void override_types(int* device_types);
  void override_positions_soa3(double* device_positions_soa3);

 private:
  struct Allocation {
    std::string name;
    ScalarType type = ScalarType::float32;
    std::size_t element_count = 0;
    void* device = nullptr;
    bool async_allocated = false;
  };

  const Allocation* find_allocation(const std::string& name) const;
  void release();

  std::vector<Allocation> allocations_;
  DeviceWorkspaceView view_{};
  DeviceWorkspaceSummary summary_{};
};

}  // namespace nep_adapters::cuda_backend
