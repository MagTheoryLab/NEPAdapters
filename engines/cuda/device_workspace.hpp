#pragma once

#include "workspace_plan.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace nep_adapters::cuda_backend {

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
  double* potential = nullptr;
  double* force_soa3 = nullptr;
  double* virial_soa9 = nullptr;
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
  double* output_virials_per_atom_row_major9 = nullptr;
  double* structure_energy = nullptr;
  double* structure_virial_row_major9 = nullptr;
  int* active_atom_indices = nullptr;

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

  float* parameters_and_q_scaler = nullptr;
  float* fp = nullptr;
  float* charge_derivative = nullptr;
  float* descriptors = nullptr;
  float* sum_fxyz = nullptr;
  float* r12_radial = nullptr;
  float* fc_radial = nullptr;
  float* fn_radial = nullptr;
  float* r12_angular = nullptr;
  float* fc_angular = nullptr;
  float* fn_angular = nullptr;
  float* f12x = nullptr;
  float* f12y = nullptr;
  float* f12z = nullptr;
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

  void copy_int_to_device(const std::string& name, const std::vector<int>& host);
  void copy_double_to_device(
      const std::string& name,
      const std::vector<double>& host);
  void copy_float_to_device(
      const std::string& name,
      const std::vector<float>& host);

 private:
  struct Allocation {
    std::string name;
    ScalarType type = ScalarType::float32;
    std::size_t element_count = 0;
    void* device = nullptr;
  };

  Allocation* find_allocation(const std::string& name);
  const Allocation* find_allocation(const std::string& name) const;
  void release();

  std::vector<Allocation> allocations_;
  DeviceWorkspaceView view_{};
  DeviceWorkspaceSummary summary_{};
};

}  // namespace nep_adapters::cuda_backend
