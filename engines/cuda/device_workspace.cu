#include "device_workspace.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace nep_adapters::cuda_backend {
namespace {

void check_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(action) + ": " + cudaGetErrorString(status));
  }
}

void set_view_pointer(
    DeviceWorkspaceView& view,
    const std::string& name,
    void* device) {
  if (name == "types") {
    view.types = static_cast<int*>(device);
  } else if (name == "positions_soa3") {
    view.positions_soa3 = static_cast<double*>(device);
  } else if (name == "spins_soa3") {
    view.spins_soa3 = static_cast<double*>(device);
  } else if (name == "potential") {
    view.potential = static_cast<double*>(device);
  } else if (name == "force_soa3") {
    view.force_soa3 = static_cast<double*>(device);
  } else if (name == "mforce_soa3") {
    view.mforce_soa3 = static_cast<double*>(device);
  } else if (name == "virial_soa9") {
    view.virial_soa9 = static_cast<double*>(device);
  } else if (name == "charge") {
    view.charge = static_cast<double*>(device);
  } else if (name == "bec_soa9") {
    view.bec_soa9 = static_cast<double*>(device);
  } else if (name == "lammps_partial_sums") {
    view.lammps_partial_sums = static_cast<double*>(device);
  } else if (name == "atom_to_structure") {
    view.atom_to_structure = static_cast<int*>(device);
  } else if (name == "structure_atom_counts") {
    view.structure_atom_counts = static_cast<int*>(device);
  } else if (name == "structure_atom_offsets") {
    view.structure_atom_offsets = static_cast<int*>(device);
  } else if (name == "boxes_row_major9") {
    view.boxes_row_major9 = static_cast<double*>(device);
  } else if (name == "box_inverse_row_major9") {
    view.box_inverse_row_major9 = static_cast<double*>(device);
  } else if (name == "pbc_flags3") {
    view.pbc_flags3 = static_cast<int*>(device);
  } else if (name == "output_forces_aos3") {
    view.output_forces_aos3 = static_cast<double*>(device);
  } else if (name == "output_mforces_aos3") {
    view.output_mforces_aos3 = static_cast<double*>(device);
  } else if (name == "output_virials_per_atom_row_major9") {
    view.output_virials_per_atom_row_major9 = static_cast<double*>(device);
  } else if (name == "structure_energy") {
    view.structure_energy = static_cast<double*>(device);
  } else if (name == "structure_virial_row_major9") {
    view.structure_virial_row_major9 = static_cast<double*>(device);
  } else if (name == "active_atom_indices") {
    view.active_atom_indices = static_cast<int*>(device);
  } else if (name == "nn_radial") {
    view.nn_radial = static_cast<int*>(device);
  } else if (name == "nl_radial_slot_major") {
    view.nl_radial_slot_major = static_cast<int*>(device);
  } else if (name == "nn_angular") {
    view.nn_angular = static_cast<int*>(device);
  } else if (name == "nl_angular_slot_major") {
    view.nl_angular_slot_major = static_cast<int*>(device);
  } else if (name == "cell_counts") {
    view.cell_counts = static_cast<int*>(device);
  } else if (name == "cell_offsets") {
    view.cell_offsets = static_cast<int*>(device);
  } else if (name == "cell_fill") {
    view.cell_fill = static_cast<int*>(device);
  } else if (name == "cell_atoms") {
    view.cell_atoms = static_cast<int*>(device);
  } else if (name == "atom_cell") {
    view.atom_cell = static_cast<int*>(device);
  } else if (name == "cell_dims") {
    view.cell_dims = static_cast<int*>(device);
  } else if (name == "structure_cell_offsets") {
    view.structure_cell_offsets = static_cast<int*>(device);
  } else if (name == "structure_cell_dims4") {
    view.structure_cell_dims4 = static_cast<int*>(device);
  } else if (name == "neighbor_overflow") {
    view.neighbor_overflow = static_cast<int*>(device);
  } else if (name == "parameters_and_q_scaler") {
    view.parameters_and_q_scaler = static_cast<float*>(device);
  } else if (name == "fp") {
    view.fp = static_cast<float*>(device);
  } else if (name == "ann_hidden_values") {
    view.ann_hidden_values = static_cast<float*>(device);
  } else if (name == "ann_hidden_delta") {
    view.ann_hidden_delta = static_cast<float*>(device);
  } else if (name == "charge_derivative") {
    view.charge_derivative = static_cast<float*>(device);
  } else if (name == "descriptors") {
    view.descriptors = static_cast<float*>(device);
  } else if (name == "d_real") {
    view.d_real = static_cast<float*>(device);
  } else if (name == "sum_fxyz") {
    view.sum_fxyz = static_cast<float*>(device);
  } else if (name == "r12_radial") {
    view.r12_radial = static_cast<float*>(device);
  } else if (name == "fc_radial") {
    view.fc_radial = static_cast<float*>(device);
  } else if (name == "fn_radial") {
    view.fn_radial = static_cast<float*>(device);
  } else if (name == "r12_angular") {
    view.r12_angular = static_cast<float*>(device);
  } else if (name == "fc_angular") {
    view.fc_angular = static_cast<float*>(device);
  } else if (name == "fn_angular") {
    view.fn_angular = static_cast<float*>(device);
  } else if (name == "f12x") {
    view.f12x = static_cast<float*>(device);
  } else if (name == "f12y") {
    view.f12y = static_cast<float*>(device);
  } else if (name == "f12z") {
    view.f12z = static_cast<float*>(device);
  } else if (name == "per_atom_virial_float_soa9") {
    view.per_atom_virial_float_soa9 = static_cast<float*>(device);
  } else if (name == "spin_density_rho0") {
    view.spin_density_rho0 = static_cast<float*>(device);
  } else if (name == "spin_density_raw1") {
    view.spin_density_raw1 = static_cast<float*>(device);
  } else if (name == "spin_density_l1_rdot") {
    view.spin_density_l1_rdot = static_cast<float*>(device);
  } else if (name == "spin_density_l1_cross") {
    view.spin_density_l1_cross = static_cast<float*>(device);
  } else if (name == "spin_density_l1_stf") {
    view.spin_density_l1_stf = static_cast<float*>(device);
  } else if (name == "spin_density_angular2") {
    view.spin_density_angular2 = static_cast<float*>(device);
  } else if (name == "spin_density_angular3") {
    view.spin_density_angular3 = static_cast<float*>(device);
  } else if (name == "spin_density_angular4") {
    view.spin_density_angular4 = static_cast<float*>(device);
  } else if (name == "spin_density_geom") {
    view.spin_density_geom = static_cast<float*>(device);
  } else if (name == "spin_density_rho0_dot") {
    view.spin_density_rho0_dot = static_cast<float*>(device);
  } else if (name == "spin_density_raw1_dot") {
    view.spin_density_raw1_dot = static_cast<float*>(device);
  } else if (name == "spin_edge_dx") {
    view.spin_edge_dx = static_cast<float*>(device);
  } else if (name == "spin_edge_dy") {
    view.spin_edge_dy = static_cast<float*>(device);
  } else if (name == "spin_edge_dz") {
    view.spin_edge_dz = static_cast<float*>(device);
  } else if (name == "spin_edge_dist") {
    view.spin_edge_dist = static_cast<float*>(device);
  } else if (name == "spin_edge_weights") {
    view.spin_edge_weights = static_cast<float*>(device);
  } else if (name == "spin_edge_weight_derivatives") {
    view.spin_edge_weight_derivatives = static_cast<float*>(device);
  } else if (name == "spin_chiral_polar") {
    view.spin_chiral_polar = static_cast<float*>(device);
  } else if (name == "spin_chiral_octupoles_raw") {
    view.spin_chiral_octupoles_raw = static_cast<float*>(device);
  } else if (name == "spin_chiral_hexadecapoles_raw") {
    view.spin_chiral_hexadecapoles_raw = static_cast<float*>(device);
  } else if (name == "spin_chiral_chirals") {
    view.spin_chiral_chirals = static_cast<float*>(device);
  } else if (name == "spin_chiral_pseudodevs") {
    view.spin_chiral_pseudodevs = static_cast<float*>(device);
  }
}

int encode_neighbor_source(NeighborSource source) {
  return source == NeighborSource::external ? 1 : 0;
}

}  // namespace

DeviceWorkspace::DeviceWorkspace(const WorkspacePlan& plan) {
  view_.neighbor_source = encode_neighbor_source(plan.neighbor_source);
  view_.atom_capacity = plan.atom_capacity;
  view_.active_atom_capacity = plan.active_atom_capacity;
  view_.structure_capacity = plan.structure_capacity;
  summary_.array_count = plan.arrays.size();

  try {
    allocations_.reserve(plan.arrays.size());
    for (const DeviceArrayPlan& array : plan.arrays) {
      Allocation allocation;
      allocation.name = array.name;
      allocation.type = array.type;
      allocation.element_count = array.element_count;

      const std::size_t bytes = array.bytes();
      if (bytes > 0) {
        cudaError_t status = cudaMallocAsync(&allocation.device, bytes, 0);
        if (status == cudaSuccess) {
          allocation.async_allocated = true;
        } else {
          cudaGetLastError();
          check_cuda(cudaMalloc(&allocation.device, bytes), array.name.c_str());
        }
        summary_.total_bytes += bytes;
      }
      set_view_pointer(view_, allocation.name, allocation.device);
      allocations_.push_back(std::move(allocation));
    }
  } catch (...) {
    release();
    throw;
  }
}

DeviceWorkspace::~DeviceWorkspace() {
  release();
}

DeviceWorkspace::DeviceWorkspace(DeviceWorkspace&& other) noexcept {
  *this = std::move(other);
}

DeviceWorkspace& DeviceWorkspace::operator=(DeviceWorkspace&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  release();
  allocations_ = std::move(other.allocations_);
  view_ = other.view_;
  summary_ = other.summary_;

  other.view_ = {};
  other.summary_ = {};
  return *this;
}

DeviceWorkspaceView DeviceWorkspace::view() const {
  return view_;
}

DeviceWorkspaceSummary DeviceWorkspace::summary() const {
  return summary_;
}

void DeviceWorkspace::reset_runtime_overrides() {
  const Allocation* types = find_allocation("types");
  if (types != nullptr) {
    set_view_pointer(view_, types->name, types->device);
  }
  const Allocation* positions = find_allocation("positions_soa3");
  if (positions != nullptr) {
    set_view_pointer(view_, positions->name, positions->device);
  }
}

void DeviceWorkspace::override_types(int* device_types) {
  view_.types = device_types;
}

void DeviceWorkspace::override_positions_soa3(double* device_positions_soa3) {
  view_.positions_soa3 = device_positions_soa3;
}

void DeviceWorkspace::copy_int_to_device(
    const std::string& name,
    const std::vector<int>& host) {
  Allocation* allocation = find_allocation(name);
  if (allocation == nullptr || allocation->type != ScalarType::int32 ||
      allocation->element_count != host.size()) {
    throw std::runtime_error("int workspace copy shape mismatch: " + name);
  }
  check_cuda(
      cudaMemcpy(
          allocation->device,
          host.data(),
          host.size() * sizeof(int),
          cudaMemcpyHostToDevice),
      name.c_str());
}

void DeviceWorkspace::copy_double_to_device(
    const std::string& name,
    const std::vector<double>& host) {
  Allocation* allocation = find_allocation(name);
  if (allocation == nullptr || allocation->type != ScalarType::float64 ||
      allocation->element_count != host.size()) {
    throw std::runtime_error("double workspace copy shape mismatch: " + name);
  }
  check_cuda(
      cudaMemcpy(
          allocation->device,
          host.data(),
          host.size() * sizeof(double),
          cudaMemcpyHostToDevice),
      name.c_str());
}

void DeviceWorkspace::copy_float_to_device(
    const std::string& name,
    const std::vector<float>& host) {
  Allocation* allocation = find_allocation(name);
  if (allocation == nullptr || allocation->type != ScalarType::float32 ||
      allocation->element_count != host.size()) {
    throw std::runtime_error("float workspace copy shape mismatch: " + name);
  }
  check_cuda(
      cudaMemcpy(
          allocation->device,
          host.data(),
          host.size() * sizeof(float),
          cudaMemcpyHostToDevice),
      name.c_str());
}

DeviceWorkspace::Allocation* DeviceWorkspace::find_allocation(
    const std::string& name) {
  for (Allocation& allocation : allocations_) {
    if (allocation.name == name) {
      return &allocation;
    }
  }
  return nullptr;
}

const DeviceWorkspace::Allocation* DeviceWorkspace::find_allocation(
    const std::string& name) const {
  for (const Allocation& allocation : allocations_) {
    if (allocation.name == name) {
      return &allocation;
    }
  }
  return nullptr;
}

void DeviceWorkspace::release() {
  for (Allocation& allocation : allocations_) {
    if (allocation.device != nullptr) {
      if (allocation.async_allocated) {
        cudaFreeAsync(allocation.device, 0);
      } else {
        cudaFree(allocation.device);
      }
      allocation.device = nullptr;
    }
  }
  allocations_.clear();
  view_ = {};
  summary_ = {};
}

}  // namespace nep_adapters::cuda_backend
