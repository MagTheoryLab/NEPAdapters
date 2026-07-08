#include "workspace_plan.hpp"

#include <stdexcept>

namespace nep_adapters::cuda_backend {
namespace {

constexpr std::size_t kLammpsReductionThreads = 256;

std::size_t scalar_size(ScalarType type) {
  switch (type) {
    case ScalarType::int32:
      return sizeof(int);
    case ScalarType::float32:
      return sizeof(float);
    case ScalarType::float64:
      return sizeof(double);
  }
  throw std::runtime_error("unknown scalar type");
}

void add_array(
    WorkspacePlan& plan,
    const char* name,
    ScalarType type,
    std::size_t element_count) {
  plan.arrays.push_back({name, type, element_count});
}

void add_common_atom_arrays(WorkspacePlan& plan) {
  add_array(plan, "types", ScalarType::int32, plan.atom_capacity);
  add_array(plan, "positions_soa3", ScalarType::float64, plan.atom_capacity * 3);
  add_array(plan, "potential", ScalarType::float64, plan.atom_capacity);
  add_array(plan, "force_soa3", ScalarType::float64, plan.atom_capacity * 3);
  add_array(plan, "virial_soa9", ScalarType::float64, plan.atom_capacity * 9);
}

void add_slot_major_neighbor_arrays(
    WorkspacePlan& plan,
    const ModelProtocol& protocol) {
  add_array(plan, "nn_radial", ScalarType::int32, plan.atom_capacity);
  add_array(
      plan,
      "nl_radial_slot_major",
      ScalarType::int32,
      plan.atom_capacity * static_cast<std::size_t>(protocol.neighbor_capacity_radial));
  add_array(plan, "nn_angular", ScalarType::int32, plan.atom_capacity);
  add_array(
      plan,
      "nl_angular_slot_major",
      ScalarType::int32,
      plan.atom_capacity * static_cast<std::size_t>(protocol.neighbor_capacity_angular));
}

void add_cell_list_arrays(WorkspacePlan& plan) {
  add_array(plan, "cell_counts", ScalarType::int32, plan.atom_capacity);
  add_array(plan, "cell_offsets", ScalarType::int32, plan.atom_capacity + 1);
  add_array(plan, "cell_fill", ScalarType::int32, plan.atom_capacity);
  add_array(plan, "cell_atoms", ScalarType::int32, plan.atom_capacity);
  add_array(plan, "atom_cell", ScalarType::int32, plan.atom_capacity);
  add_array(plan, "cell_dims", ScalarType::int32, 4);
  add_array(plan, "neighbor_overflow", ScalarType::int32, 3);
}

void add_execution_scratch(
    WorkspacePlan& plan,
    const ModelProtocol& protocol,
    bool include_basis_cache,
    bool include_angular_vectors) {
  add_array(
      plan,
      "parameters_and_q_scaler",
      ScalarType::float32,
      protocol.model_parameter_count + protocol.q_scaler_count);
  add_array(
      plan,
      "fp",
      ScalarType::float32,
      plan.atom_capacity * static_cast<std::size_t>(protocol.descriptor_dim));
  if (protocol.charge_mode > 0) {
    add_array(plan, "charge", ScalarType::float64, plan.atom_capacity);
    add_array(plan, "bec_soa9", ScalarType::float64, plan.atom_capacity * 9);
    add_array(plan, "d_real", ScalarType::float32, plan.atom_capacity);
    add_array(
        plan,
        "charge_derivative",
        ScalarType::float32,
        plan.atom_capacity * static_cast<std::size_t>(protocol.descriptor_dim));
  }
  add_array(
      plan,
      "descriptors",
      ScalarType::float32,
      plan.atom_capacity * static_cast<std::size_t>(protocol.descriptor_dim));
  add_array(
      plan,
      "sum_fxyz",
      ScalarType::float32,
      plan.atom_capacity * (static_cast<std::size_t>(protocol.n_max_angular) + 1) *
          static_cast<std::size_t>(protocol.body_channels.abc_count()));
  add_array(
      plan,
      "r12_radial",
      ScalarType::float32,
      plan.atom_capacity * static_cast<std::size_t>(protocol.neighbor_capacity_radial));
  if (include_basis_cache) {
    add_array(
        plan,
        "fc_radial",
        ScalarType::float32,
        plan.atom_capacity * static_cast<std::size_t>(protocol.neighbor_capacity_radial));
    add_array(
        plan,
        "fn_radial",
        ScalarType::float32,
        plan.atom_capacity * static_cast<std::size_t>(protocol.neighbor_capacity_radial) *
            (static_cast<std::size_t>(protocol.basis_size_radial) + 1));
  }
  add_array(
      plan,
      "r12_angular",
      ScalarType::float32,
      plan.atom_capacity * static_cast<std::size_t>(protocol.neighbor_capacity_angular));
  if (include_basis_cache) {
    add_array(
        plan,
        "fc_angular",
        ScalarType::float32,
        plan.atom_capacity * static_cast<std::size_t>(protocol.neighbor_capacity_angular));
    add_array(
        plan,
        "fn_angular",
        ScalarType::float32,
        plan.atom_capacity * static_cast<std::size_t>(protocol.neighbor_capacity_angular) *
            (static_cast<std::size_t>(protocol.basis_size_angular) + 1));
  }
  if (include_angular_vectors) {
    add_array(
        plan,
        "f12x",
        ScalarType::float32,
        plan.atom_capacity * static_cast<std::size_t>(protocol.neighbor_capacity_angular));
    add_array(
        plan,
        "f12y",
        ScalarType::float32,
        plan.atom_capacity * static_cast<std::size_t>(protocol.neighbor_capacity_angular));
    add_array(
        plan,
        "f12z",
        ScalarType::float32,
        plan.atom_capacity * static_cast<std::size_t>(protocol.neighbor_capacity_angular));
  }
}

}  // namespace

std::size_t DeviceArrayPlan::bytes() const {
  return element_count * scalar_size(type);
}

std::size_t WorkspacePlan::total_bytes() const {
  std::size_t total = 0;
  for (const DeviceArrayPlan& array : arrays) {
    total += array.bytes();
  }
  return total;
}

const DeviceArrayPlan* WorkspacePlan::find_array(const std::string& name) const {
  for (const DeviceArrayPlan& array : arrays) {
    if (array.name == name) {
      return &array;
    }
  }
  return nullptr;
}

WorkspacePlan make_workspace_plan(
    const ModelProtocol& protocol,
    std::size_t atom_capacity) {
  return make_internal_neighbor_workspace_plan(protocol, atom_capacity, 1);
}

WorkspacePlan make_internal_neighbor_workspace_plan(
    const ModelProtocol& protocol,
    std::size_t atom_capacity,
    std::size_t structure_capacity) {
  if (atom_capacity == 0) {
    throw std::runtime_error("atom_capacity must be positive");
  }
  if (structure_capacity == 0) {
    throw std::runtime_error("structure_capacity must be positive");
  }

  WorkspacePlan plan;
  plan.neighbor_source = NeighborSource::internal;
  plan.atom_capacity = atom_capacity;
  plan.active_atom_capacity = atom_capacity;
  plan.structure_capacity = structure_capacity;

  add_common_atom_arrays(plan);
  add_array(plan, "atom_to_structure", ScalarType::int32, atom_capacity);
  add_array(plan, "structure_atom_counts", ScalarType::int32, structure_capacity);
  add_array(plan, "structure_atom_offsets", ScalarType::int32, structure_capacity);
  add_array(plan, "boxes_row_major9", ScalarType::float64, structure_capacity * 9);
  add_array(plan, "box_inverse_row_major9", ScalarType::float64, structure_capacity * 9);
  add_array(plan, "pbc_flags3", ScalarType::int32, structure_capacity * 3);
  add_array(plan, "output_forces_aos3", ScalarType::float64, atom_capacity * 3);
  add_array(
      plan,
      "output_virials_per_atom_row_major9",
      ScalarType::float64,
      atom_capacity * 9);
  add_array(plan, "structure_energy", ScalarType::float64, structure_capacity);
  add_array(
      plan,
      "structure_virial_row_major9",
      ScalarType::float64,
      structure_capacity * 9);
  add_slot_major_neighbor_arrays(plan, protocol);
  add_cell_list_arrays(plan);
  add_array(plan, "structure_cell_offsets", ScalarType::int32, structure_capacity + 1);
  add_array(plan, "structure_cell_dims4", ScalarType::int32, structure_capacity * 4);
  add_execution_scratch(plan, protocol, true, true);

  return plan;
}

WorkspacePlan make_external_neighbor_workspace_plan(
    const ModelProtocol& protocol,
    std::size_t atom_capacity,
    std::size_t active_atom_capacity,
    bool include_basis_cache) {
  if (atom_capacity == 0) {
    throw std::runtime_error("atom_capacity must be positive");
  }
  if (active_atom_capacity == 0 || active_atom_capacity > atom_capacity) {
    throw std::runtime_error("active_atom_capacity must be in 1..atom_capacity");
  }

  WorkspacePlan plan;
  plan.neighbor_source = NeighborSource::external;
  plan.atom_capacity = atom_capacity;
  plan.active_atom_capacity = active_atom_capacity;
  plan.structure_capacity = 0;

  add_common_atom_arrays(plan);
  add_array(plan, "active_atom_indices", ScalarType::int32, active_atom_capacity);
  add_slot_major_neighbor_arrays(plan, protocol);
  add_array(plan, "neighbor_overflow", ScalarType::int32, 3);
  add_array(
      plan,
      "lammps_partial_sums",
      ScalarType::float64,
      7 * ((atom_capacity + kLammpsReductionThreads - 1) /
           kLammpsReductionThreads));
  add_execution_scratch(plan, protocol, include_basis_cache, include_basis_cache);

  return plan;
}

WorkspacePlan make_model_workspace_plan(const ModelProtocol& protocol) {
  WorkspacePlan plan;
  plan.atom_capacity = 0;
  add_array(
      plan,
      "ann_type_major",
      ScalarType::float32,
      protocol.ann_parameter_count);
  add_array(
      plan,
      "descriptor_coefficients",
      ScalarType::float32,
      protocol.descriptor_parameter_count);
  add_array(
      plan,
      "q_scaler",
      ScalarType::float32,
      protocol.q_scaler_count);
  return plan;
}

}  // namespace nep_adapters::cuda_backend
