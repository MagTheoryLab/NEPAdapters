#include "host_staging.hpp"

#include <algorithm>
#include <stdexcept>

namespace nep_adapters::cuda_backend {
namespace {

constexpr int kLammpsNeighborMask = 0x3fffffff;

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

int mapped_lammps_type(const NepaLammpsNeighborInput& input, int atom) {
  const int lammps_type = input.types[atom];
  require(lammps_type >= 0, "negative LAMMPS atom type");
  const int mapped = input.type_map[lammps_type];
  require(mapped >= 0, "unmapped LAMMPS atom type");
  return mapped;
}

void stage_lammps_neighbor_slots(
    const NepaLammpsNeighborInput& input,
    int atom_capacity,
    int neighbor_capacity,
    std::vector<int>& counts,
    std::vector<int>& slots) {
  counts.assign(static_cast<std::size_t>(atom_capacity), 0);
  slots.assign(
      static_cast<std::size_t>(atom_capacity) *
          static_cast<std::size_t>(neighbor_capacity),
      0);

  for (int active = 0; active < input.inum; ++active) {
    const int atom = input.ilist[active];
    require(atom >= 0 && atom < atom_capacity, "active atom out of range");
    const int neighbor_count = input.numneigh[atom];
    require(neighbor_count >= 0, "negative neighbor count");
    require(
        neighbor_count <= neighbor_capacity,
        "neighbor count exceeds CUDA workspace capacity");
    counts[static_cast<std::size_t>(atom)] = neighbor_count;

    if (neighbor_count == 0) {
      continue;
    }
    require(input.firstneigh[atom] != nullptr, "missing LAMMPS neighbor row");
    for (int slot = 0; slot < neighbor_count; ++slot) {
      const int neighbor = input.firstneigh[atom][slot] & kLammpsNeighborMask;
      require(neighbor >= 0 && neighbor < atom_capacity, "neighbor atom out of range");
      slots[static_cast<std::size_t>(atom) +
            static_cast<std::size_t>(atom_capacity) *
                static_cast<std::size_t>(slot)] = neighbor;
    }
  }
}

}  // namespace

HostBatchStaging stage_batch_for_internal_neighbors(
    const NepaStructureBatch& batch) {
  require(batch.num_structures > 0, "batch must contain structures");
  require(batch.total_atoms > 0, "batch must contain atoms");
  require(batch.atom_counts != nullptr, "missing atom counts");
  require(batch.atom_offsets != nullptr, "missing atom offsets");
  require(batch.types != nullptr, "missing atom types");
  require(batch.positions_aos3 != nullptr, "missing atom positions");
  require(batch.boxes_row_major9 != nullptr, "missing boxes");

  HostBatchStaging staging;
  staging.types.assign(batch.types, batch.types + batch.total_atoms);
  staging.positions_soa3.assign(static_cast<std::size_t>(batch.total_atoms) * 3, 0.0);
  staging.atom_to_structure.assign(static_cast<std::size_t>(batch.total_atoms), -1);
  staging.structure_atom_counts.assign(
      batch.atom_counts,
      batch.atom_counts + batch.num_structures);
  staging.structure_atom_offsets.assign(
      batch.atom_offsets,
      batch.atom_offsets + batch.num_structures);
  staging.boxes_row_major9.assign(
      batch.boxes_row_major9,
      batch.boxes_row_major9 + static_cast<std::size_t>(batch.num_structures) * 9);
  staging.pbc_flags3.assign(
      static_cast<std::size_t>(batch.num_structures) * 3,
      0);
  if (batch.pbc_flags3 != nullptr) {
    staging.pbc_flags3.assign(
        batch.pbc_flags3,
        batch.pbc_flags3 + static_cast<std::size_t>(batch.num_structures) * 3);
  }

  for (int atom = 0; atom < batch.total_atoms; ++atom) {
    staging.positions_soa3[static_cast<std::size_t>(atom)] =
        batch.positions_aos3[3 * static_cast<std::size_t>(atom)];
    staging.positions_soa3[static_cast<std::size_t>(batch.total_atoms + atom)] =
        batch.positions_aos3[3 * static_cast<std::size_t>(atom) + 1];
    staging.positions_soa3[
        static_cast<std::size_t>(2 * batch.total_atoms + atom)] =
        batch.positions_aos3[3 * static_cast<std::size_t>(atom) + 2];
  }

  for (int structure = 0; structure < batch.num_structures; ++structure) {
    const int atom_count = batch.atom_counts[structure];
    const int atom_offset = batch.atom_offsets[structure];
    require(atom_count > 0, "structure atom count must be positive");
    require(atom_offset >= 0, "structure atom offset must be non-negative");
    require(
        atom_offset + atom_count <= batch.total_atoms,
        "structure atom range exceeds batch atom count");
    for (int local = 0; local < atom_count; ++local) {
      const int atom = atom_offset + local;
      require(
          staging.atom_to_structure[static_cast<std::size_t>(atom)] < 0,
          "overlapping structure atom ranges");
      staging.atom_to_structure[static_cast<std::size_t>(atom)] = structure;
    }
  }

  for (int atom = 0; atom < batch.total_atoms; ++atom) {
    require(
        staging.atom_to_structure[static_cast<std::size_t>(atom)] >= 0,
        "batch atom is not covered by any structure");
  }

  return staging;
}

HostExternalNeighborStaging stage_lammps_external_neighbors(
    const NepaLammpsNeighborInput& input,
    const ModelProtocol& protocol) {
  require(input.nlocal >= 0, "negative nlocal");
  require(input.inum >= 0, "negative inum");
  require(input.ilist != nullptr, "missing ilist");
  require(input.numneigh != nullptr, "missing numneigh");
  require(input.firstneigh != nullptr, "missing firstneigh");
  require(input.types != nullptr, "missing types");
  require(input.type_map != nullptr, "missing type_map");
  require(input.positions != nullptr, "missing positions");

  int atom_capacity = input.nlocal;
  for (int active = 0; active < input.inum; ++active) {
    const int atom = input.ilist[active];
    require(atom >= 0, "negative active atom index");
    atom_capacity = std::max(atom_capacity, atom + 1);
    const int neighbor_count = input.numneigh[atom];
    require(neighbor_count >= 0, "negative neighbor count");
    if (neighbor_count > 0) {
      require(input.firstneigh[atom] != nullptr, "missing LAMMPS neighbor row");
    }
    for (int slot = 0; slot < neighbor_count; ++slot) {
      const int neighbor = input.firstneigh[atom][slot] & kLammpsNeighborMask;
      require(neighbor >= 0, "negative neighbor atom index");
      atom_capacity = std::max(atom_capacity, neighbor + 1);
    }
  }

  HostExternalNeighborStaging staging;
  staging.atom_capacity = atom_capacity;
  staging.active_atom_count = input.inum;
  staging.types.assign(static_cast<std::size_t>(atom_capacity), 0);
  staging.positions_soa3.assign(static_cast<std::size_t>(atom_capacity) * 3, 0.0);
  staging.active_atom_indices.assign(input.ilist, input.ilist + input.inum);

  for (int atom = 0; atom < atom_capacity; ++atom) {
    require(input.positions[atom] != nullptr, "missing LAMMPS position row");
    staging.types[static_cast<std::size_t>(atom)] = mapped_lammps_type(input, atom);
    staging.positions_soa3[static_cast<std::size_t>(atom)] = input.positions[atom][0];
    staging.positions_soa3[static_cast<std::size_t>(atom_capacity + atom)] =
        input.positions[atom][1];
    staging.positions_soa3[static_cast<std::size_t>(2 * atom_capacity + atom)] =
        input.positions[atom][2];
  }

  stage_lammps_neighbor_slots(
      input,
      atom_capacity,
      protocol.neighbor_capacity_radial,
      staging.nn_radial,
      staging.nl_radial_slot_major);
  stage_lammps_neighbor_slots(
      input,
      atom_capacity,
      protocol.neighbor_capacity_angular,
      staging.nn_angular,
      staging.nl_angular_slot_major);

  return staging;
}

}  // namespace nep_adapters::cuda_backend
