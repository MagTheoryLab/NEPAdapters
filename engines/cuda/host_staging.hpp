#pragma once

#include "model_protocol.hpp"

#include "nep_adapters/api.h"

#include <vector>

namespace nep_adapters::cuda_backend {

struct HostBatchStaging {
  std::vector<int> types;
  std::vector<double> positions_soa3;
  std::vector<int> atom_to_structure;
  std::vector<int> structure_atom_counts;
  std::vector<int> structure_atom_offsets;
  std::vector<double> boxes_row_major9;
  std::vector<int> pbc_flags3;
};

struct HostExternalNeighborStaging {
  int atom_capacity = 0;
  int active_atom_count = 0;
  std::vector<int> types;
  std::vector<double> positions_soa3;
  std::vector<int> active_atom_indices;
  std::vector<int> nn_radial;
  std::vector<int> nl_radial_slot_major;
  std::vector<int> nn_angular;
  std::vector<int> nl_angular_slot_major;
};

HostBatchStaging stage_batch_for_internal_neighbors(
    const NepaStructureBatch& batch);

HostExternalNeighborStaging stage_lammps_external_neighbors(
    const NepaLammpsNeighborInput& input,
    const ModelProtocol& protocol);

}  // namespace nep_adapters::cuda_backend
