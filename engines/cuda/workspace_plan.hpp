#pragma once

#include "model_protocol.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace nep_adapters::cuda_backend {

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

WorkspacePlan make_workspace_plan(
    const ModelProtocol& protocol,
    std::size_t atom_capacity);

WorkspacePlan make_internal_neighbor_workspace_plan(
    const ModelProtocol& protocol,
    std::size_t atom_capacity,
    std::size_t structure_capacity);

WorkspacePlan make_external_neighbor_workspace_plan(
    const ModelProtocol& protocol,
    std::size_t atom_capacity,
    std::size_t active_atom_capacity,
    bool include_basis_cache = true,
    bool include_angular_vectors = true,
    bool include_per_atom_virial_sink = false);

WorkspacePlan make_model_workspace_plan(const ModelProtocol& protocol);

}  // namespace nep_adapters::cuda_backend
