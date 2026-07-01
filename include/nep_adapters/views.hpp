#pragma once

#include <cstdint>

namespace nep_adapters {

enum class MemorySpace {
  host,
  device,
};

struct BoxView {
  const double* cell_row_major9 = nullptr;
  const std::int32_t* pbc_flags3 = nullptr;
};

struct AtomsView {
  std::int64_t count = 0;
  const std::int32_t* types = nullptr;
  const double* positions_aos3 = nullptr;
  const double* charges = nullptr;
  const double* spins_aos3 = nullptr;
  MemorySpace memory_space = MemorySpace::host;
};

struct NeighborListView {
  std::int64_t local_count = 0;
  const std::int32_t* counts = nullptr;
  const std::int32_t* offsets = nullptr;
  const std::int32_t* indices = nullptr;
  bool half_list = false;
  MemorySpace memory_space = MemorySpace::host;
};

struct OutputView {
  double* energy_per_structure = nullptr;
  double* forces_aos3 = nullptr;
  double* virials_row_major9 = nullptr;
  double* descriptors = nullptr;
};

}  // namespace nep_adapters
