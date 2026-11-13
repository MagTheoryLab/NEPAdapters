#include "nep_adapters/detail/structure_batch.hpp"
#include "nep_adapters/engine.hpp"

#include <cmath>
#include <vector>

namespace nep_adapters::detail {
namespace {
bool reject(const char* message) {
  set_last_error(message);
  return false;
}
}  // namespace

bool valid_structure_batch(const NepaStructureBatch& batch, int num_types) {
  if (batch.num_structures <= 0 || batch.total_atoms <= 0 || num_types <= 0 ||
      !batch.atom_counts || !batch.atom_offsets || !batch.types ||
      !batch.positions_aos3 || !batch.boxes_row_major9 || !batch.pbc_flags3) {
    return reject("batch must contain atoms, structures and complete input arrays");
  }

  std::vector<unsigned char> covered(static_cast<std::size_t>(batch.total_atoms), 0);
  for (int structure = 0; structure < batch.num_structures; ++structure) {
    const int count = batch.atom_counts[structure];
    const int offset = batch.atom_offsets[structure];
    if (count <= 0 || offset < 0 || offset > batch.total_atoms ||
        count > batch.total_atoms - offset) {
      return reject("structure atom range exceeds batch atom count");
    }
    for (int atom = offset; atom < offset + count; ++atom) {
      if (covered[atom]) return reject("overlapping structure atom ranges");
      covered[atom] = 1;
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (batch.pbc_flags3[3 * static_cast<std::size_t>(structure) + axis] != 1)
        return reject("NEPAdapters supports fully periodic structures only");
    }
    const double* box = batch.boxes_row_major9 + 9 * static_cast<std::size_t>(structure);
    for (int k = 0; k < 9; ++k) {
      if (!std::isfinite(box[k])) return reject("box contains non-finite values");
    }
    const double determinant =
        box[0] * (box[4] * box[8] - box[5] * box[7]) -
        box[1] * (box[3] * box[8] - box[5] * box[6]) +
        box[2] * (box[3] * box[7] - box[4] * box[6]);
    if (!std::isfinite(determinant) || std::abs(determinant) <= 1.0e-12)
      return reject("periodic box is singular or numerically degenerate");
  }
  for (int atom = 0; atom < batch.total_atoms; ++atom) {
    if (!covered[atom]) return reject("batch atom is not covered by any structure");
    if (batch.types[atom] < 0 || batch.types[atom] >= num_types)
      return reject("atom type index is outside the model type range");
    for (int axis = 0; axis < 3; ++axis) {
      const auto k = 3 * static_cast<std::size_t>(atom) + axis;
      if (!std::isfinite(batch.positions_aos3[k]))
        return reject("positions contain non-finite values");
      if (batch.spins_aos3 && !std::isfinite(batch.spins_aos3[k]))
        return reject("spins contain non-finite values");
    }
  }
  return true;
}
}  // namespace nep_adapters::detail
