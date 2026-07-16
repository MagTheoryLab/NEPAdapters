#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu.hpp"

#include "cpu_test_utils.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

int main() {
  const std::string model_path = NEP_ADAPTERS_NEP89_MODEL_PATH;
  const std::string xyz_path = NEP_ADAPTERS_NEP89_XYZ_PATH;

  if (!nep_adapters::register_cpu_engine()) {
    return EXIT_FAILURE;
  }

  const auto type_map = cpu_test::read_type_map(model_path);
  cpu_test::Frame frame =
      cpu_test::read_first_frame(xyz_path, type_map);

  NepaModel* model = nullptr;
  if (nepa_load_model("cpu", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }

  std::int32_t atom_count = static_cast<std::int32_t>(frame.types.size());
  std::int32_t atom_counts[] = {atom_count};
  std::int32_t atom_offsets[] = {0};
  std::int32_t pbc[] = {1, 1, 1};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = atom_count;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = frame.types.data();
  batch.positions_aos3 = frame.positions_aos3.data();
  batch.boxes_row_major9 = frame.box;
  batch.pbc_flags3 = pbc;

  double energy[] = {0.0};
  std::vector<double> forces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  double virial[9] = {};

  NepaFindForceResult result{};
  result.energy_per_structure = energy;
  result.forces_aos3 = forces.data();
  result.virials_row_major9 = virial;

  const NepaStatus status = nepa_find_force_batch(model, &batch, &result);
  nepa_free_model(model);
  if (status != NEPA_STATUS_OK) {
    return EXIT_FAILURE;
  }

  const double force_l1 = std::accumulate(
      forces.begin(),
      forces.end(),
      0.0,
      [](double sum, double value) { return sum + std::abs(value); });

  if (!std::isfinite(energy[0]) || !cpu_test::all_finite(forces) ||
      force_l1 <= 0.0) {
    return EXIT_FAILURE;
  }

  for (double component : virial) {
    if (!std::isfinite(component)) {
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
