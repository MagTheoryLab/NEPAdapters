#include "nep_adapters/api.h"
#include "nep_adapters/engines/cuda.hpp"

#include <cmath>
#include <cstdlib>

#ifndef NEP_ADAPTERS_CUDA_TEST_MODEL_PATH
#  error "NEP_ADAPTERS_CUDA_TEST_MODEL_PATH must be defined"
#endif

int main() {
  if (!nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }

  NepaModel* model = nullptr;
  if (nepa_load_model("cuda", NEP_ADAPTERS_CUDA_TEST_MODEL_PATH, &model) !=
          NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }

  int atom_counts[] = {2};
  int atom_offsets[] = {0};
  int types[] = {0, 1};
  double positions[] = {
      0.0, 0.0, 0.0,
      1.5, 0.0, 0.0,
  };
  double boxes[] = {
      8.0, 0.0, 0.0,
      0.0, 8.0, 0.0,
      0.0, 0.0, 8.0,
  };
  int pbc[] = {1, 1, 1};
  double energy[] = {0.0};
  double forces[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = 2;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions;
  batch.boxes_row_major9 = boxes;
  batch.pbc_flags3 = pbc;

  NepaFindForceResult result{};
  result.energy_per_structure = energy;
  result.forces_aos3 = forces;

  if (nepa_find_force_batch(model, &batch, &result) != NEPA_STATUS_OK ||
      !std::isfinite(energy[0])) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }
  for (double force : forces) {
    if (!std::isfinite(force)) {
      nepa_free_model(model);
      return EXIT_FAILURE;
    }
  }

  result.forces_aos3 = nullptr;
  if (nepa_find_force_batch(model, &batch, &result) !=
      NEPA_STATUS_INVALID_ARGUMENT) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  nepa_free_model(model);
  return EXIT_SUCCESS;
}
