#include "nep_adapters/api.h"
#if defined(NEP_ADAPTERS_TEST_CUDA_ENGINE)
#include "nep_adapters/engines/cuda.hpp"
#else
#include "nep_adapters/engines/cpu.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) return EXIT_FAILURE;
#if defined(NEP_ADAPTERS_TEST_CUDA_ENGINE)
  if (!nep_adapters::register_cuda_engine()) return EXIT_FAILURE;
  const char* backend = "cuda";
#else
  if (!nep_adapters::register_cpu_engine()) return EXIT_FAILURE;
  const char* backend = "cpu";
#endif
  NepaModel* model = nullptr;
  NepaModelInfo info{};
  if (nepa_load_model(backend, argv[1], &model) != NEPA_STATUS_OK ||
      nepa_model_info(model, &info) != NEPA_STATUS_OK) return EXIT_FAILURE;

  int counts[2] = {2, 1}, offsets[2] = {0, 0}, types[2] = {0, 0};
  int pbc[6] = {1, 1, 1, 1, 1, 1};
  double positions[6] = {0, 0, 0, 2, 0, 0};
  double spins[6] = {1, 0, 0, .2, .7, -.1};
  double boxes[18] = {6, 0, 0, 0, 6, 0, 0, 0, 6,
                     6, 0, 0, 0, 6, 0, 0, 0, 6};
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = 2;
  batch.atom_counts = counts;
  batch.atom_offsets = offsets;
  batch.types = types;
  batch.positions_aos3 = positions;
  batch.spins_aos3 = spins;
  batch.boxes_row_major9 = boxes;
  batch.pbc_flags3 = pbc;
  double energies[2] = {}, forces[6] = {};
  std::vector<double> descriptors(2 * info.descriptor_dim);
  NepaFindForceResult result{};
  result.energy_per_structure = energies;
  result.forces_aos3 = forces;
  NepaFindDescriptorResult descriptor_result{descriptors.data()};
  bool ok = true;
  auto check = [&](const char* label, NepaStatus expected) {
    std::fill_n(energies, 2, -321.0);
    std::fill_n(forces, 6, -321.0);
    std::fill(descriptors.begin(), descriptors.end(), -321.0);
    const auto force_status = nepa_find_force_batch(model, &batch, &result);
    const auto descriptor_status = nepa_find_descriptors(model, &batch, &descriptor_result);
    if (force_status != expected || descriptor_status != expected ||
        (expected != NEPA_STATUS_OK &&
         (energies[0] != -321.0 || forces[0] != -321.0 || descriptors[0] != -321.0))) {
      std::cerr << label << ": force=" << force_status << " descriptor="
                << descriptor_status << " " << nepa_last_error_message() << '\n';
      ok = false;
    }
    if (expected == NEPA_STATUS_OK) {
      ok = std::isfinite(energies[0]) &&
           std::all_of(forces, forces + 6, [](double x) { return std::isfinite(x); }) &&
           std::all_of(descriptors.begin(), descriptors.end(),
                       [](double x) { return std::isfinite(x); }) && ok;
    }
  };
  check("valid small periodic box", NEPA_STATUS_OK);
  for (int offset : {-1, std::numeric_limits<int>::max()}) {
    offsets[0] = offset;
    check("invalid offset before periodic expansion", NEPA_STATUS_INVALID_ARGUMENT);
  }
  offsets[0] = 0;
  counts[0] = std::numeric_limits<int>::max();
  check("overflowing range", NEPA_STATUS_INVALID_ARGUMENT);
  counts[0] = 1;
  check("uncovered atom", NEPA_STATUS_INVALID_ARGUMENT);
  batch.num_structures = 2;
  check("overlapping structures", NEPA_STATUS_INVALID_ARGUMENT);
  batch.num_structures = 1;
  counts[0] = 2;
  types[0] = info.num_types;
  check("invalid model type", NEPA_STATUS_INVALID_ARGUMENT);
  types[0] = 0;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (double bad : {nan, inf}) {
    positions[0] = bad;
    check("non-finite position", NEPA_STATUS_INVALID_ARGUMENT);
    positions[0] = 0;
    spins[0] = bad;
    check("non-finite spin", NEPA_STATUS_INVALID_ARGUMENT);
    spins[0] = 1;
    boxes[1] = bad;
    check("non-finite box", NEPA_STATUS_INVALID_ARGUMENT);
    boxes[1] = 0;
  }
  boxes[8] = 0;
  check("singular box", NEPA_STATUS_INVALID_ARGUMENT);
  boxes[8] = 6;
  check("valid after rejected inputs", NEPA_STATUS_OK);
  nepa_free_model(model);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
