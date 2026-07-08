#include "nep_adapters/api.h"
#include "nep_adapters/engines/cuda.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string write_zbl_model() {
  const std::string model_path =
      (std::filesystem::temp_directory_path() / "cuda_zbl_batch_force.nep").string();
  std::ofstream out(model_path);
  out << "nep4_zbl 1 C\n"
      << "zbl 0.8 1.6\n"
      << "cutoff 4 0.5 8 1\n"
      << "n_max 0 0\n"
      << "basis_size 0 0\n"
      << "l_max 0 0 0\n"
      << "ANN 3 0\n";
  const double values[] = {
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0,
      0.0, 0.0,
      1.0,
  };
  for (double value : values) {
    out << value << "\n";
  }
  return model_path;
}

struct Prediction {
  double energy = 0.0;
  std::vector<double> forces;
};

Prediction evaluate(NepaModel* model, const std::vector<double>& positions) {
  int atom_counts[] = {3};
  int atom_offsets[] = {0};
  int types[] = {0, 0, 0};
  double box[] = {
      12.0, 0.0, 0.0,
      0.0, 12.0, 0.0,
      0.0, 0.0, 12.0,
  };
  int pbc[] = {1, 1, 1};
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = 3;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions.data();
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;

  Prediction prediction;
  prediction.forces.assign(positions.size(), 0.0);
  NepaFindForceResult result{};
  result.energy_per_structure = &prediction.energy;
  result.forces_aos3 = prediction.forces.data();
  const NepaStatus status = nepa_find_force_batch(model, &batch, &result);
  if (status != NEPA_STATUS_OK) {
    std::cerr << "CUDA ZBL force status=" << status << "\n";
    std::exit(EXIT_FAILURE);
  }
  return prediction;
}

}  // namespace

int main() {
  if (!nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }

  const std::string model_path = write_zbl_model();
  NepaModel* model = nullptr;
  if (nepa_load_model("cuda", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }

  NepaModelInfo info{};
  if (nepa_model_info(model, &info) != NEPA_STATUS_OK ||
      (info.capabilities & NEPA_CAPABILITY_BATCH_FIND_FORCE) == 0 ||
      info.descriptor_dim != 1) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  const std::vector<double> positions = {
      0.10, 0.10, 0.20,
      1.18, 0.24, 0.28,
      2.45, 0.42, 0.30,
  };
  const Prediction base = evaluate(model, positions);
  if (!std::isfinite(base.energy) || std::abs(base.energy) < 1.0e-8) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }
  for (double force : base.forces) {
    if (!std::isfinite(force)) {
      nepa_free_model(model);
      return EXIT_FAILURE;
    }
  }

  constexpr double eps = 1.0e-4;
  constexpr double tolerance = 5.0e-2;
  for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
    std::vector<double> plus = positions;
    std::vector<double> minus = positions;
    plus[coordinate] += eps;
    minus[coordinate] -= eps;
    const double e_plus = evaluate(model, plus).energy;
    const double e_minus = evaluate(model, minus).energy;
    const double finite_difference_force = -(e_plus - e_minus) / (2.0 * eps);
    const double diff = std::abs(base.forces[coordinate] - finite_difference_force);
    if (diff > tolerance) {
      std::cerr << "ZBL finite-difference mismatch coordinate="
                << coordinate << " force=" << base.forces[coordinate]
                << " fd=" << finite_difference_force
                << " diff=" << diff << "\n";
      nepa_free_model(model);
      return EXIT_FAILURE;
    }
  }

  nepa_free_model(model);
  return EXIT_SUCCESS;
}
