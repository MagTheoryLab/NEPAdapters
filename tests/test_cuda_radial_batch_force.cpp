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

std::string write_radial_model() {
  const std::string model_path =
      (std::filesystem::temp_directory_path() / "cuda_radial_batch_force.nep").string();
  std::ofstream out(model_path);
  out << "nep4 1 C\n"
      << "cutoff 5 1 8 1\n"
      << "n_max 1 0\n"
      << "basis_size 2 0\n"
      << "l_max 0 0 0\n"
      << "ANN 2 0\n";
  const double values[] = {
      0.20, -0.10, 0.05, 0.15,
      0.01, -0.02,
      0.30, -0.25,
      0.04,
      0.70, -0.15, 0.05, -0.30, 0.20, -0.10, 0.0,
      0.80, 1.10,
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

Prediction evaluate(
    NepaModel* model,
    const std::vector<double>& positions,
    const double* box) {
  int atom_counts[] = {3};
  int atom_offsets[] = {0};
  int types[] = {0, 0, 0};
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
  std::vector<double> potential(3, 0.0);
  NepaFindForceResult result{};
  result.energy_per_structure = &prediction.energy;
  result.potential_per_atom = potential.data();
  result.forces_aos3 = prediction.forces.data();
  const NepaStatus status = nepa_find_force_batch(model, &batch, &result);
  if (status != NEPA_STATUS_OK) {
    std::cerr << "CUDA radial force status=" << status << "\n";
    std::exit(EXIT_FAILURE);
  }
  return prediction;
}

bool is_finite_value(double value) {
  return std::isfinite(value);
}

bool finite_difference_matches(
    NepaModel* model,
    const std::vector<double>& positions,
    const double* box) {
  const Prediction base = evaluate(model, positions, box);
  if (!is_finite_value(base.energy)) {
    return false;
  }
  for (double force : base.forces) {
    if (!is_finite_value(force)) {
      return false;
    }
  }

  constexpr double eps = 1.0e-3;
  constexpr double tolerance = 2.0e-2;
  for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
    std::vector<double> plus = positions;
    std::vector<double> minus = positions;
    plus[coordinate] += eps;
    minus[coordinate] -= eps;
    const double e_plus = evaluate(model, plus, box).energy;
    const double e_minus = evaluate(model, minus, box).energy;
    const double finite_difference_force = -(e_plus - e_minus) / (2.0 * eps);
    const double diff = std::abs(base.forces[coordinate] - finite_difference_force);
    if (diff > tolerance) {
      std::cerr << "force finite-difference mismatch coordinate=" << coordinate
                << " force=" << base.forces[coordinate]
                << " fd=" << finite_difference_force
                << " diff=" << diff << "\n";
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }

  const std::string model_path = write_radial_model();
  NepaModel* model = nullptr;
  if (nepa_load_model("cuda", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }

  NepaModelInfo info{};
  if (nepa_model_info(model, &info) != NEPA_STATUS_OK ||
      (info.capabilities & NEPA_CAPABILITY_BATCH_FIND_FORCE) == 0 ||
      info.descriptor_dim != 2) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  const std::vector<double> positions = {
      0.10, 0.20, 0.30,
      1.35, 0.10, 0.25,
      2.40, 0.35, 0.15,
  };
  const double orthorhombic_box[] = {
      12.0, 0.0, 0.0,
      0.0, 12.0, 0.0,
      0.0, 0.0, 12.0,
  };
  const double triclinic_box[] = {
      12.0, 2.0, 1.0,
      0.0, 11.0, 1.5,
      0.0, 0.0, 10.0,
  };
  const std::vector<double> triclinic_positions = {
      0.20, 0.20, 0.20,
      11.40, 1.90, 1.00,
      2.40, 0.50, 0.30,
  };
  if (!finite_difference_matches(model, positions, orthorhombic_box) ||
      !finite_difference_matches(model, triclinic_positions, triclinic_box)) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  nepa_free_model(model);
  return EXIT_SUCCESS;
}
