#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_nep3.hpp"

#include "cpu_nep3_test_utils.hpp"
#include "nep.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct LammpsInputStorage {
  int nlocal = 0;
  std::vector<int> ilist;
  std::vector<int> numneigh;
  std::vector<std::vector<int>> neighbors;
  std::vector<int*> firstneigh;
  std::vector<int> types;
  std::vector<int> type_map;
  std::vector<double> positions;
  std::vector<double*> position_rows;
};

struct LammpsPrediction {
  double total_potential = 0.0;
  double total_virial6[6] = {};
  std::vector<double> potential;
  std::vector<double> forces;
  std::vector<double*> force_rows;
  std::vector<double> virials_per_atom9;
  std::vector<double*> virial_rows;
};

LammpsInputStorage make_lammps_input(const cpu_nep3_test::Frame& frame) {
  LammpsInputStorage input;
  input.nlocal = static_cast<int>(frame.types.size());
  input.ilist.resize(static_cast<std::size_t>(input.nlocal));
  input.numneigh.assign(static_cast<std::size_t>(input.nlocal), 0);
  input.neighbors.resize(static_cast<std::size_t>(input.nlocal));
  input.firstneigh.assign(static_cast<std::size_t>(input.nlocal), nullptr);
  input.types.resize(static_cast<std::size_t>(input.nlocal));
  input.positions = frame.positions_aos3;
  input.position_rows.resize(static_cast<std::size_t>(input.nlocal));

  int max_type = 0;
  for (int i = 0; i < input.nlocal; ++i) {
    input.ilist[static_cast<std::size_t>(i)] = i;
    input.types[static_cast<std::size_t>(i)] = frame.types[static_cast<std::size_t>(i)] + 1;
    max_type = std::max(max_type, input.types[static_cast<std::size_t>(i)]);
    input.position_rows[static_cast<std::size_t>(i)] =
        input.positions.data() + 3 * static_cast<std::size_t>(i);
  }

  input.type_map.assign(static_cast<std::size_t>(max_type + 1), -1);
  for (int i = 0; i < input.nlocal; ++i) {
    input.type_map[static_cast<std::size_t>(input.types[static_cast<std::size_t>(i)])] =
        frame.types[static_cast<std::size_t>(i)];
  }

  for (int i = 0; i < input.nlocal; ++i) {
    std::vector<int>& neighbors = input.neighbors[static_cast<std::size_t>(i)];
    neighbors.reserve(static_cast<std::size_t>(input.nlocal > 0 ? input.nlocal - 1 : 0));
    for (int j = 0; j < input.nlocal; ++j) {
      if (j != i) {
        neighbors.push_back(j);
      }
    }
    input.numneigh[static_cast<std::size_t>(i)] = static_cast<int>(neighbors.size());
    input.firstneigh[static_cast<std::size_t>(i)] =
        neighbors.empty() ? nullptr : neighbors.data();
  }

  return input;
}

LammpsPrediction make_prediction_storage(int nlocal) {
  LammpsPrediction prediction;
  prediction.potential.assign(static_cast<std::size_t>(nlocal), 0.0);
  prediction.forces.assign(static_cast<std::size_t>(nlocal) * 3, 0.0);
  prediction.force_rows.resize(static_cast<std::size_t>(nlocal));
  prediction.virials_per_atom9.assign(static_cast<std::size_t>(nlocal) * 9, 0.0);
  prediction.virial_rows.resize(static_cast<std::size_t>(nlocal));

  for (int i = 0; i < nlocal; ++i) {
    prediction.force_rows[static_cast<std::size_t>(i)] =
        prediction.forces.data() + 3 * static_cast<std::size_t>(i);
    prediction.virial_rows[static_cast<std::size_t>(i)] =
        prediction.virials_per_atom9.data() + 9 * static_cast<std::size_t>(i);
  }
  return prediction;
}

LammpsPrediction run_oracle(
    const std::string& model_path,
    LammpsInputStorage& input) {
  NEP nep(model_path);
  LammpsPrediction prediction = make_prediction_storage(input.nlocal);
  nep.compute_for_lammps(
      input.nlocal,
      input.nlocal,
      input.ilist.data(),
      input.numneigh.data(),
      input.firstneigh.data(),
      input.types.data(),
      input.type_map.data(),
      input.position_rows.data(),
      prediction.total_potential,
      prediction.total_virial6,
      prediction.potential.data(),
      prediction.force_rows.data(),
      prediction.virial_rows.data());
  return prediction;
}

LammpsPrediction run_adapter(
    const std::string& model_path,
    LammpsInputStorage& input) {
  NepaModel* model = nullptr;
  if (nepa_load_model("cpu_nep3", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    std::exit(EXIT_FAILURE);
  }

  LammpsPrediction prediction = make_prediction_storage(input.nlocal);
  NepaLammpsNeighborInput lammps_input{};
  lammps_input.nlocal = input.nlocal;
  lammps_input.inum = input.nlocal;
  lammps_input.ilist = input.ilist.data();
  lammps_input.numneigh = input.numneigh.data();
  lammps_input.firstneigh = input.firstneigh.data();
  lammps_input.types = input.types.data();
  lammps_input.type_map = input.type_map.data();
  lammps_input.positions = input.position_rows.data();

  NepaLammpsNeighborResult result{};
  result.total_potential = &prediction.total_potential;
  result.total_virial6 = prediction.total_virial6;
  result.potential_per_atom = prediction.potential.data();
  result.forces = prediction.force_rows.data();
  result.virials_per_atom9 = prediction.virial_rows.data();

  const NepaStatus status =
      nepa_find_force_lammps_neighbors(model, &lammps_input, &result);
  nepa_free_model(model);
  if (status != NEPA_STATUS_OK) {
    std::exit(EXIT_FAILURE);
  }

  return prediction;
}

double max_abs_diff(const std::vector<double>& lhs, const std::vector<double>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }

  double max_diff = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

double max_abs_diff6(const double lhs[6], const double rhs[6]) {
  double max_diff = 0.0;
  for (int i = 0; i < 6; ++i) {
    max_diff = std::max(max_diff, std::abs(lhs[i] - rhs[i]));
  }
  return max_diff;
}

}  // namespace

int main() {
  const std::string model_path = NEP_ADAPTERS_PARITY_MODEL_PATH;
  const std::string xyz_path = NEP_ADAPTERS_PARITY_XYZ_PATH;
  constexpr double tolerance = 1.0e-10;

  if (!nep_adapters::register_cpu_nep3_engine()) {
    return EXIT_FAILURE;
  }

  const auto type_map = cpu_nep3_test::read_type_map(model_path);
  const cpu_nep3_test::Frame frame =
      cpu_nep3_test::read_first_frame(xyz_path, type_map);
  LammpsInputStorage input = make_lammps_input(frame);

  const LammpsPrediction oracle = run_oracle(model_path, input);
  const LammpsPrediction adapter = run_adapter(model_path, input);

  const double energy_diff =
      std::abs(adapter.total_potential - oracle.total_potential);
  const double virial_diff =
      max_abs_diff6(adapter.total_virial6, oracle.total_virial6);
  const double potential_diff =
      max_abs_diff(adapter.potential, oracle.potential);
  const double force_diff = max_abs_diff(adapter.forces, oracle.forces);
  const double atom_virial_diff =
      max_abs_diff(adapter.virials_per_atom9, oracle.virials_per_atom9);

  if (energy_diff > tolerance || virial_diff > tolerance ||
      potential_diff > tolerance || force_diff > tolerance ||
      atom_virial_diff > tolerance) {
    std::cerr << "cpu_nep3 LAMMPS-neighbor parity failed: energy_diff="
              << energy_diff << " virial_diff=" << virial_diff
              << " potential_diff=" << potential_diff
              << " force_diff=" << force_diff
              << " atom_virial_diff=" << atom_virial_diff
              << " tolerance=" << tolerance << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
