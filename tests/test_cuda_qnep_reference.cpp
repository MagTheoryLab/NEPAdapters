#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include "cpu_test_utils.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

cpu_test::Frame read_xyz_in(
    const std::string& xyz_path,
    const std::unordered_map<std::string, std::int32_t>& type_map) {
  std::ifstream input(xyz_path);
  int atom_count = 0;
  input >> atom_count;
  if (!input || atom_count <= 0) {
    std::exit(EXIT_FAILURE);
  }

  cpu_test::Frame frame;
  frame.types.resize(static_cast<std::size_t>(atom_count));
  frame.positions_aos3.resize(static_cast<std::size_t>(atom_count) * 3);
  input >> frame.box[0] >> frame.box[3] >> frame.box[6] >> frame.box[1] >>
      frame.box[4] >> frame.box[7] >> frame.box[2] >> frame.box[5] >>
      frame.box[8];

  for (int atom = 0; atom < atom_count; ++atom) {
    std::string symbol;
    input >> symbol >> frame.positions_aos3[3 * atom + 0] >>
        frame.positions_aos3[3 * atom + 1] >>
        frame.positions_aos3[3 * atom + 2];
    const auto found = type_map.find(symbol);
    if (!input || found == type_map.end()) {
      std::exit(EXIT_FAILURE);
    }
    frame.types[atom] = found->second;
  }
  return frame;
}

std::vector<double> read_columns(
    const std::string& path,
    int columns,
    int rows) {
  std::ifstream input(path);
  std::vector<double> values(static_cast<std::size_t>(rows) * columns, 0.0);
  for (double& value : values) {
    input >> value;
    if (!input) {
      std::exit(EXIT_FAILURE);
    }
  }
  return values;
}

bool near_all(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    double tolerance,
    const char* label,
    int columns) {
  double max_error = 0.0;
  std::size_t max_index = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double error = std::abs(actual[index] - expected[index]);
    if (error > max_error) {
      max_error = error;
      max_index = index;
    }
  }
  if (max_error > tolerance) {
    const std::size_t row = max_index / static_cast<std::size_t>(columns);
    const std::size_t component = max_index % static_cast<std::size_t>(columns);
    std::cerr << label << " mismatch max_error=" << max_error
              << " index=" << max_index
              << " row=" << row
              << " component=" << component
              << " actual=" << actual[max_index]
              << " expected=" << expected[max_index] << '\n';
    return false;
  }
  return true;
}

struct ChargePrediction {
  double energy = 0.0;
  std::vector<double> potential;
  std::vector<double> force;
  std::vector<double> total_virial;
  std::vector<double> per_atom_virial;
  std::vector<double> charge;
  std::vector<double> bec;
};

ChargePrediction run_small_box_charge(NepaModel* model, std::int32_t type) {
  std::int32_t atom_counts[] = {1};
  std::int32_t atom_offsets[] = {0};
  std::int32_t types[] = {type};
  double positions[] = {0.2, 0.3, 0.4};
  double box[] = {
      6.0, 0.0, 0.0,
      0.0, 6.0, 0.0,
      0.0, 0.0, 6.0,
  };
  std::int32_t pbc[] = {1, 1, 1};
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = 1;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions;
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;

  ChargePrediction prediction;
  prediction.potential.resize(1);
  prediction.force.resize(3);
  prediction.total_virial.resize(9);
  prediction.per_atom_virial.resize(9);
  prediction.charge.resize(1);
  prediction.bec.resize(9);
  NepaFindForceResult result{};
  result.energy_per_structure = &prediction.energy;
  result.potential_per_atom = prediction.potential.data();
  result.forces_aos3 = prediction.force.data();
  result.virials_row_major9 = prediction.total_virial.data();
  result.virials_per_atom_row_major9 =
      prediction.per_atom_virial.data();
  result.charge_per_atom = prediction.charge.data();
  result.bec_per_atom_row_major9 = prediction.bec.data();
  const NepaStatus status = nepa_find_charge_batch(model, &batch, &result);
  if (status != NEPA_STATUS_OK) {
    std::cerr << "small-box qNEP failed status=" << status
              << " error=" << nepa_last_error_message() << '\n';
    std::exit(EXIT_FAILURE);
  }
  return prediction;
}

}  // namespace

int main() {
  const std::string data_dir = NEP_ADAPTERS_QNEP_TEST_DATA_DIR;
  const std::string model_path = data_dir + "/nep.txt";
  const std::string xyz_path = data_dir + "/xyz.in";

  if (!nep_adapters::register_cpu_engine() ||
      !nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }

  const auto type_map = cpu_test::read_type_map(model_path);
  const cpu_test::Frame frame = read_xyz_in(xyz_path, type_map);
  const std::int32_t atom_count = static_cast<std::int32_t>(frame.types.size());
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

  NepaModel* model = nullptr;
  if (nepa_load_model("cuda", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }

  double energy[] = {0.0};
  std::vector<double> forces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  std::vector<double> total_virial(9, 0.0);
  std::vector<double> per_atom_virial(
      static_cast<std::size_t>(atom_count) * 9,
      0.0);
  std::vector<double> charge(static_cast<std::size_t>(atom_count), 0.0);
  std::vector<double> bec(static_cast<std::size_t>(atom_count) * 9, 0.0);

  NepaFindForceResult result{};
  result.energy_per_structure = energy;
  result.forces_aos3 = forces.data();
  result.virials_row_major9 = total_virial.data();
  result.virials_per_atom_row_major9 = per_atom_virial.data();
  result.charge_per_atom = charge.data();
  result.bec_per_atom_row_major9 = bec.data();

  const NepaStatus status = nepa_find_charge_batch(model, &batch, &result);
  const std::string error_message = nepa_last_error_message();
  nepa_free_model(model);
  if (std::getenv("NEP_ADAPTERS_TEST_EXPECT_PPPM_DISABLED") != nullptr) {
    if (status == NEPA_STATUS_RUNTIME_ERROR &&
        error_message.find("PPPM support is disabled") != std::string::npos) {
      return EXIT_SUCCESS;
    }
    std::cerr << "disabled PPPM did not fail closed: status=" << status
              << " error=" << error_message << '\n';
    return EXIT_FAILURE;
  }
  if (status != NEPA_STATUS_OK || !std::isfinite(energy[0]) ||
      !cpu_test::all_finite(forces) ||
      !cpu_test::all_finite(total_virial) ||
      !cpu_test::all_finite(per_atom_virial) ||
      !cpu_test::all_finite(charge)) {
    std::cerr << "CUDA qNEP status=" << status
              << " error=" << error_message << '\n';
    return EXIT_FAILURE;
  }

  const double total_charge =
      std::accumulate(charge.begin(), charge.end(), 0.0);
  if (std::abs(total_charge) > 1.0e-8) {
    std::cerr << "CUDA qNEP charge sum failed: " << total_charge << '\n';
    return EXIT_FAILURE;
  }

  const std::vector<double> force_ref =
      read_columns(data_dir + "/force_analytical_ref.out", 3, atom_count);
  const std::vector<double> per_atom_virial_ref =
      read_columns(data_dir + "/virial_ref.out", 9, atom_count);
  std::vector<double> total_virial_ref(9, 0.0);
  for (int atom = 0; atom < atom_count; ++atom) {
    for (int component = 0; component < 9; ++component) {
      total_virial_ref[component] +=
          per_atom_virial_ref[9 * atom + component];
    }
  }

  const bool force_ok =
      near_all(forces, force_ref, 2.0e-5, "qNEP force", 3);
  const bool virial_ok =
      near_all(total_virial, total_virial_ref, 1.0e-4, "qNEP virial", 9);
  const bool per_atom_virial_ok = near_all(
      per_atom_virial,
      per_atom_virial_ref,
      3.0e-5,
      "qNEP per-atom virial",
      9);

  NepaModel* cpu_small_model = nullptr;
  NepaModel* cuda_small_model = nullptr;
  if (nepa_load_model(
          "cpu",
          model_path.c_str(),
          &cpu_small_model) != NEPA_STATUS_OK ||
      nepa_load_model(
          "cuda",
          model_path.c_str(),
          &cuda_small_model) != NEPA_STATUS_OK ||
      cpu_small_model == nullptr || cuda_small_model == nullptr) {
    nepa_free_model(cpu_small_model);
    nepa_free_model(cuda_small_model);
    return EXIT_FAILURE;
  }
  const ChargePrediction cpu_small =
      run_small_box_charge(cpu_small_model, frame.types.front());
  const ChargePrediction cuda_small =
      run_small_box_charge(cuda_small_model, frame.types.front());
  nepa_free_model(cpu_small_model);
  nepa_free_model(cuda_small_model);
  const bool small_box_ok =
      std::abs(cpu_small.energy - cuda_small.energy) < 2.0e-5 &&
      near_all(
          cuda_small.potential,
          cpu_small.potential,
          2.0e-5,
          "small-box qNEP potential",
          1) &&
      near_all(
          cuda_small.force,
          cpu_small.force,
          2.0e-5,
          "small-box qNEP force",
          3) &&
      near_all(
          cuda_small.total_virial,
          cpu_small.total_virial,
          1.0e-4,
          "small-box qNEP virial",
          9) &&
      near_all(
          cuda_small.per_atom_virial,
          cpu_small.per_atom_virial,
          1.0e-4,
          "small-box qNEP per-atom virial",
          9) &&
      near_all(
          cuda_small.charge,
          cpu_small.charge,
          2.0e-5,
          "small-box qNEP charge",
          1) &&
      near_all(
          cuda_small.bec,
          cpu_small.bec,
          1.0e-4,
          "small-box qNEP BEC",
          9);
  return force_ok && virial_ok && per_atom_virial_ok && small_box_ok
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
