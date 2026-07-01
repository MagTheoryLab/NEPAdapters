#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_nep3.hpp"

#include "cpu_nep3_test_utils.hpp"
#include "nep.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Prediction {
  std::vector<double> energy;
  std::vector<double> forces_aos3;
  std::vector<double> virials_row_major9;
};

std::vector<double> positions_to_soa(const cpu_nep3_test::Frame& frame) {
  const std::size_t atom_count = frame.types.size();
  std::vector<double> positions(atom_count * 3);
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    positions[atom] = frame.positions_aos3[3 * atom + 0];
    positions[atom_count + atom] = frame.positions_aos3[3 * atom + 1];
    positions[2 * atom_count + atom] = frame.positions_aos3[3 * atom + 2];
  }
  return positions;
}

Prediction run_oracle(
    const std::string& model_path,
    const cpu_nep3_test::Frame& frame,
    int structure_count) {
  NEP nep(model_path);
  Prediction prediction;
  const std::size_t atom_count = frame.types.size();
  prediction.energy.resize(structure_count, 0.0);
  prediction.forces_aos3.resize(
      static_cast<std::size_t>(structure_count) * atom_count * 3,
      0.0);
  prediction.virials_row_major9.resize(
      static_cast<std::size_t>(structure_count) * 9,
      0.0);

  const std::vector<double> positions_soa = positions_to_soa(frame);
  std::vector<double> box(frame.box, frame.box + 9);

  for (int structure = 0; structure < structure_count; ++structure) {
    std::vector<double> potential(atom_count, 0.0);
    std::vector<double> force_soa(atom_count * 3, 0.0);
    std::vector<double> virial_soa(atom_count * 9, 0.0);

    nep.compute(frame.types, box, positions_soa, potential, force_soa, virial_soa);

    prediction.energy[structure] =
        std::accumulate(potential.begin(), potential.end(), 0.0);

    const std::size_t atom_offset =
        static_cast<std::size_t>(structure) * atom_count;
    for (std::size_t atom = 0; atom < atom_count; ++atom) {
      prediction.forces_aos3[3 * (atom_offset + atom) + 0] = force_soa[atom];
      prediction.forces_aos3[3 * (atom_offset + atom) + 1] =
          force_soa[atom_count + atom];
      prediction.forces_aos3[3 * (atom_offset + atom) + 2] =
          force_soa[2 * atom_count + atom];
    }

    for (std::size_t component = 0; component < 9; ++component) {
      const std::size_t component_offset = component * atom_count;
      for (std::size_t atom = 0; atom < atom_count; ++atom) {
        prediction.virials_row_major9[
            static_cast<std::size_t>(structure) * 9 + component] +=
            virial_soa[component_offset + atom];
      }
    }
  }

  return prediction;
}

Prediction run_adapter(
    const std::string& model_path,
    const cpu_nep3_test::Frame& frame,
    int structure_count) {
  NepaModel* model = nullptr;
  if (nepa_load_model("cpu_nep3", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    std::exit(EXIT_FAILURE);
  }

  const std::int32_t atom_count =
      static_cast<std::int32_t>(frame.types.size());
  const std::int32_t total_atoms = atom_count * structure_count;
  std::vector<std::int32_t> atom_counts(structure_count, atom_count);
  std::vector<std::int32_t> atom_offsets(structure_count, 0);
  std::vector<std::int32_t> types(total_atoms, 0);
  std::vector<double> positions(static_cast<std::size_t>(total_atoms) * 3, 0.0);
  std::vector<double> boxes(static_cast<std::size_t>(structure_count) * 9, 0.0);
  std::vector<std::int32_t> pbc(static_cast<std::size_t>(structure_count) * 3, 1);

  for (int structure = 0; structure < structure_count; ++structure) {
    atom_offsets[structure] = structure * atom_count;
    std::copy(
        frame.types.begin(),
        frame.types.end(),
        types.begin() + atom_offsets[structure]);
    std::copy(
        frame.positions_aos3.begin(),
        frame.positions_aos3.end(),
        positions.begin() + static_cast<std::size_t>(atom_offsets[structure]) * 3);
    std::copy(
        frame.box,
        frame.box + 9,
        boxes.begin() + static_cast<std::size_t>(structure) * 9);
  }

  NepaStructureBatch batch{};
  batch.num_structures = structure_count;
  batch.total_atoms = total_atoms;
  batch.atom_counts = atom_counts.data();
  batch.atom_offsets = atom_offsets.data();
  batch.types = types.data();
  batch.positions_aos3 = positions.data();
  batch.boxes_row_major9 = boxes.data();
  batch.pbc_flags3 = pbc.data();

  Prediction prediction;
  prediction.energy.resize(structure_count, 0.0);
  prediction.forces_aos3.resize(static_cast<std::size_t>(total_atoms) * 3, 0.0);
  prediction.virials_row_major9.resize(
      static_cast<std::size_t>(structure_count) * 9,
      0.0);

  NepaFindForceResult result{};
  result.energy_per_structure = prediction.energy.data();
  result.forces_aos3 = prediction.forces_aos3.data();
  result.virials_row_major9 = prediction.virials_row_major9.data();

  const NepaStatus status = nepa_find_force_batch(model, &batch, &result);
  nepa_free_model(model);
  if (status != NEPA_STATUS_OK) {
    std::exit(EXIT_FAILURE);
  }

  return prediction;
}

double max_abs_diff(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }

  double max_diff = 0.0;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    max_diff = std::max(max_diff, std::abs(lhs[index] - rhs[index]));
  }
  return max_diff;
}

std::vector<double> repeat_forces(
    const cpu_nep3_test::Frame& frame,
    int structure_count) {
  std::vector<double> values;
  values.reserve(
      static_cast<std::size_t>(structure_count) *
      frame.reference_forces_aos3.size());
  for (int structure = 0; structure < structure_count; ++structure) {
    values.insert(
        values.end(),
        frame.reference_forces_aos3.begin(),
        frame.reference_forces_aos3.end());
  }
  return values;
}

std::vector<double> repeat_virials(
    const cpu_nep3_test::Frame& frame,
    int structure_count) {
  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(structure_count) * 9);
  for (int structure = 0; structure < structure_count; ++structure) {
    values.insert(
        values.end(),
        frame.reference_virial_row_major9,
        frame.reference_virial_row_major9 + 9);
  }
  return values;
}

}  // namespace

int main() {
  const std::string model_path = NEP_ADAPTERS_PARITY_MODEL_PATH;
  const std::string xyz_path = NEP_ADAPTERS_PARITY_XYZ_PATH;
  constexpr int structure_count = 2;
  constexpr double tolerance = 1.0e-10;

  if (!nep_adapters::register_cpu_nep3_engine()) {
    return EXIT_FAILURE;
  }

  const auto type_map = cpu_nep3_test::read_type_map(model_path);
  const cpu_nep3_test::Frame frame =
      cpu_nep3_test::read_first_frame(xyz_path, type_map);

  const Prediction oracle = run_oracle(model_path, frame, structure_count);
  const Prediction adapter = run_adapter(model_path, frame, structure_count);

  const double energy_diff = max_abs_diff(adapter.energy, oracle.energy);
  const double force_diff = max_abs_diff(adapter.forces_aos3, oracle.forces_aos3);
  const double virial_diff =
      max_abs_diff(adapter.virials_row_major9, oracle.virials_row_major9);
  double baseline_energy_diff = 0.0;
  double baseline_force_diff = 0.0;
  double baseline_virial_diff = 0.0;
  const bool has_baseline_labels =
      frame.has_reference_forces && frame.has_reference_virial;
  if (has_baseline_labels) {
    const std::vector<double> expected_energy(
        static_cast<std::size_t>(structure_count),
        frame.reference_energy);
    const std::vector<double> expected_forces =
        repeat_forces(frame, structure_count);
    const std::vector<double> expected_virials =
        repeat_virials(frame, structure_count);
    baseline_energy_diff = max_abs_diff(adapter.energy, expected_energy);
    baseline_force_diff = max_abs_diff(adapter.forces_aos3, expected_forces);
    baseline_virial_diff =
        max_abs_diff(adapter.virials_row_major9, expected_virials);
  }

  if (energy_diff > tolerance || force_diff > tolerance ||
      virial_diff > tolerance ||
      (has_baseline_labels &&
       (baseline_energy_diff > tolerance ||
        baseline_force_diff > tolerance ||
        baseline_virial_diff > tolerance))) {
    std::cerr << "cpu_nep3 parity failed: energy_diff=" << energy_diff
              << " force_diff=" << force_diff
              << " virial_diff=" << virial_diff
              << " baseline_energy_diff=" << baseline_energy_diff
              << " baseline_force_diff=" << baseline_force_diff
              << " baseline_virial_diff=" << baseline_virial_diff
              << " tolerance=" << tolerance << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
