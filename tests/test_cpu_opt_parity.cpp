#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#include "nep_adapters/engines/cpu_nep3.hpp"
#include "nep_adapters/engines/cpu_opt.hpp"

#include "cpu_nep3_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Prediction {
  std::vector<double> energy;
  std::vector<double> forces_aos3;
  std::vector<double> virials_row_major9;
  std::vector<double> descriptors;
};

Prediction run_adapter(
    const char* engine_name,
    const std::string& model_path,
    const cpu_nep3_test::Frame& frame,
    int structure_count) {
  NepaModel* model = nullptr;
  if (nepa_load_model(engine_name, model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    std::exit(EXIT_FAILURE);
  }

  NepaModelInfo model_info{};
  if (nepa_model_info(model, &model_info) != NEPA_STATUS_OK ||
      model_info.descriptor_dim <= 0 ||
      !nep_adapters::has_capability(
          model_info.capabilities, nep_adapters::Capability::batch_find_force) ||
      !nep_adapters::has_capability(
          model_info.capabilities, nep_adapters::Capability::descriptors)) {
    nepa_free_model(model);
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
  prediction.descriptors.resize(
      static_cast<std::size_t>(total_atoms) * model_info.descriptor_dim,
      0.0);

  NepaFindForceResult result{};
  result.energy_per_structure = prediction.energy.data();
  result.forces_aos3 = prediction.forces_aos3.data();
  result.virials_row_major9 = prediction.virials_row_major9.data();

  const NepaStatus force_status = nepa_find_force_batch(model, &batch, &result);
  NepaFindDescriptorResult descriptor_result{};
  descriptor_result.descriptors = prediction.descriptors.data();
  const NepaStatus descriptor_status =
      nepa_find_descriptors(model, &batch, &descriptor_result);
  nepa_free_model(model);
  if (force_status != NEPA_STATUS_OK || descriptor_status != NEPA_STATUS_OK) {
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

std::vector<double> repeat_energy(
    const cpu_nep3_test::Frame& frame,
    int structure_count) {
  return std::vector<double>(
      static_cast<std::size_t>(structure_count),
      frame.reference_energy);
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

std::vector<double> repeat_descriptors(
    const cpu_nep3_test::Matrix& matrix,
    int structure_count) {
  std::vector<double> values;
  values.reserve(
      static_cast<std::size_t>(structure_count) * matrix.values.size());
  for (int structure = 0; structure < structure_count; ++structure) {
    values.insert(values.end(), matrix.values.begin(), matrix.values.end());
  }
  return values;
}

}  // namespace

int main() {
  constexpr int structure_count = 2;
  constexpr double tolerance = 1.0e-10;

  if (!nep_adapters::register_cpu_nep3_engine() ||
      !nep_adapters::register_cpu_opt_engine()) {
    return EXIT_FAILURE;
  }

  const std::string model_path = NEP_ADAPTERS_PARITY_MODEL_PATH;
  const std::string xyz_path = NEP_ADAPTERS_PARITY_XYZ_PATH;
  const auto type_map = cpu_nep3_test::read_type_map(model_path);
  const cpu_nep3_test::Frame frame =
      cpu_nep3_test::read_first_frame(xyz_path, type_map);

  const Prediction oracle =
      run_adapter("cpu_nep3", model_path, frame, structure_count);
  const Prediction opt =
      run_adapter("cpu_opt", model_path, frame, structure_count);

  const double energy_diff = max_abs_diff(opt.energy, oracle.energy);
  const double force_diff = max_abs_diff(opt.forces_aos3, oracle.forces_aos3);
  const double virial_diff =
      max_abs_diff(opt.virials_row_major9, oracle.virials_row_major9);
  const double descriptor_diff =
      max_abs_diff(opt.descriptors, oracle.descriptors);

  double baseline_energy_diff = 0.0;
  double baseline_force_diff = 0.0;
  double baseline_virial_diff = 0.0;
  double baseline_descriptor_diff = 0.0;
  const bool has_baseline_labels =
      frame.has_reference_forces && frame.has_reference_virial;
  if (has_baseline_labels) {
    baseline_energy_diff =
        max_abs_diff(opt.energy, repeat_energy(frame, structure_count));
    baseline_force_diff =
        max_abs_diff(opt.forces_aos3, repeat_forces(frame, structure_count));
    baseline_virial_diff =
        max_abs_diff(opt.virials_row_major9, repeat_virials(frame, structure_count));
  }

#if defined(NEP_ADAPTERS_EXPECTED_DESCRIPTOR_PATH)
  const cpu_nep3_test::Matrix expected_descriptors =
      cpu_nep3_test::read_matrix(NEP_ADAPTERS_EXPECTED_DESCRIPTOR_PATH);
  const std::vector<double> repeated_descriptors =
      repeat_descriptors(expected_descriptors, structure_count);
  baseline_descriptor_diff =
      max_abs_diff(opt.descriptors, repeated_descriptors);
#endif

  if (energy_diff > tolerance || force_diff > tolerance ||
      virial_diff > tolerance || descriptor_diff > tolerance ||
      (has_baseline_labels &&
       (baseline_energy_diff > tolerance ||
        baseline_force_diff > tolerance ||
        baseline_virial_diff > tolerance)) ||
      baseline_descriptor_diff > tolerance) {
    std::cerr << "cpu_opt parity failed: energy_diff=" << energy_diff
              << " force_diff=" << force_diff
              << " virial_diff=" << virial_diff
              << " descriptor_diff=" << descriptor_diff
              << " baseline_energy_diff=" << baseline_energy_diff
              << " baseline_force_diff=" << baseline_force_diff
              << " baseline_virial_diff=" << baseline_virial_diff
              << " baseline_descriptor_diff=" << baseline_descriptor_diff
              << " tolerance=" << tolerance << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
