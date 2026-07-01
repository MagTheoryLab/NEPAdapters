#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#include "nep_adapters/engines/cpu_nep3.hpp"

#include "cpu_nep3_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

double max_abs_diff(const std::vector<double>& lhs, const std::vector<double>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  double max_diff = 0.0;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    max_diff = std::max(max_diff, std::abs(lhs[index] - rhs[index]));
  }
  return max_diff;
}

double max_abs_diff9(const double lhs[9], const double rhs[9]) {
  double max_diff = 0.0;
  for (int index = 0; index < 9; ++index) {
    max_diff = std::max(max_diff, std::abs(lhs[index] - rhs[index]));
  }
  return max_diff;
}

}  // namespace

int main() {
  const std::string data_dir = NEP_ADAPTERS_CPU_NEP3_TEST_DATA_DIR;
  const std::string model_path = data_dir + "/nep.txt";
  const std::string xyz_path = data_dir + "/train.xyz";

  if (!nep_adapters::register_cpu_nep3_engine()) {
    return EXIT_FAILURE;
  }

  const auto type_map = cpu_nep3_test::read_type_map(model_path);
  cpu_nep3_test::Frame frame =
      cpu_nep3_test::read_first_frame(xyz_path, type_map);

  NepaModel* model = nullptr;
  if (nepa_load_model("cpu_nep3", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }

  NepaModelInfo model_info{};
  if (nepa_model_info(model, &model_info) != NEPA_STATUS_OK ||
      model_info.cutoff_max <= 0.0 || model_info.num_types <= 0) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }
  if (!nep_adapters::has_capability(
          model_info.capabilities, nep_adapters::Capability::batch_find_force) ||
      !nep_adapters::has_capability(
          model_info.capabilities, nep_adapters::Capability::external_neighbors) ||
      !nep_adapters::has_capability(
          model_info.capabilities, nep_adapters::Capability::descriptors) ||
      model_info.descriptor_dim <= 0) {
    nepa_free_model(model);
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
  std::vector<double> descriptors(
      static_cast<std::size_t>(atom_count) * model_info.descriptor_dim,
      0.0);
  NepaFindDescriptorResult descriptor_result{};
  descriptor_result.descriptors = descriptors.data();
  const NepaStatus descriptor_status =
      nepa_find_descriptors(model, &batch, &descriptor_result);
  nepa_free_model(model);
  if (status != NEPA_STATUS_OK || descriptor_status != NEPA_STATUS_OK) {
    return EXIT_FAILURE;
  }

  const double force_l1 = std::accumulate(
      forces.begin(),
      forces.end(),
      0.0,
      [](double sum, double value) { return sum + std::abs(value); });

  if (!std::isfinite(energy[0]) || !cpu_nep3_test::all_finite(forces) ||
      force_l1 <= 0.0) {
    return EXIT_FAILURE;
  }

  for (double component : virial) {
    if (!std::isfinite(component)) {
      return EXIT_FAILURE;
    }
  }

  constexpr double tolerance = 1.0e-10;
  const double energy_diff = std::abs(energy[0] - frame.reference_energy);
  const double force_diff = max_abs_diff(forces, frame.reference_forces_aos3);
  const double virial_diff =
      max_abs_diff9(virial, frame.reference_virial_row_major9);
  const cpu_nep3_test::Matrix expected_descriptors =
      cpu_nep3_test::read_matrix(data_dir + "/descriptor.txt");
  const double descriptor_diff =
      max_abs_diff(descriptors, expected_descriptors.values);
  if (!frame.has_reference_forces || !frame.has_reference_virial ||
      expected_descriptors.rows != static_cast<std::size_t>(atom_count) ||
      expected_descriptors.cols !=
          static_cast<std::size_t>(model_info.descriptor_dim) ||
      energy_diff > tolerance || force_diff > tolerance ||
      virial_diff > tolerance || descriptor_diff > tolerance) {
    std::cerr << "cpu_nep3 baseline failed: energy_diff=" << energy_diff
              << " force_diff=" << force_diff
              << " virial_diff=" << virial_diff
              << " descriptor_diff=" << descriptor_diff
              << " tolerance=" << tolerance << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
