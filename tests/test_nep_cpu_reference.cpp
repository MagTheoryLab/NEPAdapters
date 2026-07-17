#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#if defined(NEP_ADAPTERS_REFERENCE_CUDA)
#include "nep_adapters/engines/cuda.hpp"
#else
#include "nep_adapters/engines/cpu.hpp"
#endif

#include "cpu_test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

double max_abs_diff(
    const std::vector<double>& actual,
    const std::vector<double>& expected) {
  if (actual.size() != expected.size()) {
    return INFINITY;
  }
  double error = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    error = std::max(error, std::abs(actual[index] - expected[index]));
  }
  return error;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "usage: test_nep_cpu_reference <data dir> "
                 "[--expect-unsupported]\n";
    return EXIT_FAILURE;
  }
  const std::string data_dir = argv[1];
  const bool expect_unsupported =
      argc == 3 && std::string(argv[2]) == "--expect-unsupported";
  const std::string model_path = data_dir + "/nep.txt";

#if defined(NEP_ADAPTERS_REFERENCE_CUDA)
  constexpr const char* backend_name = "cuda";
  if (!nep_adapters::register_cuda_engine()) {
    std::cerr << "failed to register cuda engine\n";
    return EXIT_FAILURE;
  }
#else
  constexpr const char* backend_name = "cpu";
  if (!nep_adapters::register_cpu_engine()) {
    std::cerr << "failed to register cpu engine\n";
    return EXIT_FAILURE;
  }
#endif
  const auto type_map = cpu_test::read_type_map(model_path);
  const cpu_test::Frame frame =
      cpu_test::read_xyz_in(data_dir + "/xyz.in", type_map);
  const std::int32_t atom_count =
      static_cast<std::int32_t>(frame.types.size());

  NepaModel* model = nullptr;
  const NepaStatus load_status =
      nepa_load_model(backend_name, model_path.c_str(), &model);
  if (load_status != NEPA_STATUS_OK || model == nullptr) {
    if (expect_unsupported && load_status == NEPA_STATUS_UNSUPPORTED) {
      std::cout << "NEP_CPU reference contract: backend=" << backend_name
                << " unsupported_model=1 message="
                << nepa_last_error_message() << '\n';
      return EXIT_SUCCESS;
    }
    std::cerr << "failed to load " << backend_name
              << " model: status=" << static_cast<int>(load_status)
              << " message=" << nepa_last_error_message() << '\n';
    return EXIT_FAILURE;
  }
  if (expect_unsupported) {
    std::cerr << backend_name << " unexpectedly accepted an unsupported model\n";
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  NepaModelInfo info{};
  if (nepa_model_info(model, &info) != NEPA_STATUS_OK ||
      info.descriptor_dim <= 0) {
    std::cerr << "invalid model info for " << backend_name << '\n';
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

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
  std::vector<double> potential(static_cast<std::size_t>(atom_count), 0.0);
  std::vector<double> forces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  std::vector<double> virials(static_cast<std::size_t>(atom_count) * 9, 0.0);
  std::vector<double> charge;
  std::vector<double> bec;
  const bool is_charge_model = nep_adapters::has_capability(
      info.capabilities, nep_adapters::Capability::charge);
  if (is_charge_model) {
    charge.resize(static_cast<std::size_t>(atom_count), 0.0);
    bec.resize(static_cast<std::size_t>(atom_count) * 9, 0.0);
  }

  NepaFindForceResult result{};
  result.energy_per_structure = energy;
  result.potential_per_atom = potential.data();
  result.forces_aos3 = forces.data();
  result.virials_per_atom_row_major9 = virials.data();
  result.charge_per_atom = charge.empty() ? nullptr : charge.data();
  result.bec_per_atom_row_major9 = bec.empty() ? nullptr : bec.data();

  std::vector<double> descriptors(
      static_cast<std::size_t>(atom_count) * info.descriptor_dim,
      0.0);
  NepaFindDescriptorResult descriptor_result{};
  descriptor_result.descriptors = descriptors.data();

  const NepaStatus force_status = nepa_find_force_batch(model, &batch, &result);
  const NepaStatus descriptor_status =
      nepa_find_descriptors(model, &batch, &descriptor_result);
  nepa_free_model(model);
  if (force_status != NEPA_STATUS_OK || descriptor_status != NEPA_STATUS_OK) {
    std::cerr << "reference calculation failed for " << backend_name
              << ": force_status=" << static_cast<int>(force_status)
              << " descriptor_status=" << static_cast<int>(descriptor_status)
              << '\n';
    return EXIT_FAILURE;
  }

  const cpu_test::Matrix expected_forces =
      cpu_test::read_plain_matrix(data_dir + "/force_analytical_ref.out", 3);
  const cpu_test::Matrix expected_virials =
      cpu_test::read_plain_matrix(data_dir + "/virial_ref.out", 9);
  const cpu_test::Matrix expected_descriptors = cpu_test::read_plain_matrix(
      data_dir + "/descriptor_ref.out",
      static_cast<std::size_t>(info.descriptor_dim));

  const double force_error = max_abs_diff(forces, expected_forces.values);
  const double virial_error = max_abs_diff(virials, expected_virials.values);
  const double descriptor_error =
      max_abs_diff(descriptors, expected_descriptors.values);
  const double potential_sum =
      std::accumulate(potential.begin(), potential.end(), 0.0);
  const double charge_sum =
      std::accumulate(charge.begin(), charge.end(), 0.0);
#if defined(NEP_ADAPTERS_REFERENCE_CUDA)
  constexpr double force_tolerance = 2.0e-5;
  constexpr double virial_tolerance = 3.0e-5;
  constexpr double descriptor_tolerance = 5.0e-6;
  constexpr double sum_tolerance = 1.0e-6;
#else
  constexpr double force_tolerance = 1.0e-10;
  constexpr double virial_tolerance = 1.0e-10;
  constexpr double descriptor_tolerance = 1.0e-10;
  constexpr double sum_tolerance = 1.0e-10;
#endif

  if (expected_forces.rows != static_cast<std::size_t>(atom_count) ||
      expected_virials.rows != static_cast<std::size_t>(atom_count) ||
      expected_descriptors.rows != static_cast<std::size_t>(atom_count) ||
      !std::isfinite(energy[0]) ||
      std::abs(energy[0] - potential_sum) > sum_tolerance ||
      force_error > force_tolerance || virial_error > virial_tolerance ||
      descriptor_error > descriptor_tolerance ||
      (is_charge_model &&
       (!cpu_test::all_finite(charge) || !cpu_test::all_finite(bec) ||
        std::abs(charge_sum) > sum_tolerance))) {
    std::cerr << "NEP_CPU reference mismatch: force=" << force_error
              << " virial=" << virial_error
              << " descriptor=" << descriptor_error
              << " energy_sum=" << std::abs(energy[0] - potential_sum)
              << " charge_sum=" << std::abs(charge_sum) << '\n';
    return EXIT_FAILURE;
  }

  std::cout << "NEP_CPU reference parity: backend=" << backend_name
            << " atoms=" << atom_count
            << " force=" << force_error
            << " virial=" << virial_error
            << " descriptor=" << descriptor_error
            << " charge_model=" << is_charge_model << '\n';
  return EXIT_SUCCESS;
}
