#include "cpu_test_utils.hpp"
#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

NepaStatus find_force(
    NepaModel* model,
    const NepaStructureBatch& batch) {
  std::vector<double> energies(
      static_cast<std::size_t>(batch.num_structures), 0.0);
  std::vector<double> forces(
      static_cast<std::size_t>(batch.total_atoms) * 3, 0.0);
  NepaFindForceResult result{};
  result.energy_per_structure = energies.data();
  result.forces_aos3 = forces.data();
  return nepa_find_force_batch(model, &batch, &result);
}

bool rejects_model_text(
    const std::filesystem::path& path,
    const std::string& text) {
  {
    std::ofstream output(path);
    output << text;
  }
  NepaModel* model = nullptr;
  const NepaStatus status =
      nepa_load_model("cpu", path.string().c_str(), &model);
  if (model != nullptr) {
    nepa_free_model(model);
  }
  std::filesystem::remove(path);
  return status == NEPA_STATUS_RUNTIME_ERROR;
}

}  // namespace

int main() {
  if (!nep_adapters::register_cpu_engine()) {
    return EXIT_FAILURE;
  }

  const std::string data_dir = NEP_ADAPTERS_CPU_TEST_DATA_DIR;
  const std::string model_path = data_dir + "/nep.txt";
  const cpu_test::Frame frame = cpu_test::read_first_frame(
      data_dir + "/train.xyz",
      cpu_test::read_type_map(model_path));
  const std::int32_t atom_count =
      static_cast<std::int32_t>(frame.types.size());
  if (atom_count < 4) {
    std::cerr << "input validation fixture is too small\n";
    return EXIT_FAILURE;
  }

  NepaModel* model = nullptr;
  NepaModelInfo info{};
  if (nepa_load_model("cpu", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr ||
      nepa_model_info(model, &info) != NEPA_STATUS_OK) {
    return EXIT_FAILURE;
  }

  std::vector<std::int32_t> types = frame.types;
  std::int32_t counts[] = {atom_count};
  std::int32_t offsets[] = {0};
  std::int32_t pbc[] = {1, 1, 1};
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = atom_count;
  batch.atom_counts = counts;
  batch.atom_offsets = offsets;
  batch.types = types.data();
  batch.positions_aos3 = frame.positions_aos3.data();
  batch.boxes_row_major9 = frame.box;
  batch.pbc_flags3 = pbc;

  const NepaStatus valid_status = find_force(model, batch);
  types[0] = info.num_types;
  const NepaStatus invalid_type_status = find_force(model, batch);
  types[0] = frame.types[0];

  const std::int32_t first_count = atom_count / 2;
  const std::int32_t second_count = atom_count - first_count;
  std::int32_t overlap_counts[] = {first_count, second_count};
  std::int32_t overlap_offsets[] = {0, 0};
  std::int32_t overlap_pbc[] = {1, 1, 1, 1, 1, 1};
  batch.num_structures = 2;
  batch.atom_counts = overlap_counts;
  batch.atom_offsets = overlap_offsets;
  batch.pbc_flags3 = overlap_pbc;
  const NepaStatus overlap_status = find_force(model, batch);

  std::int32_t gap_counts[] = {first_count, second_count - 1};
  std::int32_t gap_offsets[] = {0, first_count + 1};
  batch.atom_counts = gap_counts;
  batch.atom_offsets = gap_offsets;
  const NepaStatus gap_status = find_force(model, batch);

  // A dense periodic cell exceeds the native neighbor capacity. The adapter
  // must preserve the native exception text instead of returning a bare
  // runtime-error status.
  constexpr std::int32_t kDenseStructures = 2;
  constexpr std::int32_t kDenseAtomsPerStructure = 8;
  constexpr std::int32_t kDenseAtoms =
      kDenseStructures * kDenseAtomsPerStructure;
  std::vector<std::int32_t> dense_types(kDenseAtoms, 0);
  std::vector<double> dense_positions(
      static_cast<std::size_t>(kDenseAtoms) * 3, 0.0);
  std::int32_t dense_counts[] = {
      kDenseAtomsPerStructure, kDenseAtomsPerStructure};
  std::int32_t dense_offsets[] = {0, kDenseAtomsPerStructure};
  double dense_boxes[] = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
  };
  std::int32_t dense_pbc[] = {1, 1, 1, 1, 1, 1};
  batch.num_structures = kDenseStructures;
  batch.total_atoms = kDenseAtoms;
  batch.atom_counts = dense_counts;
  batch.atom_offsets = dense_offsets;
  batch.types = dense_types.data();
  batch.positions_aos3 = dense_positions.data();
  batch.boxes_row_major9 = dense_boxes;
  batch.pbc_flags3 = dense_pbc;
  const NepaStatus dense_status = find_force(model, batch);
  const std::string dense_error = nepa_last_error_message();

  nepa_free_model(model);
  const std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path();
  std::string oversized_header = "nep4 95";
  for (int type = 0; type < 95; ++type) {
    oversized_header += " H";
  }
  oversized_header += "\n";
  const bool oversized_model_rejected = rejects_model_text(
      temp_dir / "nep_adapters_cpu_oversized_types.nep",
      oversized_header);
  const bool truncated_spin_rejected = rejects_model_text(
      temp_dir / "nep_adapters_cpu_truncated_spin.nep",
      "nep4_spin 1 Fe\nspin_mode 1 1\nspin_chiral\n");
  const bool unknown_element_rejected = rejects_model_text(
      temp_dir / "nep_adapters_cpu_unknown_element.nep",
      "nep4 1 NotAnElement\n");

  if (valid_status != NEPA_STATUS_OK ||
      invalid_type_status != NEPA_STATUS_INVALID_ARGUMENT ||
      overlap_status != NEPA_STATUS_INVALID_ARGUMENT ||
      gap_status != NEPA_STATUS_INVALID_ARGUMENT ||
      dense_status != NEPA_STATUS_RUNTIME_ERROR ||
      dense_error.find("neighbor capacity exceeded") == std::string::npos ||
      !oversized_model_rejected ||
      !truncated_spin_rejected ||
      !unknown_element_rejected) {
    std::cerr << "unexpected validation statuses valid=" << valid_status
              << " type=" << invalid_type_status
              << " overlap=" << overlap_status
              << " gap=" << gap_status
              << " dense=" << dense_status
              << " dense_error=" << dense_error
              << " oversized=" << oversized_model_rejected
              << " truncated_spin=" << truncated_spin_rejected
              << " unknown_element=" << unknown_element_rejected << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
