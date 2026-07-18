#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#include "nep_adapters/engines/cpu.hpp"

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

}  // namespace

int main() {
  const std::string data_dir = NEP_ADAPTERS_QNEP_TEST_DATA_DIR;
  const std::string model_path = data_dir + "/nep.txt";
  const std::string xyz_path = data_dir + "/xyz.in";

  if (!nep_adapters::register_cpu_engine()) {
    return EXIT_FAILURE;
  }

  const auto type_map = cpu_test::read_type_map(model_path);
  const cpu_test::Frame frame = read_xyz_in(xyz_path, type_map);

  NepaModel* model = nullptr;
  if (nepa_load_model("cpu", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }

  NepaModelInfo model_info{};
  if (nepa_model_info(model, &model_info) != NEPA_STATUS_OK ||
      !nep_adapters::has_capability(
          model_info.capabilities, nep_adapters::Capability::charge)) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

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

  double energy[] = {0.0};
  std::vector<double> forces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  std::vector<double> charge(static_cast<std::size_t>(atom_count), 0.0);
  std::vector<double> bec(static_cast<std::size_t>(atom_count) * 9, 0.0);

  NepaFindForceResult result{};
  result.energy_per_structure = energy;
  result.forces_aos3 = forces.data();
  result.charge_per_atom = charge.data();
  result.bec_per_atom_row_major9 = bec.data();

  const NepaStatus status = nepa_find_charge_batch(model, &batch, &result);
  nepa_free_model(model);
  if (status != NEPA_STATUS_OK || !std::isfinite(energy[0]) ||
      !cpu_test::all_finite(forces) ||
      !cpu_test::all_finite(charge) || !cpu_test::all_finite(bec)) {
    return EXIT_FAILURE;
  }

  const double total_charge =
      std::accumulate(charge.begin(), charge.end(), 0.0);
  if (std::abs(total_charge) > 1.0e-10) {
    std::cerr << "qNEP charge sum failed: " << total_charge << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
