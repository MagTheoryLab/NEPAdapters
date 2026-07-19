#include "cpu_test_utils.hpp"
#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#include "nep_adapters/engines/cpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct OwnedBatch {
  std::vector<std::int32_t> atom_counts;
  std::vector<std::int32_t> atom_offsets;
  std::vector<std::int32_t> types;
  std::vector<double> positions;
  std::vector<double> boxes;
  std::vector<std::int32_t> pbc;
  NepaStructureBatch view{};
};

OwnedBatch repeat_frame(const cpu_test::Frame& frame, int count) {
  OwnedBatch out;
  const int atoms = static_cast<int>(frame.types.size());
  for (int structure = 0; structure < count; ++structure) {
    out.atom_counts.push_back(atoms);
    out.atom_offsets.push_back(structure * atoms);
    out.types.insert(out.types.end(), frame.types.begin(), frame.types.end());
    out.positions.insert(
        out.positions.end(), frame.positions_aos3.begin(), frame.positions_aos3.end());
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        out.boxes.push_back(frame.box[3 * column + row]);
      }
    }
    out.pbc.insert(out.pbc.end(), {1, 1, 1});
  }
  out.view.num_structures = count;
  out.view.total_atoms = atoms * count;
  out.view.atom_counts = out.atom_counts.data();
  out.view.atom_offsets = out.atom_offsets.data();
  out.view.types = out.types.data();
  out.view.positions_aos3 = out.positions.data();
  out.view.boxes_row_major9 = out.boxes.data();
  out.view.pbc_flags3 = out.pbc.data();
  return out;
}

bool close(double actual, double expected, double atol, double rtol) {
  return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}

bool check_response(
    const std::string& model_path,
    const std::string& structure_path,
    const std::string& descriptor_path,
    NepaModelKind expected_kind,
    nep_adapters::Capability expected_capability,
    const std::vector<double>& golden) {
  const auto type_map = cpu_test::read_type_map(model_path);
  const cpu_test::Frame frame = cpu_test::read_first_frame(structure_path, type_map);
  OwnedBatch batch = repeat_frame(frame, 2);
  NepaModel* model = nullptr;
  if (nepa_load_model("cpu", model_path.c_str(), &model) != NEPA_STATUS_OK) {
    std::cerr << "response model load failed: " << nepa_last_error_message()
              << '\n';
    return false;
  }
  NepaModelKind kind = NEPA_MODEL_KIND_ORDINARY;
  NepaModelInfo info{};
  const bool metadata_ok =
      nepa_model_kind(model, &kind) == NEPA_STATUS_OK && kind == expected_kind &&
      nepa_model_info(model, &info) == NEPA_STATUS_OK &&
      nep_adapters::has_capability(info.capabilities, expected_capability) &&
      nep_adapters::has_capability(
          info.capabilities, nep_adapters::Capability::descriptors) &&
      info.descriptor_dim > 0;
  std::vector<double> values(golden.size() * 2, 0.0);
  NepaStatus status = NEPA_STATUS_UNSUPPORTED;
  if (expected_kind == NEPA_MODEL_KIND_DIPOLE) {
    NepaDipoleResult result{values.data()};
    status = nepa_find_dipoles(model, &batch.view, &result);
  } else {
    NepaPolarizabilityResult result{values.data()};
    status = nepa_find_polarizabilities(model, &batch.view, &result);
  }
  std::vector<double> force(static_cast<std::size_t>(batch.view.total_atoms) * 3);
  double energy[2]{};
  NepaFindForceResult force_result{};
  force_result.energy_per_structure = energy;
  force_result.forces_aos3 = force.data();
  const NepaStatus ordinary_status =
      nepa_find_force_batch(model, &batch.view, &force_result);
  std::vector<double> descriptors(
      static_cast<std::size_t>(batch.view.total_atoms) * info.descriptor_dim,
      0.0);
  NepaFindDescriptorResult descriptor_result{descriptors.data()};
  const NepaStatus descriptor_status =
      nepa_find_descriptors(model, &batch.view, &descriptor_result);
  nepa_free_model(model);
  if (!metadata_ok || status != NEPA_STATUS_OK ||
      ordinary_status != NEPA_STATUS_UNSUPPORTED ||
      descriptor_status != NEPA_STATUS_OK) {
    std::cerr << "response metadata/status mismatch kind=" << kind
              << " status=" << status << " ordinary=" << ordinary_status
              << " descriptor=" << descriptor_status
              << " error=" << nepa_last_error_message() << '\n';
    return false;
  }
  const double atol = golden.size() == 3 ? 1.0e-5 : 2.0e-4;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (!close(values[i], golden[i % golden.size()], atol, 5.0e-4)) {
      std::cerr << "response golden mismatch index=" << i
                << " actual=" << values[i]
                << " expected=" << golden[i % golden.size()] << '\n';
      return false;
    }
  }
  const cpu_test::Matrix expected_descriptors =
      cpu_test::read_matrix(descriptor_path);
  const std::size_t atoms_per_structure = frame.types.size();
  if (expected_descriptors.rows != atoms_per_structure ||
      expected_descriptors.cols != static_cast<std::size_t>(info.descriptor_dim)) {
    std::cerr << "response descriptor golden shape mismatch\n";
    return false;
  }
  for (int structure = 0; structure < batch.view.num_structures; ++structure) {
    const std::size_t offset =
        static_cast<std::size_t>(structure) * atoms_per_structure * info.descriptor_dim;
    for (std::size_t i = 0; i < expected_descriptors.values.size(); ++i) {
      if (!close(descriptors[offset + i], expected_descriptors.values[i],
                 1.0e-10, 1.0e-10)) {
        std::cerr << "response descriptor mismatch structure=" << structure
                  << " index=" << i << " actual=" << descriptors[offset + i]
                  << " expected=" << expected_descriptors.values[i] << '\n';
        return false;
      }
    }
  }
  return true;
}

bool check_dftd3(
    NepaModel* model,
    const NepaStructureBatch& batch,
    bool include_nep,
    double golden_energy,
    const std::array<double, 12>& golden_force,
    const std::array<double, 9>& golden_virial) {
  std::vector<double> energy(batch.num_structures, 0.0);
  std::vector<double> potential(batch.total_atoms, 0.0);
  std::vector<double> force(static_cast<std::size_t>(batch.total_atoms) * 3, 0.0);
  std::vector<double> virial(static_cast<std::size_t>(batch.num_structures) * 9, 0.0);
  std::vector<double> atom_virial(
      static_cast<std::size_t>(batch.total_atoms) * 9, 0.0);
  NepaDftd3Parameters parameters{"pbe", 12.0, 10.0};
  NepaDftd3Result result{
      energy.data(), potential.data(), force.data(), virial.data(), atom_virial.data()};
  const NepaStatus status = include_nep
      ? nepa_compute_with_dftd3_batch(model, &batch, &parameters, &result)
      : nepa_compute_dftd3_batch(model, &batch, &parameters, &result);
  if (status != NEPA_STATUS_OK) {
    std::cerr << "DFT-D3 failed: " << nepa_last_error_message() << '\n';
    return false;
  }
  for (int structure = 0; structure < batch.num_structures; ++structure) {
    if (!close(energy[structure], golden_energy, 1.0e-10, 2.0e-11)) {
      std::cerr << "DFT-D3 energy mismatch structure=" << structure
                << " actual=" << energy[structure]
                << " expected=" << golden_energy << '\n';
      return false;
    }
    for (std::size_t i = 0; i < golden_force.size(); ++i) {
      if (!close(force[static_cast<std::size_t>(structure) * 12 + i],
                 golden_force[i], 1.0e-10, 2.0e-11)) {
        std::cerr << "DFT-D3 force mismatch structure=" << structure
                  << " index=" << i << " actual="
                  << std::setprecision(17)
                  << force[static_cast<std::size_t>(structure) * 12 + i]
                  << " expected=" << golden_force[i] << '\n';
        return false;
      }
    }
    for (std::size_t i = 0; i < golden_virial.size(); ++i) {
      if (!close(virial[static_cast<std::size_t>(structure) * 9 + i],
                 golden_virial[i], 1.0e-10, 2.0e-11)) {
        std::cerr << "DFT-D3 virial mismatch structure=" << structure
                  << " index=" << i << " actual="
                  << virial[static_cast<std::size_t>(structure) * 9 + i]
                  << " expected=" << golden_virial[i] << '\n';
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!nep_adapters::register_cpu_engine()) {
    return EXIT_FAILURE;
  }
  const std::string root = NEP_ADAPTERS_PRODUCTION_FIXTURE_DIR;
  if (!check_response(
          root + "/dipole/nep.txt", root + "/dipole/structure.xyz",
          root + "/dipole/descriptor_golden.txt",
          NEPA_MODEL_KIND_DIPOLE, nep_adapters::Capability::dipole,
          {0.15439024567604065, 0.005705520510673523,
           0.0044387467205524445}) ||
      !check_response(
          root + "/polarizability/nep.txt",
          root + "/polarizability/structure.xyz",
          root + "/polarizability/descriptor_golden.txt",
          NEPA_MODEL_KIND_POLARIZABILITY,
          nep_adapters::Capability::polarizability,
          {100.79893493652344, 92.42485046386719, 56.936161041259766,
           3.494504451751709, -0.08088953793048859,
           0.07827239483594894})) {
    std::cerr << "response API check failed\n";
    return EXIT_FAILURE;
  }

  const std::string dft_model = root + "/dftd3/nep.txt";
  const cpu_test::Frame frame = cpu_test::read_first_frame(
      root + "/dftd3/structure.xyz", cpu_test::read_type_map(dft_model));
  OwnedBatch batch = repeat_frame(frame, 2);
  NepaModel* model = nullptr;
  if (nepa_load_model("cpu", dft_model.c_str(), &model) != NEPA_STATUS_OK) {
    return EXIT_FAILURE;
  }
  NepaModelInfo info{};
  const bool capability_ok = nepa_model_info(model, &info) == NEPA_STATUS_OK &&
      nep_adapters::has_capability(
          info.capabilities, nep_adapters::Capability::dftd3);
  const bool pure_ok = check_dftd3(
      model, batch.view, false, -0.55635719211672985,
      {-5.617457530612317e-07, 1.2286027133291515e-06,
       0.035924129236214478, -3.3706086395032071e-08,
       8.6785351689023854e-07, -0.022261267865837285,
       2.2456730222131585e-08, -8.1066044266264996e-07,
       -0.035924129238444472, 5.7299510923536356e-07,
       -1.2857957875575532e-06, 0.022261267868067282},
      {0.0050193723691757735, 3.6530891834089683e-06,
       -2.844169029763563e-08, 3.6530891833973191e-06,
       0.0050412491644913548, 1.4480575659207584e-07,
       -2.8441690294410754e-08, 1.4480575659077673e-07,
       -0.14779042335055095});
  const bool combined_ok = check_dftd3(
      model, batch.view, true, -13.0823247488847,
      {0.00024238139388109135, -0.0003215517327062171,
       -1.1658917972701766, -0.00020988752600463136,
       3.4265233601270134e-05, 0.93713949108202077,
       0.000208236332115568, -2.7753660459274229e-05,
       1.1658917970700526, -0.0002407301999920175,
       0.00031504015956382823, -0.9371394908818953},
      {-3.1930922027597188, 0.0002789806577792452,
       -4.2010820581918583e-06, 0.00027898065777815797,
       -3.1912470020934411, 1.660220811599584e-05,
       -4.2010820579579154e-06, 1.6602208116219175e-05,
       5.3416236155363208});

  std::vector<double> energy(2), force(24);
  NepaFindForceResult force_result{};
  force_result.energy_per_structure = energy.data();
  force_result.forces_aos3 = force.data();
  const bool cancel_ok = nepa_cancel_model(model) == NEPA_STATUS_OK &&
      nepa_find_force_batch(model, &batch.view, &force_result) ==
          NEPA_STATUS_CANCELLED &&
      nepa_reset_cancel(model) == NEPA_STATUS_OK &&
      nepa_find_force_batch(model, &batch.view, &force_result) == NEPA_STATUS_OK;
  nepa_free_model(model);
  if (!capability_ok) std::cerr << "DFT-D3 capability check failed\n";
  if (!pure_ok) std::cerr << "pure DFT-D3 check failed\n";
  if (!combined_ok) std::cerr << "combined DFT-D3 check failed\n";
  if (!cancel_ok) std::cerr << "cancellation check failed\n";
  return capability_ok && pure_ok && combined_ok && cancel_ok
      ? EXIT_SUCCESS
      : EXIT_FAILURE;
}
