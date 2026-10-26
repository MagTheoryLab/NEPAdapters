#include "device_model.hpp"
#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void close(double actual, double expected) {
  require(std::isfinite(actual) && std::abs(actual - expected) < 1e-10,
          "new/legacy output mismatch");
}
struct TemporaryDirectory {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("nepa-block-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  TemporaryDirectory() { std::filesystem::create_directory(path); }
  ~TemporaryDirectory() { std::filesystem::remove_all(path); }
};

// Small asymmetric two-element fixture; every coefficient is distinct. Keep
// this writer independent of the reader and the engine's packing routines.
void write_models(const std::string& old_path, const std::string& new_path,
                  int zbl, bool second_layer, bool typewise, int charge = 0) {
  constexpr int dim = 6, neurons = 3, pairs = 4, basis = 4;
  const int ann = (second_layer ? (dim + 1) * neurons + (neurons + 2) * 2
                               : (dim + 2 + (charge != 0)) * neurons) * 2 + (charge ? 2 : 1);
  std::vector<double> values(ann + 2 * pairs * basis);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = 0.08 * std::sin(i + 0.3);
  std::ofstream old(old_path), modern(new_path);
  old << std::setprecision(17);
  modern << std::setprecision(17);
  old << (zbl ? "nep4_zbl" : "nep4");
  if (charge) old << "_charge" << charge;
  old << " 2 Fe Ge\n";
  modern << "nep 2 Fe Ge\nformat_version 1\nmodel "
         << (charge ? "qnep" + std::to_string(charge) : "nep")
         << "\nenergy_reference -123.1234567890123 0.234567890123456\n";
  const std::string zbl_line = zbl == 1 ? "zbl 0.8 2.5\n" : zbl == 2 ? "zbl 0 0\n" : "";
  const std::string cutoff = typewise ? "cutoff 6 4 7 5 64 64\n" : "cutoff 6 4 64 64\n";
  const std::string fields = zbl_line + cutoff + "n_max 1 1\nbasis_size 1 1\nl_max 2 0 0\nANN 3 " +
      (second_layer ? "2\n" : "0\n");
  old << fields;
  modern << fields << "@ parameters " << values.size() << '\n';
  for (double value : values) old << value << '\n';
  for (int i = 0; i < ann; ++i) modern << values[i] << '\n';
  for (int part = 0; part < 2; ++part)
    for (int pair = 0; pair < pairs; ++pair)
      for (int b = 0; b < basis; ++b)
        modern << values[ann + part * pairs * basis + b * pairs + pair] << '\n';
  modern << "@ q_scaler " << dim << '\n';
  for (int i = 0; i < dim; ++i) {
    old << 0.5 + i * 0.1 << '\n';
    modern << 0.5 + i * 0.1 << '\n';
  }
  if (zbl == 2) {
    modern << "@ zbl_parameters 30\n";
    for (int pair = 0; pair < 3; ++pair) {
      const double z[] = {0.8, 2.5, 0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162};
      for (double value : z) { old << value << '\n'; modern << value << '\n'; }
    }
  }
}

struct Prediction {
  double total = 0, virial[6] = {}, potential[4] = {};
  double forces[4][3] = {}, per_atom_virial[4][9] = {};
};

Prediction lammps(const std::string& file, bool atom_energy) {
  NepaModel* model = nullptr;
  require(nepa_load_model("cpu", file.c_str(), &model) == NEPA_STATUS_OK, "CPU load failed");
  // Two owned centers in shuffled order; the other atoms are neighbors only.
  int ilist[] = {1, 0}, counts[] = {3, 3, 0, 0};
  int list0[] = {1, 2, 3}, list1[] = {0, 2, 3};
  int* neighbors[] = {list0, list1, nullptr, nullptr};
  int types[] = {2, 1, 1, 2}, type_map[] = {-1, 1, 0};
  double coords[4][3] = {{1, 2, 3}, {3.1, 2.2, 3.3}, {2.3, 4.2, 3.7}, {3.9, 3.5, 4.7}};
  double* rows[] = {coords[0], coords[1], coords[2], coords[3]};
  NepaLammpsNeighborInput input{};
  input.nlocal = 2; input.inum = 2; input.ilist = ilist;
  input.numneigh = counts; input.firstneigh = neighbors;
  input.types = types; input.type_map = type_map; input.positions = rows;
  Prediction prediction;
  double* forces[] = {prediction.forces[0], prediction.forces[1], prediction.forces[2], prediction.forces[3]};
  double* virials[] = {prediction.per_atom_virial[0], prediction.per_atom_virial[1], prediction.per_atom_virial[2], prediction.per_atom_virial[3]};
  NepaLammpsNeighborResult result{};
  result.total_potential = &prediction.total;
  result.total_virial6 = prediction.virial;
  result.potential_per_atom = atom_energy ? prediction.potential : nullptr;
  result.forces = forces; result.virials_per_atom9 = virials;
  const auto status = nepa_find_force_lammps_neighbors(model, &input, &result);
  nepa_free_model(model);
  require(status == NEPA_STATUS_OK, "LAMMPS neighbor evaluation failed");
  return prediction;
}
}  // namespace

int main() {
  try {
    TemporaryDirectory dir;
    const auto legacy = (dir.path / "legacy.nep").string(), modern = (dir.path / "modern.nep").string();
    require(nepa_register_cpu_engine(), "CPU registration failed");
    for (int zbl : {0, 1, 2}) for (bool second : {false, true}) for (bool typewise : {false, true}) {
      write_models(legacy, modern, zbl, second, typewise);
      const auto a = nep_adapters::cuda_backend::load_host_model_parameters(legacy);
      const auto b = nep_adapters::cuda_backend::load_host_model_parameters(modern);
      require(a.ann_type_major == b.ann_type_major, "CUDA ANN packing mismatch");
      require(a.descriptor_coefficients == b.descriptor_coefficients, "CUDA descriptor packing mismatch");
      require(a.q_scaler == b.q_scaler, "CUDA scaler packing mismatch");
      require(a.flexible_zbl_parameters == b.flexible_zbl_parameters, "CUDA ZBL packing mismatch");
      require(b.spin_baseline == std::vector<double>({-123.1234567890123, 0.234567890123456}), "reference precision lost");
      for (bool atom_energy : {false, true}) {
        const auto before = lammps(legacy, atom_energy), after = lammps(modern, atom_energy);
        close(after.total, before.total - 123.1234567890123 + 0.234567890123456);
        for (int i = 0; i < 6; ++i) close(after.virial[i], before.virial[i]);
        for (int i = 0; i < 4; ++i) {
          if (atom_energy) close(after.potential[i], before.potential[i] +
              (i == 0 ? -123.1234567890123 : i == 1 ? 0.234567890123456 : 0.));
          for (int j = 0; j < 3; ++j) close(after.forces[i][j], before.forces[i][j]);
          for (int j = 0; j < 9; ++j) close(after.per_atom_virial[i][j], before.per_atom_virial[i][j]);
        }
      }
    }
    for (int charge : {1, 2}) {
      write_models(legacy, modern, 0, false, false, charge);
      const auto a = nep_adapters::cuda_backend::load_host_model_parameters(legacy);
      const auto b = nep_adapters::cuda_backend::load_host_model_parameters(modern);
      require(a.ann_type_major_qscaled == b.ann_type_major_qscaled, "qNEP CUDA ANN packing mismatch");
      require(a.descriptor_coefficients == b.descriptor_coefficients, "qNEP CUDA coefficients mismatch");
      require(b.spin_baseline.size() == 2, "qNEP CUDA reference missing");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
