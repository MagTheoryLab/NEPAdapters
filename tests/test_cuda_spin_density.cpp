#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_opt.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kAtomCount = 2;
constexpr int kDescriptorDim = 17;
constexpr int kC4L4DescriptorDim = 69;
constexpr int kFirstDensityDescriptor = 19;
constexpr int kRaw1DotDescriptor = 55;

std::filesystem::path process_local_model_path(const std::string& label) {
  return std::filesystem::temp_directory_path() /
         (label + "_pid" + std::to_string(static_cast<long long>(getpid())) +
          ".nep");
}

std::string write_model(int active_dim) {
  const std::string path = process_local_model_path(
      "cuda_spin_density_" + std::to_string(active_dim)).string();
  std::ofstream out(path);
  out << "nep4_spin1 1 Fe\n";
  out << "spin_mode 1 10\n";
  out << "spin_baseline -2\n";
  out << "spin_n_max 0 0\n";
  out << "spin_basis_size 0 0\n";
  out << "spin_l_max 4 0 0\n";
  out << "spin_compress 1\n";
  out << "spin_cutoff 4 4\n";
  out << "spin_chiral 0\n";
  out << "spin_scaler 1\n";
  out << "spin_dof_type Fe\n";
  out << "spin_env_type Fe\n";
  out << "cutoff 4 4 64 64\n";
  out << "n_max 0 0\n";
  out << "basis_size 0 0\n";
  out << "l_max 0 0 0\n";
  out << "ANN 1 0\n";
  for (int dim = 0; dim < kDescriptorDim; ++dim) {
    out << (dim == active_dim ? 0.25 : 0.0) << "\n";
  }
  out << "0\n1\n0\n0\n0\n1\n";
  for (int dim = 0; dim < kDescriptorDim; ++dim) {
    out << "1\n";
  }
  return path;
}

std::string write_c4_l4_model(
    const std::string& label,
    int max_neighbors,
    int active_descriptor) {
  const std::string path = process_local_model_path(
      "cuda_spin_density_c4_l4_" + label).string();
  std::ofstream out(path);
  out << "nep4_spin1 1 Fe\n";
  out << "spin_mode 1 10\n";
  out << "spin_baseline -2\n";
  out << "spin_n_max 2 1\n";
  out << "spin_basis_size 3 3\n";
  out << "spin_l_max 4 0 0\n";
  out << "spin_compress 4\n";
  out << "spin_cutoff 4 4\n";
  out << "spin_chiral 1\n";
  out << "spin_scaler 1\n";
  out << "spin_dof_type Fe\n";
  out << "spin_env_type Fe\n";
  out << "cutoff 4 4 " << max_neighbors << " " << max_neighbors << "\n";
  out << "n_max 0 0\n";
  out << "basis_size 0 0\n";
  out << "l_max 0 0 0\n";
  out << "ANN 1 0\n";
  for (int dim = 0; dim < kC4L4DescriptorDim; ++dim) {
    out << (dim == active_descriptor ? 0.25 : 0.0) << "\n";
  }
  out << "0\n1\n0\n";
  out << "0\n0\n";
  for (int coefficient = 0; coefficient < 16; ++coefficient) {
    out << "1\n";
  }
  for (int dim = 0; dim < kC4L4DescriptorDim; ++dim) {
    out << "1\n";
  }
  return path;
}

std::string write_c4_l4_raw1_dot_model(int max_neighbors) {
  return write_c4_l4_model(
      "raw1_dot_capacity" +
          std::to_string(static_cast<int>(std::ceil(max_neighbors * 1.25))),
      max_neighbors,
      kRaw1DotDescriptor);
}

std::string write_c4_l4_density_model() {
  // Keep a typical 25-neighbor shape alongside the denser 64/96 cases below.
  return write_c4_l4_model(
      "density", 25, kFirstDensityDescriptor);
}

double max_abs_diff(const std::vector<double>& lhs, const std::vector<double>& rhs) {
  const double infinity = std::numeric_limits<double>::infinity();
  if (lhs.size() != rhs.size()) {
    return infinity;
  }
  double diff = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i])) {
      return infinity;
    }
    const double delta = std::abs(lhs[i] - rhs[i]);
    if (!std::isfinite(delta)) {
      return infinity;
    }
    diff = std::max(diff, delta);
  }
  return diff;
}

void print_non_finite(
    const char* label,
    const std::vector<double>& values) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      std::cerr << " nonfinite=" << label << '[' << index << "]="
                << values[index];
    }
  }
}

bool check_max_abs_diff_rejects_non_finite() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  const double max = std::numeric_limits<double>::max();
  return std::isinf(max_abs_diff({nan}, {0.0})) &&
         std::isinf(max_abs_diff({0.0}, {nan})) &&
         std::isinf(max_abs_diff({infinity}, {infinity})) &&
         std::isinf(max_abs_diff({max}, {-max}));
}

struct BatchResult {
  double energy = 0.0;
  std::vector<double> potential;
  std::vector<double> force;
  std::vector<double> virial;
  std::vector<double> mforce;
  std::vector<double> descriptor;
};

BatchResult run_batch(NepaModel* model) {
  const int atom_counts[] = {kAtomCount};
  const int atom_offsets[] = {0};
  const int types[] = {0, 0};
  const double positions[] = {0.0, 0.0, 0.0, 1.3, 0.4, 0.2};
  const double spins[] = {1.2, -0.3, 0.4, -0.5, 0.7, 1.1};
  const double box[] = {16.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 16.0};
  const int pbc[] = {0, 0, 0};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = kAtomCount;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions;
  batch.spins_aos3 = spins;
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;

  BatchResult out;
  out.potential.assign(kAtomCount, 0.0);
  out.force.assign(3 * kAtomCount, 0.0);
  out.virial.assign(9, 0.0);
  out.mforce.assign(3 * kAtomCount, 0.0);
  NepaModelInfo info{};
  if (nepa_model_info(model, &info) != NEPA_STATUS_OK) {
    std::cerr << "model info failed: " << nepa_last_error_message() << "\n";
    std::exit(EXIT_FAILURE);
  }
  out.descriptor.assign(
      static_cast<std::size_t>(kAtomCount) * info.descriptor_dim, 0.0);
  NepaFindForceResult result{};
  result.energy_per_structure = &out.energy;
  result.potential_per_atom = out.potential.data();
  result.forces_aos3 = out.force.data();
  result.virials_row_major9 = out.virial.data();
  result.mforces_aos3 = out.mforce.data();
  if (nepa_find_force_batch(model, &batch, &result) != NEPA_STATUS_OK) {
    std::cerr << "batch failed: " << nepa_last_error_message() << "\n";
    std::exit(EXIT_FAILURE);
  }
  NepaFindDescriptorResult descriptor_result{};
  descriptor_result.descriptors = out.descriptor.data();
  if (nepa_find_descriptors(model, &batch, &descriptor_result) != NEPA_STATUS_OK) {
    std::cerr << "descriptor failed: " << nepa_last_error_message() << "\n";
    std::exit(EXIT_FAILURE);
  }
  return out;
}

bool check_lammps(NepaModel* model, const BatchResult& ref) {
  int ilist[] = {0, 1};
  int numneigh[] = {1, 1};
  int neigh0[] = {1};
  int neigh1[] = {0};
  int* firstneigh[] = {neigh0, neigh1};
  int types[] = {1, 1};
  int type_map[] = {-1, 0};
  double x0[] = {0.0, 0.0, 0.0};
  double x1[] = {1.3, 0.4, 0.2};
  double s0[] = {0.6, -0.15, 0.2, 2.0};
  double s1[] = {-0.25, 0.35, 0.55, 2.0};
  double* positions[] = {x0, x1};
  double* spins[] = {s0, s1};
  double total_potential = 0.0;
  double total_virial6[6] = {};
  double potential[2] = {};
  double f0[3] = {};
  double f1[3] = {};
  double m0[3] = {};
  double m1[3] = {};
  double* forces[] = {f0, f1};
  double* mforces[] = {m0, m1};
  NepaLammpsNeighborInput input{};
  input.nlocal = 2;
  input.inum = 2;
  input.ilist = ilist;
  input.numneigh = numneigh;
  input.firstneigh = firstneigh;
  input.types = types;
  input.type_map = type_map;
  input.positions = positions;
  input.spins = spins;
  NepaLammpsNeighborResult result{};
  result.total_potential = &total_potential;
  result.total_virial6 = total_virial6;
  result.potential_per_atom = potential;
  result.forces = forces;
  result.mforces = mforces;
  if (nepa_find_force_lammps_neighbors(model, &input, &result) != NEPA_STATUS_OK) {
    std::cerr << "LAMMPS failed: " << nepa_last_error_message() << "\n";
    return false;
  }
  const std::vector<double> force = {f0[0], f0[1], f0[2], f1[0], f1[1], f1[2]};
  const std::vector<double> mforce = {m0[0], m0[1], m0[2], m1[0], m1[1], m1[2]};
  return std::abs(total_potential - ref.energy) < 1.0e-6 &&
         max_abs_diff(force, ref.force) < 1.0e-6 &&
         max_abs_diff(mforce, ref.mforce) < 1.0e-6;
}

bool check_dim(int active_dim) {
  const std::string model_path = write_model(active_dim);
  NepaModel* cpu = nullptr;
  NepaModel* gpu = nullptr;
  if (nepa_load_model("cpu_opt", model_path.c_str(), &cpu) != NEPA_STATUS_OK ||
      nepa_load_model("cuda", model_path.c_str(), &gpu) != NEPA_STATUS_OK) {
    return false;
  }
  const BatchResult c = run_batch(cpu);
  const BatchResult g = run_batch(gpu);
  const bool ok =
      std::abs(c.energy - g.energy) < 1.0e-6 &&
      max_abs_diff(c.potential, g.potential) < 1.0e-6 &&
      max_abs_diff(c.force, g.force) < 1.0e-6 &&
      max_abs_diff(c.virial, g.virial) < 1.0e-6 &&
      max_abs_diff(c.mforce, g.mforce) < 1.0e-6 &&
      check_lammps(gpu, c);
  if (!ok) {
    std::cerr << "spin density mismatch active_dim=" << active_dim
              << " energy=" << std::abs(c.energy - g.energy)
              << " force=" << max_abs_diff(c.force, g.force)
              << " virial=" << max_abs_diff(c.virial, g.virial)
              << " mforce=" << max_abs_diff(c.mforce, g.mforce);
    print_non_finite("cpu_force", c.force);
    print_non_finite("gpu_force", g.force);
    print_non_finite("cpu_virial", c.virial);
    print_non_finite("gpu_virial", g.virial);
    std::cerr << '\n';
  }
  nepa_free_model(cpu);
  nepa_free_model(gpu);
  return ok;
}

bool check_c4_l4_raw1_dot() {
  // Exercise both one-chunk and multi-chunk workspace capacities through the
  // same streaming primitive core.
  for (int max_neighbors : {64, 96}) {
    const std::string model_path =
        write_c4_l4_raw1_dot_model(max_neighbors);
    NepaModel* cpu = nullptr;
    NepaModel* gpu = nullptr;
    if (nepa_load_model("cpu_opt", model_path.c_str(), &cpu) != NEPA_STATUS_OK ||
        nepa_load_model("cuda", model_path.c_str(), &gpu) != NEPA_STATUS_OK) {
      std::cerr << "raw1-dot model load failed: "
                << nepa_last_error_message() << "\n";
      return false;
    }
    const BatchResult cpu_oracle = run_batch(cpu);
    const BatchResult optimized = run_batch(gpu);
    const double force_diff = max_abs_diff(cpu_oracle.force, optimized.force);
    const double mforce_diff = max_abs_diff(cpu_oracle.mforce, optimized.mforce);
    const bool ok = force_diff < 2.0e-4 && mforce_diff < 2.0e-4;
    if (!ok) {
      std::cerr << "optimized c4/l4 raw1-dot derivative mismatch"
                << " max_neighbors=" << max_neighbors
                << " force=" << force_diff
                << " mforce=" << mforce_diff << "\n";
    }
    nepa_free_model(cpu);
    nepa_free_model(gpu);
    if (!ok) {
      return false;
    }
  }
  return true;
}

bool check_c4_l4_density_finalize() {
  const std::string model_path = write_c4_l4_density_model();
  NepaModel* cpu = nullptr;
  NepaModel* gpu = nullptr;
  if (nepa_load_model("cpu_opt", model_path.c_str(), &cpu) != NEPA_STATUS_OK ||
      nepa_load_model("cuda", model_path.c_str(), &gpu) != NEPA_STATUS_OK) {
    std::cerr << "density model load failed: "
              << nepa_last_error_message() << "\n";
    return false;
  }
  const BatchResult cpu_oracle = run_batch(cpu);
  const BatchResult optimized = run_batch(gpu);
  const double descriptor_diff =
      max_abs_diff(cpu_oracle.descriptor, optimized.descriptor);
  const double energy_diff = std::abs(cpu_oracle.energy - optimized.energy);
  const double force_diff = max_abs_diff(cpu_oracle.force, optimized.force);
  const double mforce_diff = max_abs_diff(cpu_oracle.mforce, optimized.mforce);
  std::cout << "c4_l4_density_oracle"
            << " descriptor=" << descriptor_diff
            << " energy=" << energy_diff
            << " force=" << force_diff
            << " mforce=" << mforce_diff << "\n";
  const bool ok = descriptor_diff < 2.0e-4 && energy_diff < 2.0e-4 &&
                  force_diff < 2.0e-4 && mforce_diff < 2.0e-4;
  if (!ok) {
    std::cerr << "optimized c4/l4 density mismatch"
              << " descriptor=" << descriptor_diff
              << " energy=" << energy_diff
              << " force=" << force_diff
              << " mforce=" << mforce_diff << "\n";
  }
  nepa_free_model(cpu);
  nepa_free_model(gpu);
  return ok;
}

}  // namespace

int main() {
  if (!check_max_abs_diff_rejects_non_finite()) {
    std::cerr << "max_abs_diff accepted a non-finite comparison\n";
    return EXIT_FAILURE;
  }
  if (!nep_adapters::register_cpu_opt_engine() ||
      !nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }
  for (int active_dim : {7, 8, 9, 10, 11, 12, 13, 14, 15, 16}) {
    if (!check_dim(active_dim)) {
      return EXIT_FAILURE;
    }
  }
  if (!check_c4_l4_raw1_dot()) {
    return EXIT_FAILURE;
  }
  if (!check_c4_l4_density_finalize()) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
