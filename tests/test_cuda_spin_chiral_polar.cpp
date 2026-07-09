#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_opt.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kAtomCount = 3;
constexpr int kDescriptorDim = 69;

std::string write_model(int active_dim) {
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("cuda_spin_chiral_" + std::to_string(active_dim) + ".nep")).string();
  std::ofstream out(path);
  out << "nep4_spin1 1 Fe\n";
  out << "spin_mode 1 10\n";
  out << "spin_baseline -2\n";
  out << "spin_n_max 0 0\n";
  out << "spin_basis_size 3 3\n";
  out << "spin_l_max 4 0 0\n";
  out << "spin_compress 4\n";
  out << "spin_cutoff 4 4\n";
  out << "spin_chiral 1\n";
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
  out << "0\n";
  for (int c = 0; c < 4; ++c) {
    for (int k = 0; k < 4; ++k) {
      out << (0.2 + 0.03 * c + 0.01 * k) << "\n";
    }
  }
  out << "0\n";
  out << "0\n";
  out << "0\n";
  out << "1\n";
  for (int dim = 0; dim < kDescriptorDim; ++dim) {
    out << "1\n";
  }
  return path;
}

double max_abs_diff(const std::vector<double>& lhs, const std::vector<double>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  double diff = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    diff = std::max(diff, std::abs(lhs[i] - rhs[i]));
  }
  return diff;
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
  const int types[] = {0, 0, 0};
  const double positions[] = {
      0.0, 0.0, 0.0,
      1.3, 0.4, 0.2,
      0.2, 1.1, 0.7,
  };
  const double spins[] = {
      1.2, -0.3, 0.4,
      -0.5, 0.7, 1.1,
      0.6, 0.9, -0.8,
  };
  const double box[] = {
      16.0, 0.0, 0.0,
      0.0, 16.0, 0.0,
      0.0, 0.0, 16.0,
  };
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
  out.descriptor.assign(kAtomCount * kDescriptorDim, 0.0);
  NepaFindForceResult force_result{};
  force_result.energy_per_structure = &out.energy;
  force_result.potential_per_atom = out.potential.data();
  force_result.forces_aos3 = out.force.data();
  force_result.virials_row_major9 = out.virial.data();
  force_result.mforces_aos3 = out.mforce.data();
  if (nepa_find_force_batch(model, &batch, &force_result) != NEPA_STATUS_OK) {
    std::cerr << "batch force failed: " << nepa_last_error_message() << "\n";
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
  int ilist[] = {0, 1, 2};
  int numneigh[] = {2, 2, 2};
  int neigh0[] = {1, 2};
  int neigh1[] = {0, 2};
  int neigh2[] = {0, 1};
  int* firstneigh[] = {neigh0, neigh1, neigh2};
  int types[] = {1, 1, 1};
  int type_map[] = {-1, 0};
  double x0[] = {0.0, 0.0, 0.0};
  double x1[] = {1.3, 0.4, 0.2};
  double x2[] = {0.2, 1.1, 0.7};
  double s0[] = {0.6, -0.15, 0.2, 2.0};
  double s1[] = {-0.25, 0.35, 0.55, 2.0};
  double s2[] = {0.3, 0.45, -0.4, 2.0};
  double* positions[] = {x0, x1, x2};
  double* spins[] = {s0, s1, s2};
  double total_potential = 0.0;
  double total_virial6[6] = {};
  double potential[3] = {};
  double f0[3] = {};
  double f1[3] = {};
  double f2[3] = {};
  double m0[3] = {};
  double m1[3] = {};
  double m2[3] = {};
  double* forces[] = {f0, f1, f2};
  double* mforces[] = {m0, m1, m2};
  NepaLammpsNeighborInput input{};
  input.nlocal = 3;
  input.inum = 3;
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
  const std::vector<double> force = {
      f0[0], f0[1], f0[2], f1[0], f1[1], f1[2], f2[0], f2[1], f2[2]};
  const std::vector<double> mforce = {
      m0[0], m0[1], m0[2], m1[0], m1[1], m1[2], m2[0], m2[1], m2[2]};
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
  const double energy_diff = std::abs(c.energy - g.energy);
  const double potential_diff = max_abs_diff(c.potential, g.potential);
  const double force_diff = max_abs_diff(c.force, g.force);
  const double virial_diff = max_abs_diff(c.virial, g.virial);
  const double mforce_diff = max_abs_diff(c.mforce, g.mforce);
  const double descriptor_diff = max_abs_diff(c.descriptor, g.descriptor);
  const bool ok = energy_diff < 1.0e-6 && potential_diff < 1.0e-6 &&
                  force_diff < 1.0e-6 && virial_diff < 1.0e-6 &&
                  mforce_diff < 1.0e-6 && descriptor_diff < 1.0e-6 &&
                  check_lammps(gpu, c);
  if (!ok) {
    std::cerr << "spin chiral CPU/GPU mismatch active_dim=" << active_dim
              << " energy=" << energy_diff
              << " potential=" << potential_diff
              << " force=" << force_diff
              << " virial=" << virial_diff
              << " mforce=" << mforce_diff
              << " descriptor=" << descriptor_diff << "\n";
    std::cerr << "sizes force=" << c.force.size()
              << " virial=" << c.virial.size()
              << " mforce=" << c.mforce.size() << "\n";
    for (std::size_t i = 0; i < c.force.size(); ++i) {
      std::cerr << "force[" << i << "] cpu=" << c.force[i]
                << " gpu=" << g.force[i] << "\n";
    }
    for (std::size_t i = 0; i < c.virial.size(); ++i) {
      std::cerr << "virial[" << i << "] cpu=" << c.virial[i]
                << " gpu=" << g.virial[i] << "\n";
    }
    for (std::size_t i = 0; i < c.mforce.size(); ++i) {
      std::cerr << "mforce[" << i << "] cpu=" << c.mforce[i]
                << " gpu=" << g.mforce[i] << "\n";
    }
  }
  nepa_free_model(cpu);
  nepa_free_model(gpu);
  return ok;
}

}  // namespace

int main() {
  if (!nep_adapters::register_cpu_opt_engine() ||
      !nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }
  for (int active_dim : {59, 60, 61, 64, 65, 68}) {
    if (!check_dim(active_dim)) {
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
