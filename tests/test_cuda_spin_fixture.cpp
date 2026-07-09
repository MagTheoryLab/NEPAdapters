#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_opt.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr int kAtomCount = 4;

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

double max_abs_diff_range(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs,
    int dim,
    int begin,
    int end) {
  double diff = 0.0;
  for (int atom = 0; atom < kAtomCount; ++atom) {
    for (int d = begin; d < end; ++d) {
      const std::size_t index =
          static_cast<std::size_t>(atom) * dim + static_cast<std::size_t>(d);
      diff = std::max(diff, std::abs(lhs[index] - rhs[index]));
    }
  }
  return diff;
}

struct Prediction {
  double energy = 0.0;
  std::vector<double> potential;
  std::vector<double> force;
  std::vector<double> virial;
  std::vector<double> mforce;
  std::vector<double> descriptor;
};

Prediction run(NepaModel* model) {
  const std::int32_t atom_counts[] = {kAtomCount};
  const std::int32_t atom_offsets[] = {0};
  const std::int32_t types[] = {0, 0, 0, 0};
  const double positions[] = {
      0.2, 0.2, 0.2,
      3.7, 0.3, 0.2,
      0.4, 3.6, 0.5,
      1.8, 1.7, 3.5};
  const double spins[] = {
      1.0, 0.2, 0.0,
      0.4, -0.3, 0.7,
      -0.2, 0.8, 0.5,
      0.6, 0.1, -0.4};
  const double box[] = {
      16.0, 0.0, 0.0,
      0.0, 16.0, 0.0,
      0.0, 0.0, 16.0};
  const std::int32_t pbc[] = {0, 0, 0};

  NepaModelInfo info{};
  if (nepa_model_info(model, &info) != NEPA_STATUS_OK ||
      info.descriptor_dim != 88) {
    std::cerr << "model_info failed\n";
    std::exit(EXIT_FAILURE);
  }

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

  Prediction out;
  out.potential.assign(kAtomCount, 0.0);
  out.force.assign(static_cast<std::size_t>(kAtomCount) * 3, 0.0);
  out.virial.assign(9, 0.0);
  out.mforce.assign(static_cast<std::size_t>(kAtomCount) * 3, 0.0);
  out.descriptor.assign(
      static_cast<std::size_t>(kAtomCount) * info.descriptor_dim, 0.0);
  NepaFindForceResult force_result{};
  force_result.energy_per_structure = &out.energy;
  force_result.potential_per_atom = out.potential.data();
  force_result.forces_aos3 = out.force.data();
  force_result.virials_row_major9 = out.virial.data();
  force_result.mforces_aos3 = out.mforce.data();
  if (nepa_find_force_batch(model, &batch, &force_result) != NEPA_STATUS_OK) {
    std::cerr << "find_force_batch failed: " << nepa_last_error_message() << "\n";
    std::exit(EXIT_FAILURE);
  }
  NepaFindDescriptorResult descriptor_result{};
  descriptor_result.descriptors = out.descriptor.data();
  if (nepa_find_descriptors(model, &batch, &descriptor_result) != NEPA_STATUS_OK) {
    std::cerr << "find_descriptors failed: " << nepa_last_error_message() << "\n";
    std::exit(EXIT_FAILURE);
  }
  return out;
}

bool check_lammps(NepaModel* model, const Prediction& ref) {
  int ilist[] = {0, 1, 2, 3};
  int numneigh[] = {3, 3, 3, 3};
  int neigh0[] = {1, 2, 3};
  int neigh1[] = {0, 2, 3};
  int neigh2[] = {0, 1, 3};
  int neigh3[] = {0, 1, 2};
  int* firstneigh[] = {neigh0, neigh1, neigh2, neigh3};
  int types[] = {1, 1, 1, 1};
  int type_map[] = {-1, 0};
  double x0[] = {0.2, 0.2, 0.2};
  double x1[] = {3.7, 0.3, 0.2};
  double x2[] = {0.4, 3.6, 0.5};
  double x3[] = {1.8, 1.7, 3.5};
  double s0[] = {0.5, 0.1, 0.0, 2.0};
  double s1[] = {0.2, -0.15, 0.35, 2.0};
  double s2[] = {-0.1, 0.4, 0.25, 2.0};
  double s3[] = {0.3, 0.05, -0.2, 2.0};
  double* positions[] = {x0, x1, x2, x3};
  double* spins[] = {s0, s1, s2, s3};
  double total_potential = 0.0;
  double total_virial6[6] = {};
  double potential[4] = {};
  double f0[3] = {};
  double f1[3] = {};
  double f2[3] = {};
  double f3[3] = {};
  double m0[3] = {};
  double m1[3] = {};
  double m2[3] = {};
  double m3[3] = {};
  double* forces[] = {f0, f1, f2, f3};
  double* mforces[] = {m0, m1, m2, m3};

  NepaLammpsNeighborInput input{};
  input.nlocal = 4;
  input.inum = 4;
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
    std::cerr << "LAMMPS fixture failed: " << nepa_last_error_message() << "\n";
    return false;
  }

  const std::vector<double> force = {
      f0[0], f0[1], f0[2], f1[0], f1[1], f1[2],
      f2[0], f2[1], f2[2], f3[0], f3[1], f3[2]};
  const std::vector<double> mforce = {
      m0[0], m0[1], m0[2], m1[0], m1[1], m1[2],
      m2[0], m2[1], m2[2], m3[0], m3[1], m3[2]};
  const std::vector<double> virial6 = {
      ref.virial[0],
      ref.virial[4],
      ref.virial[8],
      ref.virial[1],
      ref.virial[2],
      ref.virial[5]};
  const bool ok = std::abs(total_potential - ref.energy) < 2.0e-4 &&
                  max_abs_diff(force, ref.force) < 2.0e-4 &&
                  max_abs_diff(mforce, ref.mforce) < 2.0e-4 &&
                  max_abs_diff(std::vector<double>(total_virial6, total_virial6 + 6),
                               virial6) < 2.0e-4;
  if (!ok) {
    std::cerr << "CUDA spin fixture LAMMPS mismatch:"
              << " energy=" << std::abs(total_potential - ref.energy)
              << " force=" << max_abs_diff(force, ref.force)
              << " mforce=" << max_abs_diff(mforce, ref.mforce)
              << " virial6="
              << max_abs_diff(std::vector<double>(total_virial6, total_virial6 + 6),
                              virial6)
              << "\n";
  }
  return ok;
}

}  // namespace

int main() {
  if (!nep_adapters::register_cpu_opt_engine() ||
      !nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }
  NepaModel* cpu = nullptr;
  NepaModel* gpu = nullptr;
  if (nepa_load_model("cpu_opt", NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE, &cpu) !=
          NEPA_STATUS_OK ||
      nepa_load_model("cuda", NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE, &gpu) !=
          NEPA_STATUS_OK) {
    return EXIT_FAILURE;
  }
  const Prediction cpu_out = run(cpu);
  const Prediction gpu_out = run(gpu);
  const double energy_diff = std::abs(cpu_out.energy - gpu_out.energy);
  const double potential_diff = max_abs_diff(cpu_out.potential, gpu_out.potential);
  const double force_diff = max_abs_diff(cpu_out.force, gpu_out.force);
  const double virial_diff = max_abs_diff(cpu_out.virial, gpu_out.virial);
  const double mforce_diff = max_abs_diff(cpu_out.mforce, gpu_out.mforce);
  const double descriptor_diff = max_abs_diff(cpu_out.descriptor, gpu_out.descriptor);
  int max_index = 0;
  for (std::size_t i = 1; i < cpu_out.descriptor.size(); ++i) {
    if (std::abs(cpu_out.descriptor[i] - gpu_out.descriptor[i]) >
        std::abs(cpu_out.descriptor[static_cast<std::size_t>(max_index)] -
                 gpu_out.descriptor[static_cast<std::size_t>(max_index)])) {
      max_index = static_cast<int>(i);
    }
  }
  const bool ok = energy_diff < 2.0e-4 && potential_diff < 2.0e-4 &&
                  force_diff < 2.0e-4 && virial_diff < 2.0e-4 &&
                  mforce_diff < 2.0e-4 && descriptor_diff < 2.0e-4 &&
                  check_lammps(gpu, cpu_out);
  if (!ok) {
    std::cerr << "CUDA spin fixture mismatch:"
              << " energy=" << energy_diff
              << " potential=" << potential_diff
              << " force=" << force_diff
              << " virial=" << virial_diff
              << " mforce=" << mforce_diff
              << " descriptor=" << descriptor_diff
              << " struct_descriptor="
              << max_abs_diff_range(cpu_out.descriptor, gpu_out.descriptor, 88, 0, 20)
              << " spin_main="
              << max_abs_diff_range(cpu_out.descriptor, gpu_out.descriptor, 88, 20, 78)
              << " spin_chiral_bulk="
              << max_abs_diff_range(cpu_out.descriptor, gpu_out.descriptor, 88, 78, 80)
              << " spin_chiral_polar="
              << max_abs_diff_range(cpu_out.descriptor, gpu_out.descriptor, 88, 80, 84)
              << " spin_chiral_pseudo="
              << max_abs_diff_range(cpu_out.descriptor, gpu_out.descriptor, 88, 84, 88)
              << " max_atom=" << (max_index / 88)
              << " max_dim=" << (max_index % 88)
              << " cpu=" << cpu_out.descriptor[static_cast<std::size_t>(max_index)]
              << " gpu=" << gpu_out.descriptor[static_cast<std::size_t>(max_index)]
              << "\n";
  }
  nepa_free_model(cpu);
  nepa_free_model(gpu);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
