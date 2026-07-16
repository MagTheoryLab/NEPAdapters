#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#include "nep_adapters/engines/cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kAtomCount = 2;
constexpr int kSpinDim = 16;
constexpr int kDescriptorDim = 1 + kSpinDim;
constexpr int kChiralSpinDim = 19;
constexpr int kChiralDescriptorDim = 1 + kChiralSpinDim;

int spin_descriptor_dim(int compress, int lmax, bool chiral = false) {
  int dim = 2 + 4 * compress;
  if (lmax >= 0) {
    dim += compress;
  }
  if (lmax >= 1) {
    dim += 3 * compress;
  }
  for (int ell = 2; ell <= lmax; ++ell) {
    dim += compress;
  }
  dim += compress;
  dim += compress;
  if (lmax >= 1) {
    dim += compress;
  }
  if (chiral) {
    dim += std::min(2, compress) + 2 * compress;
  }
  return dim;
}

void write_model(
    const std::string& path,
    bool chiral = false,
    int active_descriptor_dim = 1,
    double spin_baseline = 0.0) {
  std::ofstream out(path);
  out << std::setprecision(17);
  const int descriptor_dim = chiral ? kChiralDescriptorDim : kDescriptorDim;
  out << "nep4_spin1 1 Fe\n";
  out << "spin_mode 1 10\n";
  out << "spin_baseline " << spin_baseline << "\n";
  out << "spin_n_max 0 0\n";
  out << "spin_basis_size 0 0\n";
  out << "spin_l_max 4 0 0\n";
  out << "spin_compress 1\n";
  out << "spin_cutoff 4 4\n";
  out << "spin_chiral " << (chiral ? 1 : 0) << "\n";
  out << "spin_scaler 1\n";
  out << "spin_dof_type Fe\n";
  out << "spin_env_type Fe\n";
  out << "cutoff 4 4 64 64\n";
  out << "n_max 0 0\n";
  out << "basis_size 0 0\n";
  out << "l_max 0 0 0\n";
  out << "ANN 1 0\n";

  for (int d = 0; d < descriptor_dim; ++d) {
    out << (d == active_descriptor_dim ? 0.25 : 0.0) << "\n";
  }
  out << "0\n";  // b0
  out << "1\n";  // w1
  out << "0\n";  // b1
  out << "0\n";  // structural radial coefficient
  out << "0\n";  // structural angular coefficient
  out << "1\n";  // spin radial coefficient
  for (int d = 0; d < descriptor_dim; ++d) {
    out << "1\n";
  }
}

void write_expanded_lmax_model(const std::string& path) {
  constexpr int descriptor_dim = 10;
  std::ofstream out(path);
  out << "nep4 1 Fe\n"
      << "cutoff 4 4 64 64\n"
      << "n_max 0 0\n"
      << "basis_size 0 0\n"
      << "l_max 4 2 0 1 1 1 1\n"
      << "ANN 1 0\n";
  for (int d = 0; d < descriptor_dim; ++d) {
    out << "0\n";
  }
  out << "0\n1\n0\n";
  out << "0\n0\n";
  for (int d = 0; d < descriptor_dim; ++d) {
    out << "1\n";
  }
}

double max_abs_diff(const std::vector<double>& lhs, const std::vector<double>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  double out = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    out = std::max(out, std::abs(lhs[i] - rhs[i]));
  }
  return out;
}

std::vector<double> read_reference_vector(const std::string& path, const std::string& wanted) {
  std::ifstream in(path);
  std::string label;
  std::size_t count = 0;
  while (in >> label >> count) {
    std::vector<double> values(count, 0.0);
    for (double& value : values) {
      in >> value;
    }
    if (label == wanted) {
      return values;
    }
  }
  std::cerr << "missing spin reference block: " << wanted << '\n';
  std::exit(EXIT_FAILURE);
}

double expected_energy(const std::vector<double>& spins_aos3, double spin_baseline = 0.0) {
  double energy = 0.0;
  for (int atom = 0; atom < kAtomCount; ++atom) {
    const double sx = spins_aos3[3 * atom + 0];
    const double sy = spins_aos3[3 * atom + 1];
    const double sz = spins_aos3[3 * atom + 2];
    energy += std::tanh(0.25 * (sx * sx + sy * sy + sz * sz));
  }
  return energy + kAtomCount * spin_baseline;
}

std::vector<double> expected_mforce(const std::vector<double>& spins_aos3) {
  std::vector<double> out(spins_aos3.size(), 0.0);
  for (int atom = 0; atom < kAtomCount; ++atom) {
    const double sx = spins_aos3[3 * atom + 0];
    const double sy = spins_aos3[3 * atom + 1];
    const double sz = spins_aos3[3 * atom + 2];
    const double x = 0.25 * (sx * sx + sy * sy + sz * sz);
    const double sech2 = 1.0 - std::tanh(x) * std::tanh(x);
    out[3 * atom + 0] = -0.5 * sx * sech2;
    out[3 * atom + 1] = -0.5 * sy * sech2;
    out[3 * atom + 2] = -0.5 * sz * sech2;
  }
  return out;
}

std::vector<double> make_lammps_spins4(const std::vector<double>& spins_aos3) {
  const int atom_count = static_cast<int>(spins_aos3.size() / 3);
  std::vector<double> out(static_cast<std::size_t>(atom_count) * 4, 0.0);
  for (int atom = 0; atom < atom_count; ++atom) {
    const double sx = spins_aos3[3 * atom + 0];
    const double sy = spins_aos3[3 * atom + 1];
    const double sz = spins_aos3[3 * atom + 2];
    const double mu = std::sqrt(sx * sx + sy * sy + sz * sz);
    if (mu > 0.0) {
      out[4 * atom + 0] = sx / mu;
      out[4 * atom + 1] = sy / mu;
      out[4 * atom + 2] = sz / mu;
    }
    out[4 * atom + 3] = mu;
  }
  return out;
}

struct BatchPrediction {
  double energy = 0.0;
  std::vector<double> forces;
  std::vector<double> mforces;
};

BatchPrediction predict_batch(
    NepaModel* model,
    const std::vector<double>& positions,
    const std::vector<double>& spins,
    double spin_baseline = 0.0) {
  std::int32_t atom_counts[1] = {kAtomCount};
  std::int32_t atom_offsets[1] = {0};
  std::int32_t types[kAtomCount] = {0, 0};
  double box[9] = {8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0};
  std::int32_t pbc[3] = {1, 1, 1};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = kAtomCount;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions.data();
  batch.spins_aos3 = spins.data();
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;

  BatchPrediction prediction;
  prediction.forces.assign(static_cast<std::size_t>(kAtomCount) * 3, 0.0);
  prediction.mforces.assign(static_cast<std::size_t>(kAtomCount) * 3, 0.0);
  NepaFindForceResult result{};
  result.energy_per_structure = &prediction.energy;
  result.forces_aos3 = prediction.forces.data();
  result.mforces_aos3 = prediction.mforces.data();
  if (nepa_find_force_batch(model, &batch, &result) != NEPA_STATUS_OK) {
    std::exit(EXIT_FAILURE);
  }
  return prediction;
}

bool check_edge_derivatives(
    NepaModel* model,
    const std::vector<double>& positions,
    const std::vector<double>& spins) {
  const BatchPrediction base = predict_batch(model, positions, spins);
  const double h = 1.0e-6;
  std::vector<double> fd_force(positions.size(), 0.0);
  std::vector<double> fd_mforce(spins.size(), 0.0);
  for (std::size_t i = 0; i < positions.size(); ++i) {
    std::vector<double> plus = positions;
    std::vector<double> minus = positions;
    plus[i] += h;
    minus[i] -= h;
    fd_force[i] =
        -(predict_batch(model, plus, spins).energy -
          predict_batch(model, minus, spins).energy) / (2.0 * h);
  }
  for (std::size_t i = 0; i < spins.size(); ++i) {
    std::vector<double> plus = spins;
    std::vector<double> minus = spins;
    plus[i] += h;
    minus[i] -= h;
    fd_mforce[i] =
        -(predict_batch(model, positions, plus).energy -
          predict_batch(model, positions, minus).energy) / (2.0 * h);
  }
  const double force_diff = max_abs_diff(base.forces, fd_force);
  const double mforce_diff = max_abs_diff(base.mforces, fd_mforce);
  if (force_diff > 2.0e-8 || mforce_diff > 2.0e-8) {
    std::cerr << "edge derivative mismatch: force_diff=" << force_diff
              << " mforce_diff=" << mforce_diff << '\n';
    return false;
  }
  return true;
}

BatchPrediction predict_batch_n(
    NepaModel* model,
    int atom_count,
    const std::vector<double>& positions,
    const std::vector<double>& spins,
    std::vector<double>* virial_out = nullptr) {
  std::vector<std::int32_t> atom_counts = {atom_count};
  std::vector<std::int32_t> atom_offsets = {0};
  std::vector<std::int32_t> types(static_cast<std::size_t>(atom_count), 0);
  double box[9] = {10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0};
  std::int32_t pbc[3] = {1, 1, 1};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = atom_count;
  batch.atom_counts = atom_counts.data();
  batch.atom_offsets = atom_offsets.data();
  batch.types = types.data();
  batch.positions_aos3 = positions.data();
  batch.spins_aos3 = spins.data();
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;

  BatchPrediction prediction;
  prediction.forces.assign(static_cast<std::size_t>(atom_count) * 3, 0.0);
  prediction.mforces.assign(static_cast<std::size_t>(atom_count) * 3, 0.0);
  std::vector<double> virial(9, 0.0);
  NepaFindForceResult result{};
  result.energy_per_structure = &prediction.energy;
  result.forces_aos3 = prediction.forces.data();
  result.mforces_aos3 = prediction.mforces.data();
  result.virials_row_major9 = virial.data();
  if (nepa_find_force_batch(model, &batch, &result) != NEPA_STATUS_OK) {
    std::exit(EXIT_FAILURE);
  }
  if (virial_out) {
    *virial_out = std::move(virial);
  }
  return prediction;
}

bool check_chiral_derivatives(
    NepaModel* model,
    const std::vector<double>& positions,
    const std::vector<double>& spins) {
  const int atom_count = static_cast<int>(spins.size() / 3);
  const BatchPrediction base = predict_batch_n(model, atom_count, positions, spins);
  const double h = 1.0e-6;
  std::vector<double> fd_force(positions.size(), 0.0);
  std::vector<double> fd_mforce(spins.size(), 0.0);
  for (std::size_t i = 0; i < positions.size(); ++i) {
    std::vector<double> plus = positions;
    std::vector<double> minus = positions;
    plus[i] += h;
    minus[i] -= h;
    fd_force[i] =
        -(predict_batch_n(model, atom_count, plus, spins).energy -
          predict_batch_n(model, atom_count, minus, spins).energy) / (2.0 * h);
  }
  for (std::size_t i = 0; i < spins.size(); ++i) {
    std::vector<double> plus = spins;
    std::vector<double> minus = spins;
    plus[i] += h;
    minus[i] -= h;
    fd_mforce[i] =
        -(predict_batch_n(model, atom_count, positions, plus).energy -
          predict_batch_n(model, atom_count, positions, minus).energy) / (2.0 * h);
  }
  const double force_diff = max_abs_diff(base.forces, fd_force);
  const double mforce_diff = max_abs_diff(base.mforces, fd_mforce);
  if (force_diff > 2.0e-7 || mforce_diff > 2.0e-7) {
    std::cerr << "chiral derivative mismatch: force_diff=" << force_diff
              << " mforce_diff=" << mforce_diff << '\n';
    return false;
  }
  return true;
}

bool run_lammps_matches_batch_n(
    NepaModel* model,
    const std::vector<double>& positions,
    const std::vector<double>& spins,
    bool permute_ilist = false) {
  const int atom_count = static_cast<int>(spins.size() / 3);
  std::vector<double> batch_virial;
  const BatchPrediction batch =
      predict_batch_n(model, atom_count, positions, spins, &batch_virial);
  std::vector<int> ilist(static_cast<std::size_t>(atom_count));
  std::vector<int> numneigh(static_cast<std::size_t>(atom_count), atom_count - 1);
  std::vector<std::vector<int>> neigh_storage(static_cast<std::size_t>(atom_count));
  std::vector<int*> firstneigh(static_cast<std::size_t>(atom_count));
  for (int i = 0; i < atom_count; ++i) {
    ilist[static_cast<std::size_t>(i)] = i;
    for (int j = 0; j < atom_count; ++j) {
      if (j != i) {
        neigh_storage[static_cast<std::size_t>(i)].push_back(j);
      }
    }
    firstneigh[static_cast<std::size_t>(i)] = neigh_storage[static_cast<std::size_t>(i)].data();
  }
  if (permute_ilist && atom_count == 4) {
    ilist = {2, 0, 3, 1};
  }
  std::vector<int> types(static_cast<std::size_t>(atom_count), 1);
  int type_map[2] = {-1, 0};
  std::vector<double> position_storage = positions;
  std::vector<double> spin_storage = make_lammps_spins4(spins);
  std::vector<double*> position_rows(static_cast<std::size_t>(atom_count));
  std::vector<double*> spin_rows(static_cast<std::size_t>(atom_count));
  for (int atom = 0; atom < atom_count; ++atom) {
    position_rows[static_cast<std::size_t>(atom)] = position_storage.data() + 3 * atom;
    spin_rows[static_cast<std::size_t>(atom)] = spin_storage.data() + 4 * atom;
  }

  double total_potential = 0.0;
  double total_virial[6] = {};
  std::vector<double> forces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  std::vector<double> mforces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  std::vector<double*> force_rows(static_cast<std::size_t>(atom_count));
  std::vector<double*> mforce_rows(static_cast<std::size_t>(atom_count));
  for (int atom = 0; atom < atom_count; ++atom) {
    force_rows[static_cast<std::size_t>(atom)] = forces.data() + 3 * atom;
    mforce_rows[static_cast<std::size_t>(atom)] = mforces.data() + 3 * atom;
  }

  NepaLammpsNeighborInput input{};
  input.nlocal = atom_count;
  input.inum = atom_count;
  input.ilist = ilist.data();
  input.numneigh = numneigh.data();
  input.firstneigh = firstneigh.data();
  input.types = types.data();
  input.type_map = type_map;
  input.positions = position_rows.data();
  input.spins = spin_rows.data();

  NepaLammpsNeighborResult result{};
  result.total_potential = &total_potential;
  result.total_virial6 = total_virial;
  result.forces = force_rows.data();
  result.mforces = mforce_rows.data();

  if (nepa_find_force_lammps_neighbors(model, &input, &result) != NEPA_STATUS_OK) {
    return false;
  }
  const double batch_total_virial[6] = {
      batch_virial[0],
      batch_virial[4],
      batch_virial[8],
      0.5 * (batch_virial[1] + batch_virial[3]),
      0.5 * (batch_virial[2] + batch_virial[6]),
      0.5 * (batch_virial[5] + batch_virial[7]),
  };
  std::vector<double> batch_total_virial_vec(batch_total_virial, batch_total_virial + 6);
  std::vector<double> total_virial_vec(total_virial, total_virial + 6);
  const double energy_diff = std::abs(total_potential - batch.energy);
  const double force_diff = max_abs_diff(forces, batch.forces);
  const double mforce_diff = max_abs_diff(mforces, batch.mforces);
  const double virial_diff = max_abs_diff(total_virial_vec, batch_total_virial_vec);
  if (energy_diff > 1.0e-10 || force_diff > 1.0e-10 ||
      mforce_diff > 1.0e-10 || virial_diff > 1.0e-10) {
    std::cerr << "chiral LAMMPS mismatch: energy_diff=" << energy_diff
              << " force_diff=" << force_diff
              << " mforce_diff=" << mforce_diff
              << " virial_diff=" << virial_diff << '\n';
    return false;
  }
  return true;
}

bool check_spin_fixture_reference(NepaModel* model, const std::string& reference_path) {
  constexpr int atom_count = 4;
  const std::vector<double> positions = {
      0.2, 0.2, 0.2,
      3.7, 0.3, 0.2,
      0.4, 3.6, 0.5,
      1.8, 1.7, 3.5};
  const std::vector<double> spins = {
      1.0, 0.2, 0.0,
      0.4, -0.3, 0.7,
      -0.2, 0.8, 0.5,
      0.6, 0.1, -0.4};
  std::int32_t atom_counts[1] = {atom_count};
  std::int32_t atom_offsets[1] = {0};
  std::int32_t types[atom_count] = {0, 0, 0, 0};
  double box[9] = {4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
  std::int32_t pbc[3] = {1, 1, 1};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = atom_count;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions.data();
  batch.spins_aos3 = spins.data();
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;

  NepaModelInfo info{};
  if (nepa_model_info(model, &info) != NEPA_STATUS_OK || info.descriptor_dim != 88) {
    return false;
  }

  std::vector<double> energy(1, 0.0);
  std::vector<double> potential(atom_count, 0.0);
  std::vector<double> forces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  std::vector<double> mforces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  std::vector<double> virial(9, 0.0);
  NepaFindForceResult result{};
  result.energy_per_structure = energy.data();
  result.potential_per_atom = potential.data();
  result.forces_aos3 = forces.data();
  result.mforces_aos3 = mforces.data();
  result.virials_row_major9 = virial.data();
  if (nepa_find_force_batch(model, &batch, &result) != NEPA_STATUS_OK) {
    return false;
  }

  std::vector<double> descriptors(
      static_cast<std::size_t>(atom_count) * info.descriptor_dim, 0.0);
  NepaFindDescriptorResult descriptor_result{};
  descriptor_result.descriptors = descriptors.data();
  if (nepa_find_descriptors(model, &batch, &descriptor_result) != NEPA_STATUS_OK) {
    return false;
  }

  const double energy_diff =
      max_abs_diff(energy, read_reference_vector(reference_path, "energy_total"));
  const double potential_diff =
      max_abs_diff(potential, read_reference_vector(reference_path, "energy_atom"));
  const double force_diff =
      max_abs_diff(forces, read_reference_vector(reference_path, "force"));
  const double mforce_diff =
      max_abs_diff(mforces, read_reference_vector(reference_path, "mforce"));
  const double virial_diff =
      max_abs_diff(virial, read_reference_vector(reference_path, "virial9"));
  const double descriptor_diff =
      max_abs_diff(descriptors, read_reference_vector(reference_path, "descriptor"));
  if (energy_diff > 1.0e-10 || potential_diff > 1.0e-10 ||
      force_diff > 1.0e-10 || mforce_diff > 1.0e-10 ||
      virial_diff > 1.0e-9 || descriptor_diff > 1.0e-10) {
    std::cerr << "spin fixture reference mismatch: energy_diff=" << energy_diff
              << " potential_diff=" << potential_diff
              << " force_diff=" << force_diff
              << " mforce_diff=" << mforce_diff
              << " virial_diff=" << virial_diff
              << " descriptor_diff=" << descriptor_diff << '\n';
    return false;
  }
  return true;
}

std::vector<double> descriptors(
    NepaModel* model,
    const std::vector<double>& positions,
    const std::vector<double>& spins) {
  std::int32_t atom_counts[1] = {kAtomCount};
  std::int32_t atom_offsets[1] = {0};
  std::int32_t types[kAtomCount] = {0, 0};
  double box[9] = {8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0};
  std::int32_t pbc[3] = {1, 1, 1};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = kAtomCount;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions.data();
  batch.spins_aos3 = spins.data();
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;

  std::vector<double> out(static_cast<std::size_t>(kAtomCount) * kDescriptorDim, 0.0);
  NepaFindDescriptorResult result{};
  result.descriptors = out.data();
  if (nepa_find_descriptors(model, &batch, &result) != NEPA_STATUS_OK) {
    std::exit(EXIT_FAILURE);
  }
  return out;
}

bool run_batch(
    NepaModel* model,
    const std::vector<double>& positions,
    const std::vector<double>& spins,
    double spin_baseline = 0.0) {
  std::int32_t atom_counts[1] = {kAtomCount};
  std::int32_t atom_offsets[1] = {0};
  std::int32_t types[kAtomCount] = {0, 0};
  double box[9] = {8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0};
  std::int32_t pbc[3] = {1, 1, 1};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = kAtomCount;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions.data();
  batch.spins_aos3 = spins.data();
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;

  double energy = 0.0;
  std::vector<double> forces(static_cast<std::size_t>(kAtomCount) * 3, 0.0);
  std::vector<double> mforces(static_cast<std::size_t>(kAtomCount) * 3, 0.0);
  double virial[9] = {};
  NepaFindForceResult result{};
  result.energy_per_structure = &energy;
  result.forces_aos3 = forces.data();
  result.virials_row_major9 = virial;
  result.mforces_aos3 = mforces.data();

  if (nepa_find_force_batch(model, &batch, &result) != NEPA_STATUS_OK) {
    return false;
  }

  const std::vector<double> zeros(forces.size(), 0.0);
  const double energy_diff = std::abs(energy - expected_energy(spins, spin_baseline));
  const double force_diff = max_abs_diff(forces, zeros);
  const double mforce_diff = max_abs_diff(mforces, expected_mforce(spins));
  if (energy_diff > 1.0e-11 || force_diff > 1.0e-9 || mforce_diff > 1.0e-9) {
    std::cerr << "batch spin mismatch: energy_diff=" << energy_diff
              << " force_diff=" << force_diff
              << " mforce_diff=" << mforce_diff << '\n';
    return false;
  }
  return true;
}

bool run_multi_structure_batch_matches_single(NepaModel* model) {
  constexpr int structure_count = 3;
  std::vector<std::int32_t> atom_counts(structure_count, kAtomCount);
  std::vector<std::int32_t> atom_offsets(structure_count, 0);
  std::vector<std::int32_t> types(static_cast<std::size_t>(structure_count) * kAtomCount, 0);
  std::vector<double> boxes(static_cast<std::size_t>(structure_count) * 9, 0.0);
  std::vector<std::int32_t> pbc(static_cast<std::size_t>(structure_count) * 3, 1);
  std::vector<double> positions(static_cast<std::size_t>(structure_count) * kAtomCount * 3);
  std::vector<double> spins(positions.size());
  for (int structure = 0; structure < structure_count; ++structure) {
    atom_offsets[structure] = structure * kAtomCount;
    boxes[static_cast<std::size_t>(structure) * 9 + 0] = 8.0;
    boxes[static_cast<std::size_t>(structure) * 9 + 4] = 8.0;
    boxes[static_cast<std::size_t>(structure) * 9 + 8] = 8.0;
    const std::vector<double> one_positions = {
        0.1 * structure, 0.0, 0.0,
        1.2, 0.4 + 0.1 * structure, 0.3};
    const std::vector<double> one_spins = {
        0.2 + 0.1 * structure, -0.4, 0.5,
        -0.3, 0.1, 0.6 - 0.05 * structure};
    std::copy(
        one_positions.begin(),
        one_positions.end(),
        positions.begin() + static_cast<std::ptrdiff_t>(structure * kAtomCount * 3));
    std::copy(
        one_spins.begin(),
        one_spins.end(),
        spins.begin() + static_cast<std::ptrdiff_t>(structure * kAtomCount * 3));
  }

  NepaStructureBatch batch{};
  batch.num_structures = structure_count;
  batch.total_atoms = structure_count * kAtomCount;
  batch.atom_counts = atom_counts.data();
  batch.atom_offsets = atom_offsets.data();
  batch.types = types.data();
  batch.positions_aos3 = positions.data();
  batch.spins_aos3 = spins.data();
  batch.boxes_row_major9 = boxes.data();
  batch.pbc_flags3 = pbc.data();

  std::vector<double> energy(structure_count, 0.0);
  std::vector<double> forces(positions.size(), 0.0);
  std::vector<double> mforces(spins.size(), 0.0);
  NepaFindForceResult result{};
  result.energy_per_structure = energy.data();
  result.forces_aos3 = forces.data();
  result.mforces_aos3 = mforces.data();
  if (nepa_find_force_batch(model, &batch, &result) != NEPA_STATUS_OK) {
    return false;
  }

  for (int structure = 0; structure < structure_count; ++structure) {
    const auto begin =
        static_cast<std::size_t>(structure) * kAtomCount * 3;
    const std::vector<double> one_positions(
        positions.begin() + static_cast<std::ptrdiff_t>(begin),
        positions.begin() + static_cast<std::ptrdiff_t>(begin + kAtomCount * 3));
    const std::vector<double> one_spins(
        spins.begin() + static_cast<std::ptrdiff_t>(begin),
        spins.begin() + static_cast<std::ptrdiff_t>(begin + kAtomCount * 3));
    const BatchPrediction single = predict_batch(model, one_positions, one_spins);
    const std::vector<double> batch_forces(
        forces.begin() + static_cast<std::ptrdiff_t>(begin),
        forces.begin() + static_cast<std::ptrdiff_t>(begin + kAtomCount * 3));
    const std::vector<double> batch_mforces(
        mforces.begin() + static_cast<std::ptrdiff_t>(begin),
        mforces.begin() + static_cast<std::ptrdiff_t>(begin + kAtomCount * 3));
    if (std::abs(energy[structure] - single.energy) > 1.0e-12 ||
        max_abs_diff(batch_forces, single.forces) > 1.0e-12 ||
        max_abs_diff(batch_mforces, single.mforces) > 1.0e-12) {
      return false;
    }
  }
  return true;
}

bool run_lammps(
    NepaModel* model,
    const std::vector<double>& positions,
    const std::vector<double>& spins,
    double spin_baseline = 0.0,
    bool zero_neighbors = false) {
  int ilist[kAtomCount] = {0, 1};
  int numneigh[kAtomCount] = {
      zero_neighbors ? 0 : 1,
      zero_neighbors ? 0 : 1};
  int neigh0[1] = {1};
  int neigh1[1] = {0};
  int* firstneigh[kAtomCount] = {neigh0, neigh1};
  int types[kAtomCount] = {1, 1};
  int type_map[2] = {-1, 0};
  std::vector<double> position_storage = positions;
  std::vector<double> spin_storage = make_lammps_spins4(spins);
  double* position_rows[kAtomCount] = {
      position_storage.data(),
      position_storage.data() + 3};
  double* spin_rows[kAtomCount] = {
      spin_storage.data(),
      spin_storage.data() + 4};

  double total_potential = 0.0;
  double total_virial[6] = {};
  std::vector<double> forces(static_cast<std::size_t>(kAtomCount) * 3, 0.0);
  std::vector<double> mforces(static_cast<std::size_t>(kAtomCount) * 3, 0.0);
  double* force_rows[kAtomCount] = {forces.data(), forces.data() + 3};
  double* mforce_rows[kAtomCount] = {mforces.data(), mforces.data() + 3};

  NepaLammpsNeighborInput input{};
  input.nlocal = kAtomCount;
  input.inum = kAtomCount;
  input.ilist = ilist;
  input.numneigh = numneigh;
  input.firstneigh = firstneigh;
  input.types = types;
  input.type_map = type_map;
  input.positions = position_rows;
  input.spins = spin_rows;

  NepaLammpsNeighborResult result{};
  result.total_potential = &total_potential;
  result.total_virial6 = total_virial;
  result.forces = force_rows;
  result.mforces = mforce_rows;

  if (nepa_find_force_lammps_neighbors(model, &input, &result) != NEPA_STATUS_OK) {
    return false;
  }

  const std::vector<double> zeros(forces.size(), 0.0);
  const double energy_diff = std::abs(total_potential - expected_energy(spins, spin_baseline));
  const double force_diff = max_abs_diff(forces, zeros);
  const double mforce_diff = max_abs_diff(mforces, expected_mforce(spins));
  if (energy_diff > 1.0e-11 || force_diff > 1.0e-9 || mforce_diff > 1.0e-9) {
    std::cerr << "LAMMPS spin mismatch: energy_diff=" << energy_diff
              << " force_diff=" << force_diff
              << " mforce_diff=" << mforce_diff << '\n';
    return false;
  }
  return true;
}

bool check_lammps_neighbor_virial_ownership(
    NepaModel* model,
    const std::vector<double>& positions,
    const std::vector<double>& spins) {
  const int atom_count = static_cast<int>(spins.size() / 3);
  int ilist[1] = {0};
  std::vector<int> numneigh(static_cast<std::size_t>(atom_count), 0);
  numneigh[0] = atom_count - 1;
  std::vector<int> neighbors(static_cast<std::size_t>(atom_count - 1));
  for (int atom = 1; atom < atom_count; ++atom) {
    neighbors[static_cast<std::size_t>(atom - 1)] = atom;
  }
  std::vector<int*> firstneigh(static_cast<std::size_t>(atom_count), nullptr);
  firstneigh[0] = neighbors.data();
  std::vector<int> types(static_cast<std::size_t>(atom_count), 1);
  int type_map[2] = {-1, 0};
  std::vector<double> position_storage = positions;
  std::vector<double> spin_storage = make_lammps_spins4(spins);
  std::vector<double*> position_rows(static_cast<std::size_t>(atom_count));
  std::vector<double*> spin_rows(static_cast<std::size_t>(atom_count));
  for (int atom = 0; atom < atom_count; ++atom) {
    position_rows[static_cast<std::size_t>(atom)] = position_storage.data() + 3 * atom;
    spin_rows[static_cast<std::size_t>(atom)] = spin_storage.data() + 4 * atom;
  }

  double total_potential = 0.0;
  double total_virial[6] = {};
  std::vector<double> forces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  std::vector<double> mforces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  std::vector<double> atom_virial(static_cast<std::size_t>(atom_count) * 9, 0.0);
  std::vector<double*> force_rows(static_cast<std::size_t>(atom_count));
  std::vector<double*> mforce_rows(static_cast<std::size_t>(atom_count));
  std::vector<double*> virial_rows(static_cast<std::size_t>(atom_count));
  for (int atom = 0; atom < atom_count; ++atom) {
    force_rows[static_cast<std::size_t>(atom)] = forces.data() + 3 * atom;
    mforce_rows[static_cast<std::size_t>(atom)] = mforces.data() + 3 * atom;
    virial_rows[static_cast<std::size_t>(atom)] = atom_virial.data() + 9 * atom;
  }

  NepaLammpsNeighborInput input{};
  input.nlocal = 1;
  input.inum = 1;
  input.ilist = ilist;
  input.numneigh = numneigh.data();
  input.firstneigh = firstneigh.data();
  input.types = types.data();
  input.type_map = type_map;
  input.positions = position_rows.data();
  input.spins = spin_rows.data();

  NepaLammpsNeighborResult result{};
  result.total_potential = &total_potential;
  result.total_virial6 = total_virial;
  result.forces = force_rows.data();
  result.mforces = mforce_rows.data();
  result.virials_per_atom9 = virial_rows.data();
  if (nepa_find_force_lammps_neighbors(model, &input, &result) != NEPA_STATUS_OK) {
    return false;
  }

  double center_max = 0.0;
  double neighbor_max = 0.0;
  for (int component = 0; component < 9; ++component) {
    center_max = std::max(center_max, std::abs(atom_virial[component]));
    for (int atom = 1; atom < atom_count; ++atom) {
      neighbor_max = std::max(
          neighbor_max,
          std::abs(atom_virial[static_cast<std::size_t>(atom) * 9 + component]));
    }
  }
  double total_diff = 0.0;
  for (int component = 0; component < 6; ++component) {
    double atom_sum = 0.0;
    for (int atom = 1; atom < atom_count; ++atom) {
      const std::size_t base = static_cast<std::size_t>(atom) * 9;
      if (component < 3) {
        atom_sum += atom_virial[base + component];
      } else {
        atom_sum += 0.5 * (
            atom_virial[base + component] + atom_virial[base + component + 3]);
      }
    }
    total_diff = std::max(
        total_diff,
        std::abs(total_virial[component] - atom_sum));
  }
  if (center_max > 1.0e-12 || neighbor_max < 1.0e-12 || total_diff > 1.0e-10) {
    std::cerr << "LAMMPS spin virial ownership mismatch: center_max=" << center_max
              << " neighbor_max=" << neighbor_max
              << " total_diff=" << total_diff << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (spin_descriptor_dim(1, 4) != kSpinDim ||
      spin_descriptor_dim(1, 4, true) != kChiralSpinDim ||
      !nep_adapters::register_cpu_engine()) {
    return EXIT_FAILURE;
  }

  const std::string model_path = "spin_generated.nep";
  const std::string chiral_path = "spin_chiral_generated.nep";
  const std::string edge_path = "spin_edge_generated.nep";
  const std::string baseline_path = "spin_baseline_generated.nep";
  const std::string chiral_bulk_path = "spin_chiral_bulk_generated.nep";
  const std::string chiral_polar_path = "spin_chiral_polar_generated.nep";
  const std::string chiral_pseudo_path = "spin_chiral_pseudo_generated.nep";
  const std::string expanded_lmax_path = "expanded_lmax_generated.nep";
  write_model(model_path);
  write_model(chiral_path, true);
  write_model(edge_path, false, 3);
  write_model(baseline_path, false, 1, 1.25);
  write_model(chiral_bulk_path, true, 1 + kSpinDim);
  write_model(chiral_polar_path, true, 1 + kSpinDim + 1);
  write_model(chiral_pseudo_path, true, 1 + kSpinDim + 2);
  write_expanded_lmax_model(expanded_lmax_path);

  NepaModel* expanded_lmax_model = nullptr;
  NepaModelInfo expanded_lmax_info{};
  const bool expanded_lmax_ok =
      nepa_load_model("cpu", expanded_lmax_path.c_str(), &expanded_lmax_model) ==
          NEPA_STATUS_OK &&
      expanded_lmax_model != nullptr &&
      nepa_model_info(expanded_lmax_model, &expanded_lmax_info) == NEPA_STATUS_OK &&
      expanded_lmax_info.descriptor_dim == 10;
  nepa_free_model(expanded_lmax_model);

  NepaModel* chiral_model = nullptr;
  if (nepa_load_model("cpu", chiral_path.c_str(), &chiral_model) != NEPA_STATUS_OK ||
      chiral_model == nullptr) {
    std::cerr << "spin chiral model did not load\n";
    return EXIT_FAILURE;
  }
  NepaModelInfo chiral_info{};
  if (nepa_model_info(chiral_model, &chiral_info) != NEPA_STATUS_OK ||
      chiral_info.descriptor_dim != kChiralDescriptorDim ||
      !nep_adapters::has_capability(chiral_info.capabilities, nep_adapters::Capability::spin)) {
    nepa_free_model(chiral_model);
    std::cerr << "spin chiral model_info mismatch\n";
    return EXIT_FAILURE;
  }
  nepa_free_model(chiral_model);
#ifdef NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE
  NepaModel* fixture_model = nullptr;
  if (nepa_load_model("cpu", NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE, &fixture_model) !=
          NEPA_STATUS_OK ||
      fixture_model == nullptr) {
    std::cerr << "spin_chiral fixture did not load\n";
    return EXIT_FAILURE;
  }
  NepaModelInfo fixture_info{};
  if (nepa_model_info(fixture_model, &fixture_info) != NEPA_STATUS_OK ||
      fixture_info.descriptor_dim != 88 ||
      !nep_adapters::has_capability(fixture_info.capabilities, nep_adapters::Capability::spin)) {
    nepa_free_model(fixture_model);
    std::cerr << "spin_chiral fixture model_info mismatch\n";
    return EXIT_FAILURE;
  }
  nepa_free_model(fixture_model);
#endif

  NepaModel* model = nullptr;
  if (nepa_load_model("cpu", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }

  NepaModelInfo info{};
  if (nepa_model_info(model, &info) != NEPA_STATUS_OK ||
      info.descriptor_dim != kDescriptorDim ||
      !nep_adapters::has_capability(info.capabilities, nep_adapters::Capability::spin)) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  const std::vector<double> positions = {0.0, 0.0, 0.0, 1.2, 0.4, 0.3};
  const std::vector<double> spins = {0.2, -0.4, 0.5, -0.3, 0.1, 0.6};
  const std::vector<double> flipped_spins = {-0.2, 0.4, -0.5, 0.3, -0.1, -0.6};
  const std::vector<double> rotated_positions = {0.0, 0.0, 0.0, -0.4, 1.2, 0.3};
  const std::vector<double> rotated_spins = {0.4, 0.2, 0.5, -0.1, -0.3, 0.6};

  const std::vector<double> desc = descriptors(model, positions, spins);
  const double flip_diff = max_abs_diff(desc, descriptors(model, positions, flipped_spins));
  const double rotate_diff =
      max_abs_diff(desc, descriptors(model, rotated_positions, rotated_spins));

  const bool ok = flip_diff < 1.0e-10 && rotate_diff < 1.0e-10 &&
                  run_batch(model, positions, spins) &&
                  run_multi_structure_batch_matches_single(model) &&
                  run_lammps(model, positions, spins) &&
                  run_lammps(model, positions, spins, 0.0, true);
  nepa_free_model(model);

  NepaModel* edge_model = nullptr;
  const bool edge_ok =
      nepa_load_model("cpu", edge_path.c_str(), &edge_model) == NEPA_STATUS_OK &&
      edge_model != nullptr && check_edge_derivatives(edge_model, positions, spins) &&
      check_lammps_neighbor_virial_ownership(edge_model, positions, spins);
  nepa_free_model(edge_model);

  NepaModel* baseline_model = nullptr;
  const bool baseline_ok =
      nepa_load_model("cpu", baseline_path.c_str(), &baseline_model) == NEPA_STATUS_OK &&
      baseline_model != nullptr &&
      run_batch(baseline_model, positions, spins, 1.25) &&
      run_lammps(baseline_model, positions, spins, 1.25);
  nepa_free_model(baseline_model);

  const std::vector<double> chiral_positions = {
      0.0, 0.0, 0.0,
      1.2, 0.1, 0.0,
      0.2, 1.1, 0.3,
      0.4, 0.3, 1.4};
  const std::vector<double> chiral_spins = {
      1.0, 0.2, 0.0,
      0.4, -0.3, 0.7,
      -0.2, 0.8, 0.5,
      0.6, 0.1, -0.4};
  bool chiral_numeric_ok = true;
  for (const std::string& path : {chiral_bulk_path, chiral_polar_path, chiral_pseudo_path}) {
    NepaModel* chiral_numeric_model = nullptr;
    chiral_numeric_ok =
        chiral_numeric_ok &&
        nepa_load_model("cpu", path.c_str(), &chiral_numeric_model) == NEPA_STATUS_OK &&
        chiral_numeric_model != nullptr &&
        check_chiral_derivatives(chiral_numeric_model, chiral_positions, chiral_spins) &&
        check_lammps_neighbor_virial_ownership(
            chiral_numeric_model, chiral_positions, chiral_spins) &&
        run_lammps_matches_batch_n(chiral_numeric_model, chiral_positions, chiral_spins) &&
        run_lammps_matches_batch_n(chiral_numeric_model, chiral_positions, chiral_spins, true);
    nepa_free_model(chiral_numeric_model);
  }
#ifdef NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE
  NepaModel* fixture_numeric_model = nullptr;
  const bool fixture_numeric_ok =
      nepa_load_model("cpu", NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE, &fixture_numeric_model) ==
          NEPA_STATUS_OK &&
      fixture_numeric_model != nullptr &&
      run_lammps_matches_batch_n(fixture_numeric_model, chiral_positions, chiral_spins) &&
      run_lammps_matches_batch_n(fixture_numeric_model, chiral_positions, chiral_spins, true)
#ifdef NEP_ADAPTERS_SPIN_CHIRAL_REFERENCE
      && check_spin_fixture_reference(fixture_numeric_model, NEP_ADAPTERS_SPIN_CHIRAL_REFERENCE)
#endif
      ;
  nepa_free_model(fixture_numeric_model);
#else
  const bool fixture_numeric_ok = true;
#endif

  std::remove(model_path.c_str());
  std::remove(chiral_path.c_str());
  std::remove(edge_path.c_str());
  std::remove(baseline_path.c_str());
  std::remove(chiral_bulk_path.c_str());
  std::remove(chiral_polar_path.c_str());
  std::remove(chiral_pseudo_path.c_str());
  std::remove(expanded_lmax_path.c_str());

  if (!ok || !edge_ok || !baseline_ok || !chiral_numeric_ok || !fixture_numeric_ok ||
      !expanded_lmax_ok) {
    std::cerr << "descriptor invariance diffs: flip=" << flip_diff
              << " rotate=" << rotate_diff << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
