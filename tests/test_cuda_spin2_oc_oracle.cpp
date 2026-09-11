#include "nep_adapters/api.h"
#if defined(NEP_ADAPTERS_TEST_CPU_ENGINE)
#include "nep_adapters/engines/cpu.hpp"
#else
#include "nep_adapters/engines/cuda.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int g_descriptor_dim = 0;

struct Case {
  std::string name;
  int atoms = 0;
  std::vector<int> types;
  std::vector<double> cell, positions, spins, potential, forces, virial,
      mforces, tau, descriptor;
};

std::vector<double> read_array(std::istream& input, const std::string& expected) {
  std::string name; std::size_t count = 0; input >> name >> count;
  if (name != expected) throw std::runtime_error("expected " + expected);
  std::vector<double> values(count);
  for (double& value : values) input >> value;
  return values;
}

std::vector<Case> read_oracle(const std::string& path) {
  std::ifstream input(path); std::string token, hash; int descriptor_dim = 0;
  input >> token;
  if (token != "spin2_oc_oracle_v1" && token != "spin3_oc_oracle_v1")
    throw std::runtime_error("bad versioned O/C spin oracle");
  input >> token >> hash;
  if (token != "model_sha256" ||
      (hash != "2be994d33e029928f98cfb56b8b1b1ce74e323f324919c34dc21291cf50364b8" &&
       hash != "8132407fecffb8a4ae6434748a556f8436d22bcd5b7e2c9ee5f71ffb108f3344" &&
       hash != "1744fddc200b83a842cdf46bf57a5207bd074c7801e24d69863c66b4a961abbd" &&
       hash != "961f2e47a6ac13f516eba20390150eeb1d83d38c21debb43683a2c56a170a271" &&
       hash != "f699bd78b4314c522d04761cc6909946cc3ebf5ca5046016851c25eae58cc2e9" &&
       hash != "08bffedb58efd700be8044463a668c172c00fa82666d72dc730a975d7e047107" &&
       hash != "51d363c4a2b2f3d65374eef7623bf711063a809b35598171641b91e986f4652a" &&
       hash != "49b0e1dc04d743bec8d1eeb1342048ed86474e4043d36164dbf974a2babb842c" &&
       hash != "4fe437867031a28e0cfba62d701a7b60c85fe7d6f728653be6552ab3ee57430d" &&
       hash != "5ad1b0e0ecb4f9174615b69de8e89c3d8e949f85dd35be8564adf77cde88384b" &&
       hash != "e5df123b708ee57af52d0a442c433dbc2418ff37f552a9e1b987f869d077c974" &&
       hash != "6d9c5c66916f0b0ce2825795e9c0af2908021317628db083e0669431867fd501"))
    throw std::runtime_error("versioned O/C spin oracle model hash mismatch");
  input >> token >> descriptor_dim;
  if (token != "descriptor_dim" ||
      (descriptor_dim != 33 && descriptor_dim != 49 && descriptor_dim != 53 &&
       descriptor_dim != 67 && descriptor_dim != 79 &&
       descriptor_dim != 85 && descriptor_dim != 109 && descriptor_dim != 121 &&
       descriptor_dim != 159 &&
       descriptor_dim != 211))
    throw std::runtime_error("bad full descriptor dimension");
  g_descriptor_dim = descriptor_dim;
  input >> token >> descriptor_dim;
  if (token != "spin_descriptor_dim" ||
      descriptor_dim != g_descriptor_dim - 30)
    throw std::runtime_error("bad spin descriptor dimension");
  std::vector<Case> result;
  while (input >> token) {
    if (token != "case") throw std::runtime_error("expected case");
    Case item; input >> item.name >> item.atoms >> token;
    if (token != "types") throw std::runtime_error("expected types");
    item.types.resize(item.atoms);
    for (int& type : item.types) {
      std::string symbol; input >> symbol;
      type = symbol == "Fe" ? 0 : symbol == "Ge" ? 1 : -1;
      if (type < 0) throw std::runtime_error("unknown type");
    }
    item.cell = read_array(input, "cell");
    item.positions = read_array(input, "positions");
    item.spins = read_array(input, "spins");
    item.potential = read_array(input, "potential");
    item.forces = read_array(input, "forces");
    item.virial = read_array(input, "virial");
    item.mforces = read_array(input, "mforces");
    item.tau = read_array(input, "tau");
    item.descriptor = read_array(input, "descriptor");
    input >> token;
    if (token != "end") throw std::runtime_error("unterminated case");
    result.push_back(std::move(item));
  }
  return result;
}

double max_abs(const std::vector<double>& a, const std::vector<double>& b,
               std::size_t* worst = nullptr) {
  if (a.size() != b.size()) return INFINITY;
  double error = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double current = std::abs(a[i] - b[i]);
    if (current > error) { error = current; if (worst) *worst = i; }
  }
  return error;
}

bool within_tolerance(const std::vector<double>& actual,
                      const std::vector<double>& expected,
                      double absolute_tolerance,
                      double relative_tolerance) {
  if (actual.size() != expected.size()) return false;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double scale = std::max(std::abs(actual[i]), std::abs(expected[i]));
    if (std::abs(actual[i] - expected[i]) >
        absolute_tolerance + relative_tolerance * scale) {
      return false;
    }
  }
  return true;
}

bool within_tolerance(double actual, double expected,
                      double absolute_tolerance,
                      double relative_tolerance) {
  const double scale = std::max(std::abs(actual), std::abs(expected));
  return std::abs(actual - expected) <=
      absolute_tolerance + relative_tolerance * scale;
}

struct Evaluation {
  double energy = 0;
  std::vector<double> potential, forces, atom_virial, virial, mforces,
      transfer, descriptor;
};

Evaluation evaluate(
    NepaModel* model,
    const Case& item,
    bool descriptors = true,
    bool auxiliary_outputs = true) {
  const int counts[] = {item.atoms}, offsets[] = {0}, pbc[] = {1, 1, 1};
  NepaStructureBatch batch{};
  batch.num_structures = 1; batch.total_atoms = item.atoms;
  batch.atom_counts = counts; batch.atom_offsets = offsets;
  batch.types = item.types.data(); batch.positions_aos3 = item.positions.data();
  batch.spins_aos3 = item.spins.data(); batch.boxes_row_major9 = item.cell.data();
  batch.pbc_flags3 = pbc;
  Evaluation out;
  out.potential.resize(item.atoms); out.forces.resize(3 * item.atoms);
  out.atom_virial.resize(9 * item.atoms); out.virial.resize(9);
  out.mforces.resize(3 * item.atoms); out.transfer.resize(9 * item.atoms);
  NepaFindForceResult force{};
  force.energy_per_structure = &out.energy;
  force.potential_per_atom = out.potential.data(); force.forces_aos3 = out.forces.data();
  force.virials_row_major9 = auxiliary_outputs ? out.virial.data() : nullptr;
  force.virials_per_atom_row_major9 =
      auxiliary_outputs ? out.atom_virial.data() : nullptr;
  force.mforces_aos3 = out.mforces.data();
  force.spin_transfer_per_atom_row_major9 =
      auxiliary_outputs ? out.transfer.data() : nullptr;
  if (nepa_find_force_batch(model, &batch, &force) != NEPA_STATUS_OK)
    throw std::runtime_error(nepa_last_error_message());
  if (descriptors) {
    out.descriptor.resize(
        static_cast<std::size_t>(item.atoms) * g_descriptor_dim);
    NepaFindDescriptorResult desc{}; desc.descriptors = out.descriptor.data();
    if (nepa_find_descriptors(model, &batch, &desc) != NEPA_STATUS_OK)
      throw std::runtime_error(nepa_last_error_message());
  }
  return out;
}

#if defined(NEP_ADAPTERS_TEST_CPU_ENGINE)
Evaluation evaluate_lammps(NepaModel* model, const Case& item) {
  const int n = item.atoms;
  std::vector<int> ilist(static_cast<std::size_t>(n));
  std::vector<int> numneigh(static_cast<std::size_t>(n), n - 1);
  std::vector<std::vector<int>> neighbors(static_cast<std::size_t>(n));
  std::vector<int*> firstneigh(static_cast<std::size_t>(n));
  std::vector<int> types(static_cast<std::size_t>(n));
  std::vector<int> type_map = {-1, 0, 1};
  std::vector<double*> position_rows(static_cast<std::size_t>(n));
  std::vector<double> lammps_spins(static_cast<std::size_t>(n) * 4);
  std::vector<double*> spin_rows(static_cast<std::size_t>(n));
  for (int atom = 0; atom < n; ++atom) {
    ilist[atom] = atom;
    types[atom] = item.types[atom] + 1;
    position_rows[atom] = const_cast<double*>(item.positions.data() + 3 * atom);
    spin_rows[atom] = lammps_spins.data() + 4 * atom;
    spin_rows[atom][0] = item.spins[3 * atom];
    spin_rows[atom][1] = item.spins[3 * atom + 1];
    spin_rows[atom][2] = item.spins[3 * atom + 2];
    spin_rows[atom][3] = 1.0;
    for (int other = 0; other < n; ++other) {
      if (other != atom) neighbors[atom].push_back(other);
    }
    firstneigh[atom] = neighbors[atom].data();
  }
  Evaluation out;
  out.potential.assign(static_cast<std::size_t>(n), 0.0);
  out.forces.assign(static_cast<std::size_t>(n) * 3, 0.0);
  out.mforces.assign(static_cast<std::size_t>(n) * 3, 0.0);
  out.atom_virial.assign(static_cast<std::size_t>(n) * 9, 0.0);
  out.transfer.assign(static_cast<std::size_t>(n) * 9, 0.0);
  out.virial.assign(6, 0.0);
  std::vector<double*> force_rows(static_cast<std::size_t>(n));
  std::vector<double*> mforce_rows(static_cast<std::size_t>(n));
  std::vector<double*> virial_rows(static_cast<std::size_t>(n));
  std::vector<double*> transfer_rows(static_cast<std::size_t>(n));
  for (int atom = 0; atom < n; ++atom) {
    force_rows[atom] = out.forces.data() + 3 * atom;
    mforce_rows[atom] = out.mforces.data() + 3 * atom;
    virial_rows[atom] = out.atom_virial.data() + 9 * atom;
    transfer_rows[atom] = out.transfer.data() + 9 * atom;
  }
  NepaLammpsNeighborInput input{};
  input.nlocal = n; input.inum = n; input.ilist = ilist.data();
  input.numneigh = numneigh.data(); input.firstneigh = firstneigh.data();
  input.types = types.data(); input.type_map = type_map.data();
  input.positions = position_rows.data(); input.spins = spin_rows.data();
  NepaLammpsNeighborResult result{};
  result.total_potential = &out.energy; result.total_virial6 = out.virial.data();
  result.potential_per_atom = out.potential.data(); result.forces = force_rows.data();
  result.mforces = mforce_rows.data(); result.virials_per_atom9 = virial_rows.data();
  result.spin_transfer_per_atom_row_major9 = transfer_rows.data();
  if (nepa_find_force_lammps_neighbors(model, &input, &result) != NEPA_STATUS_OK) {
    throw std::runtime_error(nepa_last_error_message());
  }
  return out;
}
#endif

bool check_case(NepaModel* model, const Case& item, bool forward_only) {
  const Evaluation out = evaluate(model, item);
  const Evaluation steady = evaluate(model, item, false, false);
  std::vector<double> expected_virial(9), tau(3 * item.atoms);
  for (int atom = 0; atom < item.atoms; ++atom) {
    for (int k = 0; k < 9; ++k) expected_virial[k] += item.virial[9 * atom + k];
    const double* s = item.spins.data() + 3 * atom;
    const double* m = out.mforces.data() + 3 * atom;
    tau[3 * atom] = s[1] * m[2] - s[2] * m[1];
    tau[3 * atom + 1] = s[2] * m[0] - s[0] * m[2];
    tau[3 * atom + 2] = s[0] * m[1] - s[1] * m[0];
  }
  std::size_t worst = 0;
  const double descriptor = max_abs(out.descriptor, item.descriptor, &worst);
  const double potential = max_abs(out.potential, item.potential);
  const double expected_energy =
      std::accumulate(item.potential.begin(), item.potential.end(), 0.0);
  const double energy = std::abs(out.energy - expected_energy);
  const double force = max_abs(out.forces, item.forces);
  const double mforce = max_abs(out.mforces, item.mforces);
  const double virial = max_abs(out.virial, expected_virial);
  const double atom_virial = max_abs(out.atom_virial, item.virial);
  const double tau_error = max_abs(tau, item.tau);
  const double steady_potential = max_abs(steady.potential, item.potential);
  const double steady_energy = std::abs(steady.energy -
      std::accumulate(item.potential.begin(), item.potential.end(), 0.0));
  const double steady_force = max_abs(steady.forces, item.forces);
  std::size_t worst_steady_mforce = 0;
  const double steady_mforce =
      max_abs(steady.mforces, item.mforces, &worst_steady_mforce);
  std::cout << item.name << " descriptor=" << descriptor
            << " potential=" << potential << " energy=" << energy
            << " force=" << force << " mforce=" << mforce
            << " virial=" << virial << " atom_virial=" << atom_virial
            << " tau=" << tau_error
            << " steady_potential=" << steady_potential
            << " steady_energy=" << steady_energy
            << " steady_force=" << steady_force
            << " steady_mforce=" << steady_mforce
            << " steady_mforce_index=" << worst_steady_mforce
            << " steady_mforce_actual=" << steady.mforces[worst_steady_mforce]
            << " steady_mforce_expected=" << item.mforces[worst_steady_mforce]
            << " worst_descriptor=" << worst << '\n';
  constexpr double fp32_relative_tolerance = 1.0e-6;
  const bool forward = descriptor < 1.2e-3 &&
      within_tolerance(out.potential, item.potential, 1.2e-3,
                       fp32_relative_tolerance) &&
      within_tolerance(out.energy, expected_energy, 3e-3,
                       fp32_relative_tolerance);
  const bool derivative =
      within_tolerance(out.forces, item.forces, 5e-3,
                       fp32_relative_tolerance) &&
      mforce < 5e-3 &&
      within_tolerance(out.virial, expected_virial, 2e-2,
                       fp32_relative_tolerance) &&
      within_tolerance(out.atom_virial, item.virial, 2e-2,
                       fp32_relative_tolerance) &&
      tau_error < 6e-3;
  const bool steady_state =
      within_tolerance(steady.potential, item.potential, 1.2e-3,
                       fp32_relative_tolerance) &&
      within_tolerance(steady.energy, expected_energy, 3e-3,
                       fp32_relative_tolerance) &&
      within_tolerance(steady.forces, item.forces, 5e-3,
                       fp32_relative_tolerance) &&
      steady_mforce < 5e-3;
  bool lammps_ok = true;
#if defined(NEP_ADAPTERS_TEST_CPU_ENGINE)
  if (item.name == "noncollinear") {
    const Evaluation lammps = evaluate_lammps(model, item);
    const double lammps_energy = std::abs(lammps.energy - out.energy);
    const double lammps_force = max_abs(lammps.forces, out.forces);
    const double lammps_mforce = max_abs(lammps.mforces, out.mforces);
    const double lammps_transfer = max_abs(lammps.transfer, out.transfer);
    std::vector<double> expected_atom_virial(static_cast<std::size_t>(item.atoms) * 9);
    const int lammps_from_raw[9] = {0, 4, 8, 1, 2, 5, 3, 6, 7};
    for (int atom = 0; atom < item.atoms; ++atom) {
      for (int component = 0; component < 9; ++component) {
        expected_atom_virial[9 * atom + component] =
            out.atom_virial[9 * atom + lammps_from_raw[component]];
      }
    }
    const double lammps_atom_virial =
        max_abs(lammps.atom_virial, expected_atom_virial);
    double lammps_virial = 0.0;
    const double expected_lammps6[6] = {
      out.virial[0], out.virial[4], out.virial[8],
      0.5 * (out.virial[1] + out.virial[3]),
      0.5 * (out.virial[2] + out.virial[6]),
      0.5 * (out.virial[5] + out.virial[7])};
    for (int component = 0; component < 6; ++component) {
      lammps_virial = std::max(
          lammps_virial,
          std::abs(lammps.virial[component] - expected_lammps6[component]));
    }
    std::cout << "lammps energy=" << lammps_energy
              << " force=" << lammps_force
              << " mforce=" << lammps_mforce
              << " virial=" << lammps_virial
              << " atom_virial=" << lammps_atom_virial
              << " transfer=" << lammps_transfer << '\n';
    lammps_ok = lammps_energy < 1e-10 && lammps_force < 1e-10 &&
        lammps_mforce < 1e-10 && lammps_virial < 1e-10 &&
        lammps_atom_virial < 1e-10 && lammps_transfer < 1e-10;
    Case replicated = item;
    constexpr int copies = 16;
    replicated.atoms = item.atoms * copies;
    replicated.types.clear();
    replicated.positions.clear();
    replicated.spins.clear();
    for (int copy = 0; copy < copies; ++copy) {
      replicated.types.insert(replicated.types.end(), item.types.begin(), item.types.end());
      replicated.spins.insert(replicated.spins.end(), item.spins.begin(), item.spins.end());
      for (int atom = 0; atom < item.atoms; ++atom) {
        replicated.positions.push_back(item.positions[3 * atom] + 20.0 * copy);
        replicated.positions.push_back(item.positions[3 * atom + 1]);
        replicated.positions.push_back(item.positions[3 * atom + 2]);
      }
    }
    const Evaluation parallel_lammps = evaluate_lammps(model, replicated);
    lammps_ok = lammps_ok &&
        std::abs(parallel_lammps.energy - copies * lammps.energy) < 1e-10;
    for (int copy = 0; copy < copies; ++copy) {
      const auto force_begin = parallel_lammps.forces.begin() +
          static_cast<std::ptrdiff_t>(copy * item.atoms * 3);
      const auto mforce_begin = parallel_lammps.mforces.begin() +
          static_cast<std::ptrdiff_t>(copy * item.atoms * 3);
      lammps_ok = lammps_ok &&
          max_abs(std::vector<double>(force_begin, force_begin + item.atoms * 3),
                  lammps.forces) < 1e-10 &&
          max_abs(std::vector<double>(mforce_begin, mforce_begin + item.atoms * 3),
                  lammps.mforces) < 1e-10;
    }
  }
#endif
  return forward && steady_state && (forward_only || derivative) && lammps_ok;
}

bool finite_difference(NepaModel* model, const Case& reference) {
  constexpr double step = 2e-3;
  const Evaluation base = evaluate(model, reference, false);
  double force_error = 0, mforce_error = 0, strain_error = 0;
  std::size_t worst_mforce = 0;
  double worst_mforce_derivative = 0;
  for (std::size_t k = 0; k < reference.positions.size(); ++k) {
    Case plus = reference, minus = reference;
    plus.positions[k] += step; minus.positions[k] -= step;
    const double derivative = (evaluate(model, plus, false).energy -
        evaluate(model, minus, false).energy) / (2 * step);
    force_error = std::max(force_error, std::abs(derivative + base.forces[k]));
  }
  for (std::size_t k = 0; k < reference.spins.size(); ++k) {
    // The frozen production fixture exposes magnetic-force DOFs for Fe only.
    // Ge remains an allowed spin environment, so its spin can affect energy,
    // but the public mforce contract masks its conjugate derivative to zero.
    if (reference.types[k / 3] != 0) continue;
    Case plus = reference, minus = reference;
    plus.spins[k] += step; minus.spins[k] -= step;
    const double derivative = (evaluate(model, plus, false).energy -
        evaluate(model, minus, false).energy) / (2 * step);
    const double error = std::abs(derivative + base.mforces[k]);
    if (error > mforce_error) {
      mforce_error = error;
      worst_mforce = k;
      worst_mforce_derivative = derivative;
    }
  }
  for (int axis = 0; axis < 3; ++axis) {
    Case plus = reference, minus = reference;
    for (int atom = 0; atom < reference.atoms; ++atom) {
      plus.positions[3 * atom + axis] *= 1 + step;
      minus.positions[3 * atom + axis] *= 1 - step;
    }
    for (int column = 0; column < 3; ++column) {
      plus.cell[3 * axis + column] *= 1 + step;
      minus.cell[3 * axis + column] *= 1 - step;
    }
    const double derivative = (evaluate(model, plus, false).energy -
        evaluate(model, minus, false).energy) / (2 * step);
    strain_error = std::max(strain_error,
                            std::abs(derivative + base.virial[4 * axis]));
  }
  std::cout << "finite_difference force=" << force_error
            << " mforce=" << mforce_error
            << " mforce_index=" << worst_mforce
            << " fd_deds=" << worst_mforce_derivative
            << " analytic_mforce=" << base.mforces[worst_mforce]
            << " strain=" << strain_error << '\n';
  double force_scale = 0.0;
  for (double value : base.forces) {
    force_scale = std::max(force_scale, std::abs(value));
  }
  const double force_tolerance = 3e-3 + 5e-5 * force_scale;
  return force_error < force_tolerance &&
      mforce_error < 3e-3 && strain_error < 5e-3;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: test_cuda_spin2_oc_oracle MODEL ORACLE [--forward-only]\n";
    return EXIT_FAILURE;
  }
  const bool forward_only = argc == 4 && std::string(argv[3]) == "--forward-only";
  #if defined(NEP_ADAPTERS_TEST_CPU_ENGINE)
  if (!nep_adapters::register_cpu_engine()) return EXIT_FAILURE;
  const char* engine = "cpu";
  #else
  if (!nep_adapters::register_cuda_engine()) return EXIT_FAILURE;
  const char* engine = "cuda";
  #endif
  NepaModel* model = nullptr;
  if (nepa_load_model(engine, argv[1], &model) != NEPA_STATUS_OK) {
    std::cerr << nepa_last_error_message() << '\n'; return EXIT_FAILURE;
  }
  bool ok = true;
  try {
    const auto cases = read_oracle(argv[2]);
    for (const Case& item : cases) {
      ok = check_case(model, item, forward_only) && ok;
      if (!forward_only && item.name == "noncollinear")
        ok = finite_difference(model, item) && ok;
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; ok = false;
  }
  nepa_free_model(model);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
