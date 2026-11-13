#include "nep_adapters/api.h"
#if defined(NEP_ADAPTERS_TEST_CPU_ENGINE)
#include "nep_adapters/engines/cpu.hpp"
#else
#include "nep_adapters/engines/cuda.hpp"
#include <cuda_runtime.h>
#include "../engines/cuda/device_model.hpp"
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
bool g_spin_type_semantics = true;
std::vector<int> g_active;

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
  if (token != "spin_density_oracle_v1") throw std::runtime_error("bad density oracle");
  input >> token >> hash;
  if (token != "model_sha256" || hash.size() != 64) throw std::runtime_error("missing provenance");
  input >> token >> descriptor_dim;
  if (token != "descriptor_dim" || descriptor_dim < 1) throw std::runtime_error("bad dimension");
  g_descriptor_dim = descriptor_dim;
  input >> token;
  if (token != "active") throw std::runtime_error("missing active mask");
  g_active.resize(3);
  for (auto& v : g_active) input >> v;
  std::vector<Case> result;
  while (input >> token) {
    if (token != "case") throw std::runtime_error("expected case");
    Case item; input >> item.name >> item.atoms >> token;
    if (token != "types") throw std::runtime_error("expected types");
    item.types.resize(item.atoms);
    for (int& type : item.types) {
      std::string symbol; input >> symbol;
      type = symbol == "Fe" ? 0 : symbol == "Bi" ? 1 : symbol == "O" ? 2 : -1;
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

#if !defined(NEP_ADAPTERS_TEST_CPU_ENGINE)
template<class T> struct DeviceBuffer {
  T* data = nullptr;
  explicit DeviceBuffer(const std::vector<T>& host) {
    if (cudaMalloc(reinterpret_cast<void**>(&data), host.size() * sizeof(T)) != cudaSuccess)
      throw std::runtime_error("allocate test device buffer");
    if (cudaMemcpy(data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaFree(data);
      throw std::runtime_error("upload test device buffer");
    }
  }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  void copy(std::vector<T>& host) {
    if (cudaMemcpy(host.data(), data, host.size() * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess)
      throw std::runtime_error("download test device buffer");
  }
};
#endif

Evaluation evaluate_lammps(
    NepaModel* model, const Case& item, bool device = false, int device_nlocal = -1) {
  const int n = item.atoms;
  std::vector<int> ilist(static_cast<std::size_t>(n));
  std::vector<int> numneigh(static_cast<std::size_t>(n), n - 1);
  std::vector<std::vector<int>> neighbors(static_cast<std::size_t>(n));
  std::vector<int*> firstneigh(static_cast<std::size_t>(n));
  std::vector<int> types(static_cast<std::size_t>(n));
  std::vector<int> type_map = {-1, 0, 1, 2};
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
#if !defined(NEP_ADAPTERS_TEST_CPU_ENGINE)
  if (device) {
    std::vector<int> flat;
    for (const auto& row : neighbors) flat.insert(flat.end(), row.begin(), row.end());
    DeviceBuffer<int> di(ilist), dn(numneigh), dl(flat), dt(types), dm(type_map);
    DeviceBuffer<double> dx(item.positions), ds(lammps_spins), de(std::vector<double>(1)),
        dv(out.virial), df(out.forces), dh(out.mforces), da(out.potential), dav(out.atom_virial);
    NepaLammpsDeviceNeighborInput din{};
    din.nlocal = device_nlocal < 0 ? n : device_nlocal;
    din.nall = n; din.inum = din.nlocal; din.max_neighbors = n - 1;
    din.neighbor_rows = n; din.numneigh_length = n;
    din.ilist = di.data; din.numneigh = dn.data; din.neighbors = dl.data;
    din.neighbor_atom_stride = n - 1; din.neighbor_slot_stride = 1;
    din.types = dt.data; din.type_map = dm.data; din.type_map_length = 4;
    din.positions = dx.data; din.position_atom_stride = 3; din.position_component_stride = 1;
    din.spins = ds.data; din.spin_atom_stride = 4; din.spin_component_stride = 1;
    NepaLammpsDeviceNeighborResult dout{};
    dout.total_potential = de.data; dout.total_virial6 = dv.data;
    dout.potential_per_atom = da.data; dout.forces = df.data;
    dout.force_atom_stride = 3; dout.force_component_stride = 1;
    dout.mforces = dh.data; dout.mforce_atom_stride = 3; dout.mforce_component_stride = 1;
    dout.virials_per_atom9 = dav.data; dout.virial_atom_stride = 9; dout.virial_component_stride = 1;
    if (nepa_find_force_lammps_device_neighbors(model, &din, &dout) != NEPA_STATUS_OK)
      throw std::runtime_error(nepa_last_error_message());
    std::vector<double> energy(1); de.copy(energy); out.energy = energy[0];
    da.copy(out.potential); df.copy(out.forces); dh.copy(out.mforces);
    dv.copy(out.virial); dav.copy(out.atom_virial);
    std::vector<double> original_spins(lammps_spins.size()); ds.copy(original_spins);
    if (original_spins != lammps_spins) throw std::runtime_error("caller device spins mutated");
    return out;
  }
#endif
  if (nepa_find_force_lammps_neighbors(model, &input, &result) != NEPA_STATUS_OK) {
    throw std::runtime_error(nepa_last_error_message());
  }
  return out;
}

bool check_case(NepaModel* model, const Case& item, bool) {
  const auto out = evaluate(model, item);
  const auto steady = evaluate(model, item, false, false);
  const double energy = std::abs(out.energy - std::accumulate(item.potential.begin(), item.potential.end(), 0.0));
  const double force = max_abs(out.forces, item.forces), mforce = max_abs(out.mforces, item.mforces);
  const double virial = max_abs(out.virial, item.virial), potential = max_abs(out.potential, item.potential);
  double descriptor = 0;
  if (!item.descriptor.empty())
    for (int a = 0; a < item.atoms; ++a)
      for (int d = 30; d < g_descriptor_dim; ++d)
        descriptor = std::max(descriptor, std::abs(out.descriptor[a*g_descriptor_dim+d] - item.descriptor[a*g_descriptor_dim+d]));
  const double reuse = std::max({max_abs(out.forces, steady.forces), max_abs(out.mforces, steady.mforces),
    max_abs(out.potential, steady.potential)});
  std::vector<double> reduced(9);
  for (int i = 0; i < item.atoms; ++i)
    for (int k = 0; k < 9; ++k) reduced[k] += out.atom_virial[9*i+k];
  const double atom_sum = max_abs(reduced, out.virial);
  std::cout << item.name << " E=" << energy << " F=" << force << " M=" << mforce
    << " V=" << virial << " q=" << descriptor << " reuse=" << reuse << " atom_sum=" << atom_sum << '\n';
  return energy < 3e-5 && potential < 2e-5 && force < 1e-4 && mforce < 1e-4 &&
    virial < 5e-4 && descriptor < 1e-4 && reuse < 2e-6 && atom_sum < 2e-5;
}

bool check_spin_type_lammps(NepaModel* model, const Case& item) {
  const auto batch = evaluate(model, item);
  const int order[9] = {0, 4, 8, 1, 2, 5, 3, 6, 7};
#if defined(NEP_ADAPTERS_TEST_CPU_ENGINE)
  const std::vector<bool> paths{false};
#else
  const std::vector<bool> paths{false, true};
#endif
  for (bool device : paths) {
    const auto out = evaluate_lammps(model, item, device);
    if (std::abs(out.energy - batch.energy) > 5e-5 ||
        max_abs(out.forces, batch.forces) > 5e-5 ||
        max_abs(out.mforces, batch.mforces) > 5e-5) return false;
    for (int atom = 0; atom < item.atoms; ++atom)
      for (int k = 0; k < 9; ++k)
        if (std::abs(out.atom_virial[9 * atom + k] - batch.atom_virial[9 * atom + order[k]]) > 5e-5)
          return false;
    for (int k = 0; k < 6; ++k) {
      const double expected = k < 3 ? batch.virial[order[k]] :
          0.5 * (batch.virial[order[k]] + batch.virial[order[k + 3]]);
      if (std::abs(out.virial[k] - expected) > 5e-5) return false;
    }
    Case raw = item;
    for (int atom = 0; atom < raw.atoms; ++atom) {
      if (g_active[raw.types[atom]]) continue;
      raw.spins[3 * atom] = 0.125;
      raw.spins[3 * atom + 1] = -0.75;
      raw.spins[3 * atom + 2] = 1.5;
    }
    const auto masked = evaluate_lammps(model, raw, device); // Reused device workspace.
    if (std::abs(masked.energy - out.energy) > 5e-5 ||
        max_abs(masked.forces, out.forces) > 5e-5 ||
        max_abs(masked.mforces, out.mforces) > 5e-5 ||
        max_abs(masked.atom_virial, out.atom_virial) > 5e-5) return false;
#if !defined(NEP_ADAPTERS_TEST_CPU_ENGINE)
    if (device) {
      // Keep one local center and return the remaining atoms as ghosts. Both
      // zero and nonzero caller spins on excluded species must be masked before
      // LAMMPS can reverse-communicate their magnetic-force contributions.
      for (const Case* ghost_case : {&item, static_cast<const Case*>(&raw)}) {
        const auto ghosts = evaluate_lammps(model, *ghost_case, true, 1);
        bool saw_excluded_ghost = false;
        for (int atom = 1; atom < item.atoms; ++atom) {
          if (g_active[item.types[atom]]) continue;
          saw_excluded_ghost = true;
          for (int k = 0; k < 3; ++k) {
            if (ghosts.mforces[3 * atom + k] != 0.0) {
              std::cerr << "nonzero excluded ghost magnetic force\n";
              return false;
            }
          }
        }
        if (!saw_excluded_ghost) return false;
      }
    }
#endif
    std::cout << "spin_type LAMMPS " << (device ? "device" : "host")
              << " values and effective-spin masking passed\n";
  }
  return true;
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
    // The SOC fixtures select Fe; excluded Bi/O input spins are masked inside
    // the backend and are not independent magnetic-force degrees of freedom.
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
    std::cerr << "usage: test_spin_density_oracle MODEL ORACLE [--forward-only]\n";
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
    #if !defined(NEP_ADAPTERS_TEST_CPU_ENGINE)
    {
      const auto host = nep_adapters::cuda_backend::load_host_model_parameters(argv[1]);
      const nep_adapters::cuda_backend::DeviceModel device(host);
      const int* pointer = device.view().spin_reverse_index;
      if (!host.protocol.spin_soc && host.protocol.spin_l_max == 4 &&
          host.protocol.spin_order == 4) {
        const auto layout = nep_adapters::cuda_backend::make_spin_density_layout(host.protocol);
        const int geometry = (layout.lmax + 1) * (layout.lmax + 1);
        const int groups = geometry + layout.spin_components;
        std::vector<int> index(groups + 1 + layout.terms.size() + 26 + device.view().spin_layout.harmonic_count), seen(layout.terms.size(), 0);
        if (!pointer || cudaMemcpy(index.data(), pointer, index.size()*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
          throw std::runtime_error("missing cooperative reverse index");
        if (index[0] != 0 || index[groups] != static_cast<int>(layout.terms.size()))
          throw std::runtime_error("incomplete cooperative reverse index");
        for (int group = 0; group < groups; ++group) {
          if (index[group] > index[group+1]) throw std::runtime_error("invalid reverse index offsets");
          int previous = -1;
          for (int k = index[group]; k < index[group+1]; ++k) {
            const int t = index[groups+1+k];
            if (t <= previous || t >= static_cast<int>(layout.terms.size()))
              throw std::runtime_error("unstable reverse term partition");
            const auto& term = layout.terms[t];
            if ((term.neighbor ? geometry + term.spin : term.geometry) != group || ++seen[t] != 1)
              throw std::runtime_error("invalid reverse term ownership");
            previous = t;
          }
        }
        if (std::find(seen.begin(), seen.end(), 0) != seen.end())
          throw std::runtime_error("missing reverse term");
        const int* offsets = index.data() + groups + 1 + layout.terms.size();
        if (offsets[0] != 0 || offsets[25] != device.view().spin_layout.harmonic_count)
          throw std::runtime_error("incomplete component harmonics");
        for (int g = 0; g < 25; ++g) {
          int previous = -1;
          for (int k = offsets[g]; k < offsets[g+1]; ++k) {
            const int h = offsets[26+k];
            if (h <= previous || h >= device.view().spin_layout.harmonic_count ||
                nep_adapters::spin_density::harmonics[h].component != g)
              throw std::runtime_error("invalid harmonic component partition");
            previous = h;
          }
        }
      } else if (pointer) throw std::runtime_error("unexpected reverse index for serial layout");
    }
    #endif

    for (const Case& item : cases) {
      ok = check_case(model, item, forward_only) && ok;
      if (!forward_only && item.name == "large") {
        ok = finite_difference(model, item) && ok;
        if (g_spin_type_semantics) ok = check_spin_type_lammps(model, item) && ok;
      }
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; ok = false;
  }
  nepa_free_model(model);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
