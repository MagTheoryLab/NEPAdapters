#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_opt.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Case {
  int nlocal = 0;
  int nall = 0;
  long long neighbor_count = 0;
  int max_neighbors = 0;
  std::vector<int> ilist;
  std::vector<int> numneigh;
  std::vector<int> types;
  std::vector<int> type_map = {-1, 0};
  std::vector<std::vector<int>> neigh_storage;
  std::vector<int*> firstneigh;
  std::vector<double> positions;
  std::vector<double> spins;
  std::vector<double*> position_rows;
  std::vector<double*> spin_rows;
  std::vector<double> forces;
  std::vector<double> mforces;
  std::vector<double*> force_rows;
  std::vector<double*> mforce_rows;
};

int parse_int(const char* value, const char* name)
{
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0' || parsed <= 0 || parsed > 1000000) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
  return static_cast<int>(parsed);
}

double parse_double(const char* value, const char* name)
{
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (!end || *end != '\0' || parsed <= 0.0) {
    throw std::runtime_error(std::string("invalid ") + name);
  }
  return parsed;
}

void append_atom(Case& c, const int ix, const int iy, const int iz, const double spacing)
{
  const int atom = c.nall++;
  c.positions.push_back(spacing * ix);
  c.positions.push_back(spacing * iy);
  c.positions.push_back(spacing * iz);
  c.spins.push_back(std::sin(0.17 * atom));
  c.spins.push_back(std::cos(0.11 * atom));
  c.spins.push_back(0.5 + 0.25 * std::sin(0.07 * atom));
  c.spins.push_back(1.0);
}

Case make_case(
  const int nx,
  const int ny,
  const int nz,
  const int ghost_slabs,
  const double spacing,
  const double cutoff)
{
  Case c;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        append_atom(c, ix, iy, iz, spacing);
      }
    }
  }
  c.nlocal = c.nall;
  for (int ix = nx; ix < nx + ghost_slabs; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        append_atom(c, ix, iy, iz, spacing);
      }
    }
  }

  c.ilist.resize(static_cast<std::size_t>(c.nlocal));
  c.numneigh.assign(static_cast<std::size_t>(c.nall), 0);
  c.types.assign(static_cast<std::size_t>(c.nall), 1);
  c.neigh_storage.resize(static_cast<std::size_t>(c.nall));
  c.firstneigh.assign(static_cast<std::size_t>(c.nall), nullptr);
  const double cutoff2 = cutoff * cutoff;
  for (int i = 0; i < c.nlocal; ++i) {
    c.ilist[static_cast<std::size_t>(i)] = i;
    const double* xi = c.positions.data() + static_cast<std::size_t>(3) * i;
    auto& neigh = c.neigh_storage[static_cast<std::size_t>(i)];
    for (int j = 0; j < c.nall; ++j) {
      if (i == j) {
        continue;
      }
      const double* xj = c.positions.data() + static_cast<std::size_t>(3) * j;
      const double dx = xj[0] - xi[0];
      const double dy = xj[1] - xi[1];
      const double dz = xj[2] - xi[2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > 1.0e-24 && r2 < cutoff2) {
        neigh.push_back(j);
      }
    }
    c.numneigh[static_cast<std::size_t>(i)] = static_cast<int>(neigh.size());
    c.firstneigh[static_cast<std::size_t>(i)] = neigh.data();
    c.neighbor_count += static_cast<long long>(neigh.size());
    c.max_neighbors = std::max(c.max_neighbors, static_cast<int>(neigh.size()));
  }

  c.position_rows.resize(static_cast<std::size_t>(c.nall));
  c.spin_rows.resize(static_cast<std::size_t>(c.nall));
  c.forces.assign(static_cast<std::size_t>(c.nall) * 3, 0.0);
  c.mforces.assign(static_cast<std::size_t>(c.nall) * 3, 0.0);
  c.force_rows.resize(static_cast<std::size_t>(c.nall));
  c.mforce_rows.resize(static_cast<std::size_t>(c.nall));
  for (int atom = 0; atom < c.nall; ++atom) {
    c.position_rows[static_cast<std::size_t>(atom)] =
      c.positions.data() + static_cast<std::size_t>(3) * atom;
    c.spin_rows[static_cast<std::size_t>(atom)] =
      c.spins.data() + static_cast<std::size_t>(4) * atom;
    c.force_rows[static_cast<std::size_t>(atom)] =
      c.forces.data() + static_cast<std::size_t>(3) * atom;
    c.mforce_rows[static_cast<std::size_t>(atom)] =
      c.mforces.data() + static_cast<std::size_t>(3) * atom;
  }
  return c;
}

bool run_once(
  NepaModel* model,
  Case& c,
  const bool use_spins,
  double& total_potential,
  double total_virial[6])
{
  std::fill(c.forces.begin(), c.forces.end(), 0.0);
  std::fill(c.mforces.begin(), c.mforces.end(), 0.0);
  total_potential = 0.0;
  std::fill(total_virial, total_virial + 6, 0.0);

  NepaLammpsNeighborInput input{};
  input.nlocal = c.nlocal;
  input.inum = c.nlocal;
  input.ilist = c.ilist.data();
  input.numneigh = c.numneigh.data();
  input.firstneigh = c.firstneigh.data();
  input.types = c.types.data();
  input.type_map = c.type_map.data();
  input.positions = c.position_rows.data();
  input.spins = use_spins ? c.spin_rows.data() : nullptr;

  NepaLammpsNeighborResult result{};
  result.total_potential = &total_potential;
  result.total_virial6 = total_virial;
  result.forces = c.force_rows.data();
  result.mforces = c.mforce_rows.data();
  return nepa_find_force_lammps_neighbors(model, &input, &result) == NEPA_STATUS_OK;
}

}  // namespace

int main(int argc, char** argv)
{
  std::string model_path = NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE;
  const char* model_name = "spin_chiral_fixture";
  bool use_spins = true;
  int nx = 16;
  int ny = 16;
  int nz = 16;
  int ghost_slabs = 8;
  int iterations = 5;
  int warmup = 2;
  double spacing = 2.7;
  double cutoff = 6.0;

  for (int arg = 1; arg < argc; ++arg) {
    auto next = [&](const char* name) -> const char* {
      if (arg + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++arg];
    };
    if (std::strcmp(argv[arg], "--model") == 0) {
      model_path = next("--model");
    } else if (std::strcmp(argv[arg], "--nonmag") == 0) {
      model_path = NEP_ADAPTERS_NONMAG_FIXTURE;
      model_name = "nonmag_fixture";
      use_spins = false;
    } else if (std::strcmp(argv[arg], "--nx") == 0) {
      nx = parse_int(next("--nx"), "--nx");
    } else if (std::strcmp(argv[arg], "--ny") == 0) {
      ny = parse_int(next("--ny"), "--ny");
    } else if (std::strcmp(argv[arg], "--nz") == 0) {
      nz = parse_int(next("--nz"), "--nz");
    } else if (std::strcmp(argv[arg], "--ghost-slabs") == 0) {
      ghost_slabs = parse_int(next("--ghost-slabs"), "--ghost-slabs");
    } else if (std::strcmp(argv[arg], "--spacing") == 0) {
      spacing = parse_double(next("--spacing"), "--spacing");
    } else if (std::strcmp(argv[arg], "--cutoff") == 0) {
      cutoff = parse_double(next("--cutoff"), "--cutoff");
    } else if (std::strcmp(argv[arg], "--iterations") == 0) {
      iterations = parse_int(next("--iterations"), "--iterations");
    } else if (std::strcmp(argv[arg], "--warmup") == 0) {
      warmup = parse_int(next("--warmup"), "--warmup");
    } else {
      throw std::runtime_error(std::string("unknown argument: ") + argv[arg]);
    }
  }

  NepaModel* model = nullptr;
  if (!nep_adapters::register_cpu_opt_engine()) {
    std::cerr << "failed to register cpu_opt engine\n";
    return 1;
  }
  const NepaStatus load_status = nepa_load_model("cpu_opt", model_path.c_str(), &model);
  if (load_status != NEPA_STATUS_OK || !model) {
    std::cerr << "failed to load model: " << model_path << '\n';
    return 1;
  }

  Case c = make_case(nx, ny, nz, ghost_slabs, spacing, cutoff);
  double total_potential = 0.0;
  double total_virial[6] = {};
  for (int i = 0; i < warmup; ++i) {
    if (!run_once(model, c, use_spins, total_potential, total_virial)) {
      std::cerr << "warmup failed\n";
      nepa_free_model(model);
      return 1;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    if (!run_once(model, c, use_spins, total_potential, total_virial)) {
      std::cerr << "benchmark iteration failed\n";
      nepa_free_model(model);
      return 1;
    }
  }
  const auto stop = std::chrono::steady_clock::now();
  const double seconds = std::chrono::duration<double>(stop - start).count();
  const double atom_steps = static_cast<double>(c.nlocal) * iterations;

  std::cout << "{\"benchmark\":\"cpu_opt_spin_lammps\","
            << "\"model\":\"" << model_name << "\","
            << "\"nlocal\":" << c.nlocal << ','
            << "\"nall\":" << c.nall << ','
            << "\"ghost_atoms\":" << (c.nall - c.nlocal) << ','
            << "\"neighbors\":" << c.neighbor_count << ','
            << "\"max_neighbors\":" << c.max_neighbors << ','
            << "\"iterations\":" << iterations << ','
            << "\"warmup\":" << warmup << ','
            << "\"openmp\":" << NEP_ADAPTERS_BENCH_OPENMP_ENABLED << ','
            << "\"seconds\":" << seconds << ','
            << "\"evals_per_second\":" << (iterations / seconds) << ','
            << "\"atom_steps_per_second\":" << (atom_steps / seconds) << ','
            << "\"energy\":" << total_potential
            << "}\n";
  nepa_free_model(model);
  return 0;
}
