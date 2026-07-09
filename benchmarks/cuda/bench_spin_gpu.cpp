#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_opt.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string engine = "cuda";
  std::string mode = "batch";
  std::string model = "spin";
  int replicate = 6;
  int warmup = 2;
  int iterations = 10;
};

int parse_positive(const char* text, const char* name) {
  const int value = std::atoi(text);
  if (value <= 0) {
    std::cerr << "Invalid " << name << ": " << text << "\n";
    std::exit(EXIT_FAILURE);
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--engine" && i + 1 < argc) {
      options.engine = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      options.mode = argv[++i];
    } else if (arg == "--model" && i + 1 < argc) {
      options.model = argv[++i];
    } else if (arg == "--replicate" && i + 1 < argc) {
      options.replicate = parse_positive(argv[++i], "--replicate");
    } else if (arg == "--warmup" && i + 1 < argc) {
      options.warmup = parse_positive(argv[++i], "--warmup");
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = parse_positive(argv[++i], "--iterations");
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--engine cpu_opt|cuda] [--mode batch|lammps]"
                << " [--model spin|struct]"
                << " [--replicate N] [--warmup N] [--iterations N]\n";
      std::exit(EXIT_FAILURE);
    }
  }
  if ((options.engine != "cpu_opt" && options.engine != "cuda") ||
      (options.mode != "batch" && options.mode != "lammps") ||
      (options.model != "spin" && options.model != "struct")) {
    std::cerr << "Invalid --engine, --mode, or --model\n";
    std::exit(EXIT_FAILURE);
  }
  return options;
}

std::vector<std::string> split(const std::string& line) {
  std::istringstream stream(line);
  std::vector<std::string> tokens;
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

int as_int(const std::string& value) {
  return std::stoi(value);
}

std::string write_struct_model_from_spin() {
  std::ifstream input(NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE);
  if (!input) {
    std::cerr << "failed to open spin fixture\n";
    std::exit(EXIT_FAILURE);
  }
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input, line)) {
    if (!split(line).empty()) {
      lines.push_back(line);
    }
  }
  if (lines.empty()) {
    std::cerr << "empty spin fixture\n";
    std::exit(EXIT_FAILURE);
  }

  const std::vector<std::string> first = split(lines[0]);
  std::size_t cursor = 1;
  int spin_compress = 0;
  int spin_basis_size = 0;
  while (cursor < lines.size()) {
    const std::vector<std::string> tokens = split(lines[cursor]);
    if (tokens[0] == "spin_basis_size") {
      spin_basis_size = as_int(tokens[1]);
    } else if (tokens[0] == "spin_compress") {
      spin_compress = as_int(tokens[1]);
    } else if (tokens[0] == "cutoff") {
      break;
    }
    ++cursor;
  }
  const std::string cutoff = lines[cursor++];
  const std::vector<std::string> n_max_tokens = split(lines[cursor]);
  const std::string n_max = lines[cursor++];
  const std::vector<std::string> basis_tokens = split(lines[cursor]);
  const std::string basis_size = lines[cursor++];
  const std::vector<std::string> l_max_tokens = split(lines[cursor]);
  const std::string l_max = lines[cursor++];
  const std::vector<std::string> ann_tokens = split(lines[cursor]);
  const std::string ann = lines[cursor++];

  const int num_types = as_int(first[1]);
  const int n_max_radial = as_int(n_max_tokens[1]);
  const int n_max_angular = as_int(n_max_tokens[2]);
  const int basis_radial = as_int(basis_tokens[1]);
  const int basis_angular = as_int(basis_tokens[2]);
  const int l_max_3body = as_int(l_max_tokens[1]);
  int channels = l_max_3body;
  for (std::size_t i = 2; i < l_max_tokens.size(); ++i) {
    channels += as_int(l_max_tokens[i]) != 0 ? 1 : 0;
  }
  const int hidden = as_int(ann_tokens[1]);
  const int struct_dim = (n_max_radial + 1) + (n_max_angular + 1) * channels;
  const int spin_dim = 2 + 4 * spin_compress + spin_compress +
                       3 * spin_compress + 3 * spin_compress +
                       spin_compress + spin_compress + spin_compress +
                       std::min(2, spin_compress) + 2 * spin_compress;
  const int spin_model_dim = struct_dim + spin_dim;
  const int ordinary_coeff_count =
      num_types * num_types *
      ((n_max_radial + 1) * (basis_radial + 1) +
       (n_max_angular + 1) * (basis_angular + 1));
  const int spin_coeff_count =
      num_types * num_types * spin_compress * (spin_basis_size + 1);

  std::vector<double> scalars;
  for (; cursor < lines.size(); ++cursor) {
    scalars.push_back(std::stod(split(lines[cursor])[0]));
  }
  const int spin_ann_count = (spin_model_dim + 2) * hidden * num_types + 1;
  const int spin_param_count = spin_ann_count + ordinary_coeff_count + spin_coeff_count;
  if (static_cast<int>(scalars.size()) < spin_param_count + spin_model_dim) {
    std::cerr << "spin fixture parameter count is too small\n";
    std::exit(EXIT_FAILURE);
  }

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "nep_adapters_struct_from_spin.nep";
  std::ofstream out(path);
  out << "nep4";
  for (std::size_t i = 1; i < first.size(); ++i) {
    out << ' ' << first[i];
  }
  out << "\n" << cutoff << "\n" << n_max << "\n" << basis_size << "\n"
      << l_max << "\n" << ann << "\n";

  std::size_t offset = 0;
  for (int type = 0; type < num_types; ++type) {
    for (int neuron = 0; neuron < hidden; ++neuron) {
      for (int d = 0; d < struct_dim; ++d) {
        out << scalars[offset + static_cast<std::size_t>(neuron) * spin_model_dim + d]
            << "\n";
      }
    }
    offset += static_cast<std::size_t>(hidden) * spin_model_dim;
    for (int i = 0; i < hidden; ++i) {
      out << scalars[offset++] << "\n";
    }
    for (int i = 0; i < hidden; ++i) {
      out << scalars[offset++] << "\n";
    }
  }
  out << scalars[offset++] << "\n";
  for (int i = 0; i < ordinary_coeff_count; ++i) {
    out << scalars[offset++] << "\n";
  }
  offset += spin_coeff_count;
  for (int i = 0; i < struct_dim; ++i) {
    out << scalars[static_cast<std::size_t>(spin_param_count + i)] << "\n";
  }
  return path.string();
}

struct System {
  int atom_count = 0;
  std::vector<int> types0;
  std::vector<int> types1;
  std::vector<double> positions;
  std::vector<double> spins;
  std::vector<std::array<double, 3>> x;
  std::vector<std::array<double, 4>> s;
  std::vector<double*> x_ptrs;
  std::vector<double*> s_ptrs;
  std::vector<int> ilist;
  std::vector<int> numneigh;
  std::vector<int> neighbors;
  std::vector<int*> firstneigh;
  double box[9] = {};
};

System make_system(int replicate, bool build_lammps_neighbors) {
  const double base_x[12] = {
      0.2, 0.2, 0.2,
      3.7, 0.3, 0.2,
      0.4, 3.6, 0.5,
      1.8, 1.7, 3.5};
  const double base_s[12] = {
      1.0, 0.2, 0.0,
      0.4, -0.3, 0.7,
      -0.2, 0.8, 0.5,
      0.6, 0.1, -0.4};
  constexpr double spacing = 5.2;
  System system;
  system.atom_count = 4 * replicate * replicate * replicate;
  system.types0.assign(static_cast<std::size_t>(system.atom_count), 0);
  system.types1.assign(static_cast<std::size_t>(system.atom_count), 1);
  system.positions.resize(static_cast<std::size_t>(system.atom_count) * 3);
  system.spins.resize(static_cast<std::size_t>(system.atom_count) * 3);
  if (build_lammps_neighbors) {
    system.x.resize(static_cast<std::size_t>(system.atom_count));
    system.s.resize(static_cast<std::size_t>(system.atom_count));
  }
  int atom = 0;
  for (int ix = 0; ix < replicate; ++ix) {
    for (int iy = 0; iy < replicate; ++iy) {
      for (int iz = 0; iz < replicate; ++iz) {
        for (int b = 0; b < 4; ++b) {
          const double shift[3] = {spacing * ix, spacing * iy, spacing * iz};
          for (int d = 0; d < 3; ++d) {
            const double pos = base_x[3 * b + d] + shift[d] + 2.0;
            const double spin = base_s[3 * b + d];
            system.positions[3 * static_cast<std::size_t>(atom) + d] = pos;
            system.spins[3 * static_cast<std::size_t>(atom) + d] = spin;
            if (build_lammps_neighbors) {
              system.x[static_cast<std::size_t>(atom)][d] = pos;
              system.s[static_cast<std::size_t>(atom)][d] = 0.5 * spin;
            }
          }
          if (build_lammps_neighbors) {
            system.s[static_cast<std::size_t>(atom)][3] = 2.0;
          }
          ++atom;
        }
      }
    }
  }
  const double length = spacing * replicate + 8.0;
  system.box[0] = length;
  system.box[4] = length;
  system.box[8] = length;

  if (!build_lammps_neighbors) {
    return system;
  }
  system.x_ptrs.resize(static_cast<std::size_t>(system.atom_count));
  system.s_ptrs.resize(static_cast<std::size_t>(system.atom_count));
  system.ilist.resize(static_cast<std::size_t>(system.atom_count));
  system.numneigh.resize(static_cast<std::size_t>(system.atom_count));
  constexpr int max_neighbors = 128;
  system.neighbors.resize(
      static_cast<std::size_t>(system.atom_count) * max_neighbors);
  system.firstneigh.resize(static_cast<std::size_t>(system.atom_count));
  for (int i = 0; i < system.atom_count; ++i) {
    system.x_ptrs[static_cast<std::size_t>(i)] = system.x[static_cast<std::size_t>(i)].data();
    system.s_ptrs[static_cast<std::size_t>(i)] = system.s[static_cast<std::size_t>(i)].data();
    system.ilist[static_cast<std::size_t>(i)] = i;
    system.firstneigh[static_cast<std::size_t>(i)] =
        system.neighbors.data() + static_cast<std::size_t>(i) * max_neighbors;
  }
  constexpr double cutoff2 = 36.0;
  for (int ix = 0; ix < replicate; ++ix) {
    for (int iy = 0; iy < replicate; ++iy) {
      for (int iz = 0; iz < replicate; ++iz) {
        for (int b = 0; b < 4; ++b) {
          const int i = (((ix * replicate + iy) * replicate + iz) * 4 + b);
          int count = 0;
          int* neigh = system.firstneigh[static_cast<std::size_t>(i)];
          for (int dx = -1; dx <= 1; ++dx) {
            const int jx = ix + dx;
            if (jx < 0 || jx >= replicate) {
              continue;
            }
            for (int dy = -1; dy <= 1; ++dy) {
              const int jy = iy + dy;
              if (jy < 0 || jy >= replicate) {
                continue;
              }
              for (int dz = -1; dz <= 1; ++dz) {
                const int jz = iz + dz;
                if (jz < 0 || jz >= replicate) {
                  continue;
                }
                for (int jb = 0; jb < 4; ++jb) {
                  const int j =
                      (((jx * replicate + jy) * replicate + jz) * 4 + jb);
                  if (i == j) {
                    continue;
                  }
                  double r2 = 0.0;
                  for (int d = 0; d < 3; ++d) {
                    const double delta =
                        system.x[static_cast<std::size_t>(j)][d] -
                        system.x[static_cast<std::size_t>(i)][d];
                    r2 += delta * delta;
                  }
                  if (r2 < cutoff2) {
                    if (count >= max_neighbors) {
                      std::cerr << "neighbor benchmark capacity exceeded\n";
                      std::exit(EXIT_FAILURE);
                    }
                    neigh[count++] = j;
                  }
                }
              }
            }
          }
          system.numneigh[static_cast<std::size_t>(i)] = count;
        }
      }
    }
  }
  return system;
}

double elapsed_seconds(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

void run_batch(NepaModel* model, const System& system) {
  const int atom_counts[] = {system.atom_count};
  const int atom_offsets[] = {0};
  const int pbc[] = {0, 0, 0};
  std::vector<double> energy(1, 0.0);
  std::vector<double> potential(static_cast<std::size_t>(system.atom_count), 0.0);
  std::vector<double> force(static_cast<std::size_t>(system.atom_count) * 3, 0.0);
  std::vector<double> mforce(static_cast<std::size_t>(system.atom_count) * 3, 0.0);
  std::vector<double> virial(9, 0.0);
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = system.atom_count;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = system.types0.data();
  batch.positions_aos3 = system.positions.data();
  batch.spins_aos3 = system.spins.data();
  batch.boxes_row_major9 = system.box;
  batch.pbc_flags3 = pbc;
  NepaFindForceResult result{};
  result.energy_per_structure = energy.data();
  result.potential_per_atom = potential.data();
  result.forces_aos3 = force.data();
  result.mforces_aos3 = mforce.data();
  result.virials_row_major9 = virial.data();
  if (nepa_find_force_batch(model, &batch, &result) != NEPA_STATUS_OK) {
    std::cerr << "find_force_batch failed: " << nepa_last_error_message() << "\n";
    std::exit(EXIT_FAILURE);
  }
  const double l1 = std::accumulate(
      force.begin(), force.end(), 0.0,
      [](double sum, double value) { return sum + std::abs(value); });
  if (!std::isfinite(energy[0]) || l1 <= 0.0) {
    std::cerr << "invalid batch benchmark output\n";
    std::exit(EXIT_FAILURE);
  }
}

void run_lammps(NepaModel* model, const System& system) {
  int type_map[] = {-1, 0};
  double total_potential = 0.0;
  double total_virial6[6] = {};
  std::vector<double> potential(static_cast<std::size_t>(system.atom_count), 0.0);
  std::vector<std::array<double, 3>> force(static_cast<std::size_t>(system.atom_count));
  std::vector<std::array<double, 3>> mforce(static_cast<std::size_t>(system.atom_count));
  std::vector<double*> force_ptrs(static_cast<std::size_t>(system.atom_count));
  std::vector<double*> mforce_ptrs(static_cast<std::size_t>(system.atom_count));
  for (int i = 0; i < system.atom_count; ++i) {
    force[static_cast<std::size_t>(i)] = {0.0, 0.0, 0.0};
    mforce[static_cast<std::size_t>(i)] = {0.0, 0.0, 0.0};
    force_ptrs[static_cast<std::size_t>(i)] = force[static_cast<std::size_t>(i)].data();
    mforce_ptrs[static_cast<std::size_t>(i)] = mforce[static_cast<std::size_t>(i)].data();
  }
  NepaLammpsNeighborInput input{};
  input.nlocal = system.atom_count;
  input.inum = system.atom_count;
  input.ilist = const_cast<int*>(system.ilist.data());
  input.numneigh = const_cast<int*>(system.numneigh.data());
  input.firstneigh = const_cast<int**>(system.firstneigh.data());
  input.types = const_cast<int*>(system.types1.data());
  input.type_map = type_map;
  input.positions = const_cast<double**>(system.x_ptrs.data());
  input.spins = const_cast<double**>(system.s_ptrs.data());
  NepaLammpsNeighborResult result{};
  result.total_potential = &total_potential;
  result.total_virial6 = total_virial6;
  result.potential_per_atom = potential.data();
  result.forces = force_ptrs.data();
  result.mforces = mforce_ptrs.data();
  if (nepa_find_force_lammps_neighbors(model, &input, &result) != NEPA_STATUS_OK) {
    std::cerr << "find_force_lammps_neighbors failed: "
              << nepa_last_error_message() << "\n";
    std::exit(EXIT_FAILURE);
  }
  double l1 = 0.0;
  for (const auto& f : force) {
    l1 += std::abs(f[0]) + std::abs(f[1]) + std::abs(f[2]);
  }
  if (!std::isfinite(total_potential) || l1 <= 0.0) {
    std::cerr << "invalid LAMMPS benchmark output\n";
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  nep_adapters::register_cpu_opt_engine();
  nep_adapters::register_cuda_engine();
  const std::string model_path =
      options.model == "spin" ? NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE
                              : write_struct_model_from_spin();
  NepaModel* model = nullptr;
  if (nepa_load_model(options.engine.c_str(), model_path.c_str(), &model) !=
      NEPA_STATUS_OK) {
    std::cerr << "load_model failed: " << nepa_last_error_message() << "\n";
    return EXIT_FAILURE;
  }
  const System system = make_system(options.replicate, options.mode == "lammps");
  const auto run_once = [&]() {
    if (options.mode == "batch") {
      run_batch(model, system);
    } else {
      run_lammps(model, system);
    }
  };
  for (int i = 0; i < options.warmup; ++i) {
    run_once();
  }
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < options.iterations; ++i) {
    run_once();
  }
  const auto end = std::chrono::steady_clock::now();
  nepa_free_model(model);
  const double seconds = elapsed_seconds(begin, end);
  const double atom_steps =
      static_cast<double>(system.atom_count) * options.iterations;
  long long neighbor_count = 0;
  for (int count : system.numneigh) {
    neighbor_count += count;
  }
  const double avg_neighbors =
      system.numneigh.empty()
          ? 0.0
          : static_cast<double>(neighbor_count) / system.atom_count;
  std::cout << "{\"benchmark\":\"spin_chiral_fixture\","
            << "\"engine\":\"" << options.engine << "\","
            << "\"mode\":\"" << options.mode << "\","
            << "\"model\":\"" << options.model << "\","
            << "\"replicate\":" << options.replicate << ','
            << "\"atoms\":" << system.atom_count << ','
            << "\"avg_neighbors\":" << avg_neighbors << ','
            << "\"iterations\":" << options.iterations << ','
            << "\"seconds\":" << seconds << ','
            << "\"evals_per_second\":" << options.iterations / seconds << ','
            << "\"atom_steps_per_second\":" << atom_steps / seconds
            << "}\n";
  return EXIT_SUCCESS;
}
