#include "nep_adapters/api.h"
#include "nep_adapters/engines/cuda.hpp"

#include "device_operations.hpp"
#include "device_model.hpp"
#include "device_workspace.hpp"
#include "simulation_box.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace {

struct Options {
  int atoms = 1000000;
  int cubic_cells = 0;
  int radial_capacity = 256;
  int angular_capacity = 256;
  int warmup = 2;
  int iterations = 8;
  std::string mode = "all";
  std::string layout = "both";
  bool use_type_map = false;
  bool cycle_model_types = false;
  bool write_totals = false;
  bool write_per_atom = false;
  bool write_spin_transfer = false;
  bool breakdown = false;
  bool split_pipeline = false;
  bool strip_spin_model = false;
  std::string model_path;
  std::string replay_path;
  double spacing = 2.0;
  double skin = 0.0;
  double radial_cutoff = 6.0;
  double angular_cutoff = 5.0;
};

struct ReusedWorkspaceTiming {
  double total_ms = 0.0;
  double stage_ms = 0.0;
  double clear_ms = 0.0;
  double radial_cache_ms = 0.0;
  double descriptor_ms = 0.0;
  double ann_ms = 0.0;
  double force_ms = 0.0;
  double output_ms = 0.0;
};

int parse_positive_int(const char* text, const char* name) {
  const int value = std::atoi(text);
  if (value <= 0) {
    std::cerr << "Invalid " << name << ": " << text << "\n";
    std::exit(EXIT_FAILURE);
  }
  return value;
}

double parse_positive_double(const char* text, const char* name) {
  const double value = std::atof(text);
  if (value <= 0.0) {
    std::cerr << "Invalid " << name << ": " << text << "\n";
    std::exit(EXIT_FAILURE);
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--atoms" && i + 1 < argc) {
      options.atoms = parse_positive_int(argv[++i], "--atoms");
    } else if (arg == "--cubic" && i + 1 < argc) {
      options.cubic_cells = parse_positive_int(argv[++i], "--cubic");
      const long long atom_count =
          static_cast<long long>(options.cubic_cells) *
          options.cubic_cells *
          options.cubic_cells;
      if (atom_count > std::numeric_limits<int>::max()) {
        std::cerr << "Invalid --cubic: too many atoms\n";
        std::exit(EXIT_FAILURE);
      }
      options.atoms = static_cast<int>(atom_count);
    } else if (arg == "--mn-radial" && i + 1 < argc) {
      options.radial_capacity = parse_positive_int(argv[++i], "--mn-radial");
    } else if (arg == "--mn-angular" && i + 1 < argc) {
      options.angular_capacity = parse_positive_int(argv[++i], "--mn-angular");
    } else if (arg == "--warmup" && i + 1 < argc) {
      options.warmup = parse_positive_int(argv[++i], "--warmup");
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = parse_positive_int(argv[++i], "--iterations");
    } else if (arg == "--mode" && i + 1 < argc) {
      options.mode = argv[++i];
      if (options.mode != "all" && options.mode != "api" &&
          options.mode != "reused") {
        std::cerr << "Invalid --mode: " << options.mode << "\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (arg == "--layout" && i + 1 < argc) {
      options.layout = argv[++i];
      if (options.layout != "legacy" && options.layout != "nolegacy" &&
          options.layout != "both") {
        std::cerr << "Invalid --layout: " << options.layout << "\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (arg == "--type-map") {
      options.use_type_map = true;
    } else if (arg == "--cycle-model-types") {
      options.cycle_model_types = true;
    } else if (arg == "--model" && i + 1 < argc) {
      options.model_path = argv[++i];
    } else if (arg == "--replay" && i + 1 < argc) {
      options.replay_path = argv[++i];
    } else if (arg == "--spacing" && i + 1 < argc) {
      options.spacing = parse_positive_double(argv[++i], "--spacing");
    } else if (arg == "--skin" && i + 1 < argc) {
      options.skin = std::atof(argv[++i]);
      if (options.skin < 0.0) {
        std::cerr << "Invalid --skin: " << argv[i] << "\n";
        std::exit(EXIT_FAILURE);
      }
    } else if (arg == "--cutoff" && i + 1 < argc) {
      options.radial_cutoff = parse_positive_double(argv[++i], "--cutoff");
    } else if (arg == "--radial-cutoff" && i + 1 < argc) {
      options.radial_cutoff =
          parse_positive_double(argv[++i], "--radial-cutoff");
    } else if (arg == "--angular-cutoff" && i + 1 < argc) {
      options.angular_cutoff =
          parse_positive_double(argv[++i], "--angular-cutoff");
    } else if (arg == "--totals") {
      options.write_totals = true;
    } else if (arg == "--per-atom") {
      options.write_per_atom = true;
    } else if (arg == "--spin-transfer") {
      options.write_spin_transfer = true;
    } else if (arg == "--breakdown") {
      options.breakdown = true;
    } else if (arg == "--split-pipeline") {
      options.split_pipeline = true;
    } else if (arg == "--strip-spin-model") {
      options.strip_spin_model = true;
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--atoms N] [--warmup N] [--iterations N]"
                << " [--cubic N] [--mode all|api|reused]"
                << " [--layout legacy|nolegacy|both] [--type-map]"
                << " [--cycle-model-types]"
                << " [--model PATH] [--replay PATH]"
                << " [--mn-radial N] [--mn-angular N] [--spacing X]"
                << " [--skin X]"
                << " [--radial-cutoff X] [--angular-cutoff X]"
                << " [--totals] [--per-atom] [--spin-transfer] [--breakdown]"
                << " [--strip-spin-model]"
                << " [--split-pipeline]\n";
      std::exit(EXIT_FAILURE);
    }
  }
  if (options.strip_spin_model && options.model_path.empty()) {
    std::cerr << "--strip-spin-model requires --model PATH\n";
    std::exit(EXIT_FAILURE);
  }
  if (options.use_type_map && options.cycle_model_types) {
    std::cerr << "--type-map and --cycle-model-types are mutually exclusive\n";
    std::exit(EXIT_FAILURE);
  }
  return options;
}

std::vector<std::string> split_tokens(const std::string& line) {
  std::istringstream stream(line);
  std::vector<std::string> tokens;
  std::string token;
  while (stream >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

int parse_model_int(const std::string& value) {
  return std::stoi(value);
}

std::string write_struct_model_from_spin(const std::string& spin_model_path) {
  std::ifstream input(spin_model_path);
  if (!input) {
    throw std::runtime_error("failed to open spin model: " + spin_model_path);
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input, line)) {
    if (!split_tokens(line).empty()) {
      lines.push_back(line);
    }
  }
  if (lines.empty()) {
    throw std::runtime_error("empty spin model: " + spin_model_path);
  }

  const std::vector<std::string> first = split_tokens(lines[0]);
  std::size_t cursor = 1;
  int spin_compress = 0;
  int spin_basis_size = 0;
  while (cursor < lines.size()) {
    const std::vector<std::string> tokens = split_tokens(lines[cursor]);
    if (tokens[0] == "spin_basis_size") {
      spin_basis_size = parse_model_int(tokens[1]);
    } else if (tokens[0] == "spin_compress") {
      spin_compress = parse_model_int(tokens[1]);
    } else if (tokens[0] == "cutoff") {
      break;
    }
    ++cursor;
  }
  if (cursor >= lines.size()) {
    throw std::runtime_error("spin model is missing structural cutoff block");
  }

  const std::string cutoff = lines[cursor++];
  const std::vector<std::string> n_max_tokens = split_tokens(lines[cursor]);
  const std::string n_max = lines[cursor++];
  const std::vector<std::string> basis_tokens = split_tokens(lines[cursor]);
  const std::string basis_size = lines[cursor++];
  const std::vector<std::string> l_max_tokens = split_tokens(lines[cursor]);
  const std::string l_max = lines[cursor++];
  const std::vector<std::string> ann_tokens = split_tokens(lines[cursor]);
  const std::string ann = lines[cursor++];

  const int num_types = parse_model_int(first[1]);
  const int n_max_radial = parse_model_int(n_max_tokens[1]);
  const int n_max_angular = parse_model_int(n_max_tokens[2]);
  const int basis_radial = parse_model_int(basis_tokens[1]);
  const int basis_angular = parse_model_int(basis_tokens[2]);
  const int l_max_3body = parse_model_int(l_max_tokens[1]);
  int channels = l_max_3body;
  for (std::size_t i = 2; i < l_max_tokens.size(); ++i) {
    channels += parse_model_int(l_max_tokens[i]) != 0 ? 1 : 0;
  }
  const int hidden = parse_model_int(ann_tokens[1]);
  const int struct_dim = (n_max_radial + 1) + (n_max_angular + 1) * channels;
  const int spin_dim =
      2 + 4 * spin_compress + spin_compress + 3 * spin_compress +
      3 * spin_compress + spin_compress + spin_compress + spin_compress +
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
    scalars.push_back(std::stod(split_tokens(lines[cursor])[0]));
  }
  const int spin_ann_count = (spin_model_dim + 2) * hidden * num_types + 1;
  const int spin_param_count =
      spin_ann_count + ordinary_coeff_count + spin_coeff_count;
  if (static_cast<int>(scalars.size()) < spin_param_count + spin_model_dim) {
    throw std::runtime_error("spin model parameter count is too small");
  }

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      "nep_adapters_lammps_device_struct_from_spin.nep";
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
        out << scalars[offset + static_cast<std::size_t>(neuron) *
                                  spin_model_dim + d]
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

void check_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(action) + ": " + cudaGetErrorString(status));
  }
}

template <typename Fn>
double time_synchronized_phase(Fn&& fn) {
  const auto begin = std::chrono::steady_clock::now();
  fn();
  check_cuda(cudaDeviceSynchronize(), "sync bench phase");
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

template <typename T>
T* copy_to_device(const std::vector<T>& host, const char* action) {
  T* device = nullptr;
  check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(T)),
      action);
  check_cuda(
      cudaMemcpy(
          device,
          host.data(),
          host.size() * sizeof(T),
          cudaMemcpyHostToDevice),
      action);
  return device;
}

std::string write_radial_model(const Options& options) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      "nep_adapters_cuda_lammps_device_layout.nep";
  std::ofstream out(path);
  out << "nep4 1 C\n"
      << "cutoff " << options.radial_cutoff << " "
      << options.angular_cutoff << " "
      << options.radial_capacity << " "
      << options.angular_capacity << "\n"
      << "n_max 1 0\n"
      << "basis_size 2 0\n"
      << "l_max 0 0 0\n"
      << "ANN 2 0\n";
  const double values[] = {
      0.20, -0.10, 0.05, 0.15,
      0.01, -0.02,
      0.30, -0.25,
      0.04,
      0.70, -0.15, 0.05, -0.30, 0.20, -0.10, 0.0,
      0.80, 1.10,
  };
  for (double value : values) {
    out << value << "\n";
  }
  return path.string();
}

struct LayoutStorage {
  int atom_count = 0;
  int nall = 0;
  int inum = 0;
  int pitch = 0;
  int max_neighbors = 2;
  int neighbor_rows = 0;
  int numneigh_length = 0;
  std::size_t total_neighbors = 0;
  int neighbor_atom_stride = 0;
  int neighbor_slot_stride = 0;
  int position_atom_stride = 0;
  int position_component_stride = 0;
  int force_atom_stride = 0;
  int force_component_stride = 0;
  int spin_atom_stride = 0;
  int spin_component_stride = 0;
  int mforce_atom_stride = 0;
  int mforce_component_stride = 0;
  int virial_atom_stride = 0;
  int virial_component_stride = 0;

  int* ilist = nullptr;
  int* numneigh = nullptr;
  int* neighbors = nullptr;
  int* types = nullptr;
  int* type_map = nullptr;
  int type_map_length = 0;
  double* positions = nullptr;
  double* spins = nullptr;
  double* total_potential = nullptr;
  double* total_virial6 = nullptr;
  double* potential_per_atom = nullptr;
  double* forces = nullptr;
  double* mforces = nullptr;
  double* virials = nullptr;
  double* spin_transfer = nullptr;
};

struct ReplayHeader {
  char magic[8];
  std::int32_t version;
  std::int32_t nlocal;
  std::int32_t nall;
  std::int32_t inum;
  std::int32_t max_neighbors;
  std::int32_t neighbor_rows;
  std::int32_t numneigh_length;
  std::int32_t neighbor_atom_stride;
  std::int32_t neighbor_slot_stride;
  std::int32_t type_map_length;
  std::int32_t position_atom_stride;
  std::int32_t position_component_stride;
  std::int32_t force_atom_stride;
  std::int32_t force_component_stride;
  std::int64_t ilist_count;
  std::int64_t numneigh_count;
  std::int64_t neighbor_count;
  std::int64_t type_count;
  std::int64_t type_map_count;
  std::int64_t position_count;
  std::int64_t force_count;
};

template <typename T>
void read_binary(std::ifstream& in, T* data, std::size_t count, const char* label) {
  if (count == 0) {
    return;
  }
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) {
    throw std::runtime_error(std::string("short replay read: ") + label);
  }
}

struct FixedNeighborSystem {
  int atom_count = 0;
  int max_neighbors = 0;
  std::size_t total_neighbors = 0;
  std::vector<int> numneigh;
  std::vector<int> row_major_neighbors;
  std::vector<double> positions_aos3;
};

std::vector<int> make_ilist(int atom_count) {
  std::vector<int> ilist(static_cast<std::size_t>(atom_count));
  for (int atom = 0; atom < atom_count; ++atom) {
    ilist[static_cast<std::size_t>(atom)] = atom;
  }
  return ilist;
}

FixedNeighborSystem make_chain_system(int atom_count) {
  FixedNeighborSystem system;
  system.atom_count = atom_count;
  system.max_neighbors = atom_count == 1 ? 0 : 2;
  system.numneigh.assign(static_cast<std::size_t>(atom_count), 2);
  if (atom_count == 1) {
    system.numneigh[0] = 0;
  } else {
    system.numneigh[0] = 1;
    system.numneigh[static_cast<std::size_t>(atom_count - 1)] = 1;
  }
  system.row_major_neighbors.assign(
      static_cast<std::size_t>(atom_count) * system.max_neighbors,
      0);
  system.positions_aos3.assign(3 * static_cast<std::size_t>(atom_count), 0.0);
  for (int atom = 0; atom < atom_count; ++atom) {
    int slot = 0;
    if (atom > 0) {
      system.row_major_neighbors[
          static_cast<std::size_t>(atom) * system.max_neighbors + slot++] =
          atom - 1;
    }
    if (atom + 1 < atom_count) {
      system.row_major_neighbors[
          static_cast<std::size_t>(atom) * system.max_neighbors + slot] =
          atom + 1;
    }
    system.total_neighbors += static_cast<std::size_t>(system.numneigh[atom]);
    system.positions_aos3[3 * static_cast<std::size_t>(atom)] = 1.5 * atom;
    system.positions_aos3[3 * static_cast<std::size_t>(atom) + 1] =
        0.1 * (atom % 7);
    system.positions_aos3[3 * static_cast<std::size_t>(atom) + 2] =
        0.1 * (atom % 11);
  }
  return system;
}

FixedNeighborSystem make_cubic_system(
    int cells,
    double spacing,
    double cutoff) {
  FixedNeighborSystem system;
  system.atom_count = cells * cells * cells;
  system.positions_aos3.assign(
      3 * static_cast<std::size_t>(system.atom_count),
      0.0);
  for (int z = 0; z < cells; ++z) {
    for (int y = 0; y < cells; ++y) {
      for (int x = 0; x < cells; ++x) {
        const int atom = x + cells * (y + cells * z);
        system.positions_aos3[3 * static_cast<std::size_t>(atom)] =
            spacing * x;
        system.positions_aos3[3 * static_cast<std::size_t>(atom) + 1] =
            spacing * y;
        system.positions_aos3[3 * static_cast<std::size_t>(atom) + 2] =
            spacing * z;
      }
    }
  }

  struct NeighborOffset {
    int dx;
    int dy;
    int dz;
  };
  std::vector<NeighborOffset> offsets;
  const int radius = static_cast<int>(std::ceil(cutoff / spacing));
  const double cutoff2 = cutoff * cutoff;
  for (int dz = -radius; dz <= radius; ++dz) {
    for (int dy = -radius; dy <= radius; ++dy) {
      for (int dx = -radius; dx <= radius; ++dx) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        const double r2 =
            spacing * spacing * static_cast<double>(dx * dx + dy * dy + dz * dz);
        if (r2 < cutoff2) {
          offsets.push_back({dx, dy, dz});
        }
      }
    }
  }

  system.max_neighbors = static_cast<int>(offsets.size());
  system.numneigh.assign(static_cast<std::size_t>(system.atom_count), 0);
  system.row_major_neighbors.assign(
      static_cast<std::size_t>(system.atom_count) * system.max_neighbors,
      0);
  for (int z = 0; z < cells; ++z) {
    for (int y = 0; y < cells; ++y) {
      for (int x = 0; x < cells; ++x) {
        const int atom = x + cells * (y + cells * z);
        int slot = 0;
        for (const NeighborOffset& offset : offsets) {
          const int nx = x + offset.dx;
          const int ny = y + offset.dy;
          const int nz = z + offset.dz;
          if (nx < 0 || nx >= cells || ny < 0 || ny >= cells ||
              nz < 0 || nz >= cells) {
            continue;
          }
          system.row_major_neighbors[
              static_cast<std::size_t>(atom) * system.max_neighbors + slot++] =
              nx + cells * (ny + cells * nz);
        }
        system.numneigh[static_cast<std::size_t>(atom)] = slot;
        system.total_neighbors += static_cast<std::size_t>(slot);
      }
    }
  }
  return system;
}

std::vector<int> make_types(
    int atom_count,
    bool use_type_map,
    bool cycle_model_types,
    int model_type_count) {
  std::vector<int> types(
      static_cast<std::size_t>(atom_count),
      use_type_map ? 1 : 0);
  if (cycle_model_types) {
    for (int atom = 0; atom < atom_count; ++atom) {
      types[static_cast<std::size_t>(atom)] = atom % model_type_count;
    }
  }
  return types;
}

LayoutStorage make_layout(
    const FixedNeighborSystem& system,
    bool soa_layout,
    bool use_type_map,
    bool cycle_model_types,
    int model_type_count,
    bool spin_model) {
  LayoutStorage storage;
  storage.atom_count = system.atom_count;
  storage.nall = system.atom_count;
  storage.inum = system.atom_count;
  storage.max_neighbors = system.max_neighbors;
  storage.neighbor_rows = system.atom_count;
  storage.numneigh_length = system.atom_count;
  storage.total_neighbors = system.total_neighbors;
  storage.pitch = soa_layout ? system.atom_count + 32 : system.atom_count;

  std::vector<int> ilist = make_ilist(system.atom_count);
  std::vector<int> types = make_types(
      system.atom_count,
      use_type_map,
      cycle_model_types,
      model_type_count);
  storage.ilist = copy_to_device(ilist, "copy ilist");
  storage.numneigh = copy_to_device(system.numneigh, "copy numneigh");
  storage.types = copy_to_device(types, "copy types");
  if (use_type_map) {
    const std::vector<int> type_map = {-1, 0};
    storage.type_map = copy_to_device(type_map, "copy type map");
    storage.type_map_length = static_cast<int>(type_map.size());
  }

  if (soa_layout) {
    storage.neighbor_atom_stride = 1;
    storage.neighbor_slot_stride = storage.pitch;
    std::vector<int> neighbors(
        static_cast<std::size_t>(storage.max_neighbors) *
            static_cast<std::size_t>(storage.pitch),
        0);
    for (int atom = 0; atom < system.atom_count; ++atom) {
      const int count = system.numneigh[static_cast<std::size_t>(atom)];
      for (int slot = 0; slot < count; ++slot) {
        neighbors[static_cast<std::size_t>(atom) +
                  static_cast<std::size_t>(slot) * storage.pitch] =
            system.row_major_neighbors[
                static_cast<std::size_t>(atom) * storage.max_neighbors + slot];
      }
    }
    storage.neighbors = copy_to_device(neighbors, "copy soa neighbors");

    storage.position_atom_stride = 1;
    storage.position_component_stride = storage.pitch;
    storage.spin_atom_stride = 1;
    storage.spin_component_stride = storage.pitch;
    std::vector<double> positions(3 * static_cast<std::size_t>(storage.pitch), 0.0);
    std::vector<double> spins(4 * static_cast<std::size_t>(storage.pitch), 0.0);
    for (int atom = 0; atom < system.atom_count; ++atom) {
      positions[static_cast<std::size_t>(atom)] =
          system.positions_aos3[3 * static_cast<std::size_t>(atom)];
      positions[static_cast<std::size_t>(atom) + storage.pitch] =
          system.positions_aos3[3 * static_cast<std::size_t>(atom) + 1];
      positions[static_cast<std::size_t>(atom) + 2 * storage.pitch] =
          system.positions_aos3[3 * static_cast<std::size_t>(atom) + 2];
      spins[static_cast<std::size_t>(atom)] = 0.5;
      spins[static_cast<std::size_t>(atom) + storage.pitch] = 0.1;
      spins[static_cast<std::size_t>(atom) + 2 * storage.pitch] = 0.0;
      spins[static_cast<std::size_t>(atom) + 3 * storage.pitch] = 2.0;
    }
    storage.positions = copy_to_device(positions, "copy soa positions");
    if (spin_model) {
      storage.spins = copy_to_device(spins, "copy soa spins");
    }

    storage.force_atom_stride = 1;
    storage.force_component_stride = storage.pitch;
    storage.mforce_atom_stride = 1;
    storage.mforce_component_stride = storage.pitch;
    storage.virial_atom_stride = 1;
    storage.virial_component_stride = storage.pitch;
    check_cuda(cudaMalloc(
                   reinterpret_cast<void**>(&storage.forces),
                   3 * static_cast<std::size_t>(storage.pitch) * sizeof(double)),
               "allocate soa forces");
    if (spin_model) {
      check_cuda(cudaMalloc(
                     reinterpret_cast<void**>(&storage.mforces),
                     3 * static_cast<std::size_t>(storage.pitch) * sizeof(double)),
                 "allocate soa mforces");
    }
    check_cuda(cudaMalloc(
                   reinterpret_cast<void**>(&storage.virials),
                   9 * static_cast<std::size_t>(storage.pitch) * sizeof(double)),
               "allocate soa virials");
  } else {
    storage.neighbor_atom_stride = storage.max_neighbors;
    storage.neighbor_slot_stride = 1;
    storage.neighbors =
        copy_to_device(system.row_major_neighbors, "copy aos neighbors");

    storage.position_atom_stride = 3;
    storage.position_component_stride = 1;
    storage.spin_atom_stride = 4;
    storage.spin_component_stride = 1;
    storage.positions = copy_to_device(system.positions_aos3, "copy aos positions");
    if (spin_model) {
      std::vector<double> spins(4 * static_cast<std::size_t>(system.atom_count), 0.0);
      for (int atom = 0; atom < system.atom_count; ++atom) {
        spins[4 * static_cast<std::size_t>(atom) + 0] = 0.5;
        spins[4 * static_cast<std::size_t>(atom) + 1] = 0.1;
        spins[4 * static_cast<std::size_t>(atom) + 2] = 0.0;
        spins[4 * static_cast<std::size_t>(atom) + 3] = 2.0;
      }
      storage.spins = copy_to_device(spins, "copy aos spins");
    }

    storage.force_atom_stride = 3;
    storage.force_component_stride = 1;
    storage.mforce_atom_stride = 3;
    storage.mforce_component_stride = 1;
    storage.virial_atom_stride = 9;
    storage.virial_component_stride = 1;
    check_cuda(cudaMalloc(
                   reinterpret_cast<void**>(&storage.forces),
                   3 * static_cast<std::size_t>(system.atom_count) * sizeof(double)),
               "allocate aos forces");
    if (spin_model) {
      check_cuda(cudaMalloc(
                     reinterpret_cast<void**>(&storage.mforces),
                     3 * static_cast<std::size_t>(system.atom_count) * sizeof(double)),
                 "allocate aos mforces");
    }
    check_cuda(cudaMalloc(
                   reinterpret_cast<void**>(&storage.virials),
                   9 * static_cast<std::size_t>(system.atom_count) * sizeof(double)),
               "allocate aos virials");
  }

  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&storage.total_potential),
                 sizeof(double)),
             "allocate total potential");
  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&storage.total_virial6),
                 6 * sizeof(double)),
             "allocate total virial");
  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&storage.potential_per_atom),
                 static_cast<std::size_t>(system.atom_count) * sizeof(double)),
             "allocate per-atom potential");
  if (spin_model) {
    check_cuda(cudaMalloc(
                   reinterpret_cast<void**>(&storage.spin_transfer),
                   9 * static_cast<std::size_t>(system.atom_count) * sizeof(double)),
               "allocate spin-transfer output");
  }
  return storage;
}

LayoutStorage load_replay_layout(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open replay: " + path);
  }
  ReplayHeader header{};
  read_binary(in, &header, 1, "header");
  const std::string magic(header.magic, header.magic + 8);
  if (magic != "NEPAKKR1" || header.version != 1) {
    throw std::runtime_error("unsupported replay format: " + path);
  }
  if (header.nlocal <= 0 || header.nall < header.nlocal ||
      header.inum <= 0 || header.max_neighbors < 0 ||
      header.neighbor_rows <= 0 || header.numneigh_length <= 0 ||
      header.neighbor_atom_stride <= 0 || header.neighbor_slot_stride <= 0 ||
      header.position_atom_stride <= 0 ||
      header.position_component_stride <= 0 ||
      header.force_atom_stride <= 0 || header.force_component_stride <= 0) {
    throw std::runtime_error("invalid replay metadata: " + path);
  }

  std::vector<int> ilist(static_cast<std::size_t>(header.ilist_count));
  std::vector<int> numneigh(static_cast<std::size_t>(header.numneigh_count));
  std::vector<int> neighbors(static_cast<std::size_t>(header.neighbor_count));
  std::vector<int> types(static_cast<std::size_t>(header.type_count));
  std::vector<int> type_map(static_cast<std::size_t>(header.type_map_count));
  std::vector<double> positions(static_cast<std::size_t>(header.position_count));
  read_binary(in, ilist.data(), ilist.size(), "ilist");
  read_binary(in, numneigh.data(), numneigh.size(), "numneigh");
  read_binary(in, neighbors.data(), neighbors.size(), "neighbors");
  read_binary(in, types.data(), types.size(), "types");
  read_binary(in, type_map.data(), type_map.size(), "type_map");
  read_binary(in, positions.data(), positions.size(), "positions");

  LayoutStorage storage;
  storage.atom_count = header.nlocal;
  storage.nall = header.nall;
  storage.inum = header.inum;
  storage.max_neighbors = header.max_neighbors;
  storage.neighbor_rows = header.neighbor_rows;
  storage.numneigh_length = header.numneigh_length;
  storage.neighbor_atom_stride = header.neighbor_atom_stride;
  storage.neighbor_slot_stride = header.neighbor_slot_stride;
  storage.position_atom_stride = header.position_atom_stride;
  storage.position_component_stride = header.position_component_stride;
  storage.force_atom_stride = header.force_atom_stride;
  storage.force_component_stride = header.force_component_stride;
  storage.virial_atom_stride = 9;
  storage.virial_component_stride = 1;
  storage.type_map_length = header.type_map_length;
  storage.pitch = std::max(header.nall, header.position_component_stride);
  for (int atom = 0; atom < header.inum; ++atom) {
    const int row = ilist[static_cast<std::size_t>(atom)];
    if (row >= 0 && row < header.numneigh_length) {
      storage.total_neighbors +=
          static_cast<std::size_t>(numneigh[static_cast<std::size_t>(row)]);
    }
  }

  storage.ilist = copy_to_device(ilist, "copy replay ilist");
  storage.numneigh = copy_to_device(numneigh, "copy replay numneigh");
  storage.neighbors = copy_to_device(neighbors, "copy replay neighbors");
  storage.types = copy_to_device(types, "copy replay types");
  if (!type_map.empty()) {
    storage.type_map = copy_to_device(type_map, "copy replay type map");
  }
  storage.positions = copy_to_device(positions, "copy replay positions");

  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&storage.forces),
                 static_cast<std::size_t>(header.force_count) * sizeof(double)),
             "allocate replay forces");
  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&storage.virials),
                 9 * static_cast<std::size_t>(header.nlocal) * sizeof(double)),
             "allocate replay virials");
  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&storage.total_potential),
                 sizeof(double)),
             "allocate replay total potential");
  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&storage.total_virial6),
                 6 * sizeof(double)),
             "allocate replay total virial");
  check_cuda(cudaMalloc(
                 reinterpret_cast<void**>(&storage.potential_per_atom),
                 static_cast<std::size_t>(header.nlocal) * sizeof(double)),
             "allocate replay per-atom potential");
  return storage;
}

NepaLammpsDeviceNeighborInput make_device_input(const LayoutStorage& storage) {
  NepaLammpsDeviceNeighborInput input{};
  input.nlocal = storage.atom_count;
  input.nall = storage.nall > 0 ? storage.nall : storage.atom_count;
  input.inum = storage.inum > 0 ? storage.inum : storage.atom_count;
  input.max_neighbors = storage.max_neighbors;
  input.neighbor_rows =
      storage.neighbor_rows > 0 ? storage.neighbor_rows : storage.atom_count;
  input.numneigh_length =
      storage.numneigh_length > 0 ? storage.numneigh_length : storage.atom_count;
  input.ilist = storage.ilist;
  input.numneigh = storage.numneigh;
  input.neighbors = storage.neighbors;
  input.neighbor_atom_stride = storage.neighbor_atom_stride;
  input.neighbor_slot_stride = storage.neighbor_slot_stride;
  input.types = storage.types;
  input.type_map = storage.type_map;
  input.type_map_length = storage.type_map_length;
  input.positions = storage.positions;
  input.position_atom_stride = storage.position_atom_stride;
  input.position_component_stride = storage.position_component_stride;
  input.spins = storage.spins;
  input.spin_atom_stride = storage.spin_atom_stride;
  input.spin_component_stride = storage.spin_component_stride;
  return input;
}

void free_layout(LayoutStorage& storage) {
  cudaFree(storage.ilist);
  cudaFree(storage.numneigh);
  cudaFree(storage.neighbors);
  cudaFree(storage.types);
  cudaFree(storage.type_map);
  cudaFree(storage.positions);
  cudaFree(storage.spins);
  cudaFree(storage.total_potential);
  cudaFree(storage.total_virial6);
  cudaFree(storage.potential_per_atom);
  cudaFree(storage.forces);
  cudaFree(storage.mforces);
  cudaFree(storage.virials);
  cudaFree(storage.spin_transfer);
}

void run_once(
    NepaModel* model,
    const LayoutStorage& storage,
    const Options& options) {
  NepaLammpsDeviceNeighborInput input = make_device_input(storage);

  NepaLammpsDeviceNeighborResult result{};
  result.total_potential = options.write_totals ? storage.total_potential : nullptr;
  result.total_virial6 = options.write_totals ? storage.total_virial6 : nullptr;
  result.potential_per_atom =
      options.write_per_atom ? storage.potential_per_atom : nullptr;
  result.forces = storage.forces;
  result.force_atom_stride = storage.force_atom_stride;
  result.force_component_stride = storage.force_component_stride;
  result.mforces = storage.mforces;
  result.mforce_atom_stride = storage.mforce_atom_stride;
  result.mforce_component_stride = storage.mforce_component_stride;
  result.virials_per_atom9 = options.write_per_atom ? storage.virials : nullptr;
  result.virial_atom_stride = storage.virial_atom_stride;
  result.virial_component_stride = storage.virial_component_stride;
  result.spin_transfer_per_atom_row_major9 =
      options.write_spin_transfer ? storage.spin_transfer : nullptr;
  result.spin_transfer_atom_stride = 9;
  result.spin_transfer_component_stride = 1;

  const NepaStatus status =
      nepa_find_force_lammps_device_neighbors(model, &input, &result);
  if (status != NEPA_STATUS_OK) {
    const char* detail = nepa_last_error_message();
    const std::string suffix =
        detail != nullptr && detail[0] != '\0' ? std::string(": ") + detail : "";
    throw std::runtime_error(
        std::string("nepa_find_force_lammps_device_neighbors failed: ") +
        std::to_string(status) + suffix);
  }
}

double time_layout(NepaModel* model, const LayoutStorage& storage, const Options& options) {
  for (int i = 0; i < options.warmup; ++i) {
    run_once(model, storage, options);
  }
  check_cuda(cudaDeviceSynchronize(), "sync after warmup");

  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < options.iterations; ++i) {
    run_once(model, storage, options);
  }
  check_cuda(cudaDeviceSynchronize(), "sync after timing");
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count() /
         options.iterations;
}

nep_adapters::cuda_backend::SimulationBox make_nonperiodic_box() {
  nep_adapters::cuda_backend::SimulationBox box{};
  box.frac_to_cart[0] = 1.0;
  box.frac_to_cart[4] = 1.0;
  box.frac_to_cart[8] = 1.0;
  box.cart_to_frac[0] = 1.0;
  box.cart_to_frac[4] = 1.0;
  box.cart_to_frac[8] = 1.0;
  return box;
}

std::size_t workspace_atom_capacity(const LayoutStorage& storage) {
  const int atom_count = storage.nall > 0 ? storage.nall : storage.atom_count;
  if (storage.position_atom_stride == 1 &&
      storage.position_component_stride >= atom_count) {
    return static_cast<std::size_t>(storage.position_component_stride);
  }
  return static_cast<std::size_t>(atom_count);
}

void clear_potential(nep_adapters::cuda_backend::DeviceWorkspace& workspace) {
  const nep_adapters::cuda_backend::DeviceWorkspaceView view = workspace.view();
  check_cuda(
      cudaMemset(view.potential, 0, view.atom_capacity * sizeof(double)),
      "clear potential");
}

void run_reused_workspace_once(
    const nep_adapters::cuda_backend::ModelProtocol& protocol,
    const nep_adapters::cuda_backend::DeviceModel& model,
    const LayoutStorage& storage,
    nep_adapters::cuda_backend::DeviceWorkspace& workspace,
    const Options& options,
    bool check_overflow) {
  NepaLammpsDeviceNeighborInput input = make_device_input(storage);

  NepaLammpsDeviceNeighborResult result{};
  result.total_potential = options.write_totals ? storage.total_potential : nullptr;
  result.total_virial6 = options.write_totals ? storage.total_virial6 : nullptr;
  result.potential_per_atom =
      options.write_per_atom ? storage.potential_per_atom : nullptr;
  result.forces = storage.forces;
  result.force_atom_stride = storage.force_atom_stride;
  result.force_component_stride = storage.force_component_stride;
  result.mforces = storage.mforces;
  result.mforce_atom_stride = storage.mforce_atom_stride;
  result.mforce_component_stride = storage.mforce_component_stride;
  result.virials_per_atom9 = options.write_per_atom ? storage.virials : nullptr;
  result.virial_atom_stride = storage.virial_atom_stride;
  result.virial_component_stride = storage.virial_component_stride;

  nep_adapters::cuda_backend::stage_lammps_device_neighbors_on_device(
      input,
      protocol,
      workspace,
      check_overflow);
  const nep_adapters::cuda_backend::SimulationBox box = make_nonperiodic_box();
  const bool store_potential =
      options.write_totals || options.write_per_atom;
  if (store_potential) {
    clear_potential(workspace);
  }
  const bool has_angular = protocol.body_channels.channel_count() > 0;
  nep_adapters::cuda_backend::build_descriptor_core_from_positions_on_device(
      protocol,
      storage.atom_count,
      box,
      model,
      workspace,
      nep_adapters::cuda_backend::DescriptorCoreTopology::single_box);
  nep_adapters::cuda_backend::evaluate_ann_energy_on_device(
      protocol, storage.atom_count, model, workspace);
  const bool accumulate_virial = options.write_totals || options.write_per_atom;
  const auto virial_target = accumulate_virial
      ? nep_adapters::cuda_backend::VirialTarget::center_atom
      : nep_adapters::cuda_backend::VirialTarget::none;
  nep_adapters::cuda_backend::accumulate_lammps_radial_forces_on_device(
      protocol,
      storage.atom_count,
      box,
      model,
      workspace,
      virial_target,
      store_potential);
  if (has_angular) {
    nep_adapters::cuda_backend::accumulate_l2_angular_forces_on_device(
        protocol,
        storage.atom_count,
        model,
        workspace,
        virial_target);
  }
  nep_adapters::cuda_backend::write_lammps_device_outputs(
      input,
      result,
      workspace);
}

ReusedWorkspaceTiming run_reused_workspace_once_breakdown(
    const nep_adapters::cuda_backend::ModelProtocol& protocol,
    const nep_adapters::cuda_backend::DeviceModel& model,
    const LayoutStorage& storage,
    nep_adapters::cuda_backend::DeviceWorkspace& workspace,
    const Options& options,
    bool check_overflow) {
  ReusedWorkspaceTiming timing{};
  NepaLammpsDeviceNeighborInput input = make_device_input(storage);

  NepaLammpsDeviceNeighborResult result{};
  result.total_potential = options.write_totals ? storage.total_potential : nullptr;
  result.total_virial6 = options.write_totals ? storage.total_virial6 : nullptr;
  result.potential_per_atom =
      options.write_per_atom ? storage.potential_per_atom : nullptr;
  result.forces = storage.forces;
  result.force_atom_stride = storage.force_atom_stride;
  result.force_component_stride = storage.force_component_stride;
  result.mforces = storage.mforces;
  result.mforce_atom_stride = storage.mforce_atom_stride;
  result.mforce_component_stride = storage.mforce_component_stride;
  result.virials_per_atom9 = options.write_per_atom ? storage.virials : nullptr;
  result.virial_atom_stride = storage.virial_atom_stride;
  result.virial_component_stride = storage.virial_component_stride;

  const nep_adapters::cuda_backend::SimulationBox box = make_nonperiodic_box();
  timing.stage_ms = time_synchronized_phase([&]() {
    nep_adapters::cuda_backend::stage_lammps_device_neighbors_on_device(
        input,
        protocol,
        workspace,
        check_overflow);
  });
  const bool store_potential =
      options.write_totals || options.write_per_atom;
  timing.clear_ms = time_synchronized_phase([&]() {
    if (store_potential) {
      clear_potential(workspace);
    }
  });
  const bool has_angular = protocol.body_channels.channel_count() > 0;
  timing.descriptor_ms = time_synchronized_phase([&]() {
    nep_adapters::cuda_backend::build_descriptor_core_from_positions_on_device(
        protocol,
        storage.atom_count,
        box,
        model,
        workspace,
        nep_adapters::cuda_backend::DescriptorCoreTopology::single_box);
  });
  timing.ann_ms = time_synchronized_phase([&]() {
    nep_adapters::cuda_backend::evaluate_ann_energy_on_device(
        protocol, storage.atom_count, model, workspace);
  });
  timing.force_ms = time_synchronized_phase([&]() {
    const bool accumulate_virial =
        options.write_totals || options.write_per_atom;
    const auto virial_target = accumulate_virial
        ? nep_adapters::cuda_backend::VirialTarget::center_atom
        : nep_adapters::cuda_backend::VirialTarget::none;
    nep_adapters::cuda_backend::accumulate_lammps_radial_forces_on_device(
        protocol,
        storage.atom_count,
        box,
        model,
        workspace,
        virial_target,
        store_potential);
    if (has_angular) {
      nep_adapters::cuda_backend::accumulate_l2_angular_forces_on_device(
          protocol,
          storage.atom_count,
          model,
          workspace,
          virial_target);
    }
  });
  timing.output_ms = time_synchronized_phase([&]() {
    nep_adapters::cuda_backend::write_lammps_device_outputs(
        input,
        result,
        workspace);
  });
  timing.total_ms =
      timing.stage_ms + timing.clear_ms + timing.radial_cache_ms +
      timing.descriptor_ms + timing.ann_ms + timing.force_ms + timing.output_ms;
  return timing;
}

ReusedWorkspaceTiming time_reused_workspace_layout(
    const nep_adapters::cuda_backend::ModelProtocol& protocol,
    const nep_adapters::cuda_backend::DeviceModel& model,
    const LayoutStorage& storage,
    const Options& options) {
  NepaLammpsDeviceNeighborInput input = make_device_input(storage);
  nep_adapters::cuda_backend::ModelProtocol external_protocol = protocol;
  external_protocol.neighbor_capacity_radial =
      std::min(external_protocol.neighbor_capacity_radial, storage.max_neighbors);
  external_protocol.neighbor_capacity_angular =
      std::min(external_protocol.neighbor_capacity_angular, storage.max_neighbors);
  nep_adapters::cuda_backend::DeviceWorkspace workspace(
      nep_adapters::cuda_backend::make_external_neighbor_workspace_plan(
          external_protocol,
          workspace_atom_capacity(storage),
          static_cast<std::size_t>(storage.atom_count),
          false));
  for (int i = 0; i < options.warmup; ++i) {
    run_reused_workspace_once(
        external_protocol,
        model,
        storage,
        workspace,
        options,
        i == 0);
  }
  check_cuda(cudaDeviceSynchronize(), "sync reused warmup");

  if (options.breakdown) {
    ReusedWorkspaceTiming timing{};
    for (int i = 0; i < options.iterations; ++i) {
      const ReusedWorkspaceTiming iter = run_reused_workspace_once_breakdown(
          external_protocol,
          model,
          storage,
          workspace,
          options,
          false);
      timing.total_ms += iter.total_ms;
      timing.stage_ms += iter.stage_ms;
      timing.clear_ms += iter.clear_ms;
      timing.radial_cache_ms += iter.radial_cache_ms;
      timing.descriptor_ms += iter.descriptor_ms;
      timing.ann_ms += iter.ann_ms;
      timing.force_ms += iter.force_ms;
      timing.output_ms += iter.output_ms;
    }
    const double inv_iterations = 1.0 / options.iterations;
    timing.total_ms *= inv_iterations;
    timing.stage_ms *= inv_iterations;
    timing.clear_ms *= inv_iterations;
    timing.radial_cache_ms *= inv_iterations;
    timing.descriptor_ms *= inv_iterations;
    timing.ann_ms *= inv_iterations;
    timing.force_ms *= inv_iterations;
    timing.output_ms *= inv_iterations;
    return timing;
  }

  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < options.iterations; ++i) {
    run_reused_workspace_once(
        external_protocol,
        model,
        storage,
        workspace,
        options,
        false);
  }
  check_cuda(cudaDeviceSynchronize(), "sync reused timing");
  const auto end = std::chrono::steady_clock::now();
  ReusedWorkspaceTiming timing{};
  timing.total_ms =
      std::chrono::duration<double, std::milli>(end - begin).count() /
      options.iterations;
  return timing;
}

struct PairTiming {
  double legacy_ms = 0.0;
  double nolegacy_ms = 0.0;
};

void print_reused_workspace_timing(
    const std::string& prefix,
    const ReusedWorkspaceTiming& timing,
    double atom_count,
    bool breakdown) {
  const double throughput = atom_count / (1000.0 * timing.total_ms);
  std::cout << prefix << "_ms=" << timing.total_ms << '\n'
            << prefix << "_Matom_per_s=" << throughput << '\n';
  if (!breakdown) {
    return;
  }
  std::cout << prefix << "_stage_ms=" << timing.stage_ms << '\n'
            << prefix << "_clear_ms=" << timing.clear_ms << '\n'
            << prefix << "_radial_cache_ms=" << timing.radial_cache_ms << '\n'
            << prefix << "_descriptor_ms=" << timing.descriptor_ms << '\n'
            << prefix << "_ann_ms=" << timing.ann_ms << '\n'
            << prefix << "_force_ms=" << timing.force_ms << '\n'
            << prefix << "_output_ms=" << timing.output_ms << '\n';
}

double time_one_call(
    NepaModel* model,
    const LayoutStorage& storage,
    const Options& options) {
  const auto begin = std::chrono::steady_clock::now();
  run_once(model, storage, options);
  check_cuda(cudaDeviceSynchronize(), "sync timed call");
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

PairTiming time_paired_layouts(
    NepaModel* model,
    const LayoutStorage& legacy,
    const LayoutStorage& nolegacy,
    const Options& options,
    bool legacy_first) {
  for (int i = 0; i < options.warmup; ++i) {
    run_once(model, legacy, options);
    run_once(model, nolegacy, options);
  }
  check_cuda(cudaDeviceSynchronize(), "sync paired warmup");

  PairTiming timing;
  for (int i = 0; i < options.iterations; ++i) {
    if (legacy_first) {
      timing.legacy_ms += time_one_call(model, legacy, options);
      timing.nolegacy_ms += time_one_call(model, nolegacy, options);
    } else {
      timing.nolegacy_ms += time_one_call(model, nolegacy, options);
      timing.legacy_ms += time_one_call(model, legacy, options);
    }
  }
  timing.legacy_ms /= options.iterations;
  timing.nolegacy_ms /= options.iterations;
  return timing;
}

}  // namespace

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  if (!nep_adapters::register_cuda_engine()) {
    std::cerr << "failed to register CUDA engine\n";
    return EXIT_FAILURE;
  }

  const std::string model_path = options.strip_spin_model
      ? write_struct_model_from_spin(options.model_path)
      : (options.model_path.empty() ? write_radial_model(options) : options.model_path);
  nep_adapters::cuda_backend::HostModelParameters host;
  try {
    host = nep_adapters::cuda_backend::load_host_model_parameters(model_path);
  } catch (const std::exception& error) {
    std::cerr << "failed to load host model parameters: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  if (!options.model_path.empty()) {
    options.radial_cutoff = host.protocol.cutoff_radial;
    options.angular_cutoff = host.protocol.cutoff_angular;
    options.radial_capacity = host.protocol.max_neighbors_radial;
    options.angular_capacity = host.protocol.max_neighbors_angular;
  }

  NepaModel* model = nullptr;
  if (nepa_load_model("cuda", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    std::cerr << "failed to load CUDA model\n";
    return EXIT_FAILURE;
  }

  try {
    if (!options.replay_path.empty()) {
      LayoutStorage layout = load_replay_layout(options.replay_path);
      std::cout << "atoms=" << layout.atom_count << '\n'
                << "nall=" << layout.nall << '\n'
                << "inum=" << layout.inum << '\n'
                << "max_neighbors=" << layout.max_neighbors << '\n'
                << "avg_neighbors="
                << static_cast<double>(layout.total_neighbors) /
                       static_cast<double>(layout.inum)
                << '\n'
                << "neighbor_rows=" << layout.neighbor_rows << '\n'
                << "numneigh_length=" << layout.numneigh_length << '\n'
                << "neighbor_atom_stride=" << layout.neighbor_atom_stride << '\n'
                << "neighbor_slot_stride=" << layout.neighbor_slot_stride << '\n'
                << "position_atom_stride=" << layout.position_atom_stride << '\n'
                << "position_component_stride="
                << layout.position_component_stride << '\n'
                << "force_atom_stride=" << layout.force_atom_stride << '\n'
                << "force_component_stride=" << layout.force_component_stride << '\n'
                << "radial_cutoff=" << options.radial_cutoff << '\n'
                << "angular_cutoff=" << options.angular_cutoff << '\n'
                << "mn_radial=" << options.radial_capacity << '\n'
                << "mn_angular=" << options.angular_capacity << '\n'
                << "model_path=" << model_path << '\n'
                << "replay_path=" << options.replay_path << '\n'
                << "protocol_types=" << host.protocol.num_types << '\n'
                << "protocol_n_max_radial=" << host.protocol.n_max_radial << '\n'
                << "protocol_n_max_angular=" << host.protocol.n_max_angular << '\n'
                << "protocol_basis_size_radial="
                << host.protocol.basis_size_radial << '\n'
                << "protocol_basis_size_angular="
                << host.protocol.basis_size_angular << '\n'
                << "protocol_body_channels="
                << host.protocol.body_channels.channel_count() << '\n'
                << "protocol_has_zbl=" << (host.protocol.has_zbl ? 1 : 0) << '\n'
                << "warmup=" << options.warmup
                << " iterations=" << options.iterations << '\n'
                << "mode=" << options.mode << '\n'
                << "layout=replay\n"
                << "totals=" << (options.write_totals ? 1 : 0) << '\n'
                << "per_atom=" << (options.write_per_atom ? 1 : 0) << '\n'
                << "spin_transfer=" << (options.write_spin_transfer ? 1 : 0) << '\n'
                << "breakdown=" << (options.breakdown ? 1 : 0) << '\n'
                << "split_pipeline=" << (options.split_pipeline ? 1 : 0)
                << '\n';
      const auto atom_throughput = [&](double ms) {
        return static_cast<double>(layout.atom_count) / (1000.0 * ms);
      };
      if (options.mode == "all" || options.mode == "api") {
        const double ms = time_layout(model, layout, options);
        std::cout << "replay_ms=" << ms << '\n'
                  << "replay_Matom_per_s=" << atom_throughput(ms) << '\n';
      }
      if (options.mode == "all" || options.mode == "reused") {
        ReusedWorkspaceTiming reused{};
        if (options.breakdown || options.split_pipeline) {
          nep_adapters::cuda_backend::DeviceModel device_model(host);
          reused = time_reused_workspace_layout(
              host.protocol,
              device_model,
              layout,
              options);
        } else {
          reused.total_ms = time_layout(model, layout, options);
        }
        print_reused_workspace_timing(
            "reused_workspace_replay",
            reused,
            static_cast<double>(layout.atom_count),
            options.breakdown);
      }
      free_layout(layout);
      nepa_free_model(model);
      return EXIT_SUCCESS;
    }

    const FixedNeighborSystem system =
        options.cubic_cells > 0
            ? make_cubic_system(
                  options.cubic_cells,
                  options.spacing,
                  options.radial_cutoff + options.skin)
            : make_chain_system(options.atoms);
    std::cout << "atoms=" << options.atoms << '\n'
              << "cubic_cells=" << options.cubic_cells << '\n'
              << "max_neighbors=" << system.max_neighbors << '\n'
              << "avg_neighbors="
              << static_cast<double>(system.total_neighbors) /
                     static_cast<double>(system.atom_count)
              << '\n'
              << "radial_cutoff=" << options.radial_cutoff << '\n'
              << "candidate_cutoff=" << options.radial_cutoff + options.skin << '\n'
              << "angular_cutoff=" << options.angular_cutoff << '\n'
              << "mn_radial=" << options.radial_capacity << '\n'
              << "mn_angular=" << options.angular_capacity << '\n'
              << "model_path=" << model_path << '\n'
              << "protocol_types=" << host.protocol.num_types << '\n'
              << "protocol_n_max_radial=" << host.protocol.n_max_radial << '\n'
              << "protocol_n_max_angular=" << host.protocol.n_max_angular << '\n'
              << "protocol_basis_size_radial="
              << host.protocol.basis_size_radial << '\n'
              << "protocol_basis_size_angular="
              << host.protocol.basis_size_angular << '\n'
              << "protocol_body_channels="
              << host.protocol.body_channels.channel_count() << '\n'
              << "protocol_has_zbl=" << (host.protocol.has_zbl ? 1 : 0) << '\n'
              << "spacing=" << options.spacing << '\n'
              << "skin=" << options.skin << '\n'
              << "warmup=" << options.warmup
              << " iterations=" << options.iterations << '\n'
              << "mode=" << options.mode << '\n'
              << "layout=" << options.layout << '\n'
              << "type_map=" << (options.use_type_map ? 1 : 0) << '\n'
              << "cycle_model_types="
              << (options.cycle_model_types ? 1 : 0) << '\n'
              << "totals=" << (options.write_totals ? 1 : 0) << '\n'
              << "per_atom=" << (options.write_per_atom ? 1 : 0) << '\n'
              << "spin_transfer=" << (options.write_spin_transfer ? 1 : 0) << '\n'
              << "breakdown=" << (options.breakdown ? 1 : 0) << '\n'
              << "split_pipeline=" << (options.split_pipeline ? 1 : 0)
              << '\n';
    const auto atom_throughput = [&](double ms) {
      return static_cast<double>(system.atom_count) / (1000.0 * ms);
    };
    if (options.layout != "both") {
      LayoutStorage layout =
          make_layout(
              system,
              options.layout == "nolegacy",
              options.use_type_map,
              options.cycle_model_types,
              host.protocol.num_types,
              host.protocol.spin_mode != 0);
      if (options.mode == "all" || options.mode == "api") {
        const double ms = time_layout(model, layout, options);
        std::cout << options.layout << "_ms=" << ms << '\n'
                  << options.layout << "_Matom_per_s="
                  << atom_throughput(ms) << '\n';
      }
      if (options.mode == "all" || options.mode == "reused") {
        ReusedWorkspaceTiming reused{};
        if (options.breakdown || options.split_pipeline) {
          nep_adapters::cuda_backend::DeviceModel device_model(host);
          reused = time_reused_workspace_layout(
              host.protocol,
              device_model,
              layout,
              options);
        } else {
          reused.total_ms = time_layout(model, layout, options);
        }
        print_reused_workspace_timing(
            "reused_workspace_" + options.layout,
            reused,
            static_cast<double>(system.atom_count),
            options.breakdown);
      }
      free_layout(layout);
    } else {
    LayoutStorage legacy = make_layout(
        system,
        false,
        options.use_type_map,
        options.cycle_model_types,
        host.protocol.num_types,
        host.protocol.spin_mode != 0);
    LayoutStorage nolegacy = make_layout(
        system,
        true,
        options.use_type_map,
        options.cycle_model_types,
        host.protocol.num_types,
        host.protocol.spin_mode != 0);
    if (options.mode == "all" || options.mode == "api") {
      const double legacy_ms = time_layout(model, legacy, options);
      const double nolegacy_ms = time_layout(model, nolegacy, options);
      std::cout << "legacy_aos_ms=" << legacy_ms << '\n'
                << "legacy_aos_Matom_per_s=" << atom_throughput(legacy_ms)
                << '\n'
                << "nolegacy_soa_ms=" << nolegacy_ms << '\n'
                << "nolegacy_soa_Matom_per_s=" << atom_throughput(nolegacy_ms)
                << '\n'
                << "nolegacy_vs_legacy=" << nolegacy_ms / legacy_ms << '\n'
                << "legacy_vs_nolegacy_speedup=" << legacy_ms / nolegacy_ms
                << '\n';
      if (options.mode == "all") {
        const PairTiming legacy_first =
            time_paired_layouts(model, legacy, nolegacy, options, true);
        const PairTiming nolegacy_first =
            time_paired_layouts(model, legacy, nolegacy, options, false);
        std::cout
                << "paired_legacy_first_legacy_ms=" << legacy_first.legacy_ms
                << '\n'
                << "paired_legacy_first_nolegacy_ms=" << legacy_first.nolegacy_ms
                << '\n'
                << "paired_legacy_first_speedup="
                << legacy_first.legacy_ms / legacy_first.nolegacy_ms << '\n'
                << "paired_nolegacy_first_legacy_ms=" << nolegacy_first.legacy_ms
                << '\n'
                << "paired_nolegacy_first_nolegacy_ms="
                << nolegacy_first.nolegacy_ms << '\n'
                << "paired_nolegacy_first_speedup="
                << nolegacy_first.legacy_ms / nolegacy_first.nolegacy_ms
                << '\n';
      }
    }
    if (options.mode == "all" || options.mode == "reused") {
      ReusedWorkspaceTiming reused_legacy{};
      ReusedWorkspaceTiming reused_nolegacy{};
      if (options.breakdown || options.split_pipeline) {
        nep_adapters::cuda_backend::DeviceModel device_model(host);
        reused_legacy = time_reused_workspace_layout(
            host.protocol,
            device_model,
            legacy,
            options);
        reused_nolegacy = time_reused_workspace_layout(
            host.protocol,
            device_model,
            nolegacy,
            options);
      } else {
        reused_legacy.total_ms = time_layout(model, legacy, options);
        reused_nolegacy.total_ms = time_layout(model, nolegacy, options);
      }
      print_reused_workspace_timing(
          "reused_workspace_legacy",
          reused_legacy,
          static_cast<double>(system.atom_count),
          options.breakdown);
      print_reused_workspace_timing(
          "reused_workspace_nolegacy",
          reused_nolegacy,
          static_cast<double>(system.atom_count),
          options.breakdown);
      std::cout << "reused_workspace_speedup="
                << reused_legacy.total_ms / reused_nolegacy.total_ms << '\n';
    }
    free_layout(legacy);
    free_layout(nolegacy);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  nepa_free_model(model);
  return EXIT_SUCCESS;
}
