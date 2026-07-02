#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_nep3.hpp"
#if NEP_ADAPTERS_BENCH_HAS_CPU_OPT
#include "nep_adapters/engines/cpu_opt.hpp"
#endif

#include "cpu_nep3_test_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if NEP_ADAPTERS_BENCH_OPENMP_ENABLED
#include <omp.h>
#endif

namespace {

bool run_find_force(
    NepaModel* model,
    const NepaStructureBatch& batch,
    NepaFindForceResult& result) {
  return nepa_find_force_batch(model, &batch, &result) == NEPA_STATUS_OK;
}

enum class Mode {
  batch,
  lammps,
};

struct Replicate {
  int nx = 1;
  int ny = 1;
  int nz = 1;
};

using RankGrid = Replicate;

Replicate parse_replicate(const std::string& text) {
  Replicate replicate;
  char x1 = '\0';
  char x2 = '\0';
  std::istringstream stream(text);
  stream >> replicate.nx >> x1 >> replicate.ny >> x2 >> replicate.nz;
  if (!stream || x1 != 'x' || x2 != 'x' || replicate.nx <= 0 ||
      replicate.ny <= 0 || replicate.nz <= 0) {
    std::cerr << "Invalid --replicate value: " << text << '\n';
    std::exit(EXIT_FAILURE);
  }
  return replicate;
}

int rank_count(const RankGrid& grid) {
  return grid.nx * grid.ny * grid.nz;
}

Replicate rank_coords(const RankGrid& grid, int rank_id) {
  return {
      rank_id % grid.nx,
      (rank_id / grid.nx) % grid.ny,
      rank_id / (grid.nx * grid.ny),
  };
}

Mode parse_mode(const std::string& text) {
  if (text == "batch") {
    return Mode::batch;
  }
  if (text == "lammps") {
    return Mode::lammps;
  }
  std::cerr << "Invalid --mode value: " << text << '\n';
  std::exit(EXIT_FAILURE);
}

const char* mode_name(Mode mode) {
  return mode == Mode::batch ? "batch" : "lammps";
}

cpu_nep3_test::Frame make_supercell(
    const cpu_nep3_test::Frame& frame,
    const Replicate& replicate) {
  const std::size_t atom_count = frame.types.size();
  const std::size_t image_count =
      static_cast<std::size_t>(replicate.nx) * replicate.ny * replicate.nz;
  cpu_nep3_test::Frame supercell;
  supercell.types.reserve(atom_count * image_count);
  supercell.positions_aos3.reserve(atom_count * image_count * 3);
  supercell.reference_energy = frame.reference_energy * image_count;

  const double a[] = {frame.box[0], frame.box[3], frame.box[6]};
  const double b[] = {frame.box[1], frame.box[4], frame.box[7]};
  const double c[] = {frame.box[2], frame.box[5], frame.box[8]};

  for (int ix = 0; ix < replicate.nx; ++ix) {
    for (int iy = 0; iy < replicate.ny; ++iy) {
      for (int iz = 0; iz < replicate.nz; ++iz) {
        const double shift[] = {
            ix * a[0] + iy * b[0] + iz * c[0],
            ix * a[1] + iy * b[1] + iz * c[1],
            ix * a[2] + iy * b[2] + iz * c[2],
        };
        for (std::size_t atom = 0; atom < atom_count; ++atom) {
          supercell.types.push_back(frame.types[atom]);
          supercell.positions_aos3.push_back(
              frame.positions_aos3[3 * atom + 0] + shift[0]);
          supercell.positions_aos3.push_back(
              frame.positions_aos3[3 * atom + 1] + shift[1]);
          supercell.positions_aos3.push_back(
              frame.positions_aos3[3 * atom + 2] + shift[2]);
        }
      }
    }
  }

  supercell.box[0] = frame.box[0] * replicate.nx;
  supercell.box[3] = frame.box[3] * replicate.nx;
  supercell.box[6] = frame.box[6] * replicate.nx;
  supercell.box[1] = frame.box[1] * replicate.ny;
  supercell.box[4] = frame.box[4] * replicate.ny;
  supercell.box[7] = frame.box[7] * replicate.ny;
  supercell.box[2] = frame.box[2] * replicate.nz;
  supercell.box[5] = frame.box[5] * replicate.nz;
  supercell.box[8] = frame.box[8] * replicate.nz;
  return supercell;
}

struct LammpsInputStorage {
  int nlocal = 0;
  int nall = 0;
  int ghost_count = 0;
  int max_neighbors = 0;
  std::size_t neighbor_count = 0;
  std::vector<int> ilist;
  std::vector<int> numneigh;
  std::vector<std::vector<int>> neighbors;
  std::vector<int*> firstneigh;
  std::vector<int> types;
  std::vector<int> type_map;
  std::vector<double> positions;
  std::vector<double*> position_rows;
};

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct LammpsResultStorage {
  double total_potential = 0.0;
  double total_virial6[6] = {};
  std::vector<double> forces;
  std::vector<double*> force_rows;
};

Vec3 box_vector(const double* box, int column) {
  return {
      box[column],
      box[3 + column],
      box[6 + column],
  };
}

Vec3 scaled_shift(const Vec3& a, const Vec3& b, const Vec3& c, int ia, int ib, int ic) {
  return {
      ia * a.x + ib * b.x + ic * c.x,
      ia * a.y + ib * b.y + ic * c.y,
      ia * a.z + ib * b.z + ic * c.z,
  };
}

double norm(const Vec3& value) {
  return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

int image_range(const Vec3& vector, double cutoff) {
  const double length = norm(vector);
  if (length <= 0.0) {
    return 0;
  }
  return std::max(1, static_cast<int>(std::ceil(cutoff / length)));
}

double distance_squared(const std::vector<double>& positions, int i, const Vec3& rhs) {
  const double dx = rhs.x - positions[3 * static_cast<std::size_t>(i) + 0];
  const double dy = rhs.y - positions[3 * static_cast<std::size_t>(i) + 1];
  const double dz = rhs.z - positions[3 * static_cast<std::size_t>(i) + 2];
  return dx * dx + dy * dy + dz * dz;
}

Vec3 fractional_position(const double* box, const Vec3& position) {
  const double a00 = box[0];
  const double a01 = box[1];
  const double a02 = box[2];
  const double a10 = box[3];
  const double a11 = box[4];
  const double a12 = box[5];
  const double a20 = box[6];
  const double a21 = box[7];
  const double a22 = box[8];
  const double det =
      a00 * (a11 * a22 - a12 * a21) -
      a01 * (a10 * a22 - a12 * a20) +
      a02 * (a10 * a21 - a11 * a20);
  if (std::abs(det) <= 0.0) {
    std::cerr << "Invalid zero-volume benchmark box\n";
    std::exit(EXIT_FAILURE);
  }
  const double inv_det = 1.0 / det;
  return {
      ((a11 * a22 - a12 * a21) * position.x +
       (a02 * a21 - a01 * a22) * position.y +
       (a01 * a12 - a02 * a11) * position.z) *
          inv_det,
      ((a12 * a20 - a10 * a22) * position.x +
       (a00 * a22 - a02 * a20) * position.y +
       (a02 * a10 - a00 * a12) * position.z) *
          inv_det,
      ((a10 * a21 - a11 * a20) * position.x +
       (a01 * a20 - a00 * a21) * position.y +
       (a00 * a11 - a01 * a10) * position.z) *
          inv_det,
  };
}

double wrap_fraction(double value) {
  value -= std::floor(value);
  return value >= 1.0 ? value - 1.0 : value;
}

int rank_bin(double fraction, int count) {
  int bin = static_cast<int>(std::floor(wrap_fraction(fraction) * count));
  return std::min(count - 1, std::max(0, bin));
}

bool belongs_to_rank(
    const double* box,
    const Vec3& position,
    const RankGrid& grid,
    const Replicate& coords) {
  const Vec3 fraction = fractional_position(box, position);
  return rank_bin(fraction.x, grid.nx) == coords.nx &&
         rank_bin(fraction.y, grid.ny) == coords.ny &&
         rank_bin(fraction.z, grid.nz) == coords.nz;
}

LammpsInputStorage make_lammps_input(
    const cpu_nep3_test::Frame& frame,
    std::int32_t num_types,
    double cutoff,
    const RankGrid& rank_grid,
    int rank_id) {
  if (rank_id < 0 || rank_id >= rank_count(rank_grid)) {
    std::cerr << "Invalid --rank-id for --rank-grid\n";
    std::exit(EXIT_FAILURE);
  }

  LammpsInputStorage input;
  const Replicate coords = rank_coords(rank_grid, rank_id);
  const int system_atoms = static_cast<int>(frame.types.size());
  input.positions.reserve(frame.positions_aos3.size() * 27);
  input.types.reserve(frame.types.size() * 27);

  std::vector<int> local_atoms;
  local_atoms.reserve(frame.types.size());
  for (int atom = 0; atom < system_atoms; ++atom) {
    const Vec3 position = {
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 0],
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 1],
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 2],
    };
    if (belongs_to_rank(frame.box, position, rank_grid, coords)) {
      local_atoms.push_back(atom);
      input.types.push_back(frame.types[static_cast<std::size_t>(atom)] + 1);
      input.positions.push_back(position.x);
      input.positions.push_back(position.y);
      input.positions.push_back(position.z);
    }
  }

  input.nlocal = static_cast<int>(local_atoms.size());
  if (input.nlocal <= 0) {
    std::cerr << "Selected rank owns no local atoms; choose a smaller --rank-grid or another --rank-id\n";
    std::exit(EXIT_FAILURE);
  }
  input.ilist.resize(static_cast<std::size_t>(input.nlocal));

  const double cutoff_sq = cutoff * cutoff;
  const Vec3 a = box_vector(frame.box, 0);
  const Vec3 b = box_vector(frame.box, 1);
  const Vec3 c = box_vector(frame.box, 2);
  const int range_a = image_range(a, cutoff);
  const int range_b = image_range(b, cutoff);
  const int range_c = image_range(c, cutoff);

  for (int ia = -range_a; ia <= range_a; ++ia) {
    for (int ib = -range_b; ib <= range_b; ++ib) {
      for (int ic = -range_c; ic <= range_c; ++ic) {
        if (ia == 0 && ib == 0 && ic == 0) {
          continue;
        }
        const Vec3 shift = scaled_shift(a, b, c, ia, ib, ic);
        for (int atom = 0; atom < system_atoms; ++atom) {
          const Vec3 ghost_position = {
              frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 0] + shift.x,
              frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 1] + shift.y,
              frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 2] + shift.z,
          };
          bool used = false;
          for (int local = 0; local < input.nlocal && !used; ++local) {
            used = distance_squared(input.positions, local, ghost_position) < cutoff_sq;
          }
          if (!used) {
            continue;
          }
          input.types.push_back(frame.types[static_cast<std::size_t>(atom)] + 1);
          input.positions.push_back(ghost_position.x);
          input.positions.push_back(ghost_position.y);
          input.positions.push_back(ghost_position.z);
        }
      }
    }
  }

  for (int atom = 0; atom < system_atoms; ++atom) {
    if (std::find(local_atoms.begin(), local_atoms.end(), atom) != local_atoms.end()) {
      continue;
    }
    const Vec3 ghost_position = {
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 0],
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 1],
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 2],
    };
    bool used = false;
    for (int local = 0; local < input.nlocal && !used; ++local) {
      used = distance_squared(input.positions, local, ghost_position) < cutoff_sq;
    }
    if (!used) {
      continue;
    }
    input.types.push_back(frame.types[static_cast<std::size_t>(atom)] + 1);
    input.positions.push_back(ghost_position.x);
    input.positions.push_back(ghost_position.y);
    input.positions.push_back(ghost_position.z);
  }

  input.nall = static_cast<int>(input.types.size());
  input.ghost_count = input.nall - input.nlocal;
  input.numneigh.assign(static_cast<std::size_t>(input.nall), 0);
  input.neighbors.resize(static_cast<std::size_t>(input.nlocal));
  input.firstneigh.assign(static_cast<std::size_t>(input.nall), nullptr);
  input.position_rows.resize(static_cast<std::size_t>(input.nall));
  input.type_map.assign(static_cast<std::size_t>(num_types + 1), -1);

  for (std::int32_t type = 0; type < num_types; ++type) {
    input.type_map[static_cast<std::size_t>(type + 1)] = type;
  }

  for (int atom = 0; atom < input.nall; ++atom) {
    input.position_rows[static_cast<std::size_t>(atom)] =
        input.positions.data() + 3 * static_cast<std::size_t>(atom);
  }
  for (int atom = 0; atom < input.nlocal; ++atom) {
    input.ilist[static_cast<std::size_t>(atom)] = atom;
  }

  for (int i = 0; i < input.nlocal; ++i) {
    std::vector<int>& neighbors = input.neighbors[static_cast<std::size_t>(i)];
    for (int j = 0; j < input.nall; ++j) {
      if (i == j) {
        continue;
      }
      const double dx = input.positions[3 * static_cast<std::size_t>(j) + 0] -
                        input.positions[3 * static_cast<std::size_t>(i) + 0];
      const double dy = input.positions[3 * static_cast<std::size_t>(j) + 1] -
                        input.positions[3 * static_cast<std::size_t>(i) + 1];
      const double dz = input.positions[3 * static_cast<std::size_t>(j) + 2] -
                        input.positions[3 * static_cast<std::size_t>(i) + 2];
      if (dx * dx + dy * dy + dz * dz < cutoff_sq) {
        neighbors.push_back(j);
      }
    }
    input.numneigh[static_cast<std::size_t>(i)] = static_cast<int>(neighbors.size());
    input.neighbor_count += neighbors.size();
    input.max_neighbors =
        std::max(input.max_neighbors, static_cast<int>(neighbors.size()));
    input.firstneigh[static_cast<std::size_t>(i)] =
        neighbors.empty() ? nullptr : neighbors.data();
  }

  return input;
}

LammpsResultStorage make_lammps_result(int nall) {
  LammpsResultStorage result;
  result.forces.assign(static_cast<std::size_t>(nall) * 3, 0.0);
  result.force_rows.resize(static_cast<std::size_t>(nall));
  for (int atom = 0; atom < nall; ++atom) {
    result.force_rows[static_cast<std::size_t>(atom)] =
        result.forces.data() + 3 * static_cast<std::size_t>(atom);
  }
  return result;
}

bool run_lammps_force(
    NepaModel* model,
    LammpsInputStorage& input,
    LammpsResultStorage& storage) {
  std::fill(storage.forces.begin(), storage.forces.end(), 0.0);
  NepaLammpsNeighborInput lammps_input{};
  lammps_input.nlocal = input.nlocal;
  lammps_input.inum = input.nlocal;
  lammps_input.ilist = input.ilist.data();
  lammps_input.numneigh = input.numneigh.data();
  lammps_input.firstneigh = input.firstneigh.data();
  lammps_input.types = input.types.data();
  lammps_input.type_map = input.type_map.data();
  lammps_input.positions = input.position_rows.data();

  NepaLammpsNeighborResult result{};
  result.total_potential = &storage.total_potential;
  result.total_virial6 = storage.total_virial6;
  result.forces = storage.force_rows.data();
  return nepa_find_force_lammps_neighbors(model, &lammps_input, &result) ==
         NEPA_STATUS_OK;
}

}  // namespace

int main(int argc, char** argv) {
  int iterations = 10;
  int warmup = 1;
  Replicate replicate;
  RankGrid rank_grid;
  int rank_id = 0;
  std::string engine_name = "cpu_nep3";
  Mode mode = Mode::batch;

  for (int arg = 1; arg < argc; ++arg) {
    if (std::strcmp(argv[arg], "--iterations") == 0 && arg + 1 < argc) {
      iterations = std::atoi(argv[++arg]);
    } else if (std::strcmp(argv[arg], "--warmup") == 0 && arg + 1 < argc) {
      warmup = std::atoi(argv[++arg]);
    } else if (std::strcmp(argv[arg], "--replicate") == 0 && arg + 1 < argc) {
      replicate = parse_replicate(argv[++arg]);
    } else if (std::strcmp(argv[arg], "--rank-grid") == 0 && arg + 1 < argc) {
      rank_grid = parse_replicate(argv[++arg]);
    } else if (std::strcmp(argv[arg], "--rank-id") == 0 && arg + 1 < argc) {
      rank_id = std::atoi(argv[++arg]);
    } else if (std::strcmp(argv[arg], "--engine") == 0 && arg + 1 < argc) {
      engine_name = argv[++arg];
    } else if (std::strcmp(argv[arg], "--mode") == 0 && arg + 1 < argc) {
      mode = parse_mode(argv[++arg]);
    } else if (arg == 1 && argv[arg][0] != '-') {
      // Backward-compatible positional iteration count.
      iterations = std::atoi(argv[arg]);
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--engine NAME] [--mode batch|lammps]"
                << " [--iterations N] [--warmup N]"
                << " [--replicate NxMxK] [--rank-grid NxMxK] [--rank-id N]\n";
      return EXIT_FAILURE;
    }
  }

  if (iterations <= 0 || warmup < 0) {
    return EXIT_FAILURE;
  }
  if (engine_name != "cpu_nep3" && engine_name != "cpu_opt") {
    std::cerr << "Unsupported --engine value: " << engine_name << '\n';
    return EXIT_FAILURE;
  }
#if !NEP_ADAPTERS_BENCH_HAS_CPU_OPT
  if (engine_name == "cpu_opt") {
    std::cerr << "cpu_opt benchmark requested but cpu_opt was not built\n";
    return EXIT_FAILURE;
  }
#endif

  const std::string model_path = NEP_ADAPTERS_NEP89_MODEL_PATH;
  const std::string xyz_path = NEP_ADAPTERS_NEP89_XYZ_PATH;

  if (!nep_adapters::register_cpu_nep3_engine()) {
    return EXIT_FAILURE;
  }
#if NEP_ADAPTERS_BENCH_HAS_CPU_OPT
  if (!nep_adapters::register_cpu_opt_engine()) {
    return EXIT_FAILURE;
  }
#endif

  const auto type_map = cpu_nep3_test::read_type_map(model_path);
  cpu_nep3_test::Frame frame =
      cpu_nep3_test::read_first_frame(xyz_path, type_map);
  frame = make_supercell(frame, replicate);

  const std::int32_t atom_count =
      static_cast<std::int32_t>(frame.types.size());

  NepaModel* model = nullptr;
  if (nepa_load_model(engine_name.c_str(), model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }
  NepaModelInfo model_info{};
  if (nepa_model_info(model, &model_info) != NEPA_STATUS_OK ||
      model_info.cutoff_max <= 0.0 || model_info.num_types <= 0) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  std::int32_t atom_counts[] = {atom_count};
  std::int32_t atom_offsets[] = {0};
  std::int32_t pbc[] = {1, 1, 1};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = atom_count;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = frame.types.data();
  batch.positions_aos3 = frame.positions_aos3.data();
  batch.boxes_row_major9 = frame.box;
  batch.pbc_flags3 = pbc;

  double energy[] = {0.0};
  std::vector<double> forces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  double virial[9] = {};

  NepaFindForceResult result{};
  result.energy_per_structure = energy;
  result.forces_aos3 = forces.data();
  result.virials_row_major9 = virial;

  LammpsInputStorage lammps_input;
  LammpsResultStorage lammps_result;
  if (mode == Mode::lammps) {
    lammps_input = make_lammps_input(
        frame, model_info.num_types, model_info.cutoff_max, rank_grid, rank_id);
    lammps_result = make_lammps_result(lammps_input.nall);
  }

  auto run_once = [&]() {
    if (mode == Mode::batch) {
      return run_find_force(model, batch, result);
    }
    return run_lammps_force(model, lammps_input, lammps_result);
  };

  for (int i = 0; i < warmup; ++i) {
    if (!run_once()) {
      nepa_free_model(model);
      return EXIT_FAILURE;
    }
  }

  const auto started = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    if (!run_once()) {
      nepa_free_model(model);
      return EXIT_FAILURE;
    }
  }
  const auto finished = std::chrono::steady_clock::now();

  const std::vector<double>& checked_forces =
      mode == Mode::batch ? forces : lammps_result.forces;
  const double checked_energy =
      mode == Mode::batch ? energy[0] : lammps_result.total_potential;

  const double force_l1 = std::accumulate(
      checked_forces.begin(),
      checked_forces.end(),
      0.0,
      [](double sum, double value) { return sum + std::abs(value); });
  if (!std::isfinite(checked_energy) || !cpu_nep3_test::all_finite(checked_forces) ||
      force_l1 <= 0.0) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  nepa_free_model(model);

  const int total_iterations = iterations;
  const int active_atoms = mode == Mode::lammps ? lammps_input.nlocal : atom_count;
  const double seconds =
      std::chrono::duration<double>(finished - started).count();
  const double evals_per_second = static_cast<double>(total_iterations) / seconds;
  const double atom_steps_per_second =
      static_cast<double>(total_iterations) * active_atoms / seconds;
#if NEP_ADAPTERS_BENCH_OPENMP_ENABLED
  const int omp_threads = omp_get_max_threads();
#else
  const int omp_threads = 1;
#endif

  std::cout << "{\"benchmark\":\""
            << (mode == Mode::batch ? "nep89_find_force_batch" : "nep89_lammps_neighbors")
            << "\","
            << "\"engine\":\"" << engine_name << "\","
            << "\"mode\":\"" << mode_name(mode) << "\","
            << "\"model\":\"nep89\","
            << "\"openmp_enabled\":" << NEP_ADAPTERS_BENCH_OPENMP_ENABLED << ','
            << "\"omp_threads\":" << omp_threads << ','
            << "\"replicate\":\"" << replicate.nx << 'x' << replicate.ny
            << 'x' << replicate.nz << "\","
            << "\"rank_grid\":\"" << rank_grid.nx << 'x' << rank_grid.ny
            << 'x' << rank_grid.nz << "\","
            << "\"rank_id\":" << (mode == Mode::lammps ? rank_id : 0) << ','
            << "\"rank_count\":" << (mode == Mode::lammps ? rank_count(rank_grid) : 1) << ','
            << "\"atoms\":" << active_atoms << ','
            << "\"system_atoms\":" << atom_count << ','
            << "\"nlocal\":" << active_atoms << ','
            << "\"nall\":" << (mode == Mode::lammps ? lammps_input.nall : atom_count) << ','
            << "\"ghost_atoms\":" << (mode == Mode::lammps ? lammps_input.ghost_count : 0) << ','
            << "\"neighbor_count\":"
            << (mode == Mode::lammps ? lammps_input.neighbor_count : 0) << ','
            << "\"avg_neighbors\":"
            << (mode == Mode::lammps && active_atoms > 0
                    ? static_cast<double>(lammps_input.neighbor_count) / active_atoms
                    : 0.0)
            << ','
            << "\"max_neighbors\":"
            << (mode == Mode::lammps ? lammps_input.max_neighbors : 0) << ','
            << "\"warmup\":" << warmup << ','
            << "\"iterations\":" << iterations << ','
            << "\"total_iterations\":" << total_iterations << ','
            << "\"seconds\":" << seconds << ','
            << "\"evals_per_second\":" << evals_per_second << ','
            << "\"atom_steps_per_second\":" << atom_steps_per_second
            << "}\n";

  return EXIT_SUCCESS;
}
