#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu.hpp"

#include "cpu_test_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
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

void set_phase_timer_env(bool enabled) {
#if defined(_WIN32)
  if (enabled) {
    _putenv_s("NEP_CPU_PHASE_TIMER", "1");
  } else {
    _putenv_s("NEP_CPU_PHASE_TIMER", "");
  }
#else
  if (enabled) {
    setenv("NEP_CPU_PHASE_TIMER", "1", 1);
  } else {
    unsetenv("NEP_CPU_PHASE_TIMER");
  }
#endif
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

cpu_test::Frame make_supercell(
    const cpu_test::Frame& frame,
    const Replicate& replicate) {
  const std::size_t atom_count = frame.types.size();
  const std::size_t image_count =
      static_cast<std::size_t>(replicate.nx) * replicate.ny * replicate.nz;
  cpu_test::Frame supercell;
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

cpu_test::Frame make_bcc_fe32(
    const std::unordered_map<std::string, std::int32_t>& type_map) {
  const auto fe = type_map.find("Fe");
  if (fe == type_map.end()) {
    std::cerr << "The selected model does not contain Fe\n";
    std::exit(EXIT_FAILURE);
  }

  constexpr double lattice = 2.87;
  cpu_test::Frame frame;
  frame.types.reserve(32);
  frame.positions_aos3.reserve(32 * 3);
  for (int ix = 0; ix < 2; ++ix) {
    for (int iy = 0; iy < 2; ++iy) {
      for (int iz = 0; iz < 4; ++iz) {
        for (int basis = 0; basis < 2; ++basis) {
          const double offset = basis == 0 ? 0.0 : 0.5;
          frame.types.push_back(fe->second);
          frame.positions_aos3.push_back((ix + offset) * lattice);
          frame.positions_aos3.push_back((iy + offset) * lattice);
          frame.positions_aos3.push_back((iz + offset) * lattice);
        }
      }
    }
  }
  frame.box[0] = 2 * lattice;
  frame.box[4] = 2 * lattice;
  frame.box[8] = 4 * lattice;
  return frame;
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

struct CartesianCellList {
  double cutoff = 1.0;
  Vec3 origin;
  int nx = 1;
  int ny = 1;
  int nz = 1;
  const std::vector<double>* positions = nullptr;
  std::vector<std::vector<int>> cells;

  std::size_t cell_index(int ix, int iy, int iz) const {
    return (static_cast<std::size_t>(ix) * ny + iy) * nz + iz;
  }

  int coord(double value, double min_value, int count) const {
    const int index = static_cast<int>(std::floor((value - min_value) / cutoff));
    return index < 0 ? index : std::min(index, count - 1);
  }

  void build(const std::vector<double>& source_positions, int count, double cutoff_in) {
    positions = &source_positions;
    cutoff = cutoff_in;
    Vec3 lo = {
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::max(),
    };
    Vec3 hi = {
        -std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max(),
        -std::numeric_limits<double>::max(),
    };
    for (int atom = 0; atom < count; ++atom) {
      const double x = source_positions[3 * static_cast<std::size_t>(atom) + 0];
      const double y = source_positions[3 * static_cast<std::size_t>(atom) + 1];
      const double z = source_positions[3 * static_cast<std::size_t>(atom) + 2];
      lo.x = std::min(lo.x, x);
      lo.y = std::min(lo.y, y);
      lo.z = std::min(lo.z, z);
      hi.x = std::max(hi.x, x);
      hi.y = std::max(hi.y, y);
      hi.z = std::max(hi.z, z);
    }

    origin = {lo.x - cutoff, lo.y - cutoff, lo.z - cutoff};
    nx = std::max(1, static_cast<int>(std::floor((hi.x - origin.x + cutoff) / cutoff)) + 1);
    ny = std::max(1, static_cast<int>(std::floor((hi.y - origin.y + cutoff) / cutoff)) + 1);
    nz = std::max(1, static_cast<int>(std::floor((hi.z - origin.z + cutoff) / cutoff)) + 1);
    cells.assign(static_cast<std::size_t>(nx) * ny * nz, {});
    for (int atom = 0; atom < count; ++atom) {
      const int ix = coord(source_positions[3 * static_cast<std::size_t>(atom) + 0], origin.x, nx);
      const int iy = coord(source_positions[3 * static_cast<std::size_t>(atom) + 1], origin.y, ny);
      const int iz = coord(source_positions[3 * static_cast<std::size_t>(atom) + 2], origin.z, nz);
      cells[cell_index(ix, iy, iz)].push_back(atom);
    }
  }

  template <class Callback>
  void for_nearby_cells(const Vec3& position, Callback&& callback) const {
    const int cx = coord(position.x, origin.x, nx);
    const int cy = coord(position.y, origin.y, ny);
    const int cz = coord(position.z, origin.z, nz);
    const int ix0 = std::max(0, cx - 1);
    const int iy0 = std::max(0, cy - 1);
    const int iz0 = std::max(0, cz - 1);
    const int ix1 = std::min(nx - 1, cx + 1);
    const int iy1 = std::min(ny - 1, cy + 1);
    const int iz1 = std::min(nz - 1, cz + 1);
    if (ix0 > ix1 || iy0 > iy1 || iz0 > iz1) {
      return;
    }
    for (int ix = ix0; ix <= ix1; ++ix) {
      for (int iy = iy0; iy <= iy1; ++iy) {
        for (int iz = iz0; iz <= iz1; ++iz) {
          for (int atom : cells[cell_index(ix, iy, iz)]) {
            callback(atom);
          }
        }
      }
    }
  }

  bool has_atom_within(const Vec3& position, double cutoff_sq) const {
    bool found = false;
    for_nearby_cells(position, [&](int atom) {
      found = found || distance_squared(*positions, atom, position) < cutoff_sq;
    });
    return found;
  }
};

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
    const cpu_test::Frame& frame,
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
  std::vector<char> is_local_atom(static_cast<std::size_t>(system_atoms), 0);
  local_atoms.reserve(frame.types.size());
  for (int atom = 0; atom < system_atoms; ++atom) {
    const Vec3 position = {
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 0],
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 1],
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 2],
    };
    if (belongs_to_rank(frame.box, position, rank_grid, coords)) {
      is_local_atom[static_cast<std::size_t>(atom)] = 1;
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
  CartesianCellList local_cells;
  local_cells.build(input.positions, input.nlocal, cutoff);
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
          if (!local_cells.has_atom_within(ghost_position, cutoff_sq)) {
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
    if (is_local_atom[static_cast<std::size_t>(atom)]) {
      continue;
    }
    const Vec3 ghost_position = {
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 0],
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 1],
        frame.positions_aos3[3 * static_cast<std::size_t>(atom) + 2],
    };
    if (!local_cells.has_atom_within(ghost_position, cutoff_sq)) {
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

  CartesianCellList all_cells;
  all_cells.build(input.positions, input.nall, cutoff);
  for (int i = 0; i < input.nlocal; ++i) {
    std::vector<int>& neighbors = input.neighbors[static_cast<std::size_t>(i)];
    const Vec3 position_i = {
        input.positions[3 * static_cast<std::size_t>(i) + 0],
        input.positions[3 * static_cast<std::size_t>(i) + 1],
        input.positions[3 * static_cast<std::size_t>(i) + 2],
    };
    all_cells.for_nearby_cells(position_i, [&](int j) {
      if (i == j) {
        return;
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
    });
    std::sort(neighbors.begin(), neighbors.end());
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
  int structures = 1;
  std::string engine_name = "cpu";
  Mode mode = Mode::batch;
  bool phase_timer = false;
  bool fe32 = false;
  std::string model_path = NEP_ADAPTERS_NEP89_MODEL_PATH;
  std::string xyz_path = NEP_ADAPTERS_NEP89_XYZ_PATH;

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
    } else if (std::strcmp(argv[arg], "--structures") == 0 && arg + 1 < argc) {
      structures = std::atoi(argv[++arg]);
    } else if (std::strcmp(argv[arg], "--model") == 0 && arg + 1 < argc) {
      model_path = argv[++arg];
    } else if (std::strcmp(argv[arg], "--xyz") == 0 && arg + 1 < argc) {
      xyz_path = argv[++arg];
    } else if (std::strcmp(argv[arg], "--fe32") == 0) {
      fe32 = true;
    } else if (std::strcmp(argv[arg], "--engine") == 0 && arg + 1 < argc) {
      engine_name = argv[++arg];
    } else if (std::strcmp(argv[arg], "--mode") == 0 && arg + 1 < argc) {
      mode = parse_mode(argv[++arg]);
    } else if (std::strcmp(argv[arg], "--phase-timer") == 0) {
      phase_timer = true;
    } else if (arg == 1 && argv[arg][0] != '-') {
      // Backward-compatible positional iteration count.
      iterations = std::atoi(argv[arg]);
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--engine NAME] [--mode batch|lammps]"
                << " [--iterations N] [--warmup N]"
                << " [--structures N]"
                << " [--model PATH] [--xyz PATH] [--fe32]"
                << " [--replicate NxMxK] [--rank-grid NxMxK] [--rank-id N]"
                << " [--phase-timer]\n";
      return EXIT_FAILURE;
    }
  }

  if (iterations <= 0 || warmup < 0 || structures <= 0) {
    return EXIT_FAILURE;
  }
  if (engine_name != "cpu") {
    std::cerr << "Unsupported --engine value: " << engine_name << '\n';
    return EXIT_FAILURE;
  }
  if (mode == Mode::lammps && structures != 1) {
    std::cerr << "--structures is only supported in batch mode\n";
    return EXIT_FAILURE;
  }

  if (!nep_adapters::register_cpu_engine()) {
    return EXIT_FAILURE;
  }

  const auto type_map = cpu_test::read_type_map(model_path);
  cpu_test::Frame frame = fe32
      ? make_bcc_fe32(type_map)
      : cpu_test::read_first_frame(xyz_path, type_map);
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

  const std::int64_t total_atoms_64 =
      static_cast<std::int64_t>(atom_count) * structures;
  if (total_atoms_64 > std::numeric_limits<std::int32_t>::max()) {
    std::cerr << "The requested batch exceeds the C API atom-count range\n";
    return EXIT_FAILURE;
  }
  const std::int32_t total_atoms =
      static_cast<std::int32_t>(total_atoms_64);
  std::vector<std::int32_t> atom_counts(
      static_cast<std::size_t>(structures), atom_count);
  std::vector<std::int32_t> atom_offsets(
      static_cast<std::size_t>(structures));
  std::vector<std::int32_t> batch_types(
      static_cast<std::size_t>(total_atoms));
  std::vector<double> batch_positions(
      static_cast<std::size_t>(total_atoms) * 3);
  std::vector<double> batch_boxes(
      static_cast<std::size_t>(structures) * 9);
  std::vector<std::int32_t> pbc(
      static_cast<std::size_t>(structures) * 3, 1);
  for (int structure = 0; structure < structures; ++structure) {
    const std::int32_t atom_offset = structure * atom_count;
    atom_offsets[static_cast<std::size_t>(structure)] = atom_offset;
    std::copy(
        frame.types.begin(),
        frame.types.end(),
        batch_types.begin() + atom_offset);
    std::copy_n(
        frame.box,
        9,
        batch_boxes.data() + static_cast<std::size_t>(structure) * 9);
    for (std::int32_t atom = 0; atom < atom_count; ++atom) {
      const double phase =
          static_cast<double>(structure) * 0.173 +
          static_cast<double>(atom) * 0.37;
      const std::size_t source = static_cast<std::size_t>(atom) * 3;
      const std::size_t destination =
          (static_cast<std::size_t>(atom_offset) + atom) * 3;
      batch_positions[destination + 0] =
          frame.positions_aos3[source + 0] + 0.012 * std::sin(phase);
      batch_positions[destination + 1] =
          frame.positions_aos3[source + 1] + 0.010 * std::sin(1.31 * phase);
      batch_positions[destination + 2] =
          frame.positions_aos3[source + 2] + 0.011 * std::sin(1.73 * phase);
    }
  }

  NepaStructureBatch batch{};
  batch.num_structures = structures;
  batch.total_atoms = total_atoms;
  batch.atom_counts = atom_counts.data();
  batch.atom_offsets = atom_offsets.data();
  batch.types = batch_types.data();
  batch.positions_aos3 = batch_positions.data();
  batch.boxes_row_major9 = batch_boxes.data();
  batch.pbc_flags3 = pbc.data();

  std::vector<double> energy(static_cast<std::size_t>(structures), 0.0);
  std::vector<double> forces(static_cast<std::size_t>(total_atoms) * 3, 0.0);
  std::vector<double> virial(static_cast<std::size_t>(structures) * 9, 0.0);

  NepaFindForceResult result{};
  result.energy_per_structure = energy.data();
  result.forces_aos3 = forces.data();
  result.virials_row_major9 = virial.data();

  LammpsInputStorage lammps_input;
  LammpsResultStorage lammps_result;
  double setup_seconds = 0.0;
  if (mode == Mode::lammps) {
    const auto setup_started = std::chrono::steady_clock::now();
    lammps_input = make_lammps_input(
        frame, model_info.num_types, model_info.cutoff_max, rank_grid, rank_id);
    lammps_result = make_lammps_result(lammps_input.nall);
    const auto setup_finished = std::chrono::steady_clock::now();
    setup_seconds =
        std::chrono::duration<double>(setup_finished - setup_started).count();
  }

  auto run_once = [&]() {
    if (mode == Mode::batch) {
      return run_find_force(model, batch, result);
    }
    return run_lammps_force(model, lammps_input, lammps_result);
  };

  if (phase_timer) {
    set_phase_timer_env(false);
  }
  for (int i = 0; i < warmup; ++i) {
    if (!run_once()) {
      nepa_free_model(model);
      return EXIT_FAILURE;
    }
  }

  if (phase_timer) {
    set_phase_timer_env(true);
  }
  const auto started = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    if (!run_once()) {
      nepa_free_model(model);
      return EXIT_FAILURE;
    }
  }
  const auto finished = std::chrono::steady_clock::now();
  if (phase_timer) {
    set_phase_timer_env(false);
  }

  const std::vector<double>& checked_forces =
      mode == Mode::batch ? forces : lammps_result.forces;
  const double checked_energy =
      mode == Mode::batch
          ? std::accumulate(energy.begin(), energy.end(), 0.0)
          : lammps_result.total_potential;

  const double force_l1 = std::accumulate(
      checked_forces.begin(),
      checked_forces.end(),
      0.0,
      [](double sum, double value) { return sum + std::abs(value); });
  if (!std::isfinite(checked_energy) || !cpu_test::all_finite(checked_forces) ||
      force_l1 <= 0.0) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  nepa_free_model(model);

  const int total_iterations = iterations;
  const int active_atoms =
      mode == Mode::lammps ? lammps_input.nlocal : total_atoms;
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

  std::cout << std::setprecision(17)
            << "{\"benchmark\":\""
            << (mode == Mode::batch ? "nep89_find_force_batch" : "nep89_lammps_neighbors")
            << "\","
            << "\"engine\":\"" << engine_name << "\","
            << "\"mode\":\"" << mode_name(mode) << "\","
            << "\"model\":\"nep89\","
            << "\"structures\":" << (mode == Mode::batch ? structures : 1) << ','
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
            << "\"nall\":" << (mode == Mode::lammps ? lammps_input.nall : total_atoms) << ','
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
            << "\"setup_seconds\":" << setup_seconds << ','
            << "\"seconds\":" << seconds << ','
            << "\"evals_per_second\":" << evals_per_second << ','
            << "\"atom_steps_per_second\":" << atom_steps_per_second << ','
            << "\"energy_sum\":" << checked_energy << ','
            << "\"force_l1\":" << force_l1
            << "}\n";

  return EXIT_SUCCESS;
}
