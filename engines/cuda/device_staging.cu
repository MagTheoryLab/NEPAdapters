#include "device_staging.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace nep_adapters::cuda_backend {
namespace {

constexpr int kLammpsNeighborMask = 0x3fffffff;
constexpr int kBlockSize = 512;
constexpr int kWarpSize = 32;
constexpr int kStagingWarpsPerBlock = kBlockSize / kWarpSize;
constexpr int kLammpsOverflowTypeRange = 1 << 0;
constexpr int kLammpsOverflowTypeMap = 1 << 1;
constexpr int kLammpsOverflowActiveAtom = 1 << 2;
constexpr int kLammpsOverflowInputRowCount = 1 << 3;
constexpr int kLammpsOverflowNeighborIndex = 1 << 4;
constexpr int kLammpsOverflowRadialCapacity = 1 << 5;
constexpr int kLammpsOverflowAngularCapacity = 1 << 6;

void check_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(action) + ": " + cudaGetErrorString(status));
  }
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool invert_row_major3(const double* matrix, double* inverse) {
  const double det =
      matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
      matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
      matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
  if (std::abs(det) <= 1.0e-12) {
    return false;
  }
  const double inv_det = 1.0 / det;
  inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inv_det;
  inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inv_det;
  inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inv_det;
  inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inv_det;
  inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inv_det;
  inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inv_det;
  inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inv_det;
  inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inv_det;
  inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inv_det;
  return true;
}

template <typename T>
T* upload_temp(const std::vector<T>& host, const char* name) {
  if (host.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  const std::size_t bytes = host.size() * sizeof(T);
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device), bytes), name);
  try {
    check_cuda(
        cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice),
        name);
  } catch (...) {
    cudaFree(device);
    throw;
  }
  return device;
}

template <typename T>
void free_temp(T*& device) noexcept {
  if (device != nullptr) {
    cudaFree(device);
    device = nullptr;
  }
}

__global__ void stage_positions_aos_to_soa(
    int atom_count,
    int atom_stride,
    const double* positions_aos3,
    double* positions_soa3) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  positions_soa3[atom] = positions_aos3[3 * atom];
  positions_soa3[atom_stride + atom] = positions_aos3[3 * atom + 1];
  positions_soa3[2 * atom_stride + atom] = positions_aos3[3 * atom + 2];
}

__global__ void stage_atom_to_structure(
    int structure_count,
    const int* atom_counts,
    const int* atom_offsets,
    int* atom_to_structure) {
  const int structure = blockIdx.x;
  if (structure >= structure_count) {
    return;
  }
  const int atom_count = atom_counts[structure];
  const int atom_offset = atom_offsets[structure];
  for (int local = threadIdx.x; local < atom_count; local += blockDim.x) {
    atom_to_structure[atom_offset + local] = structure;
  }
}

__global__ void stage_lammps_types(
    int atom_count,
    const int* lammps_types,
    const int* type_map,
    int* mapped_types) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  mapped_types[atom] = type_map[lammps_types[atom]];
}

__global__ void stage_lammps_neighbors_slot_major(
    int active_count,
    int atom_capacity,
    int neighbor_capacity,
    const int* active_atom_indices,
    const int* row_counts,
    const int* row_major_neighbors,
    int* counts,
    int* slot_major_neighbors) {
  const int active = blockIdx.x * blockDim.x + threadIdx.x;
  if (active >= active_count) {
    return;
  }
  const int atom = active_atom_indices[active];
  const int count = row_counts[atom];
  counts[atom] = count;
  for (int slot = 0; slot < count; ++slot) {
    slot_major_neighbors[atom + atom_capacity * slot] =
        row_major_neighbors[atom * neighbor_capacity + slot];
  }
}

__global__ void stage_lammps_device_positions(
    int atom_count,
    int atom_stride,
    const double* device_positions,
    int position_atom_stride,
    int position_component_stride,
    double* positions_soa3) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  positions_soa3[atom] =
      device_positions[atom * position_atom_stride];
  positions_soa3[atom_stride + atom] =
      device_positions[atom * position_atom_stride + position_component_stride];
  positions_soa3[2 * atom_stride + atom] =
      device_positions[atom * position_atom_stride + 2 * position_component_stride];
}

__global__ void stage_lammps_device_spins(
    int atom_count,
    int atom_stride,
    const double* device_spins,
    int spin_atom_stride,
    int spin_component_stride,
    double* spins_soa3) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const double scale =
      device_spins[atom * spin_atom_stride + 3 * spin_component_stride];
  spins_soa3[atom] = device_spins[atom * spin_atom_stride] * scale;
  spins_soa3[atom_stride + atom] =
      device_spins[atom * spin_atom_stride + spin_component_stride] * scale;
  spins_soa3[2 * atom_stride + atom] =
      device_spins[atom * spin_atom_stride + 2 * spin_component_stride] * scale;
}

__global__ void stage_lammps_device_types(
    int atom_count,
    const int* lammps_types,
    const int* type_map,
    int type_map_length,
    int* mapped_types,
    int* overflow) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const int lammps_type = lammps_types[atom];
  if (lammps_type < 0 || lammps_type >= type_map_length) {
    atomicOr(overflow, kLammpsOverflowTypeRange);
    mapped_types[atom] = 0;
    return;
  }
  const int mapped_type = type_map[lammps_type];
  if (mapped_type < 0) {
    atomicOr(overflow, kLammpsOverflowTypeMap);
    mapped_types[atom] = 0;
    return;
  }
  mapped_types[atom] = mapped_type;
}

__global__ void stage_lammps_device_neighbors_dual_slot_major(
    int active_count,
    int atom_stride,
    int local_atom_count,
    int valid_atom_count,
    int neighbor_rows,
    int numneigh_length,
    int radial_capacity,
    int angular_capacity,
    int input_neighbor_capacity,
    int stage_angular,
    int check_overflow,
    const int* active_atom_indices,
    const int* row_counts,
    const int* neighbors,
    const int* neighbor_owner,
    int neighbor_atom_stride,
    int neighbor_slot_stride,
    const double* positions,
    int position_component_stride,
    double radial_cutoff_sq,
    double angular_cutoff_sq,
    int* radial_counts,
    int* radial_neighbors,
    int* angular_counts,
    int* angular_neighbors,
    int* overflow) {
  extern __shared__ int neighbor_tile[];
  int* radial_tile = neighbor_tile;
  int* angular_tile = radial_tile + kStagingWarpsPerBlock * radial_capacity;
  int* radial_tile_counts =
      angular_tile + (stage_angular ? kStagingWarpsPerBlock * angular_capacity : 0);
  int* angular_tile_counts = radial_tile_counts + kStagingWarpsPerBlock;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int active = blockIdx.x * kStagingWarpsPerBlock + warp;
  const bool has_active = active < active_count;
  const int atom = has_active ? active_atom_indices[active] : 0;
  bool valid_row = has_active && atom >= 0 && atom < local_atom_count;
  if (has_active && !valid_row) {
    if (check_overflow && lane == 0) {
      atomicOr(overflow, kLammpsOverflowActiveAtom);
    }
  }
  if (valid_row && (atom >= neighbor_rows || atom >= numneigh_length)) {
    if (lane == 0) {
      if (check_overflow) {
        atomicOr(overflow, kLammpsOverflowInputRowCount);
      }
      radial_counts[atom] = 0;
      if (stage_angular) angular_counts[atom] = 0;
    }
    valid_row = false;
  }
  const int count = valid_row ? row_counts[atom] : 0;
  if (check_overflow && lane == 0 &&
      (count < 0 || count > input_neighbor_capacity)) {
    atomicOr(overflow, kLammpsOverflowInputRowCount);
  }
  const int safe_count =
      count < 0 ? 0 : (count > input_neighbor_capacity ? input_neighbor_capacity : count);
  (void)neighbor_owner;
  int radial_count = 0;
  int angular_count = 0;
  constexpr unsigned int kFullWarpMask = 0xffffffffu;
  double xi = 0.0;
  double yi = 0.0;
  double zi = 0.0;
  if (valid_row && lane == 0) {
    xi = positions[atom];
    yi = positions[atom + position_component_stride];
    zi = positions[atom + 2 * position_component_stride];
  }
  xi = __shfl_sync(kFullWarpMask, xi, 0);
  yi = __shfl_sync(kFullWarpMask, yi, 0);
  zi = __shfl_sync(kFullWarpMask, zi, 0);
  for (int slot_base = 0; slot_base < safe_count; slot_base += kWarpSize) {
    const int slot = slot_base + lane;
    int neighbor = -1;
    bool valid_neighbor = false;
    double distance_sq = 0.0;
    if (slot < safe_count) {
      neighbor =
          neighbors[atom * neighbor_atom_stride + slot * neighbor_slot_stride] &
          kLammpsNeighborMask;
      valid_neighbor = neighbor != atom && neighbor >= 0 && neighbor < valid_atom_count;
    }
    if (check_overflow && slot < safe_count && neighbor != atom && !valid_neighbor) {
      atomicOr(overflow, kLammpsOverflowNeighborIndex);
    }
    if (valid_neighbor) {
      const double dx = positions[neighbor] - xi;
      const double dy = positions[neighbor + position_component_stride] - yi;
      const double dz = positions[neighbor + 2 * position_component_stride] - zi;
      distance_sq = dx * dx + dy * dy + dz * dz;
    }

    const bool radial_accept = valid_neighbor && distance_sq <= radial_cutoff_sq;
    const unsigned int radial_mask = __ballot_sync(kFullWarpMask, radial_accept);
    if (radial_accept) {
      const int offset =
          radial_count + __popc(radial_mask & ((1u << lane) - 1u));
      if (offset < radial_capacity) {
        radial_tile[warp * radial_capacity + offset] = neighbor;
      } else if (check_overflow) {
        atomicOr(overflow, kLammpsOverflowRadialCapacity);
      }
    }
    radial_count += __popc(radial_mask);

    if (stage_angular) {
      const bool angular_accept = valid_neighbor && distance_sq <= angular_cutoff_sq;
      const unsigned int angular_mask = __ballot_sync(kFullWarpMask, angular_accept);
      if (angular_accept) {
        const int offset =
            angular_count + __popc(angular_mask & ((1u << lane) - 1u));
        if (offset < angular_capacity) {
          angular_tile[warp * angular_capacity + offset] = neighbor;
        } else if (check_overflow) {
          atomicOr(overflow, kLammpsOverflowAngularCapacity);
        }
      }
      angular_count += __popc(angular_mask);
    }
  }
  if (lane == 0) {
    radial_tile_counts[warp] =
        radial_count < radial_capacity ? radial_count : radial_capacity;
    angular_tile_counts[warp] =
        angular_count < angular_capacity ? angular_count : angular_capacity;
    if (valid_row) {
      radial_counts[atom] = radial_tile_counts[warp];
      if (stage_angular) {
        angular_counts[atom] = angular_tile_counts[warp];
      }
    }
    if (check_overflow) {
      atomicMax(overflow + 1, radial_count);
      atomicMax(overflow + 2, angular_count);
    }
  }
  __syncthreads();

  const int radial_tile_items = kStagingWarpsPerBlock * radial_capacity;
  for (int item = threadIdx.x; item < radial_tile_items; item += blockDim.x) {
    const int slot = item / kStagingWarpsPerBlock;
    const int tile_warp = item - slot * kStagingWarpsPerBlock;
    const int tile_active = blockIdx.x * kStagingWarpsPerBlock + tile_warp;
    if (tile_active < active_count && slot < radial_tile_counts[tile_warp]) {
      const int tile_atom = active_atom_indices[tile_active];
      radial_neighbors[tile_atom + atom_stride * slot] =
          radial_tile[tile_warp * radial_capacity + slot];
    }
  }
  if (stage_angular) {
    const int angular_tile_items = kStagingWarpsPerBlock * angular_capacity;
    for (int item = threadIdx.x; item < angular_tile_items; item += blockDim.x) {
      const int slot = item / kStagingWarpsPerBlock;
      const int tile_warp = item - slot * kStagingWarpsPerBlock;
      const int tile_active = blockIdx.x * kStagingWarpsPerBlock + tile_warp;
      if (tile_active < active_count && slot < angular_tile_counts[tile_warp]) {
        const int tile_atom = active_atom_indices[tile_active];
        angular_neighbors[tile_atom + atom_stride * slot] =
            angular_tile[tile_warp * angular_capacity + slot];
      }
    }
  }
}

__global__ void count_lammps_device_neighbors_compact(
    int active_count,
    int local_atom_count,
    int valid_atom_count,
    int neighbor_rows,
    int numneigh_length,
    int input_neighbor_capacity,
    int stage_angular,
    const int* active_atom_indices,
    const int* row_counts,
    const int* neighbors,
    const int* neighbor_owner,
    int neighbor_atom_stride,
    int neighbor_slot_stride,
    const double* positions,
    int position_atom_stride,
    int position_component_stride,
    double radial_cutoff_sq,
    double angular_cutoff_sq,
    int* overflow) {
  const int active = blockIdx.x * blockDim.x + threadIdx.x;
  if (active >= active_count) {
    return;
  }
  const int atom = active_atom_indices[active];
  if (atom < 0 || atom >= local_atom_count) {
    atomicOr(overflow, kLammpsOverflowActiveAtom);
    return;
  }
  if (atom >= neighbor_rows || atom >= numneigh_length) {
    atomicOr(overflow, kLammpsOverflowInputRowCount);
    return;
  }
  const int count = row_counts[atom];
  if (count < 0 || count > input_neighbor_capacity) {
    atomicOr(overflow, kLammpsOverflowInputRowCount);
  }
  const int safe_count =
      count < 0 ? 0 : (count > input_neighbor_capacity ? input_neighbor_capacity : count);
  const double xi = positions[atom * position_atom_stride];
  const double yi =
      positions[atom * position_atom_stride + position_component_stride];
  const double zi =
      positions[atom * position_atom_stride + 2 * position_component_stride];
  (void)neighbor_owner;
  int radial_count = 0;
  int angular_count = 0;
  for (int slot = 0; slot < safe_count; ++slot) {
    const int neighbor =
        neighbors[atom * neighbor_atom_stride + slot * neighbor_slot_stride] &
        kLammpsNeighborMask;
    if (neighbor == atom) {
      continue;
    }
    if (neighbor < 0 || neighbor >= valid_atom_count) {
      atomicOr(overflow, kLammpsOverflowNeighborIndex);
      continue;
    }
    const double dx = positions[neighbor * position_atom_stride] - xi;
    const double dy =
        positions[neighbor * position_atom_stride + position_component_stride] -
        yi;
    const double dz =
        positions[neighbor * position_atom_stride +
                  2 * position_component_stride] -
        zi;
    const double distance_sq = dx * dx + dy * dy + dz * dz;
    if (distance_sq <= radial_cutoff_sq) {
      ++radial_count;
    }
    if (stage_angular && distance_sq <= angular_cutoff_sq) {
      ++angular_count;
    }
  }
  atomicMax(overflow + 1, radial_count);
  atomicMax(overflow + 2, angular_count);
}

void validate_batch_for_device_staging(
    const NepaStructureBatch& batch,
    const DeviceWorkspaceView& view) {
  require(batch.num_structures > 0, "batch must contain structures");
  require(batch.total_atoms > 0, "batch must contain atoms");
  require(static_cast<std::size_t>(batch.total_atoms) <= view.atom_capacity,
          "batch exceeds workspace atom capacity");
  require(static_cast<std::size_t>(batch.num_structures) <= view.structure_capacity,
          "batch exceeds workspace structure capacity");
  require(batch.atom_counts != nullptr, "missing atom counts");
  require(batch.atom_offsets != nullptr, "missing atom offsets");
  require(batch.types != nullptr, "missing atom types");
  require(batch.positions_aos3 != nullptr, "missing atom positions");
  if (view.spins_soa3 != nullptr) {
    require(batch.spins_aos3 != nullptr, "spin model requires atom spins");
  }
  require(batch.boxes_row_major9 != nullptr, "missing boxes");
  require(view.types != nullptr, "workspace missing types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.atom_to_structure != nullptr, "workspace missing atom_to_structure");
  require(view.structure_atom_counts != nullptr, "workspace missing structure counts");
  require(view.structure_atom_offsets != nullptr, "workspace missing structure offsets");
  require(view.boxes_row_major9 != nullptr, "workspace missing boxes");
  require(view.box_inverse_row_major9 != nullptr, "workspace missing box inverses");
  require(view.pbc_flags3 != nullptr, "workspace missing pbc flags");

  std::vector<int> covered(static_cast<std::size_t>(batch.total_atoms), 0);
  for (int structure = 0; structure < batch.num_structures; ++structure) {
    const int atom_count = batch.atom_counts[structure];
    const int atom_offset = batch.atom_offsets[structure];
    require(atom_count > 0, "structure atom count must be positive");
    require(atom_offset >= 0, "structure atom offset must be non-negative");
    require(
        atom_offset + atom_count <= batch.total_atoms,
        "structure atom range exceeds batch atom count");
    for (int local = 0; local < atom_count; ++local) {
      const int atom = atom_offset + local;
      require(covered[static_cast<std::size_t>(atom)] == 0,
              "overlapping structure atom ranges");
      covered[static_cast<std::size_t>(atom)] = 1;
    }
  }
  for (int atom = 0; atom < batch.total_atoms; ++atom) {
    require(covered[static_cast<std::size_t>(atom)] != 0,
            "batch atom is not covered by any structure");
  }
}

int lammps_atom_capacity(const NepaLammpsNeighborInput& input) {
  int atom_capacity = input.nlocal;
  for (int active = 0; active < input.inum; ++active) {
    const int atom = input.ilist[active];
    require(atom >= 0, "negative active atom index");
    atom_capacity = std::max(atom_capacity, atom + 1);
    const int neighbor_count = input.numneigh[atom];
    require(neighbor_count >= 0, "negative neighbor count");
    if (neighbor_count > 0) {
      require(input.firstneigh[atom] != nullptr, "missing LAMMPS neighbor row");
    }
    for (int slot = 0; slot < neighbor_count; ++slot) {
      const int neighbor = input.firstneigh[atom][slot] & kLammpsNeighborMask;
      require(neighbor >= 0, "negative neighbor atom index");
      atom_capacity = std::max(atom_capacity, neighbor + 1);
    }
  }
  return atom_capacity;
}

std::vector<int> flatten_lammps_neighbors(
    const NepaLammpsNeighborInput& input,
    int atom_capacity,
    int neighbor_capacity) {
  std::vector<int> row_major(
      static_cast<std::size_t>(atom_capacity) *
          static_cast<std::size_t>(neighbor_capacity),
      0);
  for (int active = 0; active < input.inum; ++active) {
    const int atom = input.ilist[active];
    const int neighbor_count = input.numneigh[atom];
    require(
        neighbor_count <= neighbor_capacity,
        "neighbor count exceeds CUDA workspace capacity");
    for (int slot = 0; slot < neighbor_count; ++slot) {
      const int neighbor = input.firstneigh[atom][slot] & kLammpsNeighborMask;
      require(neighbor >= 0 && neighbor < atom_capacity, "neighbor atom out of range");
      row_major[static_cast<std::size_t>(atom) *
                    static_cast<std::size_t>(neighbor_capacity) +
                static_cast<std::size_t>(slot)] = neighbor;
    }
  }
  return row_major;
}

}  // namespace

void stage_batch_on_device(
    const NepaStructureBatch& batch,
    DeviceWorkspace& workspace) {
  const DeviceWorkspaceView view = workspace.view();
  validate_batch_for_device_staging(batch, view);

  const std::vector<int> types(batch.types, batch.types + batch.total_atoms);
  const std::vector<int> atom_counts(
      batch.atom_counts,
      batch.atom_counts + batch.num_structures);
  const std::vector<int> atom_offsets(
      batch.atom_offsets,
      batch.atom_offsets + batch.num_structures);
  const std::vector<double> boxes(
      batch.boxes_row_major9,
      batch.boxes_row_major9 + static_cast<std::size_t>(batch.num_structures) * 9);
  std::vector<double> box_inverses(static_cast<std::size_t>(batch.num_structures) * 9);
  for (int structure = 0; structure < batch.num_structures; ++structure) {
    require(
        invert_row_major3(
            boxes.data() + static_cast<std::size_t>(structure) * 9,
            box_inverses.data() + static_cast<std::size_t>(structure) * 9),
        "singular simulation box");
  }
  std::vector<int> pbc(
      static_cast<std::size_t>(batch.num_structures) * 3,
      0);
  if (batch.pbc_flags3 != nullptr) {
    pbc.assign(
        batch.pbc_flags3,
        batch.pbc_flags3 + static_cast<std::size_t>(batch.num_structures) * 3);
  }

  check_cuda(
      cudaMemcpy(
          view.types,
          types.data(),
          types.size() * sizeof(int),
          cudaMemcpyHostToDevice),
      "copy batch types");
  check_cuda(
      cudaMemcpy(
          view.structure_atom_counts,
          atom_counts.data(),
          atom_counts.size() * sizeof(int),
          cudaMemcpyHostToDevice),
      "copy structure atom counts");
  check_cuda(
      cudaMemcpy(
          view.structure_atom_offsets,
          atom_offsets.data(),
          atom_offsets.size() * sizeof(int),
          cudaMemcpyHostToDevice),
      "copy structure atom offsets");
  check_cuda(
      cudaMemcpy(
          view.boxes_row_major9,
          boxes.data(),
          boxes.size() * sizeof(double),
          cudaMemcpyHostToDevice),
      "copy boxes");
  check_cuda(
      cudaMemcpy(
          view.box_inverse_row_major9,
          box_inverses.data(),
          box_inverses.size() * sizeof(double),
          cudaMemcpyHostToDevice),
      "copy box inverses");
  check_cuda(
      cudaMemcpy(
          view.pbc_flags3,
          pbc.data(),
          pbc.size() * sizeof(int),
          cudaMemcpyHostToDevice),
      "copy pbc flags");

  std::vector<double> positions_aos3(
      batch.positions_aos3,
      batch.positions_aos3 + static_cast<std::size_t>(batch.total_atoms) * 3);
  double* positions_aos3_device = upload_temp(positions_aos3, "upload positions_aos3");
  std::vector<double> spins_aos3;
  double* spins_aos3_device = nullptr;
  if (view.spins_soa3 != nullptr) {
    spins_aos3.assign(
        batch.spins_aos3,
        batch.spins_aos3 + static_cast<std::size_t>(batch.total_atoms) * 3);
    spins_aos3_device = upload_temp(spins_aos3, "upload spins_aos3");
  }
  int* atom_counts_device = upload_temp(atom_counts, "upload atom_counts");
  int* atom_offsets_device = upload_temp(atom_offsets, "upload atom_offsets");

  try {
    const int atom_blocks = (batch.total_atoms + kBlockSize - 1) / kBlockSize;
    stage_positions_aos_to_soa<<<atom_blocks, kBlockSize>>>(
        batch.total_atoms,
        static_cast<int>(view.atom_capacity),
        positions_aos3_device,
        view.positions_soa3);
    check_cuda(cudaGetLastError(), "stage positions AoS to SoA");
    if (view.spins_soa3 != nullptr) {
      stage_positions_aos_to_soa<<<atom_blocks, kBlockSize>>>(
          batch.total_atoms,
          static_cast<int>(view.atom_capacity),
          spins_aos3_device,
          view.spins_soa3);
      check_cuda(cudaGetLastError(), "stage spins AoS to SoA");
    }

    stage_atom_to_structure<<<batch.num_structures, kBlockSize>>>(
        batch.num_structures,
        atom_counts_device,
        atom_offsets_device,
        view.atom_to_structure);
    check_cuda(cudaGetLastError(), "stage atom_to_structure");
    check_cuda(cudaDeviceSynchronize(), "synchronize batch staging");
  } catch (...) {
    free_temp(positions_aos3_device);
    free_temp(spins_aos3_device);
    free_temp(atom_counts_device);
    free_temp(atom_offsets_device);
    throw;
  }

  free_temp(positions_aos3_device);
  free_temp(spins_aos3_device);
  free_temp(atom_counts_device);
  free_temp(atom_offsets_device);
}

void stage_lammps_external_neighbors_on_device(
    const NepaLammpsNeighborInput& input,
    const ModelProtocol& protocol,
    DeviceWorkspace& workspace) {
  require(input.nlocal >= 0, "negative nlocal");
  require(input.inum >= 0, "negative inum");
  require(input.ilist != nullptr, "missing ilist");
  require(input.numneigh != nullptr, "missing numneigh");
  require(input.firstneigh != nullptr, "missing firstneigh");
  require(input.types != nullptr, "missing types");
  require(input.type_map != nullptr, "missing type_map");
  require(input.positions != nullptr, "missing positions");
  if (protocol.spin_mode != 0) {
    require(input.spins != nullptr, "spin model requires LAMMPS spins");
  }

  const DeviceWorkspaceView view = workspace.view();
  const int atom_capacity = lammps_atom_capacity(input);
  require(static_cast<std::size_t>(atom_capacity) <= view.atom_capacity,
          "LAMMPS input exceeds workspace atom capacity");
  require(static_cast<std::size_t>(input.inum) <= view.active_atom_capacity,
          "LAMMPS input exceeds workspace active atom capacity");

  int max_lammps_type = 0;
  std::vector<int> lammps_types(static_cast<std::size_t>(atom_capacity), 0);
  std::vector<double> positions_aos3(static_cast<std::size_t>(atom_capacity) * 3, 0.0);
  std::vector<double> spins_aos3(static_cast<std::size_t>(atom_capacity) * 3, 0.0);
  for (int atom = 0; atom < atom_capacity; ++atom) {
    require(input.positions[atom] != nullptr, "missing LAMMPS position row");
    const int lammps_type = input.types[atom];
    require(lammps_type >= 0, "negative LAMMPS atom type");
    max_lammps_type = std::max(max_lammps_type, lammps_type);
    lammps_types[static_cast<std::size_t>(atom)] = lammps_type;
    positions_aos3[3 * static_cast<std::size_t>(atom)] = input.positions[atom][0];
    positions_aos3[3 * static_cast<std::size_t>(atom) + 1] = input.positions[atom][1];
    positions_aos3[3 * static_cast<std::size_t>(atom) + 2] = input.positions[atom][2];
    if (protocol.spin_mode != 0) {
      require(input.spins[atom] != nullptr, "missing LAMMPS spin row");
      const double spin_scale = input.spins[atom][3];
      spins_aos3[3 * static_cast<std::size_t>(atom)] =
          input.spins[atom][0] * spin_scale;
      spins_aos3[3 * static_cast<std::size_t>(atom) + 1] =
          input.spins[atom][1] * spin_scale;
      spins_aos3[3 * static_cast<std::size_t>(atom) + 2] =
          input.spins[atom][2] * spin_scale;
    }
  }

  std::vector<int> type_map(
      input.type_map,
      input.type_map + static_cast<std::size_t>(max_lammps_type + 1));
  for (int lammps_type : lammps_types) {
    require(type_map[static_cast<std::size_t>(lammps_type)] >= 0,
            "unmapped LAMMPS atom type");
  }

  std::vector<int> active_atom_indices(input.ilist, input.ilist + input.inum);
  std::vector<int> row_counts(static_cast<std::size_t>(atom_capacity), 0);
  for (int active = 0; active < input.inum; ++active) {
    const int atom = input.ilist[active];
    require(atom >= 0 && atom < atom_capacity, "active atom out of range");
    row_counts[static_cast<std::size_t>(atom)] = input.numneigh[atom];
  }

  const std::vector<int> radial_neighbors = flatten_lammps_neighbors(
      input,
      atom_capacity,
      protocol.neighbor_capacity_radial);
  const std::vector<int> angular_neighbors = flatten_lammps_neighbors(
      input,
      atom_capacity,
      protocol.neighbor_capacity_angular);

  check_cuda(
      cudaMemcpy(
          view.active_atom_indices,
          active_atom_indices.data(),
          active_atom_indices.size() * sizeof(int),
          cudaMemcpyHostToDevice),
      "copy active atom indices");
  check_cuda(
      cudaMemset(view.nn_radial, 0, view.atom_capacity * sizeof(int)),
      "zero radial neighbor counts");
  check_cuda(
      cudaMemset(view.nn_angular, 0, view.atom_capacity * sizeof(int)),
      "zero angular neighbor counts");
  check_cuda(
      cudaMemset(
          view.nl_radial_slot_major,
          0,
          view.atom_capacity *
              static_cast<std::size_t>(protocol.neighbor_capacity_radial) *
              sizeof(int)),
      "zero radial neighbor list");
  check_cuda(
      cudaMemset(
          view.nl_angular_slot_major,
          0,
          view.atom_capacity *
              static_cast<std::size_t>(protocol.neighbor_capacity_angular) *
              sizeof(int)),
      "zero angular neighbor list");

  int* lammps_types_device = upload_temp(lammps_types, "upload lammps types");
  int* type_map_device = upload_temp(type_map, "upload type map");
  double* positions_aos3_device = upload_temp(positions_aos3, "upload lammps positions");
  double* spins_aos3_device = protocol.spin_mode != 0
      ? upload_temp(spins_aos3, "upload lammps spins")
      : nullptr;
  int* active_device = upload_temp(active_atom_indices, "upload active atoms");
  int* counts_device = upload_temp(row_counts, "upload neighbor counts");
  int* radial_device = upload_temp(radial_neighbors, "upload radial neighbors");
  int* angular_device = upload_temp(angular_neighbors, "upload angular neighbors");

  try {
    const int atom_blocks = (atom_capacity + kBlockSize - 1) / kBlockSize;
    stage_lammps_types<<<atom_blocks, kBlockSize>>>(
        atom_capacity,
        lammps_types_device,
        type_map_device,
        view.types);
    check_cuda(cudaGetLastError(), "stage LAMMPS types");

    stage_positions_aos_to_soa<<<atom_blocks, kBlockSize>>>(
        atom_capacity,
        static_cast<int>(view.atom_capacity),
        positions_aos3_device,
        view.positions_soa3);
    check_cuda(cudaGetLastError(), "stage LAMMPS positions");
    if (protocol.spin_mode != 0) {
      stage_positions_aos_to_soa<<<atom_blocks, kBlockSize>>>(
          atom_capacity,
          static_cast<int>(view.atom_capacity),
          spins_aos3_device,
          view.spins_soa3);
      check_cuda(cudaGetLastError(), "stage LAMMPS spins");
    }

    const int active_blocks = (input.inum + kBlockSize - 1) / kBlockSize;
    stage_lammps_neighbors_slot_major<<<active_blocks, kBlockSize>>>(
        input.inum,
        atom_capacity,
        protocol.neighbor_capacity_radial,
        active_device,
        counts_device,
        radial_device,
        view.nn_radial,
        view.nl_radial_slot_major);
    check_cuda(cudaGetLastError(), "stage radial LAMMPS neighbors");

    stage_lammps_neighbors_slot_major<<<active_blocks, kBlockSize>>>(
        input.inum,
        atom_capacity,
        protocol.neighbor_capacity_angular,
        active_device,
        counts_device,
        angular_device,
        view.nn_angular,
        view.nl_angular_slot_major);
    check_cuda(cudaGetLastError(), "stage angular LAMMPS neighbors");
    check_cuda(cudaDeviceSynchronize(), "synchronize LAMMPS staging");
  } catch (...) {
    free_temp(lammps_types_device);
    free_temp(type_map_device);
    free_temp(positions_aos3_device);
    free_temp(spins_aos3_device);
    free_temp(active_device);
    free_temp(counts_device);
    free_temp(radial_device);
    free_temp(angular_device);
    throw;
  }

  free_temp(lammps_types_device);
  free_temp(type_map_device);
  free_temp(positions_aos3_device);
  free_temp(spins_aos3_device);
  free_temp(active_device);
  free_temp(counts_device);
  free_temp(radial_device);
  free_temp(angular_device);
}

LammpsDeviceNeighborCounts count_lammps_device_neighbors_on_device(
    const NepaLammpsDeviceNeighborInput& input,
    const ModelProtocol& protocol) {
  require(input.nlocal >= 0, "negative nlocal");
  require(input.nall >= input.nlocal, "nall must be at least nlocal");
  require(input.inum > 0, "device LAMMPS input must have active atoms");
  require(input.max_neighbors >= 0, "negative max_neighbors");
  require(input.neighbor_rows > 0, "invalid device neighbor row count");
  require(input.numneigh_length > 0, "invalid device numneigh length");
  require(input.ilist != nullptr, "missing device ilist");
  require(input.numneigh != nullptr, "missing device numneigh");
  require(input.neighbors != nullptr, "missing device neighbors");
  require(input.neighbor_atom_stride > 0, "invalid neighbor atom stride");
  require(input.neighbor_slot_stride > 0, "invalid neighbor slot stride");
  require(input.positions != nullptr, "missing device positions");
  require(input.position_atom_stride > 0, "invalid position atom stride");
  require(input.position_component_stride > 0, "invalid position component stride");
  if (protocol.spin_mode != 0) {
    require(input.spins != nullptr, "spin model requires device spins");
    require(input.spin_atom_stride > 0, "invalid spin atom stride");
    require(input.spin_component_stride > 0, "invalid spin component stride");
  }
  require(input.nlocal <= input.neighbor_rows,
          "device LAMMPS local atoms exceed neighbor rows");
  require(input.nlocal <= input.numneigh_length,
          "device LAMMPS local atoms exceed numneigh length");

  int* overflow = nullptr;
  check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&overflow), 3 * sizeof(int)),
      "allocate compact neighbor count scratch");
  try {
    check_cuda(
        cudaMemset(overflow, 0, 3 * sizeof(int)),
        "zero compact neighbor count scratch");
    const bool stage_angular = protocol.body_channels.channel_count() > 0;
    const int active_blocks = (input.inum + kBlockSize - 1) / kBlockSize;
    count_lammps_device_neighbors_compact<<<active_blocks, kBlockSize>>>(
        input.inum,
        input.nlocal,
        input.nall,
        input.neighbor_rows,
        input.numneigh_length,
        input.max_neighbors,
        stage_angular ? 1 : 0,
        input.ilist,
        input.numneigh,
        input.neighbors,
        input.neighbor_owner,
        input.neighbor_atom_stride,
        input.neighbor_slot_stride,
        input.positions,
        input.position_atom_stride,
        input.position_component_stride,
        std::max(protocol.cutoff_radial, protocol.zbl_outer) *
            std::max(protocol.cutoff_radial, protocol.zbl_outer),
        protocol.cutoff_angular * protocol.cutoff_angular,
        overflow);
    check_cuda(cudaGetLastError(), "count compact device LAMMPS neighbors");
    int host_overflow[3] = {};
    check_cuda(
        cudaMemcpy(
            host_overflow,
            overflow,
            3 * sizeof(int),
            cudaMemcpyDeviceToHost),
        "copy compact device LAMMPS neighbor counts");
    if (host_overflow[0] != 0) {
      std::string message = "invalid device LAMMPS neighbor input:";
      if ((host_overflow[0] & kLammpsOverflowActiveAtom) != 0) {
        message += " active atom index out of range";
      }
      if ((host_overflow[0] & kLammpsOverflowInputRowCount) != 0) {
        message += " input row count out of range";
      }
      if ((host_overflow[0] & kLammpsOverflowNeighborIndex) != 0) {
        message += " neighbor index out of range";
      }
      cudaFree(overflow);
      overflow = nullptr;
      throw std::runtime_error(message);
    }
    LammpsDeviceNeighborCounts counts{};
    counts.max_radial = host_overflow[1];
    counts.max_angular = host_overflow[2];
    cudaFree(overflow);
    overflow = nullptr;
    return counts;
  } catch (...) {
    cudaFree(overflow);
    throw;
  }
}

LammpsDeviceNeighborCounts stage_lammps_device_neighbors_on_device(
    const NepaLammpsDeviceNeighborInput& input,
    const ModelProtocol& protocol,
    DeviceWorkspace& workspace,
    bool check_overflow) {
  require(input.nlocal >= 0, "negative nlocal");
  require(input.nall >= input.nlocal, "nall must be at least nlocal");
  require(input.inum > 0, "device LAMMPS input must have active atoms");
  require(input.max_neighbors >= 0, "negative max_neighbors");
  require(input.neighbor_rows > 0, "invalid device neighbor row count");
  require(input.numneigh_length > 0, "invalid device numneigh length");
  require(input.ilist != nullptr, "missing device ilist");
  require(input.numneigh != nullptr, "missing device numneigh");
  require(input.neighbors != nullptr, "missing device neighbors");
  require(input.neighbor_atom_stride > 0, "invalid neighbor atom stride");
  require(input.neighbor_slot_stride > 0, "invalid neighbor slot stride");
  require(input.types != nullptr, "missing device types");
  require(input.positions != nullptr, "missing device positions");
  require(input.position_atom_stride > 0, "invalid position atom stride");
  require(input.position_component_stride > 0, "invalid position component stride");

  workspace.reset_runtime_overrides();
  const DeviceWorkspaceView view = workspace.view();
  require(view.neighbor_source == 1, "device LAMMPS staging requires external workspace");
  require(static_cast<std::size_t>(input.nall) <= view.atom_capacity,
          "device LAMMPS input exceeds workspace atom capacity");
  require(static_cast<std::size_t>(input.inum) <= view.active_atom_capacity,
          "device LAMMPS input exceeds active atom capacity");
  require(input.nlocal <= input.neighbor_rows,
          "device LAMMPS local atoms exceed neighbor rows");
  require(input.nlocal <= input.numneigh_length,
          "device LAMMPS local atoms exceed numneigh length");
  require(view.types != nullptr, "workspace missing types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  if (protocol.spin_mode != 0) {
    require(view.spins_soa3 != nullptr, "workspace missing spins");
  }
  require(view.nn_radial != nullptr, "workspace missing radial counts");
  require(view.neighbor_overflow != nullptr, "workspace missing neighbor overflow flag");
  const bool stage_angular = protocol.body_channels.channel_count() > 0;
  require(view.nl_radial_slot_major != nullptr,
          "workspace missing radial neighbors");
  if (stage_angular) {
    require(view.nn_angular != nullptr, "workspace missing angular counts");
    require(view.nl_angular_slot_major != nullptr,
            "workspace missing angular neighbors");
  }

  if (check_overflow) {
    check_cuda(cudaMemset(view.neighbor_overflow, 0, 3 * sizeof(int)),
               "zero device LAMMPS staging overflow");
  }

  const int atom_blocks = (input.nall + kBlockSize - 1) / kBlockSize;
  if (input.type_map != nullptr) {
    require(input.type_map_length > 0, "invalid device LAMMPS type map length");
    stage_lammps_device_types<<<atom_blocks, kBlockSize>>>(
        input.nall,
        input.types,
        input.type_map,
        input.type_map_length,
        view.types,
        view.neighbor_overflow);
    check_cuda(cudaGetLastError(), "stage device LAMMPS types");
  } else {
    workspace.override_types(const_cast<int*>(input.types));
  }

  if (input.position_atom_stride == 1 &&
      static_cast<std::size_t>(input.position_component_stride) ==
          view.atom_capacity) {
    workspace.override_positions_soa3(
        const_cast<double*>(input.positions));
  } else {
    stage_lammps_device_positions<<<atom_blocks, kBlockSize>>>(
        input.nall,
        static_cast<int>(view.atom_capacity),
        input.positions,
        input.position_atom_stride,
        input.position_component_stride,
        view.positions_soa3);
    check_cuda(cudaGetLastError(), "stage device LAMMPS positions");
  }
  const double* staged_positions =
      input.position_atom_stride == 1 &&
              static_cast<std::size_t>(input.position_component_stride) ==
                  view.atom_capacity
          ? input.positions
          : view.positions_soa3;
  if (protocol.spin_mode != 0) {
    stage_lammps_device_spins<<<atom_blocks, kBlockSize>>>(
        input.nall,
        static_cast<int>(view.atom_capacity),
        input.spins,
        input.spin_atom_stride,
        input.spin_component_stride,
        view.spins_soa3);
    check_cuda(cudaGetLastError(), "stage device LAMMPS spins");
  }

  const int active_blocks =
      (input.inum + kStagingWarpsPerBlock - 1) / kStagingWarpsPerBlock;
  if (check_overflow) {
    check_cuda(cudaMemset(view.neighbor_overflow, 0, 3 * sizeof(int)),
               "zero device LAMMPS fill overflow");
  }
  const std::size_t neighbor_tile_ints =
      kStagingWarpsPerBlock *
          static_cast<std::size_t>(protocol.neighbor_capacity_radial) +
      (stage_angular
           ? kStagingWarpsPerBlock *
               static_cast<std::size_t>(protocol.neighbor_capacity_angular)
           : 0) +
      2 * kStagingWarpsPerBlock;
  stage_lammps_device_neighbors_dual_slot_major
      <<<active_blocks, kBlockSize, neighbor_tile_ints * sizeof(int)>>>(
      input.inum,
      static_cast<int>(view.atom_capacity),
      input.nlocal,
      input.nall,
      input.neighbor_rows,
      input.numneigh_length,
      protocol.neighbor_capacity_radial,
      protocol.neighbor_capacity_angular,
      input.max_neighbors,
      stage_angular ? 1 : 0,
      check_overflow ? 1 : 0,
      input.ilist,
      input.numneigh,
      input.neighbors,
      input.neighbor_owner,
      input.neighbor_atom_stride,
      input.neighbor_slot_stride,
      staged_positions,
      static_cast<int>(view.atom_capacity),
      std::max(protocol.cutoff_radial, protocol.zbl_outer) *
          std::max(protocol.cutoff_radial, protocol.zbl_outer),
      protocol.cutoff_angular * protocol.cutoff_angular,
      view.nn_radial,
      view.nl_radial_slot_major,
      stage_angular ? view.nn_angular : nullptr,
      stage_angular ? view.nl_angular_slot_major : nullptr,
      view.neighbor_overflow);
  check_cuda(cudaGetLastError(), "stage device LAMMPS neighbors");

  if (!check_overflow) {
    return {protocol.neighbor_capacity_radial, protocol.neighbor_capacity_angular};
  }

  int host_overflow[3] = {};
  check_cuda(
      cudaMemcpy(
          host_overflow,
          view.neighbor_overflow,
          3 * sizeof(int),
          cudaMemcpyDeviceToHost),
      "copy device LAMMPS staging overflow");
  if (host_overflow[0] != 0) {
    std::string message = "invalid device LAMMPS neighbor input:";
    if ((host_overflow[0] & kLammpsOverflowTypeRange) != 0) {
      message += " type out of range";
    }
    if ((host_overflow[0] & kLammpsOverflowTypeMap) != 0) {
      message += " unmapped type";
    }
    if ((host_overflow[0] & kLammpsOverflowActiveAtom) != 0) {
      message += " active atom index out of range";
    }
    if ((host_overflow[0] & kLammpsOverflowInputRowCount) != 0) {
      message += " input row count out of range";
    }
    if ((host_overflow[0] & kLammpsOverflowNeighborIndex) != 0) {
      message += " neighbor index out of range";
    }
    if ((host_overflow[0] & kLammpsOverflowRadialCapacity) != 0) {
      message += " radial neighbor count exceeds workspace capacity";
    }
    if ((host_overflow[0] & kLammpsOverflowAngularCapacity) != 0) {
      message += " angular neighbor count exceeds workspace capacity";
    }
    message += " max_radial=" + std::to_string(host_overflow[1]);
    message += " max_angular=" + std::to_string(host_overflow[2]);
    message += " capacity_radial=" +
        std::to_string(protocol.neighbor_capacity_radial);
    message += " capacity_angular=" +
        std::to_string(protocol.neighbor_capacity_angular);
    message += " cutoff_radial=" + std::to_string(protocol.cutoff_radial);
    message += " cutoff_angular=" + std::to_string(protocol.cutoff_angular);
    throw std::runtime_error(message);
  }
  return {host_overflow[1], host_overflow[2]};
}

}  // namespace nep_adapters::cuda_backend
