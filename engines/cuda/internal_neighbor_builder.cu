#include "device_operations.hpp"
#include "simulation_box_device.cuh"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nep_adapters::cuda_backend {
namespace {

constexpr int kBlockSize = 32;

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

double reciprocal_row_norm(const SimulationBox& box, int row) {
  const double x = box.cart_to_frac[3 * row + 0];
  const double y = box.cart_to_frac[3 * row + 1];
  const double z = box.cart_to_frac[3 * row + 2];
  return std::sqrt(x * x + y * y + z * z);
}

int cell_dim_for_fractional_axis(const SimulationBox& box, int axis, double cutoff) {
  const double reciprocal_norm = reciprocal_row_norm(box, axis);
  require(reciprocal_norm > 0.0, "box reciprocal axis must be positive");
  require(cutoff > 0.0, "cutoff must be positive");
  return std::max(1, static_cast<int>(std::floor(1.0 / (cutoff * reciprocal_norm))));
}

std::array<int, 3> fit_cell_dims_to_capacity(
    int nx,
    int ny,
    int nz,
    std::size_t capacity) {
  require(capacity > 0, "cell capacity must be positive");
  std::array<int, 3> dims = {nx, ny, nz};
  while (static_cast<std::size_t>(dims[0]) *
             static_cast<std::size_t>(dims[1]) *
             static_cast<std::size_t>(dims[2]) >
         capacity) {
    int axis = 0;
    if (dims[1] > dims[axis]) {
      axis = 1;
    }
    if (dims[2] > dims[axis]) {
      axis = 2;
    }
    require(dims[axis] > 1, "cannot fit cell grid into workspace capacity");
    --dims[axis];
  }
  return dims;
}

__device__ int wrap_index(int value, int extent) {
  if (value < 0) {
    return value + extent;
  }
  if (value >= extent) {
    return value - extent;
  }
  return value;
}

__device__ bool duplicate_pbc_offset(int offset, int extent) {
  if (extent == 1) {
    return offset != 0;
  }
  if (extent == 2) {
    return offset > 0;
  }
  return false;
}

__device__ double reciprocal_row_norm_device(const SimulationBox& box, int row) {
  const double x = box.cart_to_frac[3 * row + 0];
  const double y = box.cart_to_frac[3 * row + 1];
  const double z = box.cart_to_frac[3 * row + 2];
  return sqrt(x * x + y * y + z * z);
}

__device__ int cell_dim_for_fractional_axis_device(
    const SimulationBox& box,
    int axis,
    double cutoff) {
  const double reciprocal_norm = reciprocal_row_norm_device(box, axis);
  int dim = 1;
  if (reciprocal_norm > 0.0 && cutoff > 0.0) {
    dim = static_cast<int>(floor(1.0 / (cutoff * reciprocal_norm)));
  }
  return dim > 1 ? dim : 1;
}

__device__ void fit_cell_dims_to_capacity_device(
    int capacity,
    int& nx,
    int& ny,
    int& nz,
    int* overflow) {
  while (nx * ny * nz > capacity) {
    int axis = 0;
    if (ny > nx) {
      axis = 1;
    }
    if ((axis == 0 && nz > nx) || (axis == 1 && nz > ny)) {
      axis = 2;
    }
    if ((axis == 0 && nx <= 1) ||
        (axis == 1 && ny <= 1) ||
        (axis == 2 && nz <= 1)) {
      atomicExch(overflow, 1);
      return;
    }
    if (axis == 0) {
      --nx;
    } else if (axis == 1) {
      --ny;
    } else {
      --nz;
    }
  }
}

__device__ __forceinline__ int orthorhombic_component_to_cell(
    double coordinate,
    double inverse_length,
    int dim,
    int pbc) {
  double scaled = coordinate * inverse_length;
  if (pbc) {
    scaled -= floor(scaled);
  }
  int cell = static_cast<int>(floor(scaled * dim));
  if (cell < 0) {
    cell = 0;
  }
  if (cell >= dim) {
    cell = dim - 1;
  }
  return cell;
}

__device__ __forceinline__ void orthorhombic_minimum_image_delta(
    double dx,
    double dy,
    double dz,
    double lx,
    double ly,
    double lz,
    double inv_lx,
    double inv_ly,
    double inv_lz,
    int pbc_x,
    int pbc_y,
    int pbc_z,
    float& x12,
    float& y12,
    float& z12) {
  if (pbc_x) {
    dx -= nearbyint(dx * inv_lx) * lx;
  }
  if (pbc_y) {
    dy -= nearbyint(dy * inv_ly) * ly;
  }
  if (pbc_z) {
    dz -= nearbyint(dz * inv_lz) * lz;
  }
  x12 = static_cast<float>(dx);
  y12 = static_cast<float>(dy);
  z12 = static_cast<float>(dz);
}

__global__ void plan_batched_cells(
    int structure_count,
    double cutoff,
    const int* structure_atom_counts,
    const double* boxes_row_major9,
    const double* box_inverse_row_major9,
    const int* pbc_flags3,
    int* structure_cell_offsets,
    int* structure_cell_dims4,
    int* overflow) {
  const int structure = blockIdx.x * blockDim.x + threadIdx.x;
  if (structure >= structure_count) {
    return;
  }

  const int atom_count = structure_atom_counts[structure];
  if (atom_count <= 0) {
    atomicExch(overflow, 1);
    return;
  }
  const SimulationBox box = load_structure_box(
      structure,
      boxes_row_major9,
      box_inverse_row_major9,
      pbc_flags3);
  int nx = cell_dim_for_fractional_axis_device(box, 0, cutoff);
  int ny = cell_dim_for_fractional_axis_device(box, 1, cutoff);
  int nz = cell_dim_for_fractional_axis_device(box, 2, cutoff);
  fit_cell_dims_to_capacity_device(atom_count, nx, ny, nz, overflow);
  const int cell_count = nx * ny * nz;
  structure_cell_offsets[structure] = cell_count;
  structure_cell_dims4[4 * structure + 0] = nx;
  structure_cell_dims4[4 * structure + 1] = ny;
  structure_cell_dims4[4 * structure + 2] = nz;
  structure_cell_dims4[4 * structure + 3] = cell_count;
}

__global__ void assign_cells(
    int atom_count,
    int atom_stride,
    int nx,
    int ny,
    int nz,
    SimulationBox box,
    const double* positions_soa3,
    int* atom_cell,
    int* cell_counts) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const double x = positions_soa3[atom];
  const double y = positions_soa3[atom_stride + atom];
  const double z = positions_soa3[2 * atom_stride + atom];
  const int ix = fractional_component_to_cell(
      fractional_component(box, 0, x, y, z),
      nx,
      box.pbc[0]);
  const int iy = fractional_component_to_cell(
      fractional_component(box, 1, x, y, z),
      ny,
      box.pbc[1]);
  const int iz = fractional_component_to_cell(
      fractional_component(box, 2, x, y, z),
      nz,
      box.pbc[2]);
  const int cell = ix + nx * (iy + ny * iz);
  atom_cell[atom] = cell;
  atomicAdd(&cell_counts[cell], 1);
}

__global__ void assign_cells_batched(
    int atom_count,
    int atom_stride,
    const int* atom_to_structure,
    const int* structure_cell_offsets,
    const int* structure_cell_dims4,
    const double* boxes_row_major9,
    const double* box_inverse_row_major9,
    const int* pbc_flags3,
    const double* positions_soa3,
    int* atom_cell,
    int* cell_counts) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int structure = atom_to_structure[atom];
  const int* dims = structure_cell_dims4 + 4 * structure;
  const int nx = dims[0];
  const int ny = dims[1];
  const int nz = dims[2];
  const int cell_base = structure_cell_offsets[structure];
  const SimulationBox box = load_structure_box(
      structure,
      boxes_row_major9,
      box_inverse_row_major9,
      pbc_flags3);

  const double x = positions_soa3[atom];
  const double y = positions_soa3[atom_stride + atom];
  const double z = positions_soa3[2 * atom_stride + atom];
  const int ix = fractional_component_to_cell(
      fractional_component(box, 0, x, y, z),
      nx,
      box.pbc[0]);
  const int iy = fractional_component_to_cell(
      fractional_component(box, 1, x, y, z),
      ny,
      box.pbc[1]);
  const int iz = fractional_component_to_cell(
      fractional_component(box, 2, x, y, z),
      nz,
      box.pbc[2]);
  const int cell = cell_base + ix + nx * (iy + ny * iz);
  atom_cell[atom] = cell;
  atomicAdd(&cell_counts[cell], 1);
}

__global__ void assign_cells_batched_orthorhombic(
    int atom_count,
    int atom_stride,
    const int* atom_to_structure,
    const int* structure_cell_offsets,
    const int* structure_cell_dims4,
    const double* boxes_row_major9,
    const double* box_inverse_row_major9,
    const int* pbc_flags3,
    const double* positions_soa3,
    int* atom_cell,
    int* cell_counts) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int structure = atom_to_structure[atom];
  const int* dims = structure_cell_dims4 + 4 * structure;
  const int nx = dims[0];
  const int ny = dims[1];
  const int nz = dims[2];
  const int cell_base = structure_cell_offsets[structure];
  const int box_offset = 9 * structure;
  const int pbc_offset = 3 * structure;

  const double inv_lx = box_inverse_row_major9[box_offset + 0];
  const double inv_ly = box_inverse_row_major9[box_offset + 4];
  const double inv_lz = box_inverse_row_major9[box_offset + 8];
  const double x = positions_soa3[atom];
  const double y = positions_soa3[atom_stride + atom];
  const double z = positions_soa3[2 * atom_stride + atom];
  const int ix = orthorhombic_component_to_cell(
      x, inv_lx, nx, pbc_flags3[pbc_offset + 0]);
  const int iy = orthorhombic_component_to_cell(
      y, inv_ly, ny, pbc_flags3[pbc_offset + 1]);
  const int iz = orthorhombic_component_to_cell(
      z, inv_lz, nz, pbc_flags3[pbc_offset + 2]);
  const int cell = cell_base + ix + nx * (iy + ny * iz);
  atom_cell[atom] = cell;
  atomicAdd(&cell_counts[cell], 1);
}

__global__ void scatter_cells(
    int atom_count,
    const int* atom_cell,
    const int* cell_offsets,
    int* cell_fill,
    int* cell_atoms) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const int cell = atom_cell[atom];
  const int slot = atomicAdd(&cell_fill[cell], 1);
  cell_atoms[cell_offsets[cell] + slot] = atom;
}

__global__ void build_neighbors_from_cells(
    int atom_count,
    int atom_stride,
    int nx,
    int ny,
    int nz,
    SimulationBox box,
    double cutoff_radial_sq,
    double cutoff_angular_sq,
    int radial_capacity,
    int angular_capacity,
    const double* positions_soa3,
    const int* atom_cell,
    const int* cell_offsets,
    const int* cell_atoms,
    int* nn_radial,
    int* nl_radial_slot_major,
    int* nn_angular,
    int* nl_angular_slot_major,
    int* overflow) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int cell = atom_cell[atom];
  const int ix = cell % nx;
  const int iy = (cell / nx) % ny;
  const int iz = cell / (nx * ny);

  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  int radial_count = 0;
  int angular_count = 0;

  for (int dz = -1; dz <= 1; ++dz) {
    if (box.pbc[2] && duplicate_pbc_offset(dz, nz)) {
      continue;
    }
    int cz = iz + dz;
    if (box.pbc[2]) {
      cz = wrap_index(cz, nz);
    } else if (cz < 0 || cz >= nz) {
      continue;
    }
    for (int dy = -1; dy <= 1; ++dy) {
      if (box.pbc[1] && duplicate_pbc_offset(dy, ny)) {
        continue;
      }
      int cy = iy + dy;
      if (box.pbc[1]) {
        cy = wrap_index(cy, ny);
      } else if (cy < 0 || cy >= ny) {
        continue;
      }
      for (int dx = -1; dx <= 1; ++dx) {
        if (box.pbc[0] && duplicate_pbc_offset(dx, nx)) {
          continue;
        }
        int cx = ix + dx;
        if (box.pbc[0]) {
          cx = wrap_index(cx, nx);
        } else if (cx < 0 || cx >= nx) {
          continue;
        }

        const int neighbor_cell = cx + nx * (cy + ny * cz);
        const int begin = cell_offsets[neighbor_cell];
        const int end = cell_offsets[neighbor_cell + 1];
        for (int offset = begin; offset < end; ++offset) {
          const int neighbor = cell_atoms[offset];
          if (neighbor == atom) {
            continue;
          }
          float dxij = 0.0f;
          float dyij = 0.0f;
          float dzij = 0.0f;
          minimum_image_delta(
              box,
              xi - positions_soa3[neighbor],
              yi - positions_soa3[atom_stride + neighbor],
              zi - positions_soa3[2 * atom_stride + neighbor],
              dxij,
              dyij,
              dzij);
          const double rsq = dxij * dxij + dyij * dyij + dzij * dzij;

          if (rsq < cutoff_radial_sq) {
            if (radial_count < radial_capacity) {
              nl_radial_slot_major[atom + atom_stride * radial_count] = neighbor;
            } else {
              atomicExch(overflow, 1);
            }
            ++radial_count;
          }
          if (rsq < cutoff_angular_sq) {
            if (angular_count < angular_capacity) {
              nl_angular_slot_major[atom + atom_stride * angular_count] = neighbor;
            } else {
              atomicExch(overflow, 1);
            }
            ++angular_count;
          }
        }
      }
    }
  }

  nn_radial[atom] = min(radial_count, radial_capacity);
  nn_angular[atom] = min(angular_count, angular_capacity);
}

__global__ void build_neighbors_from_batched_cells(
    int atom_count,
    int atom_stride,
    const int* atom_to_structure,
    const int* structure_cell_offsets,
    const int* structure_cell_dims4,
    const double* boxes_row_major9,
    const double* box_inverse_row_major9,
    const int* pbc_flags3,
    double cutoff_radial_sq,
    double cutoff_angular_sq,
    int radial_capacity,
    int angular_capacity,
    const double* positions_soa3,
    const int* atom_cell,
    const int* cell_offsets,
    const int* cell_atoms,
    int* nn_radial,
    int* nl_radial_slot_major,
    int* nn_angular,
    int* nl_angular_slot_major,
    int* overflow) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int structure = atom_to_structure[atom];
  const int* dims = structure_cell_dims4 + 4 * structure;
  const int nx = dims[0];
  const int ny = dims[1];
  const int nz = dims[2];
  const int cell_base = structure_cell_offsets[structure];
  const SimulationBox box = load_structure_box(
      structure,
      boxes_row_major9,
      box_inverse_row_major9,
      pbc_flags3);
  const int local_cell = atom_cell[atom] - cell_base;
  const int ix = local_cell % nx;
  const int iy = (local_cell / nx) % ny;
  const int iz = local_cell / (nx * ny);

  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  int radial_count = 0;
  int angular_count = 0;

  for (int dz = -1; dz <= 1; ++dz) {
    if (box.pbc[2] && duplicate_pbc_offset(dz, nz)) {
      continue;
    }
    int cz = iz + dz;
    if (box.pbc[2]) {
      cz = wrap_index(cz, nz);
    } else if (cz < 0 || cz >= nz) {
      continue;
    }
    for (int dy = -1; dy <= 1; ++dy) {
      if (box.pbc[1] && duplicate_pbc_offset(dy, ny)) {
        continue;
      }
      int cy = iy + dy;
      if (box.pbc[1]) {
        cy = wrap_index(cy, ny);
      } else if (cy < 0 || cy >= ny) {
        continue;
      }
      for (int dx = -1; dx <= 1; ++dx) {
        if (box.pbc[0] && duplicate_pbc_offset(dx, nx)) {
          continue;
        }
        int cx = ix + dx;
        if (box.pbc[0]) {
          cx = wrap_index(cx, nx);
        } else if (cx < 0 || cx >= nx) {
          continue;
        }

        const int neighbor_cell = cell_base + cx + nx * (cy + ny * cz);
        const int begin = cell_offsets[neighbor_cell];
        const int end = cell_offsets[neighbor_cell + 1];
        for (int offset = begin; offset < end; ++offset) {
          const int neighbor = cell_atoms[offset];
          if (neighbor == atom) {
            continue;
          }
          float dxij = 0.0f;
          float dyij = 0.0f;
          float dzij = 0.0f;
          minimum_image_delta(
              box,
              xi - positions_soa3[neighbor],
              yi - positions_soa3[atom_stride + neighbor],
              zi - positions_soa3[2 * atom_stride + neighbor],
              dxij,
              dyij,
              dzij);
          const double rsq = dxij * dxij + dyij * dyij + dzij * dzij;

          if (rsq < cutoff_radial_sq) {
            if (radial_count < radial_capacity) {
              nl_radial_slot_major[atom + atom_stride * radial_count] = neighbor;
            } else {
              atomicExch(overflow, 1);
            }
            ++radial_count;
          }
          if (rsq < cutoff_angular_sq) {
            if (angular_count < angular_capacity) {
              nl_angular_slot_major[atom + atom_stride * angular_count] = neighbor;
            } else {
              atomicExch(overflow, 1);
            }
            ++angular_count;
          }
        }
      }
    }
  }

  nn_radial[atom] = min(radial_count, radial_capacity);
  nn_angular[atom] = min(angular_count, angular_capacity);
}

__global__ void build_neighbors_from_batched_orthorhombic_cells(
    int atom_count,
    int atom_stride,
    const int* atom_to_structure,
    const int* structure_cell_offsets,
    const int* structure_cell_dims4,
    const double* boxes_row_major9,
    const double* box_inverse_row_major9,
    const int* pbc_flags3,
    double cutoff_radial_sq,
    double cutoff_angular_sq,
    int radial_capacity,
    int angular_capacity,
    const double* positions_soa3,
    const int* atom_cell,
    const int* cell_offsets,
    const int* cell_atoms,
    int* nn_radial,
    int* nl_radial_slot_major,
    int* nn_angular,
    int* nl_angular_slot_major,
    int* overflow) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int structure = atom_to_structure[atom];
  const int* dims = structure_cell_dims4 + 4 * structure;
  const int nx = dims[0];
  const int ny = dims[1];
  const int nz = dims[2];
  const int cell_base = structure_cell_offsets[structure];
  const int box_offset = 9 * structure;
  const int pbc_offset = 3 * structure;
  const double lx = boxes_row_major9[box_offset + 0];
  const double ly = boxes_row_major9[box_offset + 4];
  const double lz = boxes_row_major9[box_offset + 8];
  const double inv_lx = box_inverse_row_major9[box_offset + 0];
  const double inv_ly = box_inverse_row_major9[box_offset + 4];
  const double inv_lz = box_inverse_row_major9[box_offset + 8];
  const int pbc_x = pbc_flags3[pbc_offset + 0];
  const int pbc_y = pbc_flags3[pbc_offset + 1];
  const int pbc_z = pbc_flags3[pbc_offset + 2];

  const int local_cell = atom_cell[atom] - cell_base;
  const int ix = local_cell % nx;
  const int iy = (local_cell / nx) % ny;
  const int iz = local_cell / (nx * ny);

  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  int radial_count = 0;
  int angular_count = 0;

  for (int dz = -1; dz <= 1; ++dz) {
    if (pbc_z && duplicate_pbc_offset(dz, nz)) {
      continue;
    }
    int cz = iz + dz;
    if (pbc_z) {
      cz = wrap_index(cz, nz);
    } else if (cz < 0 || cz >= nz) {
      continue;
    }
    for (int dy = -1; dy <= 1; ++dy) {
      if (pbc_y && duplicate_pbc_offset(dy, ny)) {
        continue;
      }
      int cy = iy + dy;
      if (pbc_y) {
        cy = wrap_index(cy, ny);
      } else if (cy < 0 || cy >= ny) {
        continue;
      }
      for (int dx = -1; dx <= 1; ++dx) {
        if (pbc_x && duplicate_pbc_offset(dx, nx)) {
          continue;
        }
        int cx = ix + dx;
        if (pbc_x) {
          cx = wrap_index(cx, nx);
        } else if (cx < 0 || cx >= nx) {
          continue;
        }

        const int neighbor_cell = cell_base + cx + nx * (cy + ny * cz);
        const int begin = cell_offsets[neighbor_cell];
        const int end = cell_offsets[neighbor_cell + 1];
        for (int offset = begin; offset < end; ++offset) {
          const int neighbor = cell_atoms[offset];
          if (neighbor == atom) {
            continue;
          }
          float dxij = 0.0f;
          float dyij = 0.0f;
          float dzij = 0.0f;
          orthorhombic_minimum_image_delta(
              xi - positions_soa3[neighbor],
              yi - positions_soa3[atom_stride + neighbor],
              zi - positions_soa3[2 * atom_stride + neighbor],
              lx,
              ly,
              lz,
              inv_lx,
              inv_ly,
              inv_lz,
              pbc_x,
              pbc_y,
              pbc_z,
              dxij,
              dyij,
              dzij);
          const double rsq = dxij * dxij + dyij * dyij + dzij * dzij;

          if (rsq < cutoff_radial_sq) {
            if (radial_count < radial_capacity) {
              const int radial_offset = atom + atom_stride * radial_count;
              nl_radial_slot_major[radial_offset] = neighbor;
            } else {
              atomicExch(overflow, 1);
            }
            ++radial_count;
          }
          if (rsq < cutoff_angular_sq) {
            if (angular_count < angular_capacity) {
              const int angular_offset = atom + atom_stride * angular_count;
              nl_angular_slot_major[angular_offset] = neighbor;
            } else {
              atomicExch(overflow, 1);
            }
            ++angular_count;
          }
        }
      }
    }
  }

  nn_radial[atom] = min(radial_count, radial_capacity);
  nn_angular[atom] = min(angular_count, angular_capacity);
}

}  // namespace

void build_internal_neighbors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace) {
  DeviceWorkspaceView view = workspace.view();
  require(atom_count > 0, "atom_count must be positive");
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.neighbor_source == 0, "neighbor builder requires internal workspace");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_radial != nullptr, "workspace missing radial counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.nn_angular != nullptr, "workspace missing angular counts");
  require(view.nl_angular_slot_major != nullptr, "workspace missing angular neighbors");
  require(view.cell_counts != nullptr, "workspace missing cell counts");
  require(view.cell_offsets != nullptr, "workspace missing cell offsets");
  require(view.cell_fill != nullptr, "workspace missing cell fill");
  require(view.cell_atoms != nullptr, "workspace missing cell atoms");
  require(view.atom_cell != nullptr, "workspace missing atom_cell");
  require(view.cell_dims != nullptr, "workspace missing cell dims");
  require(view.neighbor_overflow != nullptr, "workspace missing overflow flag");

  const std::array<int, 3> dims = fit_cell_dims_to_capacity(
      cell_dim_for_fractional_axis(box, 0, protocol.cutoff_max),
      cell_dim_for_fractional_axis(box, 1, protocol.cutoff_max),
      cell_dim_for_fractional_axis(box, 2, protocol.cutoff_max),
      view.atom_capacity);
  const int nx = dims[0];
  const int ny = dims[1];
  const int nz = dims[2];
  const int cell_count = nx * ny * nz;
  require(static_cast<std::size_t>(cell_count) <= view.atom_capacity,
          "cell count exceeds workspace cell capacity");

  const int cell_dims[] = {nx, ny, nz, cell_count};
  check_cuda(
      cudaMemcpy(
          view.cell_dims,
          cell_dims,
          sizeof(cell_dims),
          cudaMemcpyHostToDevice),
      "copy cell dims");
  check_cuda(cudaMemset(view.cell_counts, 0, view.atom_capacity * sizeof(int)),
             "zero cell counts");
  check_cuda(cudaMemset(view.cell_fill, 0, view.atom_capacity * sizeof(int)),
             "zero cell fill");
  check_cuda(cudaMemset(view.nn_radial, 0, view.atom_capacity * sizeof(int)),
             "zero radial counts");
  check_cuda(cudaMemset(view.nn_angular, 0, view.atom_capacity * sizeof(int)),
             "zero angular counts");
  check_cuda(
      cudaMemset(
          view.nl_radial_slot_major,
          0,
          view.atom_capacity *
              static_cast<std::size_t>(protocol.neighbor_capacity_radial) *
              sizeof(int)),
      "zero radial neighbors");
  check_cuda(
      cudaMemset(
          view.nl_angular_slot_major,
          0,
          view.atom_capacity *
              static_cast<std::size_t>(protocol.neighbor_capacity_angular) *
              sizeof(int)),
      "zero angular neighbors");
  check_cuda(cudaMemset(view.neighbor_overflow, 0, sizeof(int)),
             "zero neighbor overflow");

  const int blocks = (atom_count + kBlockSize - 1) / kBlockSize;
  assign_cells<<<blocks, kBlockSize>>>(
      atom_count,
      static_cast<int>(view.atom_capacity),
      nx,
      ny,
      nz,
      box,
      view.positions_soa3,
      view.atom_cell,
      view.cell_counts);
  check_cuda(cudaGetLastError(), "assign cells");

  thrust::device_ptr<int> counts(view.cell_counts);
  thrust::device_ptr<int> offsets(view.cell_offsets);
  thrust::exclusive_scan(thrust::device, counts, counts + cell_count, offsets);
  const int total_atoms_in_cells =
      thrust::reduce(thrust::device, counts, counts + cell_count, 0);
  require(total_atoms_in_cells == atom_count, "cell binning lost atoms");
  check_cuda(
      cudaMemcpy(
          view.cell_offsets + cell_count,
          &total_atoms_in_cells,
          sizeof(int),
          cudaMemcpyHostToDevice),
      "copy final cell offset");

  scatter_cells<<<blocks, kBlockSize>>>(
      atom_count,
      view.atom_cell,
      view.cell_offsets,
      view.cell_fill,
      view.cell_atoms);
  check_cuda(cudaGetLastError(), "scatter cells");

  build_neighbors_from_cells<<<blocks, kBlockSize>>>(
      atom_count,
      static_cast<int>(view.atom_capacity),
      nx,
      ny,
      nz,
      box,
      protocol.cutoff_neighbor * protocol.cutoff_neighbor,
      protocol.cutoff_angular * protocol.cutoff_angular,
      protocol.neighbor_capacity_radial,
      protocol.neighbor_capacity_angular,
      view.positions_soa3,
      view.atom_cell,
      view.cell_offsets,
      view.cell_atoms,
      view.nn_radial,
      view.nl_radial_slot_major,
      view.nn_angular,
      view.nl_angular_slot_major,
      view.neighbor_overflow);
  check_cuda(cudaGetLastError(), "build internal neighbors");
  check_cuda(cudaDeviceSynchronize(), "synchronize internal neighbor build");

  int overflow = 0;
  check_cuda(
      cudaMemcpy(
          &overflow,
          view.neighbor_overflow,
          sizeof(int),
          cudaMemcpyDeviceToHost),
      "copy neighbor overflow");
  require(overflow == 0, "neighbor list exceeded workspace capacity");
}

void build_internal_neighbors_batched(
    const ModelProtocol& protocol,
    int structure_count,
    int atom_count,
    DeviceWorkspace& workspace,
    bool orthorhombic_fast_path) {
  DeviceWorkspaceView view = workspace.view();
  require(structure_count > 0, "structure_count must be positive");
  require(atom_count > 0, "atom_count must be positive");
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(static_cast<std::size_t>(structure_count) <= view.structure_capacity,
          "structure_count exceeds workspace structure capacity");
  require(view.neighbor_source == 0, "neighbor builder requires internal workspace");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.atom_to_structure != nullptr, "workspace missing atom_to_structure");
  require(view.structure_atom_counts != nullptr, "workspace missing structure counts");
  require(view.boxes_row_major9 != nullptr, "workspace missing boxes");
  require(view.box_inverse_row_major9 != nullptr, "workspace missing box inverses");
  require(view.pbc_flags3 != nullptr, "workspace missing pbc flags");
  require(view.nn_radial != nullptr, "workspace missing radial counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.nn_angular != nullptr, "workspace missing angular counts");
  require(view.nl_angular_slot_major != nullptr, "workspace missing angular neighbors");
  require(view.cell_counts != nullptr, "workspace missing cell counts");
  require(view.cell_offsets != nullptr, "workspace missing cell offsets");
  require(view.cell_fill != nullptr, "workspace missing cell fill");
  require(view.cell_atoms != nullptr, "workspace missing cell atoms");
  require(view.atom_cell != nullptr, "workspace missing atom_cell");
  require(view.structure_cell_offsets != nullptr,
          "workspace missing structure cell offsets");
  require(view.structure_cell_dims4 != nullptr,
          "workspace missing structure cell dims");
  require(view.neighbor_overflow != nullptr, "workspace missing overflow flag");

  check_cuda(cudaMemset(view.structure_cell_offsets, 0,
                        (view.structure_capacity + 1) * sizeof(int)),
             "zero structure cell offsets");
  check_cuda(cudaMemset(view.structure_cell_dims4, 0,
                        view.structure_capacity * 4 * sizeof(int)),
             "zero structure cell dims");
  check_cuda(cudaMemset(view.neighbor_overflow, 0, sizeof(int)),
             "zero batched neighbor overflow");

  const int structure_blocks = (structure_count + kBlockSize - 1) / kBlockSize;
  plan_batched_cells<<<structure_blocks, kBlockSize>>>(
      structure_count,
      protocol.cutoff_max,
      view.structure_atom_counts,
      view.boxes_row_major9,
      view.box_inverse_row_major9,
      view.pbc_flags3,
      view.structure_cell_offsets,
      view.structure_cell_dims4,
      view.neighbor_overflow);
  check_cuda(cudaGetLastError(), "plan batched cells");
  check_cuda(cudaDeviceSynchronize(), "synchronize batched cell planning");

  int overflow = 0;
  check_cuda(
      cudaMemcpy(
          &overflow,
          view.neighbor_overflow,
          sizeof(int),
          cudaMemcpyDeviceToHost),
      "copy batched cell planning overflow");
  require(overflow == 0, "batched cell planning exceeded workspace capacity");

  thrust::device_ptr<int> structure_cell_counts(view.structure_cell_offsets);
  thrust::exclusive_scan(
      thrust::device,
      structure_cell_counts,
      structure_cell_counts + structure_count,
      structure_cell_counts);

  check_cuda(cudaMemset(view.cell_counts, 0, view.atom_capacity * sizeof(int)),
             "zero batched cell counts");
  check_cuda(cudaMemset(view.cell_fill, 0, view.atom_capacity * sizeof(int)),
             "zero batched cell fill");
  check_cuda(cudaMemset(view.nn_radial, 0, view.atom_capacity * sizeof(int)),
             "zero batched radial counts");
  check_cuda(cudaMemset(view.nn_angular, 0, view.atom_capacity * sizeof(int)),
             "zero batched angular counts");
  check_cuda(
      cudaMemset(
          view.nl_radial_slot_major,
          0,
          view.atom_capacity *
              static_cast<std::size_t>(protocol.neighbor_capacity_radial) *
              sizeof(int)),
      "zero batched radial neighbors");
  check_cuda(
      cudaMemset(
          view.nl_angular_slot_major,
          0,
          view.atom_capacity *
              static_cast<std::size_t>(protocol.neighbor_capacity_angular) *
              sizeof(int)),
      "zero batched angular neighbors");
  check_cuda(cudaMemset(view.neighbor_overflow, 0, sizeof(int)),
             "zero batched neighbor overflow");

  const int blocks = (atom_count + kBlockSize - 1) / kBlockSize;
  if (orthorhombic_fast_path) {
    assign_cells_batched_orthorhombic<<<blocks, kBlockSize>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        view.atom_to_structure,
        view.structure_cell_offsets,
        view.structure_cell_dims4,
        view.boxes_row_major9,
        view.box_inverse_row_major9,
        view.pbc_flags3,
        view.positions_soa3,
        view.atom_cell,
        view.cell_counts);
  } else {
    assign_cells_batched<<<blocks, kBlockSize>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        view.atom_to_structure,
        view.structure_cell_offsets,
        view.structure_cell_dims4,
        view.boxes_row_major9,
        view.box_inverse_row_major9,
        view.pbc_flags3,
        view.positions_soa3,
        view.atom_cell,
        view.cell_counts);
  }
  check_cuda(cudaGetLastError(), "assign batched cells");

  thrust::device_ptr<int> counts(view.cell_counts);
  thrust::device_ptr<int> offsets(view.cell_offsets);
  thrust::exclusive_scan(
      thrust::device,
      counts,
      counts + view.atom_capacity,
      offsets);
  const int total_atoms_in_cells =
      thrust::reduce(
          thrust::device,
          counts,
          counts + view.atom_capacity,
          0);
  require(total_atoms_in_cells == atom_count, "batched cell binning lost atoms");
  check_cuda(
      cudaMemcpy(
          view.cell_offsets + view.atom_capacity,
          &total_atoms_in_cells,
          sizeof(int),
          cudaMemcpyHostToDevice),
      "copy final batched cell offset");

  scatter_cells<<<blocks, kBlockSize>>>(
      atom_count,
      view.atom_cell,
      view.cell_offsets,
      view.cell_fill,
      view.cell_atoms);
  check_cuda(cudaGetLastError(), "scatter batched cells");

  if (orthorhombic_fast_path) {
    build_neighbors_from_batched_orthorhombic_cells<<<blocks, kBlockSize>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        view.atom_to_structure,
        view.structure_cell_offsets,
        view.structure_cell_dims4,
        view.boxes_row_major9,
        view.box_inverse_row_major9,
        view.pbc_flags3,
        protocol.cutoff_neighbor * protocol.cutoff_neighbor,
        protocol.cutoff_angular * protocol.cutoff_angular,
        protocol.neighbor_capacity_radial,
        protocol.neighbor_capacity_angular,
        view.positions_soa3,
        view.atom_cell,
        view.cell_offsets,
        view.cell_atoms,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.nn_angular,
        view.nl_angular_slot_major,
        view.neighbor_overflow);
  } else {
    build_neighbors_from_batched_cells<<<blocks, kBlockSize>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        view.atom_to_structure,
        view.structure_cell_offsets,
        view.structure_cell_dims4,
        view.boxes_row_major9,
        view.box_inverse_row_major9,
        view.pbc_flags3,
        protocol.cutoff_neighbor * protocol.cutoff_neighbor,
        protocol.cutoff_angular * protocol.cutoff_angular,
        protocol.neighbor_capacity_radial,
        protocol.neighbor_capacity_angular,
        view.positions_soa3,
        view.atom_cell,
        view.cell_offsets,
        view.cell_atoms,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.nn_angular,
        view.nl_angular_slot_major,
        view.neighbor_overflow);
  }
  check_cuda(cudaGetLastError(), "build batched neighbors");
  check_cuda(cudaDeviceSynchronize(), "synchronize batched neighbor builder");

  overflow = 0;
  check_cuda(
      cudaMemcpy(
          &overflow,
          view.neighbor_overflow,
          sizeof(int),
          cudaMemcpyDeviceToHost),
      "copy batched neighbor overflow");
  require(overflow == 0, "batched neighbor list exceeded CUDA workspace capacity");
}

}  // namespace nep_adapters::cuda_backend
