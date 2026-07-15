#include "device_operations.hpp"
#include "simulation_box_device.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nep_adapters::cuda_backend {
namespace {

constexpr int kBlockSize = 128;

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

__global__ void build_radial_distance_cache(
    int atom_count,
    int atom_stride,
    int radial_capacity,
    SimulationBox box,
    const double* positions_soa3,
    const int* nn_radial,
    const int* nl_radial_slot_major,
    float* r12_radial) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  const int count = nn_radial[atom];
  for (int slot = 0; slot < count; ++slot) {
    const int neighbor = nl_radial_slot_major[atom + atom_stride * slot];
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    minimum_image_delta(
        box,
        xi - positions_soa3[neighbor],
        yi - positions_soa3[atom_stride + neighbor],
        zi - positions_soa3[2 * atom_stride + neighbor],
        dx,
        dy,
        dz);
    r12_radial[atom + atom_stride * slot] =
        static_cast<float>(sqrt(dx * dx + dy * dy + dz * dz));
  }
}

__global__ void build_radial_distance_cache_batched(
    int atom_count,
    int atom_stride,
    int radial_capacity,
    const int* atom_to_structure,
    const double* boxes_row_major9,
    const double* box_inverse_row_major9,
    const int* pbc_flags3,
    const double* positions_soa3,
    const int* nn_radial,
    const int* nl_radial_slot_major,
    float* r12_radial) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const SimulationBox box = load_structure_box(
      atom_to_structure[atom],
      boxes_row_major9,
      box_inverse_row_major9,
      pbc_flags3);
  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  const int count = nn_radial[atom];
  for (int slot = 0; slot < count; ++slot) {
    const int neighbor = nl_radial_slot_major[atom + atom_stride * slot];
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    minimum_image_delta(
        box,
        xi - positions_soa3[neighbor],
        yi - positions_soa3[atom_stride + neighbor],
        zi - positions_soa3[2 * atom_stride + neighbor],
        dx,
        dy,
        dz);
    r12_radial[atom + atom_stride * slot] =
        static_cast<float>(sqrt(dx * dx + dy * dy + dz * dz));
  }
}

__global__ void build_angular_delta_cache(
    int atom_count,
    int atom_stride,
    int angular_capacity,
    SimulationBox box,
    const double* positions_soa3,
    const int* nn_angular,
    const int* nl_angular_slot_major,
    float* f12x,
    float* f12y,
    float* f12z) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  const int count = nn_angular[atom];
  for (int slot = 0; slot < count; ++slot) {
    const int offset = atom + atom_stride * slot;
    const int neighbor = nl_angular_slot_major[offset];
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    minimum_image_delta(
        box,
        positions_soa3[neighbor] - xi,
        positions_soa3[atom_stride + neighbor] - yi,
        positions_soa3[2 * atom_stride + neighbor] - zi,
        dx,
        dy,
        dz);
    f12x[offset] = dx;
    f12y[offset] = dy;
    f12z[offset] = dz;
  }
}

__global__ void build_angular_delta_cache_batched(
    int atom_count,
    int atom_stride,
    int angular_capacity,
    const int* atom_to_structure,
    const double* boxes_row_major9,
    const double* box_inverse_row_major9,
    const int* pbc_flags3,
    const double* positions_soa3,
    const int* nn_angular,
    const int* nl_angular_slot_major,
    float* f12x,
    float* f12y,
    float* f12z) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const SimulationBox box = load_structure_box(
      atom_to_structure[atom],
      boxes_row_major9,
      box_inverse_row_major9,
      pbc_flags3);
  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  const int count = nn_angular[atom];
  for (int slot = 0; slot < count; ++slot) {
    const int neighbor = nl_angular_slot_major[atom + atom_stride * slot];
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    minimum_image_delta(
        box,
        positions_soa3[neighbor] - xi,
        positions_soa3[atom_stride + neighbor] - yi,
        positions_soa3[2 * atom_stride + neighbor] - zi,
        dx,
        dy,
        dz);
    f12x[atom + atom_stride * slot] = dx;
    f12y[atom + atom_stride * slot] = dy;
    f12z[atom + atom_stride * slot] = dz;
  }
}

__global__ void build_pair_geometry_cache_shared_list(
    int atom_count,
    int atom_stride,
    SimulationBox box,
    const double* positions_soa3,
    const int* nn,
    const int* nl_slot_major,
    float* r12_radial,
    float* f12x,
    float* f12y,
    float* f12z) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  const int count = nn[atom];
  for (int slot = 0; slot < count; ++slot) {
    const int offset = atom + atom_stride * slot;
    const int neighbor = nl_slot_major[offset];
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    minimum_image_delta(
        box,
        positions_soa3[neighbor] - xi,
        positions_soa3[atom_stride + neighbor] - yi,
        positions_soa3[2 * atom_stride + neighbor] - zi,
        dx,
        dy,
        dz);
    r12_radial[offset] = sqrtf(dx * dx + dy * dy + dz * dz);
    f12x[offset] = dx;
    f12y[offset] = dy;
    f12z[offset] = dz;
  }
}

__global__ void build_pair_geometry_cache_shared_list_batched(
    int atom_count,
    int atom_stride,
    const int* atom_to_structure,
    const double* boxes_row_major9,
    const double* box_inverse_row_major9,
    const int* pbc_flags3,
    const double* positions_soa3,
    const int* nn,
    const int* nl_slot_major,
    float* r12_radial,
    float* f12x,
    float* f12y,
    float* f12z) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const SimulationBox box = load_structure_box(
      atom_to_structure[atom],
      boxes_row_major9,
      box_inverse_row_major9,
      pbc_flags3);
  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  const int count = nn[atom];
  for (int slot = 0; slot < count; ++slot) {
    const int offset = atom + atom_stride * slot;
    const int neighbor = nl_slot_major[offset];
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    minimum_image_delta(
        box,
        positions_soa3[neighbor] - xi,
        positions_soa3[atom_stride + neighbor] - yi,
        positions_soa3[2 * atom_stride + neighbor] - zi,
        dx,
        dy,
        dz);
    r12_radial[offset] = sqrtf(dx * dx + dy * dy + dz * dz);
    f12x[offset] = dx;
    f12y[offset] = dy;
    f12z[offset] = dz;
  }
}

bool can_share_radial_and_angular_geometry(const ModelProtocol& protocol) {
  return protocol.neighbor_capacity_radial == protocol.neighbor_capacity_angular &&
         protocol.cutoff_radial == protocol.cutoff_angular;
}

bool needs_angular_geometry(const ModelProtocol& protocol) {
  return protocol.body_channels.channel_count() > 0;
}

}  // namespace

void build_pair_geometry_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace) {
  const DeviceWorkspaceView view = workspace.view();
  require(atom_count > 0, "atom_count must be positive");
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_radial != nullptr, "workspace missing radial counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.r12_radial != nullptr, "workspace missing radial distance cache");
  const bool needs_angular = needs_angular_geometry(protocol);
  if (needs_angular) {
    require(view.nn_angular != nullptr, "workspace missing angular counts");
    require(view.nl_angular_slot_major != nullptr,
            "workspace missing angular neighbors");
    require(view.f12x != nullptr && view.f12y != nullptr && view.f12z != nullptr,
            "workspace missing angular delta cache");
  }

  const int blocks = (atom_count + kBlockSize - 1) / kBlockSize;
  if (needs_angular && can_share_radial_and_angular_geometry(protocol)) {
    build_pair_geometry_cache_shared_list<<<blocks, kBlockSize>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        box,
        view.positions_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.r12_radial,
        view.f12x,
        view.f12y,
        view.f12z);
    check_cuda(cudaGetLastError(), "build shared pair geometry cache");
    check_cuda(cudaDeviceSynchronize(), "synchronize pair geometry cache");
    return;
  }

  build_radial_distance_cache<<<blocks, kBlockSize>>>(
      atom_count,
      static_cast<int>(view.atom_capacity),
      protocol.neighbor_capacity_radial,
      box,
      view.positions_soa3,
      view.nn_radial,
      view.nl_radial_slot_major,
      view.r12_radial);
  check_cuda(cudaGetLastError(), "build radial distance cache");

  if (!needs_angular) {
    check_cuda(cudaDeviceSynchronize(), "synchronize pair geometry cache");
    return;
  }

  build_angular_delta_cache<<<blocks, kBlockSize>>>(
      atom_count,
      static_cast<int>(view.atom_capacity),
      protocol.neighbor_capacity_angular,
      box,
      view.positions_soa3,
      view.nn_angular,
      view.nl_angular_slot_major,
      view.f12x,
      view.f12y,
      view.f12z);
  check_cuda(cudaGetLastError(), "build angular delta cache");
  check_cuda(cudaDeviceSynchronize(), "synchronize pair geometry cache");
}

void build_pair_geometry_cache_batched(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace) {
  const DeviceWorkspaceView view = workspace.view();
  require(atom_count > 0, "atom_count must be positive");
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.atom_to_structure != nullptr, "workspace missing atom_to_structure");
  require(view.boxes_row_major9 != nullptr, "workspace missing boxes");
  require(view.box_inverse_row_major9 != nullptr, "workspace missing box inverses");
  require(view.pbc_flags3 != nullptr, "workspace missing pbc flags");
  require(view.nn_radial != nullptr, "workspace missing radial counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.r12_radial != nullptr, "workspace missing radial distance cache");
  const bool needs_angular = needs_angular_geometry(protocol);
  if (needs_angular) {
    require(view.nn_angular != nullptr, "workspace missing angular counts");
    require(view.nl_angular_slot_major != nullptr, "workspace missing angular neighbors");
    require(view.f12x != nullptr && view.f12y != nullptr && view.f12z != nullptr,
            "workspace missing angular delta cache");
  }

  const int blocks = (atom_count + kBlockSize - 1) / kBlockSize;
  if (needs_angular && can_share_radial_and_angular_geometry(protocol)) {
    build_pair_geometry_cache_shared_list_batched<<<blocks, kBlockSize>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        view.atom_to_structure,
        view.boxes_row_major9,
        view.box_inverse_row_major9,
        view.pbc_flags3,
        view.positions_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.r12_radial,
        view.f12x,
        view.f12y,
        view.f12z);
    check_cuda(cudaGetLastError(), "build shared batched pair geometry cache");
    check_cuda(cudaDeviceSynchronize(), "synchronize batched pair geometry cache");
    return;
  }

  build_radial_distance_cache_batched<<<blocks, kBlockSize>>>(
      atom_count,
      static_cast<int>(view.atom_capacity),
      protocol.neighbor_capacity_radial,
      view.atom_to_structure,
      view.boxes_row_major9,
      view.box_inverse_row_major9,
      view.pbc_flags3,
      view.positions_soa3,
      view.nn_radial,
      view.nl_radial_slot_major,
      view.r12_radial);
  check_cuda(cudaGetLastError(), "build batched radial distance cache");

  if (!needs_angular) {
    check_cuda(cudaDeviceSynchronize(), "synchronize batched pair geometry cache");
    return;
  }

  build_angular_delta_cache_batched<<<blocks, kBlockSize>>>(
      atom_count,
      static_cast<int>(view.atom_capacity),
      protocol.neighbor_capacity_angular,
      view.atom_to_structure,
      view.boxes_row_major9,
      view.box_inverse_row_major9,
      view.pbc_flags3,
      view.positions_soa3,
      view.nn_angular,
      view.nl_angular_slot_major,
      view.f12x,
      view.f12y,
      view.f12z);
  check_cuda(cudaGetLastError(), "build batched angular delta cache");
  check_cuda(cudaDeviceSynchronize(), "synchronize batched pair geometry cache");
}

}  // namespace nep_adapters::cuda_backend
