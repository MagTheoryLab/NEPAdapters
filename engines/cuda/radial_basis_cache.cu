#include "radial_basis_cache.hpp"
#include "simulation_box_device.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace nep_adapters::cuda_backend {
namespace {

constexpr float kPi = 3.1415927f;

void check_cuda(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(message) + ": " + cudaGetErrorString(status));
  }
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

__device__ __forceinline__ float find_fc(float rc, float rcinv, float r) {
  if (r >= rc) {
    return 0.0f;
  }
  return 0.5f * cosf(kPi * r * rcinv) + 0.5f;
}

__global__ void build_radial_basis_cache(
    int atom_count,
    int atom_stride,
    int radial_capacity,
    int basis_size,
    float rc,
    const int* __restrict__ nn_radial,
    const float* __restrict__ r12_radial,
    float* __restrict__ fc_radial,
    float* __restrict__ fn_radial) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const float rcinv = 1.0f / rc;
  const int count = nn_radial[atom];
  for (int slot = 0; slot < count; ++slot) {
    const int offset = atom + atom_stride * slot;
    const float r = r12_radial[offset];
    const float fc = find_fc(rc, rcinv, r);
    fc_radial[offset] = fc;

    const float x = 2.0f * (r * rcinv - 1.0f) * (r * rcinv - 1.0f) - 1.0f;
    const float half_fc = 0.5f * fc;
    fn_radial[offset] = fc;
    if (basis_size >= 1) {
      fn_radial[offset + atom_stride * radial_capacity] =
          (x + 1.0f) * half_fc;
    }

    float t_minus_2 = 1.0f;
    float t_minus_1 = x;
    for (int k = 2; k <= basis_size; ++k) {
      const float t = 2.0f * x * t_minus_1 - t_minus_2;
      t_minus_2 = t_minus_1;
      t_minus_1 = t;
      fn_radial[offset + atom_stride * radial_capacity * k] =
          (t + 1.0f) * half_fc;
    }
  }
}

__device__ __forceinline__ void write_radial_basis_values(
    int atom_stride,
    int radial_capacity,
    int offset,
    int basis_size,
    float rc,
    float r,
    float* __restrict__ fc_radial,
    float* __restrict__ fn_radial) {
  const float rcinv = 1.0f / rc;
  const float fc = find_fc(rc, rcinv, r);
  fc_radial[offset] = fc;

  const float x = 2.0f * (r * rcinv - 1.0f) * (r * rcinv - 1.0f) - 1.0f;
  const float half_fc = 0.5f * fc;
  fn_radial[offset] = fc;
  if (basis_size >= 1) {
    fn_radial[offset + atom_stride * radial_capacity] =
        (x + 1.0f) * half_fc;
  }

  float t_minus_2 = 1.0f;
  float t_minus_1 = x;
  for (int k = 2; k <= basis_size; ++k) {
    const float t = 2.0f * x * t_minus_1 - t_minus_2;
    t_minus_2 = t_minus_1;
    t_minus_1 = t;
    fn_radial[offset + atom_stride * radial_capacity * k] =
        (t + 1.0f) * half_fc;
  }
}

__global__ void build_radial_geometry_basis_cache(
    int atom_count,
    int atom_stride,
    int radial_capacity,
    int basis_size,
    float rc,
    SimulationBox box,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    float* __restrict__ r12_radial,
    float* __restrict__ fc_radial,
    float* __restrict__ fn_radial) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  const int count = nn_radial[atom];
  for (int slot = 0; slot < count; ++slot) {
    const int offset = atom + atom_stride * slot;
    const int neighbor = nl_radial[offset];
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
    const float r = static_cast<float>(sqrt(dx * dx + dy * dy + dz * dz));
    r12_radial[offset] = r;
    write_radial_basis_values(
        atom_stride,
        radial_capacity,
        offset,
        basis_size,
        rc,
        r,
        fc_radial,
        fn_radial);
  }
}

__global__ void build_radial_geometry_basis_cache_batched(
    int atom_count,
    int atom_stride,
    int radial_capacity,
    int basis_size,
    float rc,
    const int* __restrict__ atom_to_structure,
    const double* __restrict__ boxes_row_major9,
    const double* __restrict__ box_inverse_row_major9,
    const int* __restrict__ pbc_flags3,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    float* __restrict__ r12_radial,
    float* __restrict__ fc_radial,
    float* __restrict__ fn_radial) {
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
    const int offset = atom + atom_stride * slot;
    const int neighbor = nl_radial[offset];
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
    const float r = static_cast<float>(sqrt(dx * dx + dy * dy + dz * dz));
    r12_radial[offset] = r;
    write_radial_basis_values(
        atom_stride,
        radial_capacity,
        offset,
        basis_size,
        rc,
        r,
        fc_radial,
        fn_radial);
  }
}

}  // namespace

void build_radial_basis_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.cutoff_radial > 0.0, "radial cutoff must be positive");
  require(protocol.basis_size_radial >= 0, "radial basis size must be non-negative");
  require(
      protocol.neighbor_capacity_radial > 0,
      "radial neighbor capacity must be positive");

  const DeviceWorkspaceView view = workspace.view();
  require(
      static_cast<std::size_t>(atom_count) <= view.atom_capacity,
      "atom_count exceeds workspace atom capacity");
  require(view.nn_radial != nullptr, "workspace missing radial neighbor counts");
  require(view.r12_radial != nullptr, "workspace missing radial distance cache");
  require(view.fc_radial != nullptr, "workspace missing radial cutoff cache");
  require(view.fn_radial != nullptr, "workspace missing radial basis cache");

  const int threads = 128;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    build_radial_basis_cache<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.neighbor_capacity_radial,
        protocol.basis_size_radial,
        static_cast<float>(protocol.cutoff_radial),
        view.nn_radial,
        view.r12_radial,
        view.fc_radial,
        view.fn_radial);
  }
  check_cuda(cudaGetLastError(), "build radial basis cache kernel launch failed");
  check_cuda(cudaDeviceSynchronize(), "build radial basis cache kernel failed");
}

void build_radial_geometry_basis_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.cutoff_radial > 0.0, "radial cutoff must be positive");
  require(protocol.basis_size_radial >= 0, "radial basis size must be non-negative");
  require(
      protocol.neighbor_capacity_radial > 0,
      "radial neighbor capacity must be positive");

  const DeviceWorkspaceView view = workspace.view();
  require(
      static_cast<std::size_t>(atom_count) <= view.atom_capacity,
      "atom_count exceeds workspace atom capacity");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_radial != nullptr, "workspace missing radial neighbor counts");
  require(view.nl_radial_slot_major != nullptr,
          "workspace missing radial neighbor list");
  require(view.r12_radial != nullptr, "workspace missing radial distance cache");
  require(view.fc_radial != nullptr, "workspace missing radial cutoff cache");
  require(view.fn_radial != nullptr, "workspace missing radial basis cache");

  const int threads = 128;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    build_radial_geometry_basis_cache<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.neighbor_capacity_radial,
        protocol.basis_size_radial,
        static_cast<float>(protocol.cutoff_radial),
        box,
        view.positions_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.r12_radial,
        view.fc_radial,
        view.fn_radial);
  }
  check_cuda(cudaGetLastError(),
             "build radial geometry/basis cache kernel launch failed");
  check_cuda(cudaDeviceSynchronize(),
             "build radial geometry/basis cache kernel failed");
}

void build_radial_geometry_basis_cache_batched(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.cutoff_radial > 0.0, "radial cutoff must be positive");
  require(protocol.basis_size_radial >= 0, "radial basis size must be non-negative");
  require(
      protocol.neighbor_capacity_radial > 0,
      "radial neighbor capacity must be positive");

  const DeviceWorkspaceView view = workspace.view();
  require(
      static_cast<std::size_t>(atom_count) <= view.atom_capacity,
      "atom_count exceeds workspace atom capacity");
  require(view.atom_to_structure != nullptr, "workspace missing atom_to_structure");
  require(view.boxes_row_major9 != nullptr, "workspace missing boxes");
  require(view.box_inverse_row_major9 != nullptr, "workspace missing box inverses");
  require(view.pbc_flags3 != nullptr, "workspace missing pbc flags");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_radial != nullptr, "workspace missing radial neighbor counts");
  require(view.nl_radial_slot_major != nullptr,
          "workspace missing radial neighbor list");
  require(view.r12_radial != nullptr, "workspace missing radial distance cache");
  require(view.fc_radial != nullptr, "workspace missing radial cutoff cache");
  require(view.fn_radial != nullptr, "workspace missing radial basis cache");

  const int threads = 128;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    build_radial_geometry_basis_cache_batched<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.neighbor_capacity_radial,
        protocol.basis_size_radial,
        static_cast<float>(protocol.cutoff_radial),
        view.atom_to_structure,
        view.boxes_row_major9,
        view.box_inverse_row_major9,
        view.pbc_flags3,
        view.positions_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.r12_radial,
        view.fc_radial,
        view.fn_radial);
  }
  check_cuda(cudaGetLastError(),
             "build batched radial geometry/basis cache kernel launch failed");
  check_cuda(cudaDeviceSynchronize(),
             "build batched radial geometry/basis cache kernel failed");
}

}  // namespace nep_adapters::cuda_backend
