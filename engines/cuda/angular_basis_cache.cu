#include "angular_basis_cache.hpp"

#include <cuda_runtime.h>

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

__device__ __forceinline__ void write_zero_fn(
    int atom_stride,
    int angular_capacity,
    int offset,
    int basis_size,
    float* fn_angular) {
  for (int k = 0; k <= basis_size; ++k) {
    fn_angular[offset + atom_stride * angular_capacity * k] = 0.0f;
  }
}

__global__ void build_angular_basis_cache(
    int atom_count,
    int atom_stride,
    int angular_capacity,
    int basis_size,
    float rc,
    const int* __restrict__ nn_angular,
    const float* __restrict__ f12x,
    const float* __restrict__ f12y,
    const float* __restrict__ f12z,
    float* __restrict__ r12_angular,
    float* __restrict__ fc_angular,
    float* __restrict__ fn_angular) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const float rcinv = 1.0f / rc;
  const int count = nn_angular[atom];
  const int limit = angular_capacity;
  for (int slot = 0; slot < limit; ++slot) {
    const int offset = atom + atom_stride * slot;
    if (slot >= count) {
      r12_angular[offset] = 0.0f;
      fc_angular[offset] = 0.0f;
      write_zero_fn(atom_stride, angular_capacity, offset, basis_size, fn_angular);
      continue;
    }

    const float dx = f12x[offset];
    const float dy = f12y[offset];
    const float dz = f12z[offset];
    const float r = sqrtf(dx * dx + dy * dy + dz * dz);
    const float fc = find_fc(rc, rcinv, r);
    r12_angular[offset] = r;
    fc_angular[offset] = fc;

    const float x = 2.0f * (r * rcinv - 1.0f) * (r * rcinv - 1.0f) - 1.0f;
    const float half_fc = 0.5f * fc;
    fn_angular[offset] = fc;
    if (basis_size >= 1) {
      fn_angular[offset + atom_stride * angular_capacity] = (x + 1.0f) * half_fc;
    }

    float t_minus_2 = 1.0f;
    float t_minus_1 = x;
    for (int k = 2; k <= basis_size; ++k) {
      const float t = 2.0f * x * t_minus_1 - t_minus_2;
      t_minus_2 = t_minus_1;
      t_minus_1 = t;
      fn_angular[offset + atom_stride * angular_capacity * k] =
          (t + 1.0f) * half_fc;
    }
  }
}

}  // namespace

void build_angular_basis_cache_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.cutoff_angular > 0.0, "angular cutoff must be positive");
  require(protocol.basis_size_angular >= 0, "angular basis size must be non-negative");
  require(
      protocol.neighbor_capacity_angular > 0,
      "angular neighbor capacity must be positive");

  const DeviceWorkspaceView view = workspace.view();
  require(
      static_cast<std::size_t>(atom_count) <= view.atom_capacity,
      "atom_count exceeds workspace atom capacity");
  require(view.nn_angular != nullptr, "workspace missing angular neighbor counts");
  require(view.f12x != nullptr && view.f12y != nullptr && view.f12z != nullptr,
          "workspace missing angular pair geometry");
  require(view.r12_angular != nullptr, "workspace missing angular distance cache");
  require(view.fc_angular != nullptr, "workspace missing angular cutoff cache");
  require(view.fn_angular != nullptr, "workspace missing angular basis cache");

  const int threads = 128;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    build_angular_basis_cache<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.neighbor_capacity_angular,
        protocol.basis_size_angular,
        static_cast<float>(protocol.cutoff_angular),
        view.nn_angular,
        view.f12x,
        view.f12y,
        view.f12z,
        view.r12_angular,
        view.fc_angular,
        view.fn_angular);
  }
  check_cuda(cudaGetLastError(), "build angular basis cache kernel launch failed");
  check_cuda(cudaDeviceSynchronize(), "build angular basis cache kernel failed");
}

}  // namespace nep_adapters::cuda_backend
