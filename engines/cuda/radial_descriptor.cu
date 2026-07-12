#include "device_operations.hpp"
#include "simulation_box_device.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nep_adapters::cuda_backend {
namespace {

constexpr float kPi = 3.1415927f;
constexpr int kMaxFusedRadialDescriptors = 32;

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

__device__ __forceinline__ float radial_cutoff(float rc, float rcinv, float r) {
  if (r >= rc) {
    return 0.0f;
  }
  return 0.5f * cosf(kPi * r * rcinv) + 0.5f;
}

__global__ void build_radial_descriptors(
    int atom_count,
    int atom_stride,
    int descriptor_dim,
    int num_types,
    int num_type_pairs,
    int n_max_radial,
    int basis_size_radial,
    int radial_capacity,
    const int* __restrict__ types,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fn_radial,
    const float* __restrict__ descriptor_coefficients,
    float* __restrict__ descriptors) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  for (int d = 0; d < descriptor_dim; ++d) {
    descriptors[atom + atom_stride * d] = 0.0f;
  }

  const int type1 = types[atom];
  const int radial_count = nn_radial[atom];
  for (int n = 0; n <= n_max_radial; ++n) {
    float q = 0.0f;
    for (int slot = 0; slot < radial_count; ++slot) {
      const int offset = atom + atom_stride * slot;
      const int neighbor = nl_radial[offset];
      const int type2 = types[neighbor];
      const int type_pair = type1 * num_types + type2;
      for (int k = 0; k <= basis_size_radial; ++k) {
        const int coefficient_index =
            (n * (basis_size_radial + 1) + k) * num_type_pairs + type_pair;
        const int fn_index = offset + atom_stride * radial_capacity * k;
        q += fn_radial[fn_index] * descriptor_coefficients[coefficient_index];
      }
    }
    descriptors[atom + atom_stride * n] = q;
  }
}

__global__ void build_radial_descriptors_from_geometry(
    int atom_count,
    int atom_stride,
    int descriptor_dim,
    int num_types,
    int num_type_pairs,
    int n_max_radial,
    int basis_size_radial,
    float rc,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ descriptor_coefficients,
    float* __restrict__ descriptors) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  for (int d = 0; d < descriptor_dim; ++d) {
    descriptors[atom + atom_stride * d] = 0.0f;
  }

  float q[kMaxFusedRadialDescriptors];
  for (int n = 0; n <= n_max_radial; ++n) {
    q[n] = 0.0f;
  }

  const int type1 = types[atom];
  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  const float rcinv = 1.0f / rc;
  const int radial_count = nn_radial[atom];
  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
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
    const float r = sqrtf(dx * dx + dy * dy + dz * dz);
    const float fc = radial_cutoff(rc, rcinv, r);
    const float x = 2.0f * (r * rcinv - 1.0f) * (r * rcinv - 1.0f) - 1.0f;
    const float half_fc = 0.5f * fc;
    const int type2 = types[neighbor];
    const int type_pair = type1 * num_types + type2;

    float t_minus_2 = 1.0f;
    float t_minus_1 = x;
    for (int k = 0; k <= basis_size_radial; ++k) {
      float fn = fc;
      if (k == 1) {
        fn = (x + 1.0f) * half_fc;
      } else if (k >= 2) {
        const float t = 2.0f * x * t_minus_1 - t_minus_2;
        t_minus_2 = t_minus_1;
        t_minus_1 = t;
        fn = (t + 1.0f) * half_fc;
      }
      for (int n = 0; n <= n_max_radial; ++n) {
        const int coefficient_index =
            (n * (basis_size_radial + 1) + k) * num_type_pairs + type_pair;
        q[n] += fn * descriptor_coefficients[coefficient_index];
      }
    }
  }

  for (int n = 0; n <= n_max_radial; ++n) {
    descriptors[atom + atom_stride * n] = q[n];
  }
}

}  // namespace

void build_radial_descriptors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.descriptor_dim > 0, "descriptor_dim must be positive");
  require(protocol.n_max_radial >= 0, "n_max_radial must be non-negative");
  require(protocol.basis_size_radial >= 0, "basis_size_radial must be non-negative");
  require(
      protocol.neighbor_capacity_radial > 0,
      "radial neighbor capacity must be positive");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView workspace_view = workspace.view();
  require(
      static_cast<std::size_t>(atom_count) <= workspace_view.atom_capacity,
      "atom_count exceeds workspace atom capacity");
  require(
      model_view.descriptor_coefficients != nullptr,
      "model missing descriptor coefficients");
  require(
      model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
      "model descriptor coefficient buffer is too small");
  require(workspace_view.types != nullptr, "workspace missing atom types");
  require(workspace_view.nn_radial != nullptr, "workspace missing radial counts");
  require(
      workspace_view.nl_radial_slot_major != nullptr,
      "workspace missing radial neighbor list");
  require(workspace_view.fn_radial != nullptr, "workspace missing radial basis cache");
  require(workspace_view.descriptors != nullptr, "workspace missing descriptor cache");

  const int threads = 32;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    build_radial_descriptors<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(workspace_view.atom_capacity),
        protocol.descriptor_dim,
        protocol.num_types,
        protocol.num_types * protocol.num_types,
        protocol.n_max_radial,
        protocol.basis_size_radial,
        protocol.neighbor_capacity_radial,
        workspace_view.types,
        workspace_view.nn_radial,
        workspace_view.nl_radial_slot_major,
        workspace_view.fn_radial,
        model_view.descriptor_coefficients,
        workspace_view.descriptors);
  }
  check_cuda(cudaGetLastError(), "build radial descriptors kernel launch failed");
  check_cuda(cudaDeviceSynchronize(), "build radial descriptors kernel failed");
}

bool build_radial_descriptors_from_geometry_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.descriptor_dim > 0, "descriptor_dim must be positive");
  require(protocol.n_max_radial >= 0, "n_max_radial must be non-negative");
  require(protocol.basis_size_radial >= 0, "basis_size_radial must be non-negative");
  require(protocol.cutoff_radial > 0.0, "radial cutoff must be positive");
  if (protocol.n_max_radial + 1 > kMaxFusedRadialDescriptors) {
    return false;
  }

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView workspace_view = workspace.view();
  require(
      static_cast<std::size_t>(atom_count) <= workspace_view.atom_capacity,
      "atom_count exceeds workspace atom capacity");
  require(
      model_view.descriptor_coefficients != nullptr,
      "model missing descriptor coefficients");
  require(
      model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
      "model descriptor coefficient buffer is too small");
  require(workspace_view.types != nullptr, "workspace missing atom types");
  require(workspace_view.positions_soa3 != nullptr, "workspace missing positions");
  require(workspace_view.nn_radial != nullptr, "workspace missing radial counts");
  require(
      workspace_view.nl_radial_slot_major != nullptr,
      "workspace missing radial neighbor list");
  require(workspace_view.descriptors != nullptr, "workspace missing descriptor cache");

  const int threads = 32;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    build_radial_descriptors_from_geometry<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(workspace_view.atom_capacity),
        protocol.descriptor_dim,
        protocol.num_types,
        protocol.num_types * protocol.num_types,
        protocol.n_max_radial,
        protocol.basis_size_radial,
        static_cast<float>(protocol.cutoff_radial),
        box,
        workspace_view.types,
        workspace_view.positions_soa3,
        workspace_view.nn_radial,
        workspace_view.nl_radial_slot_major,
        model_view.descriptor_coefficients,
        workspace_view.descriptors);
  }
  check_cuda(
      cudaGetLastError(),
      "build radial descriptors from geometry kernel launch failed");
  check_cuda(
      cudaDeviceSynchronize(),
      "build radial descriptors from geometry kernel failed");
  return true;
}

}  // namespace nep_adapters::cuda_backend
