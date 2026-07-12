#include "device_operations.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nep_adapters::cuda_backend {
namespace {

constexpr int kThreads = 256;

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

__device__ __forceinline__ int public_virial_to_internal(int component) {
  constexpr int map[] = {0, 3, 4, 6, 1, 5, 7, 8, 2};
  return map[component];
}

__global__ void pack_atom_outputs(
    int atom_count,
    int atom_stride,
    const double* force_soa3,
    const double* mforce_soa3,
    const double* virial_soa9,
    double* output_forces_aos3,
    double* output_mforces_aos3,
    double* output_virials_per_atom_row_major9) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  output_forces_aos3[3 * atom + 0] = force_soa3[atom];
  output_forces_aos3[3 * atom + 1] = force_soa3[atom_stride + atom];
  output_forces_aos3[3 * atom + 2] = force_soa3[2 * atom_stride + atom];
  if (output_mforces_aos3 != nullptr && mforce_soa3 != nullptr) {
    output_mforces_aos3[3 * atom + 0] = mforce_soa3[atom];
    output_mforces_aos3[3 * atom + 1] = mforce_soa3[atom_stride + atom];
    output_mforces_aos3[3 * atom + 2] = mforce_soa3[2 * atom_stride + atom];
  }

  for (int component = 0; component < 9; ++component) {
    const int internal = public_virial_to_internal(component);
    output_virials_per_atom_row_major9[9 * atom + component] =
        virial_soa9[internal * atom_stride + atom];
  }
}

__global__ void reduce_structure_outputs(
    int structure_count,
    int atom_stride,
    const int* structure_atom_counts,
    const int* structure_atom_offsets,
    const double* potential,
    const double* virial_soa9,
    double* structure_energy,
    double* structure_virial_row_major9) {
  const int structure = blockIdx.x;
  const int quantity = blockIdx.y;
  if (structure >= structure_count || quantity >= 10) {
    return;
  }

  const int atom_count = structure_atom_counts[structure];
  const int atom_offset = structure_atom_offsets[structure];
  double sum = 0.0;
  for (int local = threadIdx.x; local < atom_count; local += blockDim.x) {
    const int atom = atom_offset + local;
    if (quantity == 0) {
      sum += potential[atom];
    } else {
      const int internal = public_virial_to_internal(quantity - 1);
      sum += virial_soa9[internal * atom_stride + atom];
    }
  }

  extern __shared__ double partial[];
  partial[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    if (quantity == 0) {
      structure_energy[structure] = partial[0];
    } else {
      structure_virial_row_major9[9 * structure + (quantity - 1)] = partial[0];
    }
  }
}

}  // namespace

void prepare_batched_outputs(
    int structure_count,
    int atom_count,
    DeviceWorkspace& workspace) {
  require(structure_count > 0, "structure_count must be positive");
  require(atom_count > 0, "atom_count must be positive");
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(static_cast<std::size_t>(structure_count) <= view.structure_capacity,
          "structure_count exceeds workspace structure capacity");
  require(view.structure_atom_counts != nullptr,
          "workspace missing structure atom counts");
  require(view.structure_atom_offsets != nullptr,
          "workspace missing structure atom offsets");
  require(view.potential != nullptr, "workspace missing potential");
  require(view.force_soa3 != nullptr, "workspace missing force output");
  require(view.virial_soa9 != nullptr, "workspace missing virial output");
  require(view.output_forces_aos3 != nullptr, "workspace missing packed forces");
  require(view.output_virials_per_atom_row_major9 != nullptr,
          "workspace missing packed per-atom virials");
  require(view.structure_energy != nullptr, "workspace missing structure energy");
  require(view.structure_virial_row_major9 != nullptr,
          "workspace missing structure virial");

  const int atom_blocks = (atom_count + kThreads - 1) / kThreads;
  pack_atom_outputs<<<atom_blocks, kThreads>>>(
      atom_count,
      static_cast<int>(view.atom_capacity),
      view.force_soa3,
      view.mforce_soa3,
      view.virial_soa9,
      view.output_forces_aos3,
      view.output_mforces_aos3,
      view.output_virials_per_atom_row_major9);
  check_cuda(cudaGetLastError(), "pack batched atom outputs");

  const dim3 reduce_blocks(structure_count, 10);
  reduce_structure_outputs<<<
      reduce_blocks,
      kThreads,
      kThreads * sizeof(double)>>>(
      structure_count,
      static_cast<int>(view.atom_capacity),
      view.structure_atom_counts,
      view.structure_atom_offsets,
      view.potential,
      view.virial_soa9,
      view.structure_energy,
      view.structure_virial_row_major9);
  check_cuda(cudaGetLastError(), "reduce batched structure outputs");
  check_cuda(cudaDeviceSynchronize(), "prepare batched outputs");
}

}  // namespace nep_adapters::cuda_backend
