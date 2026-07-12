#include "device_operations.hpp"

#include <cuda_runtime.h>

#include <cstddef>
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

__device__ __forceinline__ double lammps_voigt_component(
    const double* virial_soa9,
    int atom_stride,
    int atom,
    int component) {
  switch (component) {
    case 0:
      return virial_soa9[atom];
    case 1:
      return virial_soa9[atom_stride + atom];
    case 2:
      return virial_soa9[2 * atom_stride + atom];
    case 3:
      return 0.5 * (virial_soa9[3 * atom_stride + atom] +
                    virial_soa9[6 * atom_stride + atom]);
    case 4:
      return 0.5 * (virial_soa9[4 * atom_stride + atom] +
                    virial_soa9[7 * atom_stride + atom]);
    case 5:
      return 0.5 * (virial_soa9[5 * atom_stride + atom] +
                    virial_soa9[8 * atom_stride + atom]);
  }
  return 0.0;
}

__global__ void write_lammps_atom_outputs(
    int nlocal,
    int atom_stride,
    const double* potential,
    const double* virial_soa9,
    double* potential_per_atom,
    double* virials_per_atom9,
    int virial_atom_stride,
    int virial_component_stride) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= nlocal) {
    return;
  }

  if (potential_per_atom != nullptr) {
    potential_per_atom[atom] = potential[atom];
  }

  if (virials_per_atom9 != nullptr) {
    for (int component = 0; component < 9; ++component) {
      virials_per_atom9[
          atom * virial_atom_stride + component * virial_component_stride] =
          virial_soa9[component * atom_stride + atom];
    }
  }
}

__global__ void write_lammps_forces(
    int nlocal,
    int atom_stride,
    const double* force_soa3,
    double* forces,
    int force_atom_stride,
    int force_component_stride) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= nlocal) {
    return;
  }

  forces[atom * force_atom_stride] = force_soa3[atom];
  forces[atom * force_atom_stride + force_component_stride] =
      force_soa3[atom_stride + atom];
  forces[atom * force_atom_stride + 2 * force_component_stride] =
      force_soa3[2 * atom_stride + atom];
}

__global__ void reduce_lammps_partial_sums(
    int nlocal,
    int atom_stride,
    int partial_block_count,
    const double* potential,
    const double* virial_soa9,
    double* partial_sums) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  const int component = blockIdx.y;
  double sum = 0.0;
  if (atom < nlocal) {
    if (component == 0) {
      sum = potential[atom];
    } else {
      sum = lammps_voigt_component(
          virial_soa9,
          atom_stride,
          atom,
          component - 1);
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
    partial_sums[
        static_cast<std::size_t>(component) * partial_block_count + blockIdx.x] =
        partial[0];
  }
}

__global__ void finalize_lammps_totals(
    int partial_block_count,
    const double* partial_sums,
    double* total_potential,
    double* total_virial6) {
  const int component = blockIdx.x;
  double sum = 0.0;
  for (int block = threadIdx.x; block < partial_block_count; block += blockDim.x) {
    sum += partial_sums[
        static_cast<std::size_t>(component) * partial_block_count + block];
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

  if (threadIdx.x != 0) {
    return;
  }
  if (component == 0) {
    *total_potential = partial[0];
    return;
  }
  total_virial6[component - 1] = partial[0];
}

}  // namespace

void write_lammps_device_outputs(
    const NepaLammpsDeviceNeighborInput& input,
    const NepaLammpsDeviceNeighborResult& result,
    DeviceWorkspace& workspace) {
  require(input.nlocal >= 0, "negative nlocal");
  require(input.nall >= input.nlocal, "nall must be at least nlocal");
  const bool write_totals =
      result.total_potential != nullptr || result.total_virial6 != nullptr;
  require(!write_totals ||
              (result.total_potential != nullptr &&
               result.total_virial6 != nullptr),
          "incomplete device total output");
  require(result.forces != nullptr, "missing device force output");
  require(result.force_atom_stride > 0, "invalid force atom stride");
  require(result.force_component_stride > 0, "invalid force component stride");
  if (result.mforces != nullptr) {
    require(result.mforce_atom_stride > 0, "invalid mforce atom stride");
    require(result.mforce_component_stride > 0, "invalid mforce component stride");
  }
  if (result.virials_per_atom9 != nullptr) {
    require(result.virial_atom_stride > 0, "invalid virial atom stride");
    require(result.virial_component_stride > 0, "invalid virial component stride");
  }

  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(input.nall) <= view.atom_capacity,
          "nall exceeds workspace atom capacity");
  require(view.potential != nullptr, "workspace missing potential");
  require(view.force_soa3 != nullptr, "workspace missing force output");
  if (result.mforces != nullptr) {
    require(view.mforce_soa3 != nullptr, "workspace missing mforce output");
  }
  require(view.virial_soa9 != nullptr, "workspace missing virial output");
  if (write_totals) {
    require(view.lammps_partial_sums != nullptr,
            "workspace missing LAMMPS reduction partials");
  }

  const int force_blocks = (input.nall + kThreads - 1) / kThreads;
  if (force_blocks > 0) {
    write_lammps_forces<<<force_blocks, kThreads>>>(
        input.nall,
        static_cast<int>(view.atom_capacity),
        view.force_soa3,
        result.forces,
        result.force_atom_stride,
        result.force_component_stride);
    check_cuda(cudaGetLastError(), "write device LAMMPS forces");
    if (result.mforces != nullptr) {
      write_lammps_forces<<<force_blocks, kThreads>>>(
          input.nall,
          static_cast<int>(view.atom_capacity),
          view.mforce_soa3,
          result.mforces,
          result.mforce_atom_stride,
          result.mforce_component_stride);
      check_cuda(cudaGetLastError(), "write device LAMMPS mforces");
    }
  }

  const int atom_blocks = (input.nlocal + kThreads - 1) / kThreads;
  if (atom_blocks > 0 &&
      (result.potential_per_atom != nullptr ||
       result.virials_per_atom9 != nullptr)) {
    write_lammps_atom_outputs<<<atom_blocks, kThreads>>>(
        input.nlocal,
        static_cast<int>(view.atom_capacity),
        view.potential,
        view.virial_soa9,
        result.potential_per_atom,
        result.virials_per_atom9,
        result.virial_atom_stride,
        result.virial_component_stride);
    check_cuda(cudaGetLastError(), "write device LAMMPS atom outputs");
  }

  if (!write_totals) {
    return;
  }

  if (input.nlocal == 0) {
    check_cuda(
        cudaMemset(result.total_potential, 0, sizeof(double)),
        "clear empty device LAMMPS total potential");
    check_cuda(
        cudaMemset(result.total_virial6, 0, 6 * sizeof(double)),
        "clear empty device LAMMPS total virial");
    return;
  }

  const int partial_blocks = (input.nlocal + kThreads - 1) / kThreads;
  const dim3 reduction_grid(partial_blocks, 7);
  reduce_lammps_partial_sums<<<
      reduction_grid,
      kThreads,
      kThreads * sizeof(double)>>>(
      input.nlocal,
      static_cast<int>(view.atom_capacity),
      partial_blocks,
      view.potential,
      view.virial_soa9,
      view.lammps_partial_sums);
  check_cuda(cudaGetLastError(), "reduce device LAMMPS partial sums");

  finalize_lammps_totals<<<7, kThreads, kThreads * sizeof(double)>>>(
      partial_blocks,
      view.lammps_partial_sums,
      result.total_potential,
      result.total_virial6);
  check_cuda(cudaGetLastError(), "finalize device LAMMPS totals");
}

}  // namespace nep_adapters::cuda_backend
