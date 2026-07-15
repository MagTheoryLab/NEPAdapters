#include "device_operations.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace nep_adapters::cuda_backend {
namespace {

constexpr int kWarpSize = 32;
constexpr int kAnnThreadsPerAtom = 4;
constexpr int kAnnAtomsPerWarp = kWarpSize / kAnnThreadsPerAtom;
constexpr int kPreferredAnnWarpsPerBlock = 4;
constexpr int kAnnScheduleThreads = 256;
// Reorder only independent ANN work inside a bounded window. Descriptors and
// derivatives remain in atom order so descriptor and force accesses stay coalesced.
constexpr std::size_t kDefaultDynamicSharedMemoryBytes = 48 * 1024;
constexpr unsigned kFullWarpMask = 0xffffffffu;

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

__device__ __forceinline__ float ann_subwarp_sum(float value) {
  for (int offset = kAnnThreadsPerAtom / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(
        kFullWarpMask, value, offset, kAnnThreadsPerAtom);
  }
  return value;
}

__global__ void build_local_type_schedule(
    int atom_count,
    int num_types,
    const int* __restrict__ types,
    int* __restrict__ scheduled_atoms,
    int* __restrict__ schedule_identity) {
  extern __shared__ int shared_group_storage[];
  int* counts = shared_group_storage;
  int* offsets = counts + num_types;
  __shared__ int active_type_count;
  __shared__ int invalid_offset;
  for (int type = threadIdx.x; type < num_types; type += blockDim.x) {
    counts[type] = 0;
  }
  __syncthreads();

  const int atom_base = blockIdx.x * kAnnScheduleWindowAtoms;
  const int local_count = min(kAnnScheduleWindowAtoms, atom_count - atom_base);
  for (int local_atom = threadIdx.x; local_atom < local_count;
       local_atom += blockDim.x) {
    const int atom = atom_base + local_atom;
    const int type = types[atom];
    if (type >= 0 && type < num_types) {
      atomicAdd(counts + type, 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int next = atom_base;
    int active = 0;
    for (int type = 0; type < num_types; ++type) {
      offsets[type] = next;
      next += counts[type];
      active += counts[type] > 0 ? 1 : 0;
    }
    active_type_count = active;
    invalid_offset = next;
    schedule_identity[blockIdx.x] = active <= 1 ? 1 : 0;
  }
  __syncthreads();

  if (active_type_count <= 1) {
    return;
  }

  for (int local_atom = threadIdx.x; local_atom < local_count;
       local_atom += blockDim.x) {
    const int atom = atom_base + local_atom;
    const int type = types[atom];
    if (type >= 0 && type < num_types) {
      const int grouped = atomicAdd(offsets + type, 1);
      scheduled_atoms[grouped] = atom;
    } else {
      const int grouped = atomicAdd(&invalid_offset, 1);
      scheduled_atoms[grouped] = atom;
    }
  }
}

__global__ void evaluate_ann_energy_scheduled_subwarp(
    int atom_count,
    int atom_stride,
    int version,
    int descriptor_dim,
    int hidden_neurons,
    int num_types,
    const int* __restrict__ types,
    const int* __restrict__ scheduled_atoms,
    const int* __restrict__ schedule_identity,
    const float* __restrict__ descriptors,
    const float* __restrict__ ann_type_major,
    const float* __restrict__ spin_baseline,
    double* __restrict__ potential,
    float* __restrict__ fp) {
  extern __shared__ float shared_storage[];
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int atom_in_warp = lane / kAnnThreadsPerAtom;
  const int sublane = lane & (kAnnThreadsPerAtom - 1);
  const int atoms_per_block = warps_per_block * kAnnAtomsPerWarp;
  const int warp_atom_base =
      blockIdx.x * atoms_per_block + warp * kAnnAtomsPerWarp;
  const int grouped_atom = warp_atom_base + atom_in_warp;
  const bool atom_is_valid = grouped_atom < atom_count;
  const bool identity_schedule =
      schedule_identity == nullptr ||
      schedule_identity[grouped_atom / kAnnScheduleWindowAtoms] != 0;
  int atom = -1;
  if (atom_is_valid) {
    atom = grouped_atom;
    if (!identity_schedule) {
      atom = scheduled_atoms[grouped_atom];
    }
  }

  const int shared_stride =
      kAnnAtomsPerWarp * (descriptor_dim + hidden_neurons);
  float* descriptor_shared = shared_storage + warp * shared_stride;
  float* hidden_delta_shared =
      descriptor_shared + kAnnAtomsPerWarp * descriptor_dim;
  const int shared_descriptor_count = kAnnAtomsPerWarp * descriptor_dim;
  if (identity_schedule) {
    for (int index = lane;
         index < shared_descriptor_count;
         index += kWarpSize) {
      const int descriptor = index / kAnnAtomsPerWarp;
      const int source_offset = index & (kAnnAtomsPerWarp - 1);
      const int source_atom = warp_atom_base + source_offset;
      descriptor_shared[index] = source_atom < atom_count
          ? descriptors[source_atom + atom_stride * descriptor]
          : 0.0f;
    }
  } else {
    for (int base = 0; base < shared_descriptor_count; base += kWarpSize) {
      const int index = base + lane;
      const int source_offset = index & (kAnnAtomsPerWarp - 1);
      const int source_atom = __shfl_sync(
          kFullWarpMask,
          atom,
          source_offset * kAnnThreadsPerAtom);
      if (index < shared_descriptor_count) {
        const int descriptor = index / kAnnAtomsPerWarp;
        const int source_scheduled_atom = warp_atom_base + source_offset;
        descriptor_shared[index] = source_scheduled_atom < atom_count
            ? descriptors[source_atom + atom_stride * descriptor]
            : 0.0f;
      }
    }
  }
  __syncwarp();

  const int type = atom_is_valid ? types[atom] : -1;
  const bool type_is_valid = type >= 0 && type < num_types;
  const int safe_type = type_is_valid ? type : 0;
  const int w0_count = hidden_neurons * descriptor_dim;
  const int type_extra_bias_count = version == 5 ? 1 : 0;
  const int type_block_size =
      w0_count + hidden_neurons + hidden_neurons + type_extra_bias_count;
  const float* w0 = ann_type_major + safe_type * type_block_size;
  const float* b0 = w0 + w0_count;
  const float* w1 = b0 + hidden_neurons;
  const float* b1 = ann_type_major + num_types * type_block_size;

  float energy = 0.0f;
  for (int neuron = 0; neuron < hidden_neurons; ++neuron) {
    float partial = 0.0f;
    if (type_is_valid) {
      for (int descriptor = sublane;
           descriptor < descriptor_dim;
           descriptor += kAnnThreadsPerAtom) {
        partial += w0[neuron * descriptor_dim + descriptor] *
            descriptor_shared[
                descriptor * kAnnAtomsPerWarp + atom_in_warp];
      }
    }
    const float dot = ann_subwarp_sum(partial);
    if (sublane == 0 && type_is_valid) {
      const float value = tanhf(dot - b0[neuron]);
      energy += w1[neuron] * value;
      hidden_delta_shared[
          neuron * kAnnAtomsPerWarp + atom_in_warp] =
          w1[neuron] * (1.0f - value * value);
    }
  }
  __syncwarp();

  if (type_is_valid) {
    for (int descriptor = sublane;
         descriptor < descriptor_dim;
         descriptor += kAnnThreadsPerAtom) {
      float derivative = 0.0f;
      for (int neuron = 0; neuron < hidden_neurons; ++neuron) {
        derivative += hidden_delta_shared[
            neuron * kAnnAtomsPerWarp + atom_in_warp] *
            w0[neuron * descriptor_dim + descriptor];
      }
      fp[atom + atom_stride * descriptor] = derivative;
    }
  }
  if (sublane == 0 && type_is_valid) {
    const float type_bias = version == 5 ? w1[hidden_neurons] : 0.0f;
    const float baseline =
        spin_baseline != nullptr ? spin_baseline[type] : 0.0f;
    potential[atom] =
        static_cast<double>(energy - type_bias - b1[0] + baseline);
  }
}

__global__ void evaluate_qnep_ann(
    int atom_count,
    int atom_stride,
    int descriptor_dim,
    int hidden_neurons,
    int num_types,
    const int* __restrict__ types,
    const float* __restrict__ descriptors,
    const float* __restrict__ ann_type_major,
    double* __restrict__ potential,
    double* __restrict__ charge,
    float* __restrict__ fp,
    float* __restrict__ charge_derivative) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int type = types[atom];
  if (type < 0 || type >= num_types) {
    return;
  }
  const int w0_count = hidden_neurons * descriptor_dim;
  const int type_block_size =
      w0_count + hidden_neurons + hidden_neurons + hidden_neurons;
  const float* w0 = ann_type_major + type * type_block_size;
  const float* b0 = w0 + w0_count;
  const float* energy_w1 = b0 + hidden_neurons;
  const float* charge_w1 = energy_w1 + hidden_neurons;
  const float* b1 = ann_type_major + num_types * type_block_size + 1;

  float energy = 0.0f;
  float q_charge = 0.0f;
  for (int d = 0; d < descriptor_dim; ++d) {
    fp[atom + atom_stride * d] = 0.0f;
    charge_derivative[atom + atom_stride * d] = 0.0f;
  }

  for (int neuron = 0; neuron < hidden_neurons; ++neuron) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < descriptor_dim; ++d) {
      const float q = descriptors[atom + atom_stride * d];
      w0_times_q += w0[neuron * descriptor_dim + d] * q;
    }
    const float x1 = tanhf(w0_times_q - b0[neuron]);
    const float tanh_derivative = 1.0f - x1 * x1;
    energy += energy_w1[neuron] * x1;
    q_charge += charge_w1[neuron] * x1;
    for (int d = 0; d < descriptor_dim; ++d) {
      const float y1 = tanh_derivative * w0[neuron * descriptor_dim + d];
      fp[atom + atom_stride * d] += energy_w1[neuron] * y1;
      charge_derivative[atom + atom_stride * d] += charge_w1[neuron] * y1;
    }
  }
  potential[atom] = static_cast<double>(energy - b1[0]);
  charge[atom] = static_cast<double>(q_charge);
}

__global__ void zero_total_charge_kernel(
    int atom_count,
    double* __restrict__ charge) {
  extern __shared__ double partial[];
  double sum = 0.0;
  for (int atom = threadIdx.x; atom < atom_count; atom += blockDim.x) {
    sum += charge[atom];
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const double mean = partial[0] / static_cast<double>(atom_count);
  for (int atom = threadIdx.x; atom < atom_count; atom += blockDim.x) {
    charge[atom] -= mean;
  }
}

__global__ void add_charge_chain_to_fp(
    int atom_count,
    int atom_stride,
    int descriptor_dim,
    const float* __restrict__ charge_derivative,
    const float* __restrict__ d_real,
    float* __restrict__ fp) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = atom_count * descriptor_dim;
  if (index >= total) {
    return;
  }
  const int atom = index % atom_count;
  const int descriptor = index / atom_count;
  const int offset = atom + atom_stride * descriptor;
  fp[offset] += charge_derivative[offset] * d_real[atom];
}

}  // namespace

static void build_local_type_schedule_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.charge_mode == 0,
          "type-scheduled ANN requires a non-charge model");
  require(protocol.num_types > 1,
          "type-scheduled ANN requires more than one model type");
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr, "workspace missing atom types");
  require(view.ann_scheduled_atoms != nullptr,
          "workspace missing ANN atom schedule");
  require(view.ann_schedule_identity != nullptr,
          "workspace missing ANN schedule identity flags");

  const int blocks =
      (atom_count + kAnnScheduleWindowAtoms - 1) /
      kAnnScheduleWindowAtoms;
  if (blocks == 0) {
    return;
  }

  const std::size_t shared_bytes =
      2 * static_cast<std::size_t>(protocol.num_types) * sizeof(int);
  if (shared_bytes > kDefaultDynamicSharedMemoryBytes) {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "get CUDA device for type grouping");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "get CUDA properties for type grouping");
    const std::size_t shared_limit = std::max(
        static_cast<std::size_t>(properties.sharedMemPerBlock),
        static_cast<std::size_t>(properties.sharedMemPerBlockOptin));
    require(shared_bytes <= shared_limit,
            "local type grouping exceeds CUDA shared-memory capacity");
    check_cuda(
        cudaFuncSetAttribute(
            build_local_type_schedule,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)),
        "configure local type-group shared memory");
  }
  build_local_type_schedule<<<
      blocks, kAnnScheduleThreads, shared_bytes>>>(
      atom_count,
      protocol.num_types,
      view.types,
      view.ann_scheduled_atoms,
      view.ann_schedule_identity);
  check_cuda(cudaGetLastError(), "build local type schedule launch failed");
}

void evaluate_ann_energy_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.version == 4 || protocol.version == 5,
          "ANN energy kernel supports NEP4/NEP5");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.descriptor_dim > 0, "descriptor_dim must be positive");
  require(protocol.hidden_neurons > 0, "hidden_neurons must be positive");

  const bool uses_type_schedule = protocol.num_types > 1;
  if (uses_type_schedule) {
    build_local_type_schedule_on_device(protocol, atom_count, workspace);
  }

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView workspace_view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= workspace_view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(model_view.ann_type_major_qscaled != nullptr,
          "model missing q-scaled ANN parameters");
  require(model_view.q_scaler != nullptr, "model missing q_scaler");
  require(model_view.ann_type_major_qscaled_count >= protocol.ann_parameter_count,
          "q-scaled ANN parameter buffer is too small");
  require(model_view.q_scaler_count >= static_cast<std::size_t>(protocol.descriptor_dim),
          "q_scaler buffer is too small");
  require(workspace_view.types != nullptr, "workspace missing atom types");
  require(workspace_view.descriptors != nullptr, "workspace missing descriptors");
  require(workspace_view.potential != nullptr, "workspace missing potential output");
  require(workspace_view.fp != nullptr, "workspace missing descriptor derivative cache");
  if (uses_type_schedule) {
    require(workspace_view.ann_scheduled_atoms != nullptr,
            "workspace missing ANN atom schedule");
    require(workspace_view.ann_schedule_identity != nullptr,
            "workspace missing ANN schedule identity flags");
  }

  int warps_per_block = kPreferredAnnWarpsPerBlock;
  auto shared_bytes_for_warps = [&](int warps) {
    return static_cast<std::size_t>(warps * kAnnAtomsPerWarp) *
        static_cast<std::size_t>(
            protocol.descriptor_dim + protocol.hidden_neurons) *
        sizeof(float);
  };
  std::size_t shared_bytes = shared_bytes_for_warps(warps_per_block);
  if (shared_bytes > kDefaultDynamicSharedMemoryBytes) {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "get CUDA device for scheduled ANN");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "get CUDA properties for scheduled ANN");
    const std::size_t shared_limit = std::max(
        static_cast<std::size_t>(properties.sharedMemPerBlock),
        static_cast<std::size_t>(properties.sharedMemPerBlockOptin));
    while (warps_per_block > 1 && shared_bytes > shared_limit) {
      warps_per_block /= 2;
      shared_bytes = shared_bytes_for_warps(warps_per_block);
    }
    require(shared_bytes <= shared_limit,
            "scheduled ANN subwarp state exceeds CUDA shared-memory capacity");
    check_cuda(
        cudaFuncSetAttribute(
            evaluate_ann_energy_scheduled_subwarp,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)),
        "configure scheduled ANN shared memory");
  }
  const int threads = warps_per_block * kWarpSize;
  const int atoms_per_block = warps_per_block * kAnnAtomsPerWarp;
  const int blocks =
      (atom_count + atoms_per_block - 1) / atoms_per_block;
  if (blocks > 0) {
    evaluate_ann_energy_scheduled_subwarp<<<blocks, threads, shared_bytes>>>(
        atom_count,
        static_cast<int>(workspace_view.atom_capacity),
        protocol.version,
        protocol.descriptor_dim,
        protocol.hidden_neurons,
        protocol.num_types,
        workspace_view.types,
        workspace_view.ann_scheduled_atoms,
        workspace_view.ann_schedule_identity,
        workspace_view.descriptors,
        model_view.ann_type_major_qscaled,
        model_view.spin_baseline,
        workspace_view.potential,
        workspace_view.fp);
  }
  check_cuda(cudaGetLastError(), "evaluate ANN energy kernel launch failed");
}

void evaluate_qnep_ann_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.version == 4, "qNEP ANN kernel supports NEP4 charge models");
  require(protocol.charge_mode > 0, "qNEP ANN kernel requires charge mode");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.descriptor_dim > 0, "descriptor_dim must be positive");
  require(protocol.hidden_neurons > 0, "hidden_neurons must be positive");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView workspace_view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= workspace_view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(model_view.ann_type_major_qscaled != nullptr,
          "model missing q-scaled ANN parameters");
  require(model_view.ann_type_major_qscaled_count >= protocol.ann_parameter_count,
          "q-scaled ANN parameter buffer is too small");
  require(workspace_view.types != nullptr, "workspace missing atom types");
  require(workspace_view.descriptors != nullptr, "workspace missing descriptors");
  require(workspace_view.potential != nullptr, "workspace missing potential output");
  require(workspace_view.charge != nullptr, "workspace missing charge output");
  require(workspace_view.fp != nullptr, "workspace missing descriptor derivative cache");
  require(workspace_view.charge_derivative != nullptr,
          "workspace missing charge derivative cache");

  const int threads = 32;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    evaluate_qnep_ann<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(workspace_view.atom_capacity),
        protocol.descriptor_dim,
        protocol.hidden_neurons,
        protocol.num_types,
        workspace_view.types,
        workspace_view.descriptors,
        model_view.ann_type_major_qscaled,
        workspace_view.potential,
        workspace_view.charge,
        workspace_view.fp,
        workspace_view.charge_derivative);
  }
  check_cuda(cudaGetLastError(), "evaluate qNEP ANN kernel launch failed");
  check_cuda(cudaDeviceSynchronize(), "evaluate qNEP ANN kernel failed");
}

void zero_total_charge_on_device(
    int atom_count,
    DeviceWorkspace& workspace) {
  require(atom_count > 0, "atom_count must be positive");
  const DeviceWorkspaceView view = workspace.view();
  require(view.charge != nullptr, "workspace missing charge output");
  zero_total_charge_kernel<<<1, 256, 256 * sizeof(double)>>>(atom_count, view.charge);
  check_cuda(cudaGetLastError(), "zero qNEP total charge kernel launch failed");
  check_cuda(cudaDeviceSynchronize(), "zero qNEP total charge kernel failed");
}

void add_charge_chain_to_fp_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.charge_mode > 0, "charge chain requires charge mode");
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.fp != nullptr, "workspace missing descriptor derivative cache");
  require(view.charge_derivative != nullptr,
          "workspace missing charge derivative cache");
  require(view.d_real != nullptr, "workspace missing D_real cache");
  const int total = atom_count * protocol.descriptor_dim;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  if (blocks > 0) {
    add_charge_chain_to_fp<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.descriptor_dim,
        view.charge_derivative,
        view.d_real,
        view.fp);
  }
  check_cuda(cudaGetLastError(), "add qNEP charge chain kernel launch failed");
  check_cuda(cudaDeviceSynchronize(), "add qNEP charge chain kernel failed");
}

}  // namespace nep_adapters::cuda_backend
