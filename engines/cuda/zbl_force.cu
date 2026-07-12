#include "device_operations.hpp"
#include "simulation_box_device.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace nep_adapters::cuda_backend {
namespace {

constexpr float kPi = 3.1415927f;
constexpr float kCoulomb = 14.399645f;

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

__device__ __forceinline__ void find_fc_and_fcp_zbl(
    float inner,
    float outer,
    float r,
    float& fc,
    float& fcp) {
  if (r < inner) {
    fc = 1.0f;
    fcp = 0.0f;
  } else if (r < outer) {
    const float factor = kPi / (outer - inner);
    fc = 0.5f * cosf(factor * (r - inner)) + 0.5f;
    fcp = -0.5f * sinf(factor * (r - inner)) * factor;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

__device__ __forceinline__ void add_phi_zbl(
    float a,
    float b,
    float x,
    float& phi,
    float& phip) {
  const float value = a * expf(-b * x);
  phi += value;
  phip -= b * value;
}

__device__ __forceinline__ void find_f_and_fp_zbl(
    float zizj,
    float a_inv,
    float inner,
    float outer,
    float r,
    float rinv,
    float& f,
    float& fp) {
  const float x = r * a_inv;
  f = 0.0f;
  fp = 0.0f;
  add_phi_zbl(0.18175f, 3.1998f, x, f, fp);
  add_phi_zbl(0.50986f, 0.94229f, x, f, fp);
  add_phi_zbl(0.28022f, 0.4029f, x, f, fp);
  add_phi_zbl(0.02817f, 0.20162f, x, f, fp);

  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * rinv - f * rinv * rinv;
  f *= rinv;

  float fc = 0.0f;
  float fcp = 0.0f;
  find_fc_and_fcp_zbl(inner, outer, r, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

template <bool AccumulateEnergyVirial, bool VirialToNeighbor>
__global__ void accumulate_zbl_forces(
    int atom_count,
    int atom_stride,
    int num_types,
    float zbl_inner,
    float zbl_outer,
    SimulationBox box,
    const int* __restrict__ types,
    const int* __restrict__ atomic_numbers,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    double* __restrict__ potential,
    double* __restrict__ force_soa3,
    double* __restrict__ virial_soa9) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int type1 = types[atom];
  if (type1 < 0 || type1 >= num_types) {
    return;
  }
  const int zi = atomic_numbers[type1];
  const float pow_zi = powf(static_cast<float>(zi), 0.23f);
  const float zbl_outer_squared = zbl_outer * zbl_outer;
  const double x1 = positions_soa3[atom];
  const double y1 = positions_soa3[atom_stride + atom];
  const double z1 = positions_soa3[2 * atom_stride + atom];

  float s_pe = 0.0f;
  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_sxx = 0.0f;
  float s_sxy = 0.0f;
  float s_sxz = 0.0f;
  float s_syx = 0.0f;
  float s_syy = 0.0f;
  float s_syz = 0.0f;
  float s_szx = 0.0f;
  float s_szy = 0.0f;
  float s_szz = 0.0f;

  for (int slot = 0; slot < nn_radial[atom]; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    const int type2 = types[neighbor];
    if (type2 < 0 || type2 >= num_types) {
      continue;
    }
    const int zj = atomic_numbers[type2];
    float x12 = 0.0f;
    float y12 = 0.0f;
    float z12 = 0.0f;
    minimum_image_delta(
        box,
        positions_soa3[neighbor] - x1,
        positions_soa3[atom_stride + neighbor] - y1,
        positions_soa3[2 * atom_stride + neighbor] - z1,
        x12,
        y12,
        z12);
    const float r2 = x12 * x12 + y12 * y12 + z12 * z12;
    if (r2 <= 0.0f || r2 >= zbl_outer_squared) {
      continue;
    }
    const float r = sqrtf(r2);

    const float rinv = 1.0f / r;
    const float a_inv = (pow_zi + powf(static_cast<float>(zj), 0.23f)) * 2.134563f;
    const float zizj = kCoulomb * static_cast<float>(zi * zj);
    float f = 0.0f;
    float fp = 0.0f;
    find_f_and_fp_zbl(zizj, a_inv, zbl_inner, zbl_outer, r, rinv, f, fp);

    const float force_scale = 0.5f * fp * rinv;
    const float f12x = x12 * force_scale;
    const float f12y = y12 * force_scale;
    const float f12z = z12 * force_scale;

    s_fx += 2.0f * f12x;
    s_fy += 2.0f * f12y;
    s_fz += 2.0f * f12z;
    if constexpr (AccumulateEnergyVirial) {
      if constexpr (VirialToNeighbor) {
        atomicAdd(&virial_soa9[neighbor], -static_cast<double>(x12 * f12x));
        atomicAdd(
            &virial_soa9[atom_stride + neighbor],
            -static_cast<double>(y12 * f12y));
        atomicAdd(
            &virial_soa9[2 * atom_stride + neighbor],
            -static_cast<double>(z12 * f12z));
        atomicAdd(
            &virial_soa9[3 * atom_stride + neighbor],
            -static_cast<double>(x12 * f12y));
        atomicAdd(
            &virial_soa9[4 * atom_stride + neighbor],
            -static_cast<double>(x12 * f12z));
        atomicAdd(
            &virial_soa9[5 * atom_stride + neighbor],
            -static_cast<double>(y12 * f12z));
        atomicAdd(
            &virial_soa9[6 * atom_stride + neighbor],
            -static_cast<double>(y12 * f12x));
        atomicAdd(
            &virial_soa9[7 * atom_stride + neighbor],
            -static_cast<double>(z12 * f12x));
        atomicAdd(
            &virial_soa9[8 * atom_stride + neighbor],
            -static_cast<double>(z12 * f12y));
      } else {
        s_sxx -= x12 * f12x;
        s_sxy -= x12 * f12y;
        s_sxz -= x12 * f12z;
        s_syx -= y12 * f12x;
        s_syy -= y12 * f12y;
        s_syz -= y12 * f12z;
        s_szx -= z12 * f12x;
        s_szy -= z12 * f12y;
        s_szz -= z12 * f12z;
      }
      s_pe += 0.5f * f;
    }
  }

  force_soa3[atom] += static_cast<double>(s_fx);
  force_soa3[atom_stride + atom] += static_cast<double>(s_fy);
  force_soa3[2 * atom_stride + atom] += static_cast<double>(s_fz);
  if constexpr (AccumulateEnergyVirial) {
    potential[atom] += static_cast<double>(s_pe);
    if constexpr (!VirialToNeighbor) {
      virial_soa9[atom] += static_cast<double>(s_sxx);
      virial_soa9[atom_stride + atom] += static_cast<double>(s_syy);
      virial_soa9[2 * atom_stride + atom] += static_cast<double>(s_szz);
      virial_soa9[3 * atom_stride + atom] += static_cast<double>(s_sxy);
      virial_soa9[4 * atom_stride + atom] += static_cast<double>(s_sxz);
      virial_soa9[5 * atom_stride + atom] += static_cast<double>(s_syz);
      virial_soa9[6 * atom_stride + atom] += static_cast<double>(s_syx);
      virial_soa9[7 * atom_stride + atom] += static_cast<double>(s_szx);
      virial_soa9[8 * atom_stride + atom] += static_cast<double>(s_szy);
    }
  }
}

template <bool VirialToNeighbor>
__global__ void accumulate_zbl_forces_batched(
    int atom_count,
    int atom_stride,
    int num_types,
    float zbl_inner,
    float zbl_outer,
    const int* __restrict__ atom_to_structure,
    const double* __restrict__ boxes_row_major9,
    const double* __restrict__ box_inverse_row_major9,
    const int* __restrict__ pbc_flags3,
    const int* __restrict__ types,
    const int* __restrict__ atomic_numbers,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    double* __restrict__ potential,
    double* __restrict__ force_soa3,
    double* __restrict__ virial_soa9) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const SimulationBox box = load_structure_box(
      atom_to_structure[atom],
      boxes_row_major9,
      box_inverse_row_major9,
      pbc_flags3);
  const int type1 = types[atom];
  if (type1 < 0 || type1 >= num_types) {
    return;
  }
  const int zi = atomic_numbers[type1];
  const float pow_zi = powf(static_cast<float>(zi), 0.23f);
  const float zbl_outer_squared = zbl_outer * zbl_outer;
  const double x1 = positions_soa3[atom];
  const double y1 = positions_soa3[atom_stride + atom];
  const double z1 = positions_soa3[2 * atom_stride + atom];

  float s_pe = 0.0f;
  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_sxx = 0.0f;
  float s_sxy = 0.0f;
  float s_sxz = 0.0f;
  float s_syx = 0.0f;
  float s_syy = 0.0f;
  float s_syz = 0.0f;
  float s_szx = 0.0f;
  float s_szy = 0.0f;
  float s_szz = 0.0f;

  for (int slot = 0; slot < nn_radial[atom]; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    const int type2 = types[neighbor];
    if (type2 < 0 || type2 >= num_types) {
      continue;
    }
    const int zj = atomic_numbers[type2];
    float x12 = 0.0f;
    float y12 = 0.0f;
    float z12 = 0.0f;
    minimum_image_delta(
        box,
        positions_soa3[neighbor] - x1,
        positions_soa3[atom_stride + neighbor] - y1,
        positions_soa3[2 * atom_stride + neighbor] - z1,
        x12,
        y12,
        z12);
    const float r2 = x12 * x12 + y12 * y12 + z12 * z12;
    if (r2 <= 0.0f || r2 >= zbl_outer_squared) {
      continue;
    }
    const float r = sqrtf(r2);

    const float rinv = 1.0f / r;
    const float a_inv = (pow_zi + powf(static_cast<float>(zj), 0.23f)) * 2.134563f;
    const float zizj = kCoulomb * static_cast<float>(zi * zj);
    float f = 0.0f;
    float fp = 0.0f;
    find_f_and_fp_zbl(zizj, a_inv, zbl_inner, zbl_outer, r, rinv, f, fp);

    const float force_scale = 0.5f * fp * rinv;
    const float f12x = x12 * force_scale;
    const float f12y = y12 * force_scale;
    const float f12z = z12 * force_scale;

    s_fx += 2.0f * f12x;
    s_fy += 2.0f * f12y;
    s_fz += 2.0f * f12z;
    if constexpr (VirialToNeighbor) {
      atomicAdd(&virial_soa9[neighbor], -static_cast<double>(x12 * f12x));
      atomicAdd(
          &virial_soa9[atom_stride + neighbor],
          -static_cast<double>(y12 * f12y));
      atomicAdd(
          &virial_soa9[2 * atom_stride + neighbor],
          -static_cast<double>(z12 * f12z));
      atomicAdd(
          &virial_soa9[3 * atom_stride + neighbor],
          -static_cast<double>(x12 * f12y));
      atomicAdd(
          &virial_soa9[4 * atom_stride + neighbor],
          -static_cast<double>(x12 * f12z));
      atomicAdd(
          &virial_soa9[5 * atom_stride + neighbor],
          -static_cast<double>(y12 * f12z));
      atomicAdd(
          &virial_soa9[6 * atom_stride + neighbor],
          -static_cast<double>(y12 * f12x));
      atomicAdd(
          &virial_soa9[7 * atom_stride + neighbor],
          -static_cast<double>(z12 * f12x));
      atomicAdd(
          &virial_soa9[8 * atom_stride + neighbor],
          -static_cast<double>(z12 * f12y));
    } else {
      s_sxx -= x12 * f12x;
      s_sxy -= x12 * f12y;
      s_sxz -= x12 * f12z;
      s_syx -= y12 * f12x;
      s_syy -= y12 * f12y;
      s_syz -= y12 * f12z;
      s_szx -= z12 * f12x;
      s_szy -= z12 * f12y;
      s_szz -= z12 * f12z;
    }
    s_pe += 0.5f * f;
  }

  potential[atom] += static_cast<double>(s_pe);
  force_soa3[atom] += static_cast<double>(s_fx);
  force_soa3[atom_stride + atom] += static_cast<double>(s_fy);
  force_soa3[2 * atom_stride + atom] += static_cast<double>(s_fz);
  if constexpr (!VirialToNeighbor) {
    virial_soa9[atom] += static_cast<double>(s_sxx);
    virial_soa9[atom_stride + atom] += static_cast<double>(s_syy);
    virial_soa9[2 * atom_stride + atom] += static_cast<double>(s_szz);
    virial_soa9[3 * atom_stride + atom] += static_cast<double>(s_sxy);
    virial_soa9[4 * atom_stride + atom] += static_cast<double>(s_sxz);
    virial_soa9[5 * atom_stride + atom] += static_cast<double>(s_syz);
    virial_soa9[6 * atom_stride + atom] += static_cast<double>(s_syx);
    virial_soa9[7 * atom_stride + atom] += static_cast<double>(s_szx);
    virial_soa9[8 * atom_stride + atom] += static_cast<double>(s_szy);
  }
}

}  // namespace

void accumulate_zbl_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_energy_virial,
    bool virial_to_neighbor) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.has_zbl, "ZBL force kernel requires a ZBL model");
  require(!protocol.flexible_zbl, "flexible ZBL is not supported yet");
  require(protocol.zbl_inner >= 0.0, "ZBL inner cutoff must be non-negative");
  require(protocol.zbl_outer > protocol.zbl_inner, "ZBL outer cutoff must exceed inner");
  require(protocol.zbl_outer <= protocol.cutoff_radial,
          "current ZBL force kernel reuses the radial neighbor list");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(model_view.atomic_numbers != nullptr, "model missing atomic numbers");
  require(model_view.atomic_numbers_count >= static_cast<std::size_t>(protocol.num_types),
          "model atomic number buffer is too small");
  require(view.types != nullptr, "workspace missing atom types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_radial != nullptr, "workspace missing radial counts");
  require(
      view.nl_radial_slot_major != nullptr,
      "workspace missing radial neighbors");
  require(view.potential != nullptr, "workspace missing potential output");
  require(view.force_soa3 != nullptr, "workspace missing force output");
  require(view.virial_soa9 != nullptr, "workspace missing virial output");

  const int threads = 128;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    const auto launch = [&](auto accumulate_tag, auto virial_to_neighbor_tag) {
      constexpr bool kAccumulateEnergyVirial = decltype(accumulate_tag)::value;
      constexpr bool kVirialToNeighbor =
          decltype(virial_to_neighbor_tag)::value;
      accumulate_zbl_forces<kAccumulateEnergyVirial, kVirialToNeighbor>
          <<<blocks, threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.num_types,
          static_cast<float>(protocol.zbl_inner),
          static_cast<float>(protocol.zbl_outer),
          box,
          view.types,
          model_view.atomic_numbers,
          view.positions_soa3,
          view.nn_radial,
          view.nl_radial_slot_major,
          view.potential,
          view.force_soa3,
          view.virial_soa9);
    };
    if (accumulate_energy_virial) {
      if (virial_to_neighbor) {
        launch(std::true_type{}, std::true_type{});
      } else {
        launch(std::true_type{}, std::false_type{});
      }
    } else {
      launch(std::false_type{}, std::false_type{});
    }
  }
  check_cuda(cudaGetLastError(), "accumulate ZBL forces kernel launch failed");
}

void accumulate_zbl_forces_batched(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool virial_to_neighbor) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.has_zbl, "ZBL force kernel requires a ZBL model");
  require(!protocol.flexible_zbl, "flexible ZBL is not supported yet");
  require(protocol.zbl_inner >= 0.0, "ZBL inner cutoff must be non-negative");
  require(protocol.zbl_outer > protocol.zbl_inner, "ZBL outer cutoff must exceed inner");
  require(protocol.zbl_outer <= protocol.cutoff_radial,
          "current ZBL force kernel reuses the radial neighbor list");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(model_view.atomic_numbers != nullptr, "model missing atomic numbers");
  require(model_view.atomic_numbers_count >= static_cast<std::size_t>(protocol.num_types),
          "model atomic number buffer is too small");
  require(view.atom_to_structure != nullptr, "workspace missing atom_to_structure");
  require(view.boxes_row_major9 != nullptr, "workspace missing boxes");
  require(view.box_inverse_row_major9 != nullptr, "workspace missing box inverses");
  require(view.pbc_flags3 != nullptr, "workspace missing pbc flags");
  require(view.types != nullptr, "workspace missing atom types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_radial != nullptr, "workspace missing radial counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.potential != nullptr, "workspace missing potential output");
  require(view.force_soa3 != nullptr, "workspace missing force output");
  require(view.virial_soa9 != nullptr, "workspace missing virial output");

  const int threads = 128;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    const auto launch = [&](auto virial_to_neighbor_tag) {
      constexpr bool kVirialToNeighbor =
          decltype(virial_to_neighbor_tag)::value;
      accumulate_zbl_forces_batched<kVirialToNeighbor><<<blocks, threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.num_types,
          static_cast<float>(protocol.zbl_inner),
          static_cast<float>(protocol.zbl_outer),
          view.atom_to_structure,
          view.boxes_row_major9,
          view.box_inverse_row_major9,
          view.pbc_flags3,
          view.types,
          model_view.atomic_numbers,
          view.positions_soa3,
          view.nn_radial,
          view.nl_radial_slot_major,
          view.potential,
          view.force_soa3,
          view.virial_soa9);
    };
    if (virial_to_neighbor) {
      launch(std::true_type{});
    } else {
      launch(std::false_type{});
    }
  }
  check_cuda(cudaGetLastError(), "accumulate batched ZBL forces launch failed");
  check_cuda(cudaDeviceSynchronize(), "accumulate batched ZBL forces failed");
}

}  // namespace nep_adapters::cuda_backend
