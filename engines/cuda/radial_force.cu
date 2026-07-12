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
constexpr int kMaxCachedRadialBasisDerivatives = 16;

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

__device__ __forceinline__ void find_fc_and_fcp(
    float rc,
    float rcinv,
    float r,
    float& fc,
    float& fcp) {
  if (r < rc) {
    const float x = r * rcinv;
    fc = 0.5f * cosf(kPi * x) + 0.5f;
    fcp = -1.5707963f * sinf(kPi * x) * rcinv;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

__device__ __forceinline__ void find_fn_and_fnp(
    int n,
    float rcinv,
    float r,
    float fc,
    float fcp,
    float& fn,
    float& fnp) {
  if (n == 0) {
    fn = fc;
    fnp = fcp;
    return;
  }

  const float r_scaled = r * rcinv;
  const float x = 2.0f * (r_scaled - 1.0f) * (r_scaled - 1.0f) - 1.0f;
  if (n == 1) {
    fn = (x + 1.0f) * 0.5f;
    fnp = 2.0f * (r_scaled - 1.0f) * rcinv * fc + fn * fcp;
    fn *= fc;
    return;
  }

  float t0 = 1.0f;
  float t1 = x;
  float t2 = x;
  float u0 = 1.0f;
  float u1 = 2.0f * x;
  for (int m = 2; m <= n; ++m) {
    t2 = 2.0f * x * t1 - t0;
    t0 = t1;
    t1 = t2;
    const float u2 = 2.0f * x * u1 - u0;
    u0 = u1;
    u1 = u2;
  }
  fn = (t2 + 1.0f) * 0.5f;
  fnp = n * u0 * 2.0f * (r_scaled - 1.0f) * rcinv;
  fnp = fnp * fc + fn * fcp;
  fn *= fc;
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

template <bool IncludeZblForce, bool VirialToNeighbor>
__global__ void accumulate_radial_forces(
    int atom_count,
    int atom_stride,
    int num_types,
    int n_max_radial,
    int basis_size_radial,
    int radial_capacity,
    float cutoff_radial,
    float zbl_inner,
    float zbl_outer,
    SimulationBox box,
    const int* __restrict__ types,
    const int* __restrict__ atomic_numbers,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ descriptor_coefficients,
    double* __restrict__ force_soa3,
    double* __restrict__ virial_soa9,
    int accumulate_virial) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int type1 = types[atom];
  const double x1 = positions_soa3[atom];
  const double y1 = positions_soa3[atom_stride + atom];
  const double z1 = positions_soa3[2 * atom_stride + atom];
  const float rcinv = 1.0f / cutoff_radial;
  const int radial_basis_count =
      (n_max_radial + 1) * (basis_size_radial + 1);
  float pow_zi = 0.0f;
  float zbl_outer_squared = 0.0f;
  int zi = 0;
  if constexpr (IncludeZblForce) {
    zi = atomic_numbers[type1];
    pow_zi = powf(static_cast<float>(zi), 0.23f);
    zbl_outer_squared = zbl_outer * zbl_outer;
  }

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
    const float r = sqrtf(r2);
    if (r <= 0.0f) {
      continue;
    }

    float fc = 0.0f;
    float fcp = 0.0f;
    find_fc_and_fcp(cutoff_radial, rcinv, r, fc, fcp);
    float fnp_cache[kMaxCachedRadialBasisDerivatives];
    const bool cache_basis =
        basis_size_radial + 1 <= kMaxCachedRadialBasisDerivatives;
    if (cache_basis) {
      for (int k = 0; k <= basis_size_radial; ++k) {
        float fn = 0.0f;
        float fnp = 0.0f;
        find_fn_and_fnp(k, rcinv, r, fc, fcp, fn, fnp);
        fnp_cache[k] = fnp;
      }
    }
    const float rinv = 1.0f / r;
    float f12x = 0.0f;
    float f12y = 0.0f;
    float f12z = 0.0f;
      float f21x = 0.0f;
      float f21y = 0.0f;
      float f21z = 0.0f;

    for (int n = 0; n <= n_max_radial; ++n) {
      float gnp12 = 0.0f;
      float gnp21 = 0.0f;
      for (int k = 0; k <= basis_size_radial; ++k) {
        float fnp = 0.0f;
        if (cache_basis) {
          fnp = fnp_cache[k];
        } else {
          float fn = 0.0f;
          find_fn_and_fnp(k, rcinv, r, fc, fcp, fn, fnp);
        }
        const int coefficient_base = n * (basis_size_radial + 1) + k;
        gnp12 += fnp * descriptor_coefficients[
                            (type1 * num_types + type2) * radial_basis_count +
                            coefficient_base];
        gnp21 += fnp * descriptor_coefficients[
                            (type2 * num_types + type1) * radial_basis_count +
                            coefficient_base];
      }
      const float tmp12 = fp[atom + atom_stride * n] * gnp12 * rinv;
      f12x += tmp12 * x12;
      f12y += tmp12 * y12;
      f12z += tmp12 * z12;
      const float tmp21 = fp[neighbor + atom_stride * n] * gnp21 * rinv;
      f21x -= tmp21 * x12;
      f21y -= tmp21 * y12;
      f21z -= tmp21 * z12;
    }

    s_fx += f12x - f21x;
    s_fy += f12y - f21y;
    s_fz += f12z - f21z;
    if constexpr (IncludeZblForce) {
      if (r2 < zbl_outer_squared) {
        const int zj = atomic_numbers[type2];
        const float a_inv =
            (pow_zi + powf(static_cast<float>(zj), 0.23f)) * 2.134563f;
        const float zizj = kCoulomb * static_cast<float>(zi * zj);
        float zbl_f = 0.0f;
        float zbl_fp = 0.0f;
        find_f_and_fp_zbl(
            zizj,
            a_inv,
            zbl_inner,
            zbl_outer,
            r,
            rinv,
            zbl_f,
            zbl_fp);
        const float zbl_force_scale = zbl_fp * rinv;
        s_fx += x12 * zbl_force_scale;
        s_fy += y12 * zbl_force_scale;
        s_fz += z12 * zbl_force_scale;
      }
    }
    if (accumulate_virial) {
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
      s_sxx += x12 * f21x;
      s_syy += y12 * f21y;
      s_szz += z12 * f21z;
      s_sxy += x12 * f21y;
      s_sxz += x12 * f21z;
      s_syx += y12 * f21x;
      s_syz += y12 * f21z;
      s_szx += z12 * f21x;
      s_szy += z12 * f21y;
      }
    }
  }

  force_soa3[atom] += static_cast<double>(s_fx);
  force_soa3[atom_stride + atom] += static_cast<double>(s_fy);
  force_soa3[2 * atom_stride + atom] += static_cast<double>(s_fz);
  if (!accumulate_virial) {
    return;
  }
  if constexpr (VirialToNeighbor) {
    return;
  }
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

template <bool VirialToNeighbor, bool FloatVirialSink>
__global__ void accumulate_lammps_radial_forces(
    int atom_count,
    int atom_stride,
    int num_types,
    int n_max_radial,
    int basis_size_radial,
    float cutoff_radial,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ descriptor_coefficients,
    float* __restrict__ virial_float_soa9,
    double* __restrict__ force_soa3,
    double* __restrict__ virial_soa9,
    int accumulate_virial) {
  static_assert(!(VirialToNeighbor && FloatVirialSink));
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int type1 = types[atom];
  const double x1 = positions_soa3[atom];
  const double y1 = positions_soa3[atom_stride + atom];
  const double z1 = positions_soa3[2 * atom_stride + atom];
  const float rcinv = 1.0f / cutoff_radial;
  const int radial_basis_count =
      (n_max_radial + 1) * (basis_size_radial + 1);

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
    const int pair_offset = atom + atom_stride * slot;
    const int neighbor = nl_radial[pair_offset];
    const int type2 = types[neighbor];
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
    const float r = sqrtf(r2);
    if (r <= 0.0f) {
      continue;
    }

    float fc = 0.0f;
    float fcp = 0.0f;
    find_fc_and_fcp(cutoff_radial, rcinv, r, fc, fcp);
    float fnp_cache[kMaxCachedRadialBasisDerivatives];
    const bool cache_basis =
        basis_size_radial + 1 <= kMaxCachedRadialBasisDerivatives;
    if (cache_basis) {
      for (int k = 0; k <= basis_size_radial; ++k) {
        float fn = 0.0f;
        float fnp = 0.0f;
        find_fn_and_fnp(k, rcinv, r, fc, fcp, fn, fnp);
        fnp_cache[k] = fnp;
      }
    }

    const float rinv = 1.0f / r;
    float f12x = 0.0f;
    float f12y = 0.0f;
    float f12z = 0.0f;
    for (int n = 0; n <= n_max_radial; ++n) {
      float gnp12 = 0.0f;
      for (int k = 0; k <= basis_size_radial; ++k) {
        float fnp = 0.0f;
        if (cache_basis) {
          fnp = fnp_cache[k];
        } else {
          float fn = 0.0f;
          find_fn_and_fnp(k, rcinv, r, fc, fcp, fn, fnp);
        }
        const int coefficient_base = n * (basis_size_radial + 1) + k;
        const int coefficient_index =
            (type1 * num_types + type2) * radial_basis_count +
            coefficient_base;
        gnp12 += fnp * descriptor_coefficients[coefficient_index];
      }
      const float tmp12 = fp[atom + atom_stride * n] * gnp12 * rinv;
      f12x += tmp12 * x12;
      f12y += tmp12 * y12;
      f12z += tmp12 * z12;
    }

    s_fx += f12x;
    s_fy += f12y;
    s_fz += f12z;
    atomicAdd(&force_soa3[neighbor], -static_cast<double>(f12x));
    atomicAdd(
        &force_soa3[atom_stride + neighbor],
        -static_cast<double>(f12y));
    atomicAdd(
        &force_soa3[2 * atom_stride + neighbor],
        -static_cast<double>(f12z));

    if constexpr (FloatVirialSink) {
      atomic_add_per_atom_virial_float(
          atom_stride,
          neighbor,
          x12,
          y12,
          z12,
          f12x,
          f12y,
          f12z,
          virial_float_soa9);
    } else {
      if (accumulate_virial) {
        if constexpr (VirialToNeighbor) {
          atomicAdd(
              &virial_soa9[neighbor], -static_cast<double>(x12 * f12x));
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
          s_syy -= y12 * f12y;
          s_szz -= z12 * f12z;
          s_sxy -= x12 * f12y;
          s_sxz -= x12 * f12z;
          s_syx -= y12 * f12x;
          s_syz -= y12 * f12z;
          s_szx -= z12 * f12x;
          s_szy -= z12 * f12y;
        }
      }
    }
  }

  atomicAdd(&force_soa3[atom], static_cast<double>(s_fx));
  atomicAdd(&force_soa3[atom_stride + atom], static_cast<double>(s_fy));
  atomicAdd(&force_soa3[2 * atom_stride + atom], static_cast<double>(s_fz));
  if (!accumulate_virial) {
    return;
  }
  if constexpr (VirialToNeighbor || FloatVirialSink) {
    return;
  }
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

template <bool VirialToNeighbor>
__global__ void accumulate_radial_forces_batched(
    int atom_count,
    int atom_stride,
    int num_types,
    int n_max_radial,
    int basis_size_radial,
    int radial_capacity,
    float cutoff_radial,
    const int* __restrict__ atom_to_structure,
    const double* __restrict__ boxes_row_major9,
    const double* __restrict__ box_inverse_row_major9,
    const int* __restrict__ pbc_flags3,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ descriptor_coefficients,
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
  const double x1 = positions_soa3[atom];
  const double y1 = positions_soa3[atom_stride + atom];
  const double z1 = positions_soa3[2 * atom_stride + atom];
  const float rcinv = 1.0f / cutoff_radial;
  const int type_pairs = num_types * num_types;

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
    const float r = sqrtf(x12 * x12 + y12 * y12 + z12 * z12);
    if (r <= 0.0f) {
      continue;
    }

    float fc = 0.0f;
    float fcp = 0.0f;
    find_fc_and_fcp(cutoff_radial, rcinv, r, fc, fcp);
    const float rinv = 1.0f / r;
    float f12x = 0.0f;
    float f12y = 0.0f;
    float f12z = 0.0f;
    float f21x = 0.0f;
    float f21y = 0.0f;
    float f21z = 0.0f;

    for (int n = 0; n <= n_max_radial; ++n) {
      float gnp12 = 0.0f;
      float gnp21 = 0.0f;
      for (int k = 0; k <= basis_size_radial; ++k) {
        float fn = 0.0f;
        float fnp = 0.0f;
        find_fn_and_fnp(k, rcinv, r, fc, fcp, fn, fnp);
        const int coefficient_base =
            (n * (basis_size_radial + 1) + k) * type_pairs;
        gnp12 += fnp * descriptor_coefficients[
                            coefficient_base + type1 * num_types + type2];
        gnp21 += fnp * descriptor_coefficients[
                            coefficient_base + type2 * num_types + type1];
      }
      const float tmp12 = fp[atom + atom_stride * n] * gnp12 * rinv;
      const float tmp21 = fp[neighbor + atom_stride * n] * gnp21 * rinv;
      f12x += tmp12 * x12;
      f12y += tmp12 * y12;
      f12z += tmp12 * z12;
      f21x -= tmp21 * x12;
      f21y -= tmp21 * y12;
      f21z -= tmp21 * z12;
    }

    s_fx += f12x - f21x;
    s_fy += f12y - f21y;
    s_fz += f12z - f21z;
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
      s_sxx += x12 * f21x;
      s_syy += y12 * f21y;
      s_szz += z12 * f21z;
      s_sxy += x12 * f21y;
      s_sxz += x12 * f21z;
      s_syx += y12 * f21x;
      s_syz += y12 * f21z;
      s_szx += z12 * f21x;
      s_szy += z12 * f21y;
    }
  }

  force_soa3[atom] += static_cast<double>(s_fx);
  force_soa3[atom_stride + atom] += static_cast<double>(s_fy);
  force_soa3[2 * atom_stride + atom] += static_cast<double>(s_fz);
  if constexpr (VirialToNeighbor) {
    return;
  }
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

}  // namespace

void accumulate_radial_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial,
    bool clear_outputs,
    bool virial_to_neighbor) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.n_max_radial >= 0, "n_max_radial must be non-negative");
  require(protocol.basis_size_radial >= 0, "basis_size_radial must be non-negative");
  require(protocol.cutoff_radial > 0.0, "radial cutoff must be positive");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr, "workspace missing atom types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_radial != nullptr, "workspace missing radial counts");
  require(
      view.nl_radial_slot_major != nullptr,
      "workspace missing radial neighbors");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.force_soa3 != nullptr, "workspace missing force output");
  if (accumulate_virial) {
    require(view.virial_soa9 != nullptr, "workspace missing virial output");
  }
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_type_pair_major != nullptr,
          "model missing type-pair-major descriptor coefficients");

  if (clear_outputs) {
    check_cuda(
        cudaMemset(
            view.force_soa3,
            0,
            view.atom_capacity * 3 * sizeof(double)),
        "clear force output");
  }
  if (accumulate_virial && clear_outputs) {
    check_cuda(
        cudaMemset(
            view.virial_soa9,
            0,
            view.atom_capacity * 9 * sizeof(double)),
        "clear virial output");
  }

  const int threads = 32;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    if (virial_to_neighbor) {
      accumulate_radial_forces<false, true><<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.num_types,
        protocol.n_max_radial,
        protocol.basis_size_radial,
        protocol.neighbor_capacity_radial,
        static_cast<float>(protocol.cutoff_radial),
        0.0f,
        0.0f,
        box,
        view.types,
        nullptr,
        view.positions_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.fp,
        model_view.descriptor_coefficients_type_pair_major,
        view.force_soa3,
        view.virial_soa9,
        accumulate_virial ? 1 : 0);
    } else {
      accumulate_radial_forces<false, false><<<blocks, threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.num_types,
          protocol.n_max_radial,
          protocol.basis_size_radial,
          protocol.neighbor_capacity_radial,
          static_cast<float>(protocol.cutoff_radial),
          0.0f,
          0.0f,
          box,
          view.types,
          nullptr,
          view.positions_soa3,
          view.nn_radial,
          view.nl_radial_slot_major,
          view.fp,
          model_view.descriptor_coefficients_type_pair_major,
          view.force_soa3,
          view.virial_soa9,
          accumulate_virial ? 1 : 0);
    }
  }
  check_cuda(cudaGetLastError(), "accumulate radial forces kernel launch failed");
}

void accumulate_lammps_radial_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial,
    bool virial_to_neighbor) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.n_max_radial >= 0, "n_max_radial must be non-negative");
  require(protocol.basis_size_radial >= 0, "basis_size_radial must be non-negative");
  require(protocol.cutoff_radial > 0.0, "radial cutoff must be positive");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr, "workspace missing atom types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_radial != nullptr, "workspace missing radial counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.force_soa3 != nullptr, "workspace missing force output");
  if (accumulate_virial) {
    require(view.virial_soa9 != nullptr, "workspace missing virial output");
  }
  require(model_view.descriptor_coefficients_type_pair_major != nullptr,
          "model missing type-pair-major descriptor coefficients");

  check_cuda(
      cudaMemset(
          view.force_soa3,
          0,
          view.atom_capacity * 3 * sizeof(double)),
      "clear force output");
  if (accumulate_virial) {
    check_cuda(
        cudaMemset(
            view.virial_soa9,
            0,
            view.atom_capacity * 9 * sizeof(double)),
        "clear virial output");
  }

  const int threads = 64;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    const auto launch = [&](auto virial_to_neighbor_tag) {
      constexpr bool kVirialToNeighbor =
          decltype(virial_to_neighbor_tag)::value;
      accumulate_lammps_radial_forces<kVirialToNeighbor, false>
          <<<blocks, threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.num_types,
          protocol.n_max_radial,
          protocol.basis_size_radial,
          static_cast<float>(protocol.cutoff_radial),
          box,
          view.types,
          view.positions_soa3,
          view.nn_radial,
          view.nl_radial_slot_major,
          view.fp,
          model_view.descriptor_coefficients_type_pair_major,
          nullptr,
          view.force_soa3,
          view.virial_soa9,
          accumulate_virial ? 1 : 0);
    };
    if (virial_to_neighbor) {
      launch(std::true_type{});
    } else {
      launch(std::false_type{});
    }
  }
  check_cuda(cudaGetLastError(), "accumulate LAMMPS radial forces kernel launch failed");
}

void accumulate_lammps_radial_forces_to_per_atom_sink(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.n_max_radial >= 0, "n_max_radial must be non-negative");
  require(protocol.basis_size_radial >= 0,
          "basis_size_radial must be non-negative");
  require(protocol.cutoff_radial > 0.0, "radial cutoff must be positive");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr && view.positions_soa3 != nullptr,
          "workspace missing radial force atom inputs");
  require(view.nn_radial != nullptr && view.nl_radial_slot_major != nullptr,
          "workspace missing radial neighbors");
  require(view.fp != nullptr && view.force_soa3 != nullptr &&
              view.virial_soa9 != nullptr,
          "workspace missing radial sink inputs or outputs");
  require(view.per_atom_virial_float_soa9 != nullptr,
          "workspace missing per-atom virial sink");
  require(model_view.descriptor_coefficients_type_pair_major != nullptr,
          "model missing type-pair-major descriptor coefficients");

  check_cuda(
      cudaMemset(
          view.force_soa3,
          0,
          view.atom_capacity * 3 * sizeof(double)),
      "clear force output");
  check_cuda(
      cudaMemset(
          view.virial_soa9,
          0,
          view.atom_capacity * 9 * sizeof(double)),
      "clear virial output");

  const int threads = 64;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    accumulate_lammps_radial_forces<false, true><<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.num_types,
        protocol.n_max_radial,
        protocol.basis_size_radial,
        static_cast<float>(protocol.cutoff_radial),
        box,
        view.types,
        view.positions_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.fp,
        model_view.descriptor_coefficients_type_pair_major,
        view.per_atom_virial_float_soa9,
        view.force_soa3,
        view.virial_soa9,
        1);
  }
  check_cuda(
      cudaGetLastError(),
      "accumulate LAMMPS radial forces to per-atom sink failed");
}

void accumulate_radial_and_zbl_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.n_max_radial >= 0, "n_max_radial must be non-negative");
  require(protocol.basis_size_radial >= 0, "basis_size_radial must be non-negative");
  require(protocol.cutoff_radial > 0.0, "radial cutoff must be positive");
  require(protocol.has_zbl, "fused radial/ZBL force kernel requires ZBL");
  require(!protocol.flexible_zbl, "flexible ZBL is not supported yet");
  require(protocol.zbl_inner >= 0.0, "ZBL inner cutoff must be non-negative");
  require(protocol.zbl_outer > protocol.zbl_inner, "ZBL outer cutoff must exceed inner");
  require(protocol.zbl_outer <= protocol.cutoff_radial,
          "fused radial/ZBL force kernel reuses the radial neighbor list");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(model_view.atomic_numbers != nullptr, "model missing atomic numbers");
  require(model_view.atomic_numbers_count >= static_cast<std::size_t>(protocol.num_types),
          "model atomic number buffer is too small");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_type_pair_major != nullptr,
          "model missing type-pair-major descriptor coefficients");
  require(view.types != nullptr, "workspace missing atom types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_radial != nullptr, "workspace missing radial counts");
  require(
      view.nl_radial_slot_major != nullptr,
      "workspace missing radial neighbors");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.force_soa3 != nullptr, "workspace missing force output");

  check_cuda(
      cudaMemset(
          view.force_soa3,
          0,
          view.atom_capacity * 3 * sizeof(double)),
      "clear force output");

  const int threads = 64;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    accumulate_radial_forces<true, false><<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.num_types,
        protocol.n_max_radial,
        protocol.basis_size_radial,
        protocol.neighbor_capacity_radial,
        static_cast<float>(protocol.cutoff_radial),
        static_cast<float>(protocol.zbl_inner),
        static_cast<float>(protocol.zbl_outer),
        box,
        view.types,
        model_view.atomic_numbers,
        view.positions_soa3,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.fp,
        model_view.descriptor_coefficients_type_pair_major,
        view.force_soa3,
        view.virial_soa9,
        0);
  }
  check_cuda(
      cudaGetLastError(),
      "accumulate fused radial/ZBL forces kernel launch failed");
}

void accumulate_radial_forces_batched(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool virial_to_neighbor) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.n_max_radial >= 0, "n_max_radial must be non-negative");
  require(protocol.basis_size_radial >= 0, "basis_size_radial must be non-negative");
  require(protocol.cutoff_radial > 0.0, "radial cutoff must be positive");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.atom_to_structure != nullptr, "workspace missing atom_to_structure");
  require(view.boxes_row_major9 != nullptr, "workspace missing boxes");
  require(view.box_inverse_row_major9 != nullptr, "workspace missing box inverses");
  require(view.pbc_flags3 != nullptr, "workspace missing pbc flags");
  require(view.types != nullptr, "workspace missing atom types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_radial != nullptr, "workspace missing radial counts");
  require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.force_soa3 != nullptr, "workspace missing force output");
  require(view.virial_soa9 != nullptr, "workspace missing virial output");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");

  check_cuda(
      cudaMemset(
          view.force_soa3,
          0,
          static_cast<std::size_t>(atom_count) * 3 * sizeof(double)),
      "clear batched force output");
  check_cuda(
      cudaMemset(
          view.virial_soa9,
          0,
          static_cast<std::size_t>(atom_count) * 9 * sizeof(double)),
      "clear batched virial output");

  const int threads = 128;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    const auto launch = [&](auto virial_to_neighbor_tag) {
      constexpr bool kVirialToNeighbor =
          decltype(virial_to_neighbor_tag)::value;
      accumulate_radial_forces_batched<kVirialToNeighbor><<<blocks, threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.num_types,
          protocol.n_max_radial,
          protocol.basis_size_radial,
          protocol.neighbor_capacity_radial,
          static_cast<float>(protocol.cutoff_radial),
          view.atom_to_structure,
          view.boxes_row_major9,
          view.box_inverse_row_major9,
          view.pbc_flags3,
          view.types,
          view.positions_soa3,
          view.nn_radial,
          view.nl_radial_slot_major,
          view.fp,
          model_view.descriptor_coefficients,
          view.force_soa3,
          view.virial_soa9);
    };
    if (virial_to_neighbor) {
      launch(std::true_type{});
    } else {
      launch(std::false_type{});
    }
  }
  check_cuda(cudaGetLastError(), "accumulate batched radial forces launch failed");
  check_cuda(cudaDeviceSynchronize(), "accumulate batched radial forces failed");
}

}  // namespace nep_adapters::cuda_backend
