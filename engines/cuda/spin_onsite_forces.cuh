#pragma once

// Internal CUDA implementation fragment for spin_onsite.cu.
// Included inside nep_adapters::cuda_backend's anonymous namespace.
// Contains the onsite term and the unified compact spin force core.

__global__ void accumulate_spin_onsite_mforces(
    int atom_count,
    int atom_stride,
    int struct_dim,
    const double* __restrict__ spins_soa3,
    const float* __restrict__ fp,
    double* __restrict__ mforce_soa3) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const double sx = spins_soa3[atom];
  const double sy = spins_soa3[atom_stride + atom];
  const double sz = spins_soa3[2 * atom_stride + atom];
  const double s2 = sx * sx + sy * sy + sz * sz;
  const double scale =
      2.0 * static_cast<double>(fp[atom + atom_stride * struct_dim]) +
      4.0 * static_cast<double>(fp[atom + atom_stride * (struct_dim + 1)]) * s2;
  mforce_soa3[atom] -= scale * sx;
  mforce_soa3[atom_stride + atom] -= scale * sy;
  mforce_soa3[2 * atom_stride + atom] -= scale * sz;
}

// Float helpers used by the unified density and chiral core.

__device__ __forceinline__ void cross3f(
    const float* a,
    const float* b,
    float* out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ float dot3f(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ void stf_outer3f(
    const float* a,
    const float* b,
    float* out) {
  const float trace = dot3f(a, b) / 3.0f;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      float value = 0.5f * (a[i] * b[j] + a[j] * b[i]);
      if (i == j) {
        value -= trace;
      }
      out[3 * i + j] = value;
    }
  }
}

template <int C>
__device__ __forceinline__ bool load_spin_edge_f32(
    int atom,
    int neighbor,
    int atom_stride,
    int num_types,
    int spin_basis_size,
    float spin_cutoff,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ spins_soa3,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    float* rhat,
    float& dist,
    float* si,
    float* sj,
    float* weights,
    float* weight_derivatives) {
  compute_spin_edge_geometry_f32(
      atom,
      neighbor,
      atom_stride,
      box,
      positions_soa3,
      rhat,
      dist);
  #pragma unroll
  for (int d = 0; d < 3; ++d) {
    si[d] = static_cast<float>(spins_soa3[d * atom_stride + atom]);
    sj[d] = static_cast<float>(spins_soa3[d * atom_stride + neighbor]);
  }
  if (!(dist > 1.0e-12f && dist < spin_cutoff)) {
    return false;
  }
  const int type_pair = types[atom] * num_types + types[neighbor];
  evaluate_spin_edge_weights_f32<C, true>(
      spin_basis_size,
      spin_cutoff,
      dist,
      num_types,
      type_pair,
      descriptor_coefficients,
      spin_coefficient_offset,
      weights,
      weight_derivatives);
  return true;
}

__device__ __forceinline__ void fill_spin_monomials2f(
    const float* u,
    float* m2) {
  const float x = u[0];
  const float y = u[1];
  const float z = u[2];
  m2[0] = x * x;
  m2[1] = y * y;
  m2[2] = z * z;
  m2[3] = x * y;
  m2[4] = x * z;
  m2[5] = y * z;
}

__device__ __forceinline__ void fill_spin_monomialsf(
    const float* u,
    float* m3,
    float* m4) {
  const float x = u[0];
  const float y = u[1];
  const float z = u[2];
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float x3 = x2 * x;
  const float y3 = y2 * y;
  const float z3 = z2 * z;
  m3[0] = x3;
  m3[1] = y3;
  m3[2] = z3;
  m3[3] = x2 * y;
  m3[4] = x2 * z;
  m3[5] = x * y2;
  m3[6] = y2 * z;
  m3[7] = x * z2;
  m3[8] = y * z2;
  m3[9] = x * y * z;
  m4[0] = x2 * x2;
  m4[1] = y2 * y2;
  m4[2] = z2 * z2;
  m4[3] = x3 * y;
  m4[4] = x3 * z;
  m4[5] = x * y3;
  m4[6] = y3 * z;
  m4[7] = x * z3;
  m4[8] = y * z3;
  m4[9] = x2 * y2;
  m4[10] = x2 * z2;
  m4[11] = y2 * z2;
  m4[12] = x2 * y * z;
  m4[13] = x * y2 * z;
  m4[14] = x * y * z2;
}

__device__ __forceinline__ float dot_spin_termsf(
    const float* lhs,
    const float* rhs,
    int count) {
  float out = 0.0f;
  for (int k = 0; k < count; ++k) {
    out += lhs[k] * rhs[k];
  }
  return out;
}

__device__ __forceinline__ void load_spin_chiral_qoh_reduced_f32(
    int atom_stride,
    int atom,
    int channel,
    const float* __restrict__ qmat,
    const float* __restrict__ octupoles_raw_cache,
    const float* __restrict__ hexadecapoles_raw_cache,
    float* __restrict__ q_reduced,
    float* __restrict__ o_reduced,
    float* __restrict__ h_reduced) {
  constexpr int ChiC = 2;
  constexpr int C = 4;
  const int o_base = spin_atom_cache_index<C * kSpinChiralOReducedCount>(
      atom_stride, atom, channel * kSpinChiralOReducedCount);
  const int h_base = spin_atom_cache_index<ChiC * kSpinChiralHReducedCount>(
      atom_stride, atom, channel * kSpinChiralHReducedCount);

  q_reduced[0] = qmat[0] - qmat[8];
  q_reduced[1] = -qmat[5];
  q_reduced[2] = 0.5f * qmat[2];
  q_reduced[3] = qmat[1];
  q_reduced[4] = qmat[0] - qmat[4];

  for (int k = 0; k < kSpinChiralOReducedCount; ++k) {
    o_reduced[k] = octupoles_raw_cache[o_base + k];
  }
  for (int k = 0; k < kSpinChiralHReducedCount; ++k) {
    h_reduced[k] = hexadecapoles_raw_cache[h_base + k];
  }
}

__device__ __forceinline__ float contract_spin_chiral_qoh_reduced_f32(
    const float* __restrict__ q_reduced,
    const float* __restrict__ o_reduced,
    const float* __restrict__ h_reduced) {
  float value = 0.0f;
#pragma unroll 1
  for (int term = 0; term < kSpinChiralQohReducedCount; ++term) {
    const unsigned short packed = kSpinChiralQohReducedPacked[term];
    const int q = packed >> 8;
    const int o = (packed >> 4) & 0x0f;
    const int h = packed & 0x0f;
    value += kSpinChiralQohReducedCoeff[term] *
             q_reduced[q] * o_reduced[o] * h_reduced[h];
  }
  return value;
}

__device__ __forceinline__ void reconstruct_spin_rank3_raw_f32(
    const float* __restrict__ polar,
    const float* __restrict__ reduced,
    float* __restrict__ raw) {
  raw[0] = 0.6f * polar[0] - 0.6f * reduced[4] - 0.4f * reduced[6];
  raw[1] = 0.6f * polar[1] + 0.4f * reduced[0] - 0.6f * reduced[2];
  raw[2] = 0.6f * polar[2] + 0.4f * reduced[1] - 0.6f * reduced[3];
  raw[3] = 0.2f * polar[1] - 0.2f * reduced[0] + 0.8f * reduced[2];
  raw[4] = 0.2f * polar[2] - 0.2f * reduced[1] + 0.8f * reduced[3];
  raw[5] = 0.2f * polar[0] + 0.8f * reduced[4] + 0.2f * reduced[6];
  raw[6] = 0.2f * polar[2] - 0.2f * reduced[1] - 0.2f * reduced[3];
  raw[7] = 0.2f * polar[0] - 0.2f * reduced[4] + 0.2f * reduced[6];
  raw[8] = 0.2f * polar[1] - 0.2f * reduced[0] - 0.2f * reduced[2];
  raw[9] = reduced[5];
}

__device__ __forceinline__ float spin_rank3_symmetric_component_f32(
    const float* __restrict__ raw,
    int a,
    int b,
    int c) {
  const int nx = (a == 0) + (b == 0) + (c == 0);
  const int ny = (a == 1) + (b == 1) + (c == 1);
  if (nx == 3) {
    return raw[0];
  }
  if (ny == 3) {
    return raw[1];
  }
  if (nx == 0 && ny == 0) {
    return raw[2];
  }
  if (nx == 2) {
    return raw[ny == 1 ? 3 : 4];
  }
  if (ny == 2) {
    return raw[nx == 1 ? 5 : 6];
  }
  if (nx == 1 && ny == 0) {
    return raw[7];
  }
  if (nx == 0 && ny == 1) {
    return raw[8];
  }
  return raw[9];
}

__device__ __forceinline__ int spin_rank3_symmetric_index_f32(
    int a,
    int b,
    int c) {
  const int nx = (a == 0) + (b == 0) + (c == 0);
  const int ny = (a == 1) + (b == 1) + (c == 1);
  if (nx == 3) {
    return 0;
  }
  if (ny == 3) {
    return 1;
  }
  if (nx == 0 && ny == 0) {
    return 2;
  }
  if (nx == 2) {
    return ny == 1 ? 3 : 4;
  }
  if (ny == 2) {
    return nx == 1 ? 5 : 6;
  }
  if (nx == 1 && ny == 0) {
    return 7;
  }
  if (nx == 0 && ny == 1) {
    return 8;
  }
  return 9;
}

__device__ __forceinline__ void build_spin_chiral_pseudodev_center_f32(
    const float* __restrict__ geom,
    const float* __restrict__ polar,
    const float* __restrict__ octupole_reduced,
    float* __restrict__ pseudodev) {
  float octupole_raw[kSpinDeg3Count];
  reconstruct_spin_rank3_raw_f32(polar, octupole_reduced, octupole_raw);

  float unsymmetrized[9] = {};
#pragma unroll
  for (int b = 0; b < 3; ++b) {
#pragma unroll
    for (int e = 0; e < 3; ++e) {
      unsymmetrized[b] +=
          geom[6 + e] *
              spin_rank3_symmetric_component_f32(octupole_raw, 1, e, b) -
          geom[3 + e] *
              spin_rank3_symmetric_component_f32(octupole_raw, 2, e, b);
      unsymmetrized[3 + b] +=
          geom[e] *
              spin_rank3_symmetric_component_f32(octupole_raw, 2, e, b) -
          geom[6 + e] *
              spin_rank3_symmetric_component_f32(octupole_raw, 0, e, b);
      unsymmetrized[6 + b] +=
          geom[3 + e] *
              spin_rank3_symmetric_component_f32(octupole_raw, 0, e, b) -
          geom[e] *
              spin_rank3_symmetric_component_f32(octupole_raw, 1, e, b);
    }
  }
#pragma unroll
  for (int a = 0; a < 3; ++a) {
#pragma unroll
    for (int b = 0; b < 3; ++b) {
      pseudodev[3 * a + b] =
          0.5f * (unsymmetrized[3 * a + b] + unsymmetrized[3 * b + a]);
    }
  }
}

__device__ __forceinline__ void add_spin_chiral_qoh_reduced_pull_f32(
    float pull,
    const float* __restrict__ q_reduced,
    const float* __restrict__ o_reduced,
    const float* __restrict__ h_reduced,
    float* __restrict__ grad_q,
    float* __restrict__ grad_o_reduced,
    float* __restrict__ grad_h_reduced) {
  float grad_q_reduced[5] = {};
#pragma unroll 1
  for (int term = 0; term < kSpinChiralQohReducedCount; ++term) {
    const unsigned short packed = kSpinChiralQohReducedPacked[term];
    const int q = packed >> 8;
    const int o = (packed >> 4) & 0x0f;
    const int h = packed & 0x0f;
    const float scale = pull * kSpinChiralQohReducedCoeff[term];
    grad_q_reduced[q] += scale * o_reduced[o] * h_reduced[h];
    grad_o_reduced[o] += scale * q_reduced[q] * h_reduced[h];
    grad_h_reduced[h] += scale * q_reduced[q] * o_reduced[o];
  }

  grad_q[0] += grad_q_reduced[0] + grad_q_reduced[4];
  grad_q[1] += grad_q_reduced[3];
  grad_q[2] += 0.5f * grad_q_reduced[2];
  grad_q[4] -= grad_q_reduced[4];
  grad_q[5] -= grad_q_reduced[1];
  grad_q[8] -= grad_q_reduced[0];
}

__device__ __forceinline__ void add_spin_chiral_o_reduced_terms_f32(
    const float* __restrict__ grad_o_reduced,
    float* __restrict__ grad_o) {

  grad_o[0] -= grad_o_reduced[6];
  grad_o[1] += grad_o_reduced[0];
  grad_o[2] += grad_o_reduced[1];
  grad_o[3] += grad_o_reduced[2];
  grad_o[4] += grad_o_reduced[3];
  grad_o[5] += grad_o_reduced[4];
  grad_o[6] -= 3.0f * grad_o_reduced[1] + grad_o_reduced[3];
  grad_o[7] += -grad_o_reduced[4] + 3.0f * grad_o_reduced[6];
  grad_o[8] -= 3.0f * grad_o_reduced[0] + grad_o_reduced[2];
  grad_o[9] += grad_o_reduced[5];
}

__device__ __forceinline__ void add_spin_chiral_h_reduced_terms_f32(
    const float* __restrict__ grad_h_reduced,
    float* __restrict__ grad_h) {

  constexpr float OneSeventh = 1.0f / 7.0f;
  grad_h[0] += (-grad_h_reduced[7] + 4.0f * grad_h_reduced[8]) * OneSeventh;
  grad_h[1] +=
      (2.0f * grad_h_reduced[5] + grad_h_reduced[7] + grad_h_reduced[8]) *
      OneSeventh;
  grad_h[2] += (-2.0f * grad_h_reduced[5] + 2.0f * grad_h_reduced[8]) *
               OneSeventh;
  grad_h[3] += (grad_h_reduced[1] - 4.0f * grad_h_reduced[3]) * OneSeventh;
  grad_h[4] += (-grad_h_reduced[0] + 4.0f * grad_h_reduced[2]) * OneSeventh;
  grad_h[5] += (grad_h_reduced[1] + 3.0f * grad_h_reduced[3]) * OneSeventh;
  grad_h[6] += (-2.0f * grad_h_reduced[4] + 4.0f * grad_h_reduced[6]) *
               OneSeventh;
  grad_h[7] += (-grad_h_reduced[0] - 3.0f * grad_h_reduced[2]) * OneSeventh;
  grad_h[8] += (-2.0f * grad_h_reduced[4] - 3.0f * grad_h_reduced[6]) *
               OneSeventh;
  grad_h[9] += (-12.0f * grad_h_reduced[5] - 9.0f * grad_h_reduced[8]) *
               OneSeventh;
  grad_h[10] +=
      (12.0f * grad_h_reduced[5] + 6.0f * grad_h_reduced[7] -
       15.0f * grad_h_reduced[8]) * OneSeventh;
  grad_h[11] += (-6.0f * grad_h_reduced[7] + 3.0f * grad_h_reduced[8]) *
                OneSeventh;
  grad_h[12] += (12.0f * grad_h_reduced[4] - 3.0f * grad_h_reduced[6]) *
                OneSeventh;
  grad_h[13] += (6.0f * grad_h_reduced[0] - 3.0f * grad_h_reduced[2]) *
                OneSeventh;
  grad_h[14] += (-6.0f * grad_h_reduced[1] + 3.0f * grad_h_reduced[3]) *
                OneSeventh;
}

__device__ __forceinline__ void add_spin_rank3_reconstruction_pull_f32(
    const float* __restrict__ grad_raw,
    float* __restrict__ grad_polar,
    float* __restrict__ grad_reduced) {
  grad_polar[0] +=
      0.6f * grad_raw[0] + 0.2f * grad_raw[5] + 0.2f * grad_raw[7];
  grad_polar[1] +=
      0.6f * grad_raw[1] + 0.2f * grad_raw[3] + 0.2f * grad_raw[8];
  grad_polar[2] +=
      0.6f * grad_raw[2] + 0.2f * grad_raw[4] + 0.2f * grad_raw[6];

  grad_reduced[0] +=
      0.4f * grad_raw[1] - 0.2f * grad_raw[3] - 0.2f * grad_raw[8];
  grad_reduced[1] +=
      0.4f * grad_raw[2] - 0.2f * grad_raw[4] - 0.2f * grad_raw[6];
  grad_reduced[2] +=
      -0.6f * grad_raw[1] + 0.8f * grad_raw[3] - 0.2f * grad_raw[8];
  grad_reduced[3] +=
      -0.6f * grad_raw[2] + 0.8f * grad_raw[4] - 0.2f * grad_raw[6];
  grad_reduced[4] +=
      -0.6f * grad_raw[0] + 0.8f * grad_raw[5] - 0.2f * grad_raw[7];
  grad_reduced[5] += grad_raw[9];
  grad_reduced[6] +=
      -0.4f * grad_raw[0] + 0.2f * grad_raw[5] + 0.2f * grad_raw[7];
}

__device__ __forceinline__ void add_spin_chiral_pseudodev_center_pull_f32(
    const float* __restrict__ geom,
    const float* __restrict__ octupole_raw,
    const float* __restrict__ grad_pseudodev,
    float* __restrict__ grad_geom,
    float* __restrict__ grad_octupole_raw) {
  float grad_unsymmetrized[9];
#pragma unroll
  for (int a = 0; a < 3; ++a) {
#pragma unroll
    for (int b = 0; b < 3; ++b) {
      grad_unsymmetrized[3 * a + b] =
          0.5f *
          (grad_pseudodev[3 * a + b] + grad_pseudodev[3 * b + a]);
    }
  }

#pragma unroll
  for (int b = 0; b < 3; ++b) {
#pragma unroll
    for (int e = 0; e < 3; ++e) {
      const float g0 = grad_unsymmetrized[b];
      const float g1 = grad_unsymmetrized[3 + b];
      const float g2 = grad_unsymmetrized[6 + b];
      const int o0 = spin_rank3_symmetric_index_f32(0, e, b);
      const int o1 = spin_rank3_symmetric_index_f32(1, e, b);
      const int o2 = spin_rank3_symmetric_index_f32(2, e, b);

      grad_geom[6 + e] += g0 * octupole_raw[o1];
      grad_octupole_raw[o1] += g0 * geom[6 + e];
      grad_geom[3 + e] -= g0 * octupole_raw[o2];
      grad_octupole_raw[o2] -= g0 * geom[3 + e];

      grad_geom[e] += g1 * octupole_raw[o2];
      grad_octupole_raw[o2] += g1 * geom[e];
      grad_geom[6 + e] -= g1 * octupole_raw[o0];
      grad_octupole_raw[o0] -= g1 * geom[6 + e];

      grad_geom[3 + e] += g2 * octupole_raw[o0];
      grad_octupole_raw[o0] += g2 * geom[3 + e];
      grad_geom[e] -= g2 * octupole_raw[o1];
      grad_octupole_raw[o1] -= g2 * geom[e];
    }
  }
}

__device__ __forceinline__ void project_rank2_spin_gradientf(
    const float* grad,
    float* terms) {
  const float trace = (grad[0] + grad[4] + grad[8]) / 3.0f;
  terms[0] = grad[0] - trace;
  terms[1] = grad[4] - trace;
  terms[2] = grad[8] - trace;
  terms[3] = grad[1] + grad[3];
  terms[4] = grad[2] + grad[6];
  terms[5] = grad[5] + grad[7];
}

__device__ __forceinline__ void evaluate_spin_polynomial_gradientf(
    int degree,
    const float* terms,
    const float* u,
    float* grad) {
  const float x = u[0];
  const float y = u[1];
  const float z = u[2];
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  if (degree == 3) {
    grad[0] = 3.0f * terms[0] * x2 +
              2.0f * terms[3] * x * y +
              2.0f * terms[4] * x * z +
              terms[5] * y2 + terms[7] * z2 + terms[9] * y * z;
    grad[1] = 3.0f * terms[1] * y2 + terms[3] * x2 +
              2.0f * terms[5] * x * y +
              2.0f * terms[6] * y * z +
              terms[8] * z2 + terms[9] * x * z;
    grad[2] = 3.0f * terms[2] * z2 + terms[4] * x2 +
              terms[6] * y2 + 2.0f * terms[7] * x * z +
              2.0f * terms[8] * y * z + terms[9] * x * y;
    return;
  }

  const float x3 = x2 * x;
  const float y3 = y2 * y;
  const float z3 = z2 * z;
  grad[0] = 4.0f * terms[0] * x3 +
            3.0f * terms[3] * x2 * y +
            3.0f * terms[4] * x2 * z + terms[5] * y3 +
            terms[7] * z3 + 2.0f * terms[9] * x * y2 +
            2.0f * terms[10] * x * z2 +
            2.0f * terms[12] * x * y * z +
            terms[13] * y2 * z + terms[14] * y * z2;
  grad[1] = 4.0f * terms[1] * y3 + terms[3] * x3 +
            3.0f * terms[5] * x * y2 +
            3.0f * terms[6] * y2 * z + terms[8] * z3 +
            2.0f * terms[9] * x2 * y +
            2.0f * terms[11] * y * z2 + terms[12] * x2 * z +
            2.0f * terms[13] * x * y * z + terms[14] * x * z2;
  grad[2] = 4.0f * terms[2] * z3 + terms[4] * x3 +
            terms[6] * y3 + 3.0f * terms[7] * x * z2 +
            3.0f * terms[8] * y * z2 +
            2.0f * terms[10] * x2 * z +
            2.0f * terms[11] * y2 * z + terms[12] * x2 * y +
            terms[13] * x * y2 + 2.0f * terms[14] * x * y * z;
}

__device__ __forceinline__ void add_stf_outer_gradientf(
    const float* grad,
    const float* a,
    const float* b,
    float* grad_a,
    float* grad_b) {
  const float trace_grad = (grad[0] + grad[4] + grad[8]) / 3.0f;
  for (int p = 0; p < 3; ++p) {
    float ga = -trace_grad * b[p];
    float gb = -trace_grad * a[p];
    for (int q = 0; q < 3; ++q) {
      ga += 0.5f * (grad[3 * p + q] + grad[3 * q + p]) * b[q];
      gb += 0.5f * (grad[3 * p + q] + grad[3 * q + p]) * a[q];
    }
    grad_a[p] += ga;
    grad_b[p] += gb;
  }
}

template <SpinVirialMode VirialMode>
__device__ void apply_edge_gradients_f32(
    int atom,
    int neighbor,
    int atom_stride,
    float dist,
    const float* rhat,
    float grad_weight,
    float weight_derivative,
    const float* grad_rhat,
    const float* grad_si,
    const float* grad_sj,
    double* force_soa3,
    double* mforce_soa3,
    double* virial_soa9,
    double* cpu_virial) {
  const float grad_dist = grad_weight * weight_derivative;
  float dot_r = 0.0f;
  for (int d = 0; d < 3; ++d) {
    dot_r += grad_rhat[d] * rhat[d];
  }
  float grad_rij[3];
  for (int d = 0; d < 3; ++d) {
    grad_rij[d] =
        grad_dist * rhat[d] + (grad_rhat[d] - dot_r * rhat[d]) / dist;
    atomicAdd(force_soa3 + d * atom_stride + atom,
              static_cast<double>(grad_rij[d]));
    atomicAdd(force_soa3 + d * atom_stride + neighbor,
              -static_cast<double>(grad_rij[d]));
    atomicAdd(mforce_soa3 + d * atom_stride + atom,
              -static_cast<double>(grad_si[d]));
    atomicAdd(mforce_soa3 + d * atom_stride + neighbor,
              -static_cast<double>(grad_sj[d]));
  }
  if constexpr (VirialMode != SpinVirialMode::disabled) {
    for (int a = 0; a < 3; ++a) {
      const double rij_a = static_cast<double>(rhat[a] * dist);
      for (int b = 0; b < 3; ++b) {
        const int component = virial_internal_component(a * 3 + b);
        const double value = -rij_a * static_cast<double>(grad_rij[b]);
        if constexpr (VirialMode == SpinVirialMode::cpu_atom_decomposition) {
          cpu_virial[component] += value;
        } else {
          atomicAdd(virial_soa9 + component * atom_stride + atom, value);
        }
      }
    }
  }
}

__device__ int real_spherical_harmonics_spinf(
    const float* rhat,
    int ell,
    float* out) {
  const float x = rhat[0];
  const float y = rhat[1];
  const float z = rhat[2];
  if (ell == 2) {
    out[0] = sqrtf(static_cast<float>(15.0 / (4.0 * kPi))) * x * y;
    out[1] = sqrtf(static_cast<float>(15.0 / (4.0 * kPi))) * y * z;
    out[2] = sqrtf(static_cast<float>(5.0 / (16.0 * kPi))) *
             (2.0f * z * z - x * x - y * y);
    out[3] = sqrtf(static_cast<float>(15.0 / (4.0 * kPi))) * x * z;
    out[4] = sqrtf(static_cast<float>(15.0 / (16.0 * kPi))) *
             (x * x - y * y);
    return 5;
  }
  if (ell == 3) {
    const float rho2 = x * x + y * y;
    out[0] = sqrtf(static_cast<float>(35.0 / (32.0 * kPi))) *
             y * (3.0f * x * x - y * y);
    out[1] = sqrtf(static_cast<float>(105.0 / (4.0 * kPi))) * x * y * z;
    out[2] = sqrtf(static_cast<float>(21.0 / (32.0 * kPi))) *
             y * (4.0f * z * z - rho2);
    out[3] = sqrtf(static_cast<float>(7.0 / (16.0 * kPi))) *
             z * (2.0f * z * z - 3.0f * rho2);
    out[4] = sqrtf(static_cast<float>(21.0 / (32.0 * kPi))) *
             x * (4.0f * z * z - rho2);
    out[5] = sqrtf(static_cast<float>(105.0 / (16.0 * kPi))) *
             z * (x * x - y * y);
    out[6] = sqrtf(static_cast<float>(35.0 / (32.0 * kPi))) *
             x * (x * x - 3.0f * y * y);
    return 7;
  }
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  out[0] = 0.75f * sqrtf(static_cast<float>(35.0 / kPi)) *
           x * y * (x2 - y2);
  out[1] = 0.75f * sqrtf(static_cast<float>(35.0 / (2.0 * kPi))) *
           y * z * (3.0f * x2 - y2);
  out[2] = 0.75f * sqrtf(static_cast<float>(5.0 / kPi)) *
           x * y * (7.0f * z2 - 1.0f);
  out[3] = 0.75f * sqrtf(static_cast<float>(5.0 / (2.0 * kPi))) *
           y * z * (7.0f * z2 - 3.0f);
  out[4] = (3.0f / 16.0f) * sqrtf(static_cast<float>(1.0 / kPi)) *
           (35.0f * z2 * z2 - 30.0f * z2 + 3.0f);
  out[5] = 0.75f * sqrtf(static_cast<float>(5.0 / (2.0 * kPi))) *
           x * z * (7.0f * z2 - 3.0f);
  out[6] = 0.375f * sqrtf(static_cast<float>(5.0 / kPi)) *
           (x2 - y2) * (7.0f * z2 - 1.0f);
  out[7] = 0.75f * sqrtf(static_cast<float>(35.0 / (2.0 * kPi))) *
           x * z * (x2 - 3.0f * y2);
  out[8] = (3.0f / 16.0f) * sqrtf(static_cast<float>(35.0 / kPi)) *
           (x2 * x2 - 6.0f * x2 * y2 + y2 * y2);
  return 9;
}

__device__ void add_real_spherical_harmonics_gradientf(
    const float* r,
    int ell,
    const float* grad_y,
    float* grad_r) {
  const float x = r[0];
  const float y = r[1];
  const float z = r[2];
  if (ell == 2) {
    const float a = sqrtf(static_cast<float>(15.0 / (4.0 * kPi)));
    const float b = sqrtf(static_cast<float>(5.0 / (16.0 * kPi)));
    const float c = sqrtf(static_cast<float>(15.0 / (16.0 * kPi)));
    grad_r[0] += grad_y[0] * a * y - grad_y[2] * 2.0f * b * x +
                 grad_y[3] * a * z + grad_y[4] * 2.0f * c * x;
    grad_r[1] += grad_y[0] * a * x + grad_y[1] * a * z -
                 grad_y[2] * 2.0f * b * y - grad_y[4] * 2.0f * c * y;
    grad_r[2] += grad_y[1] * a * y + grad_y[2] * 4.0f * b * z +
                 grad_y[3] * a * x;
    return;
  }
  if (ell == 3) {
    const float x2 = x * x;
    const float y2 = y * y;
    const float z2 = z * z;
    const float rho2 = x2 + y2;
    const float a = sqrtf(static_cast<float>(35.0 / (32.0 * kPi)));
    const float b = sqrtf(static_cast<float>(105.0 / (4.0 * kPi)));
    const float c = sqrtf(static_cast<float>(21.0 / (32.0 * kPi)));
    const float d = sqrtf(static_cast<float>(7.0 / (16.0 * kPi)));
    const float e = sqrtf(static_cast<float>(105.0 / (16.0 * kPi)));
    grad_r[0] += grad_y[0] * 6.0f * a * x * y +
                 grad_y[1] * b * y * z -
                 grad_y[2] * 2.0f * c * x * y -
                 grad_y[3] * 6.0f * d * x * z +
                 grad_y[4] * c * (4.0f * z2 - 3.0f * x2 - y2) +
                 grad_y[5] * 2.0f * e * x * z +
                 grad_y[6] * 3.0f * a * (x2 - y2);
    grad_r[1] += grad_y[0] * 3.0f * a * (x2 - y2) +
                 grad_y[1] * b * x * z +
                 grad_y[2] * c * (4.0f * z2 - x2 - 3.0f * y2) -
                 grad_y[3] * 6.0f * d * y * z -
                 grad_y[4] * 2.0f * c * x * y -
                 grad_y[5] * 2.0f * e * y * z -
                 grad_y[6] * 6.0f * a * x * y;
    grad_r[2] += grad_y[1] * b * x * y +
                 grad_y[2] * 8.0f * c * y * z +
                 grad_y[3] * d * (6.0f * z2 - 3.0f * rho2) +
                 grad_y[4] * 8.0f * c * x * z +
                 grad_y[5] * e * (x2 - y2);
    return;
  }
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float a = 0.75f * sqrtf(static_cast<float>(35.0 / kPi));
  const float b = 0.75f * sqrtf(static_cast<float>(35.0 / (2.0 * kPi)));
  const float c = 0.75f * sqrtf(static_cast<float>(5.0 / kPi));
  const float d = 0.75f * sqrtf(static_cast<float>(5.0 / (2.0 * kPi)));
  const float e = (3.0f / 16.0f) * sqrtf(static_cast<float>(1.0 / kPi));
  const float f = 0.375f * sqrtf(static_cast<float>(5.0 / kPi));
  const float g = (3.0f / 16.0f) * sqrtf(static_cast<float>(35.0 / kPi));
  grad_r[0] += grad_y[0] * a * y * (3.0f * x2 - y2) +
               grad_y[1] * b * 6.0f * x * y * z +
               grad_y[2] * c * y * (7.0f * z2 - 1.0f) +
               grad_y[5] * d * z * (7.0f * z2 - 3.0f) +
               grad_y[6] * 2.0f * f * x * (7.0f * z2 - 1.0f) +
               grad_y[7] * b * z * (3.0f * x2 - 3.0f * y2) +
               grad_y[8] * g * (4.0f * x * x2 - 12.0f * x * y2);
  grad_r[1] += grad_y[0] * a * x * (x2 - 3.0f * y2) +
               grad_y[1] * b * z * (3.0f * x2 - 3.0f * y2) +
               grad_y[2] * c * x * (7.0f * z2 - 1.0f) +
               grad_y[3] * d * z * (7.0f * z2 - 3.0f) -
               grad_y[6] * 2.0f * f * y * (7.0f * z2 - 1.0f) -
               grad_y[7] * b * 6.0f * x * y * z +
               grad_y[8] * g * (-12.0f * x2 * y + 4.0f * y * y2);
  grad_r[2] += grad_y[1] * b * y * (3.0f * x2 - y2) +
               grad_y[2] * c * 14.0f * x * y * z +
               grad_y[3] * d * y * (21.0f * z2 - 3.0f) +
               grad_y[4] * e * (140.0f * z2 * z - 60.0f * z) +
               grad_y[5] * d * x * (21.0f * z2 - 3.0f) +
               grad_y[6] * f * 14.0f * z * (x2 - y2) +
               grad_y[7] * b * x * (x2 - 3.0f * y2);
}


// Unified compact density-force core.

template <int C>
struct SpinDensityPullShared {
  float rho0[C * 3];
  float l1[C * 9];
  float angular2[C * 15];
  float angular3[C * 21];
  float angular4[C * 27];
  float geom[C * 6];
  float rho0_dot[C * 3];
  float l1_dot[C * 9];
};

static_assert(
    sizeof(SpinDensityPullShared<4>) == 372 * sizeof(float),
    "spin density pulls must remain tightly packed");

template <int C>
struct SpinDensityPullSharedPadded {
  SpinDensityPullShared<C> values;
  // Shift each atom's base by one bank so same-component subwarp reads do not
  // alias across the eight centers in a tile.
  float bank_padding;
};

template <int C, int AtomsPerWarp>
struct SpinDensityForceTileShared {
  SpinDensityPullSharedPadded<C> pulls[AtomsPerWarp];
};

template <
    int C,
    int LMax,
    SpinVirialMode VirialMode,
    int AtomsPerWarp,
    int EdgesPerAtomBatch>
__global__ void __launch_bounds__(32, 8)
accumulate_spin_density_forces_tile_f32(
    int atom_count,
    int atom_stride,
    int struct_dim,
    int num_types,
    int spin_basis_size,
    float spin_cutoff,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    const float* __restrict__ density_rho0_cache,
    const float* __restrict__ density_angular2_cache,
    const float* __restrict__ density_angular3_cache,
    const float* __restrict__ density_angular4_cache,
    const float* __restrict__ density_geom_cache,
    const float* __restrict__ density_rho0_dot_cache,
    const float* __restrict__ density_raw1_cache,
    const float* __restrict__ density_raw1_dot_cache,
    double* __restrict__ force_soa3,
    double* __restrict__ mforce_soa3,
    double* __restrict__ virial_soa9) {
  using Layout = SpinStaticLayout<C, LMax>;
  static_assert(
      AtomsPerWarp * EdgesPerAtomBatch == 32,
      "spin density tile must fill one warp");
  const int lane = threadIdx.x;
  const int atom_in_tile = lane / EdgesPerAtomBatch;
  const int edge_lane = lane - atom_in_tile * EdgesPerAtomBatch;
  const int atom = blockIdx.x * AtomsPerWarp + atom_in_tile;
  const bool active_atom = atom < atom_count;
  constexpr unsigned int FullWarpMask = 0xffffffffu;
  __shared__ SpinDensityForceTileShared<C, AtomsPerWarp> shared;
  constexpr int cache_stride = 1;
  double direct_center_mforce_lane = 0.0;

  const float* rho0_pull_base;
  const float* l1_pull_base;
  const float* angular2_pull_base;
  const float* angular3_pull_base;
  const float* angular4_pull_base;
  const float* geom_pull_base;
  const float* rho0_dot_pull_base;
  const float* l1_dot_pull_base;
  SpinDensityPullShared<C>* atom_pulls =
      &shared.pulls[atom_in_tile].values;
  const double spin_lane = active_atom && edge_lane < 3
      ? spins_soa3[edge_lane * atom_stride + atom]
      : 0.0;
  const double si0[3] = {
      __shfl_sync(FullWarpMask, spin_lane, 0, EdgesPerAtomBatch),
      __shfl_sync(FullWarpMask, spin_lane, 1, EdgesPerAtomBatch),
      __shfl_sync(FullWarpMask, spin_lane, 2, EdgesPerAtomBatch)};
  const double spin_trace =
      (si0[0] * si0[0] + si0[1] * si0[1] + si0[2] * si0[2]) / 3.0;

  for (int component = edge_lane;
       active_atom && component < C * 3;
       component += EdgesPerAtomBatch) {
    const int c = component / 3;
    const float alpha0 =
        fp[atom + atom_stride * (struct_dim + Layout::Rho0Offset + c)];
    const float alpha0_dot =
        fp[atom + atom_stride * (struct_dim + Layout::Rho0DotOffset + c)];
    const int index = spin_atom_cache_index<C * 3>(
        atom_stride, atom, component);
    const float rho0 = density_rho0_cache[index];
    const float rho0_dot = density_rho0_dot_cache[index];
    atom_pulls->rho0[component] =
        2.0f * alpha0 * rho0 + alpha0_dot * rho0_dot;
    atom_pulls->rho0_dot[component] =
        alpha0_dot * rho0;
  }

  for (int component = edge_lane;
       active_atom && LMax >= 1 && component < C * 9;
       component += EdgesPerAtomBatch) {
    const int c = component / 9;
    const int k = component - c * 9;
    const float alpha_rdot =
        fp[atom + atom_stride * (struct_dim + Layout::L1RdotOffset + c)];
    const float alpha_cross =
        fp[atom + atom_stride * (struct_dim + Layout::L1CrossOffset + c)];
    const float alpha_stf =
        fp[atom + atom_stride * (struct_dim + Layout::L1StfOffset + c)];
    const float alpha_raw1 =
        fp[atom + atom_stride * (struct_dim + Layout::Raw1DotOffset + c)];
    const int raw1_base = spin_atom_cache_index<C * 9>(
        atom_stride, atom, c * 9);
    const int a = k / 3;
    const int b = k - 3 * a;
    const float raw = density_raw1_cache[raw1_base + k];
    const float raw_transpose = density_raw1_cache[raw1_base + 3 * b + a];
    float mat = 0.0f;
    float stf = 0.5f * (raw + raw_transpose);
    if (a == b) {
      const float rdot =
          density_raw1_cache[raw1_base] +
          density_raw1_cache[raw1_base + 4] +
          density_raw1_cache[raw1_base + 8];
      mat = 2.0f * alpha_rdot * rdot;
      stf -= rdot / 3.0f;
    } else {
      mat = 2.0f * alpha_cross * (raw - raw_transpose);
    }
    const int index = spin_atom_cache_index<C * 9>(
        atom_stride, atom, component);
    const float raw_dot = density_raw1_dot_cache[index];
    atom_pulls->l1[component] =
        mat + 2.0f * alpha_stf * stf + alpha_raw1 * raw_dot;
    atom_pulls->l1_dot[component] =
        alpha_raw1 * raw;

  }

  for (int component = edge_lane;
       active_atom && component < C * 6;
       component += EdgesPerAtomBatch) {
    const int c = component / 6;
    const int k = component - c * 6;
    int a;
    int b;
    if (k < 3) {
      a = 0;
      b = k;
    } else if (k < 5) {
      a = 1;
      b = k - 2;
    } else {
      a = 2;
      b = 2;
    }
    const double alpha_geom = static_cast<double>(
        fp[atom + atom_stride * (struct_dim + Layout::GeomOffset + c)]);
    double ss = si0[a] * si0[b];
    if (a == b) {
      ss -= spin_trace;
    }
    atom_pulls->geom[component] = static_cast<float>(alpha_geom * ss);
  }

  for (int component = edge_lane;
       active_atom && LMax >= 2 && component < C * 15;
       component += EdgesPerAtomBatch) {
    const int c = component / 15;
    const float alpha =
        fp[atom + atom_stride * (struct_dim + Layout::Angular2Offset + c)];
    const int index = spin_atom_cache_index<C * 15>(
        atom_stride, atom, component);
    atom_pulls->angular2[component] =
        2.0f * alpha * density_angular2_cache[index];
  }
  for (int component = edge_lane;
       active_atom && LMax >= 3 && component < C * 21;
       component += EdgesPerAtomBatch) {
    const int c = component / 21;
    const float alpha =
        fp[atom + atom_stride * (struct_dim + Layout::Angular3Offset + c)];
    const int index = spin_atom_cache_index<C * 21>(
        atom_stride, atom, component);
    atom_pulls->angular3[component] =
        2.0f * alpha * density_angular3_cache[index];
  }
  for (int component = edge_lane;
       active_atom && LMax >= 4 && component < C * 27;
       component += EdgesPerAtomBatch) {
    const int c = component / 27;
    const float alpha =
        fp[atom + atom_stride * (struct_dim + Layout::Angular4Offset + c)];
    const int index = spin_atom_cache_index<C * 27>(
        atom_stride, atom, component);
    atom_pulls->angular4[component] =
        2.0f * alpha * density_angular4_cache[index];
  }

  if (active_atom && edge_lane < 3) {
    double grad_spin_i_direct = 0.0;
    for (int c = 0; c < C; ++c) {
      const double alpha_geom = static_cast<double>(
          fp[atom + atom_stride * (struct_dim + Layout::GeomOffset + c)]);
      double gs = 0.0;
      for (int b = 0; b < 3; ++b) {
        int packed_component = 0;
        if (edge_lane == 0) {
          packed_component = b;
        } else if (edge_lane == 1) {
          packed_component = b == 0 ? 1 : b + 2;
        } else {
          packed_component = b == 0 ? 2 : (b == 1 ? 4 : 5);
        }
        const double geom_ab = static_cast<double>(
            density_geom_cache[
                spin_atom_cache_index<C * 6>(
                    atom_stride, atom, c * 6 + packed_component)]);
        gs += 2.0 * geom_ab * si0[b];
      }
      grad_spin_i_direct += alpha_geom * gs;
    }
    direct_center_mforce_lane = -grad_spin_i_direct;
  }
  __syncwarp(FullWarpMask);

  rho0_pull_base = atom_pulls->rho0;
  l1_pull_base = atom_pulls->l1;
  angular2_pull_base = atom_pulls->angular2;
  angular3_pull_base = atom_pulls->angular3;
  angular4_pull_base = atom_pulls->angular4;
  geom_pull_base = atom_pulls->geom;
  rho0_dot_pull_base = atom_pulls->rho0_dot;
  l1_dot_pull_base = atom_pulls->l1_dot;

  float center_force[3] = {};
  float center_mforce[3] = {};
  float center_virial[VirialMode == SpinVirialMode::disabled ? 1 : 9] = {};
  const int radial_count = active_atom ? nn_radial[atom] : 0;
  for (int slot = edge_lane;
       slot < radial_count;
       slot += EdgesPerAtomBatch) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    float rhat[3];
    float dist = 0.0f;
    float si[3];
    float sj[3];
    float weights[C];
    float weight_derivatives[C];
    if (!load_spin_edge_f32<C>(
        atom,
        neighbor,
        atom_stride,
        num_types,
        spin_basis_size,
        spin_cutoff,
        box,
        types,
        positions_soa3,
        spins_soa3,
        descriptor_coefficients,
        spin_coefficient_offset,
        rhat,
        dist,
        si,
        sj,
        weights,
        weight_derivatives)) {
      continue;
    }

    const float dot = dot3f(si, sj);
    float grad_weight[C] = {};
    float grad_rhat[3] = {};
    float grad_si[3] = {};
    float grad_sj[3] = {};
    float grad_dot = 0.0f;

    const float sj2 = dot3f(sj, sj);
    const float ri_dot_si = dot3f(rhat, si);
    const float ri_dot_sj = dot3f(rhat, sj);
    const float bond_axis = ri_dot_si * ri_dot_sj;
    float scalar_weighted = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float alpha = fp[atom + atom_stride * (struct_dim + 2 + c)];
      grad_weight[c] += alpha * dot;
      scalar_weighted += alpha * weights[c];
    }
    grad_dot += scalar_weighted;
    scalar_weighted = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float alpha = fp[atom + atom_stride * (struct_dim + 2 + C + c)];
      grad_weight[c] += alpha * dot * dot;
      scalar_weighted += alpha * weights[c];
    }
    grad_dot += 2.0f * dot * scalar_weighted;
    scalar_weighted = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float alpha = fp[atom + atom_stride * (struct_dim + 2 + 2 * C + c)];
      grad_weight[c] += alpha * sj2;
      scalar_weighted += alpha * weights[c];
    }
    for (int d = 0; d < 3; ++d) {
      grad_sj[d] += 2.0f * scalar_weighted * sj[d];
    }
    scalar_weighted = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float alpha = fp[atom + atom_stride * (struct_dim + 2 + 3 * C + c)];
      grad_weight[c] += alpha * bond_axis;
      scalar_weighted += alpha * weights[c];
    }
    for (int d = 0; d < 3; ++d) {
      grad_si[d] += scalar_weighted * ri_dot_sj * rhat[d];
      grad_sj[d] += scalar_weighted * ri_dot_si * rhat[d];
      grad_rhat[d] +=
          scalar_weighted * (ri_dot_sj * si[d] + ri_dot_si * sj[d]);
    }

    for (int c = 0; c < C; ++c) {
      const float b[3] = {
          rho0_pull_base[(c * 3) * cache_stride],
          rho0_pull_base[(c * 3 + 1) * cache_stride],
          rho0_pull_base[(c * 3 + 2) * cache_stride]};
      const float bd[3] = {
          rho0_dot_pull_base[(c * 3) * cache_stride],
          rho0_dot_pull_base[(c * 3 + 1) * cache_stride],
          rho0_dot_pull_base[(c * 3 + 2) * cache_stride]};
      const float u[3] = {
          b[0] + dot * bd[0],
          b[1] + dot * bd[1],
          b[2] + dot * bd[2]};
      const float w = weights[c];
      grad_weight[c] += dot3f(sj, u);
      for (int d = 0; d < 3; ++d) {
        grad_sj[d] += w * u[d];
      }
      grad_dot += w * dot3f(sj, bd);

      const float* gbase = geom_pull_base + c * 6 * cache_stride;
      const float kr[3] = {
          gbase[0 * cache_stride] * rhat[0] +
              gbase[1 * cache_stride] * rhat[1] +
              gbase[2 * cache_stride] * rhat[2],
          gbase[1 * cache_stride] * rhat[0] +
              gbase[3 * cache_stride] * rhat[1] +
              gbase[4 * cache_stride] * rhat[2],
          gbase[2 * cache_stride] * rhat[0] +
              gbase[4 * cache_stride] * rhat[1] +
              gbase[5 * cache_stride] * rhat[2]};
      for (int d = 0; d < 3; ++d) {
        grad_weight[c] += rhat[d] * kr[d];
        grad_rhat[d] += 2.0f * w * kr[d];
      }
      if constexpr (LMax >= 1) {
        const float* mbase = l1_pull_base + c * 9 * cache_stride;
        const float* mdbase = l1_dot_pull_base + c * 9 * cache_stride;
        const float v1[3] = {
            mbase[0 * cache_stride] * sj[0] +
                mbase[1 * cache_stride] * sj[1] +
                mbase[2 * cache_stride] * sj[2],
            mbase[3 * cache_stride] * sj[0] +
                mbase[4 * cache_stride] * sj[1] +
                mbase[5 * cache_stride] * sj[2],
            mbase[6 * cache_stride] * sj[0] +
                mbase[7 * cache_stride] * sj[1] +
                mbase[8 * cache_stride] * sj[2]};
        const float v2[3] = {
            mdbase[0 * cache_stride] * sj[0] +
                mdbase[1 * cache_stride] * sj[1] +
                mdbase[2 * cache_stride] * sj[2],
            mdbase[3 * cache_stride] * sj[0] +
                mdbase[4 * cache_stride] * sj[1] +
                mdbase[5 * cache_stride] * sj[2],
            mdbase[6 * cache_stride] * sj[0] +
                mdbase[7 * cache_stride] * sj[1] +
                mdbase[8 * cache_stride] * sj[2]};
        const float u1[3] = {
            v1[0] + dot * v2[0],
            v1[1] + dot * v2[1],
            v1[2] + dot * v2[2]};
        const float mt[3] = {
            mbase[0 * cache_stride] * rhat[0] +
                mbase[3 * cache_stride] * rhat[1] +
                mbase[6 * cache_stride] * rhat[2] +
                dot * (mdbase[0 * cache_stride] * rhat[0] +
                       mdbase[3 * cache_stride] * rhat[1] +
                       mdbase[6 * cache_stride] * rhat[2]),
            mbase[1 * cache_stride] * rhat[0] +
                mbase[4 * cache_stride] * rhat[1] +
                mbase[7 * cache_stride] * rhat[2] +
                dot * (mdbase[1 * cache_stride] * rhat[0] +
                       mdbase[4 * cache_stride] * rhat[1] +
                       mdbase[7 * cache_stride] * rhat[2]),
            mbase[2 * cache_stride] * rhat[0] +
                mbase[5 * cache_stride] * rhat[1] +
                mbase[8 * cache_stride] * rhat[2] +
                dot * (mdbase[2 * cache_stride] * rhat[0] +
                       mdbase[5 * cache_stride] * rhat[1] +
                       mdbase[8 * cache_stride] * rhat[2])};
        for (int d = 0; d < 3; ++d) {
          grad_weight[c] += rhat[d] * u1[d];
          grad_rhat[d] += w * u1[d];
          grad_sj[d] += w * mt[d];
        }
        grad_dot += w * dot3f(rhat, v2);
      }
    }

    #pragma unroll
    for (int ell = 2; ell <= LMax; ++ell) {
      const int width = (2 * ell + 1) * 3;
      const float* angular =
          ell == 2 ? angular2_pull_base :
          ell == 3 ? angular3_pull_base : angular4_pull_base;
      float ylm[9];
      const int ylm_width = real_spherical_harmonics_spinf(rhat, ell, ylm);
      float ge[27] = {};
      for (int c = 0; c < C; ++c) {
        for (int m = 0; m < ylm_width; ++m) {
          for (int d = 0; d < 3; ++d) {
            const int k = m * 3 + d;
            const float gd = angular[(c * width + k) * cache_stride];
            grad_weight[c] += gd * ylm[m] * sj[d];
            ge[k] += gd * weights[c];
          }
        }
      }
      float grad_ylm[9] = {};
      for (int m = 0; m < ylm_width; ++m) {
        for (int d = 0; d < 3; ++d) {
          const float g = ge[m * 3 + d];
          grad_sj[d] += g * ylm[m];
          grad_ylm[m] += g * sj[d];
        }
      }
      add_real_spherical_harmonics_gradientf(rhat, ell, grad_ylm, grad_rhat);
    }

    for (int d = 0; d < 3; ++d) {
      grad_si[d] += grad_dot * sj[d];
      grad_sj[d] += grad_dot * si[d];
    }

    float grad_dist = 0.0f;
    for (int c = 0; c < C; ++c) {
      grad_dist += grad_weight[c] * weight_derivatives[c];
    }
    float dot_r = 0.0f;
    for (int d = 0; d < 3; ++d) {
      dot_r += grad_rhat[d] * rhat[d];
    }
    float grad_rij[3];
    for (int d = 0; d < 3; ++d) {
      grad_rij[d] =
          grad_dist * rhat[d] + (grad_rhat[d] - dot_r * rhat[d]) / dist;
      center_force[d] += grad_rij[d];
      center_mforce[d] -= grad_si[d];
      atomicAdd(force_soa3 + d * atom_stride + neighbor,
                -static_cast<double>(grad_rij[d]));
      atomicAdd(mforce_soa3 + d * atom_stride + neighbor,
                -static_cast<double>(grad_sj[d]));
    }
    if constexpr (VirialMode != SpinVirialMode::disabled) {
      for (int a = 0; a < 3; ++a) {
        const float rij_a = rhat[a] * dist;
        for (int b = 0; b < 3; ++b) {
          center_virial[virial_internal_component(a * 3 + b)] -=
              rij_a * grad_rij[b];
        }
      }
    }
  }

  for (int offset = EdgesPerAtomBatch / 2; offset > 0; offset >>= 1) {
    for (int d = 0; d < 3; ++d) {
      center_force[d] += __shfl_down_sync(
          FullWarpMask, center_force[d], offset, EdgesPerAtomBatch);
      center_mforce[d] += __shfl_down_sync(
          FullWarpMask, center_mforce[d], offset, EdgesPerAtomBatch);
    }
    if constexpr (VirialMode != SpinVirialMode::disabled) {
      for (int component = 0; component < 9; ++component) {
        center_virial[component] += __shfl_down_sync(
            FullWarpMask,
            center_virial[component],
            offset,
            EdgesPerAtomBatch);
      }
    }
  }
  double direct_center_mforce[3] = {};
  for (int d = 0; d < 3; ++d) {
    direct_center_mforce[d] = __shfl_sync(
        FullWarpMask,
        direct_center_mforce_lane,
        d,
        EdgesPerAtomBatch);
  }
  if (active_atom && edge_lane == 0) {
    for (int d = 0; d < 3; ++d) {
      atomicAdd(force_soa3 + d * atom_stride + atom,
                static_cast<double>(center_force[d]));
      const double center_mforce_total =
          static_cast<double>(center_mforce[d]) + direct_center_mforce[d];
      atomicAdd(mforce_soa3 + d * atom_stride + atom,
                center_mforce_total);
    }
    if constexpr (VirialMode != SpinVirialMode::disabled) {
      for (int component = 0; component < 9; ++component) {
        const double value = static_cast<double>(center_virial[component]);
        if constexpr (VirialMode == SpinVirialMode::cpu_atom_decomposition) {
          atomicAdd(virial_soa9 + component * atom_stride, value);
        } else {
          virial_soa9[component * atom_stride + atom] += value;
        }
      }
    }
  }
}


// Unified compact chiral descriptor-finalize and force core.

template <int C>
__global__ void build_spin_chiral_descriptors_f32(
    int atom_count,
    int atom_stride,
    int struct_dim,
    SpinCoreLayout layout,
    const double* __restrict__ spins_soa3,
    const float* __restrict__ density_geom_cache,
    const float* __restrict__ density_raw1_cache,
    const float* __restrict__ chiral_polar_cache,
    const float* __restrict__ chiral_octupoles_raw_cache,
    const float* __restrict__ chiral_hexadecapoles_raw_cache,
    float* __restrict__ chiral_chirals_cache,
    float* __restrict__ descriptors) {
  constexpr int ChiC = C < 2 ? C : 2;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int atom = tid / C;
  const int c = tid - atom * C;
  if (atom >= atom_count) {
    return;
  }

  float geom_packed[6];
  float geom[9];
  float polar[3];
  for (int k = 0; k < 6; ++k) {
    geom_packed[k] = density_geom_cache[
        spin_atom_cache_index<C * 6>(
            atom_stride, atom, c * 6 + k)];
  }
  unpack_spin_symmetric6f(geom_packed, geom);
  for (int k = 0; k < 3; ++k) {
    polar[k] = chiral_polar_cache[
        spin_atom_cache_index<C * 3>(
            atom_stride, atom, c * 3 + k)];
  }

  float octupole_reduced[kSpinChiralOReducedCount];
  const int octupole_base =
      spin_atom_cache_index<C * kSpinChiralOReducedCount>(
          atom_stride, atom, c * kSpinChiralOReducedCount);
  for (int k = 0; k < kSpinChiralOReducedCount; ++k) {
    octupole_reduced[k] = chiral_octupoles_raw_cache[octupole_base + k];
  }

  float chiral_value = 0.0f;
  if (c < ChiC) {
    float q_reduced[5];
    float h_reduced[9];
    q_reduced[0] = geom[0] - geom[8];
    q_reduced[1] = -geom[5];
    q_reduced[2] = 0.5f * geom[2];
    q_reduced[3] = geom[1];
    q_reduced[4] = geom[0] - geom[4];
    const int hexadecapole_base =
        spin_atom_cache_index<ChiC * kSpinChiralHReducedCount>(
            atom_stride, atom, c * kSpinChiralHReducedCount);
    for (int k = 0; k < kSpinChiralHReducedCount; ++k) {
      h_reduced[k] = chiral_hexadecapoles_raw_cache[hexadecapole_base + k];
    }
    chiral_value =
        contract_spin_chiral_qoh_reduced_f32(
            q_reduced, octupole_reduced, h_reduced);
    chiral_chirals_cache[
        spin_atom_cache_index<ChiC>(atom_stride, atom, c)] =
        chiral_value;
  }

  float pseudodevs[9];
  build_spin_chiral_pseudodev_center_f32(
      geom, polar, octupole_reduced, pseudodevs);

  const float si[3] = {
      static_cast<float>(spins_soa3[atom]),
      static_cast<float>(spins_soa3[atom_stride + atom]),
      static_cast<float>(spins_soa3[2 * atom_stride + atom])};
  float chiral_q0 = 0.0f;
  const int raw1_base = spin_atom_cache_index<C * 9>(
      atom_stride, atom, c * 9);
  float raw1[9];
  for (int k = 0; k < 9; ++k) {
    raw1[k] = density_raw1_cache[raw1_base + k];
  }
  if (c < ChiC) {
    float l1_cross[3];
    spin_raw1_crossf(raw1, l1_cross);
    chiral_q0 = -chiral_value * dot3f(si, l1_cross);
  }

  const float l1_rdot = spin_raw1_tracef(raw1);
  float chiral_q1 = 0.0f;
  for (int b = 0; b < 3; ++b) {
    float raw1_transpose_si = 0.0f;
    for (int a = 0; a < 3; ++a) {
      raw1_transpose_si +=
          raw1[3 * a + b] * si[a];
    }
    chiral_q1 +=
        polar[b] * (si[b] * l1_rdot - raw1_transpose_si);
  }

  float chiral_q2 = 0.0f;
  for (int b = 0; b < 3; ++b) {
    const float raw1_row[3] = {
        raw1[3 * b], raw1[3 * b + 1], raw1[3 * b + 2]};
    float spin_cross_raw1[3];
    cross3f(si, raw1_row, spin_cross_raw1);
    for (int a = 0; a < 3; ++a) {
      chiral_q2 += pseudodevs[3 * a + b] * spin_cross_raw1[a];
    }
  }
  if (c < ChiC) {
    descriptors[
        atom + atom_stride * (struct_dim + layout.chiral_offset + c)] =
        chiral_q0;
  }
  descriptors[
      atom + atom_stride * (struct_dim + layout.chiral_offset + ChiC + c)] =
          chiral_q1;
  descriptors[
      atom +
      atom_stride * (struct_dim + layout.chiral_offset + ChiC + C + c)] =
          chiral_q2;
}

template <int C>
struct SpinChiralPullShared {
  static constexpr int ChiC = C < 2 ? C : 2;
  float geom_terms[C * kSpinDeg2Count];
  float polar[C * 3];
  float raw1[C * 9];
  float rdot[C];
  float cross[C * 3];
  float octupole_terms[C * kSpinDeg3Count];
  float hexadecapole_terms[ChiC * kSpinDeg4Count];
  float direct_mforce[3];
};

static_assert(
    sizeof(SpinChiralPullShared<4>) == 161 * sizeof(float),
    "spin chiral center pulls must remain tightly packed");

template <int C, int AtomsPerWarp>
struct SpinChiralForceTileShared {
  // The 161-float stride naturally rotates equal-component accesses through
  // shared-memory banks for the eight independent centers in a warp.
  SpinChiralPullShared<C> pulls[AtomsPerWarp];
};

template <
    int C,
    SpinVirialMode VirialMode,
    int AtomsPerWarp,
    int EdgesPerAtomBatch>
__global__ void __launch_bounds__(32, 8)
accumulate_spin_chiral_forces_tile_f32(
    int atom_count,
    int atom_stride,
    int struct_dim,
    int num_types,
    int spin_basis_size,
    SpinCoreLayout layout,
    float spin_cutoff,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    const float* __restrict__ density_geom_cache,
    const float* __restrict__ density_raw1_cache,
    const float* __restrict__ chiral_polar_cache,
    const float* __restrict__ chiral_octupoles_raw_cache,
    const float* __restrict__ chiral_hexadecapoles_raw_cache,
    const float* __restrict__ chiral_chirals_cache,
    double* __restrict__ force_soa3,
    double* __restrict__ mforce_soa3,
    double* __restrict__ virial_soa9) {
  constexpr int ChiC = C < 2 ? C : 2;
  static_assert(
      AtomsPerWarp * EdgesPerAtomBatch == 32,
      "spin chiral tile must fill one warp");
  const int lane = threadIdx.x;
  const int atom_in_tile = lane / EdgesPerAtomBatch;
  const int edge_lane = lane - atom_in_tile * EdgesPerAtomBatch;
  const int atom = blockIdx.x * AtomsPerWarp + atom_in_tile;
  const bool active_atom = atom < atom_count;
  constexpr unsigned int FullWarpMask = 0xffffffffu;
  __shared__ SpinChiralForceTileShared<C, AtomsPerWarp> shared;
  SpinChiralPullShared<C>* atom_pulls = &shared.pulls[atom_in_tile];

  const double spin_lane = active_atom && edge_lane < 3
      ? spins_soa3[edge_lane * atom_stride + atom]
      : 0.0;
  const float si[3] = {
      static_cast<float>(__shfl_sync(
          FullWarpMask, spin_lane, 0, EdgesPerAtomBatch)),
      static_cast<float>(__shfl_sync(
          FullWarpMask, spin_lane, 1, EdgesPerAtomBatch)),
      static_cast<float>(__shfl_sync(
          FullWarpMask, spin_lane, 2, EdgesPerAtomBatch))};

  float direct_grad_si[3] = {};
  if (active_atom && edge_lane < C) {
    const int c = edge_lane;
    const float alpha_q0 = c < ChiC
        ? fp[atom + atom_stride * (struct_dim + layout.chiral_offset + c)]
        : 0.0f;
    const float alpha_q1 =
        fp[atom + atom_stride * (
            struct_dim + layout.chiral_offset + ChiC + c)];
    const float alpha_q2 = fp[
        atom + atom_stride * (
            struct_dim + layout.chiral_offset + ChiC + C + c)];

    float geom_packed[6];
    float geom[9];
    float polar[3];
    float raw1[9];
    float l1_cross[3];
    float octupole_reduced[kSpinChiralOReducedCount];
    const int geom_base = spin_atom_cache_index<C * 6>(
        atom_stride, atom, c * 6);
    const int raw1_base = spin_atom_cache_index<C * 9>(
        atom_stride, atom, c * 9);
    const int vector_base = spin_atom_cache_index<C * 3>(
        atom_stride, atom, c * 3);
    const int octupole_base =
        spin_atom_cache_index<C * kSpinChiralOReducedCount>(
            atom_stride, atom, c * kSpinChiralOReducedCount);
#pragma unroll
    for (int k = 0; k < 6; ++k) {
      geom_packed[k] = density_geom_cache[geom_base + k];
    }
    unpack_spin_symmetric6f(geom_packed, geom);
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      raw1[k] = density_raw1_cache[raw1_base + k];
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      polar[k] = chiral_polar_cache[vector_base + k];
    }
    spin_raw1_crossf(raw1, l1_cross);
#pragma unroll
    for (int k = 0; k < kSpinChiralOReducedCount; ++k) {
      octupole_reduced[k] = chiral_octupoles_raw_cache[octupole_base + k];
    }
    const float l1_rdot = spin_raw1_tracef(raw1);

    float grad_geom[9] = {};
    float grad_polar[3] = {};
    float grad_raw1[9] = {};
    float grad_l1_cross[3] = {};
    float grad_octupole_reduced[kSpinChiralOReducedCount] = {};
    float grad_hexadecapole_reduced[kSpinChiralHReducedCount] = {};
    float grad_l1_rdot = alpha_q1 * dot3f(polar, si);

    if (c < ChiC) {
      const float chiral = chiral_chirals_cache[
          spin_atom_cache_index<ChiC>(atom_stride, atom, c)];
      const float si_dot_cross = dot3f(si, l1_cross);
      const float grad_chiral = -alpha_q0 * si_dot_cross;
      for (int d = 0; d < 3; ++d) {
        grad_l1_cross[d] -= alpha_q0 * chiral * si[d];
        direct_grad_si[d] -= alpha_q0 * chiral * l1_cross[d];
      }

      float q_reduced[5];
      float h_reduced[kSpinChiralHReducedCount];
      q_reduced[0] = geom[0] - geom[8];
      q_reduced[1] = -geom[5];
      q_reduced[2] = 0.5f * geom[2];
      q_reduced[3] = geom[1];
      q_reduced[4] = geom[0] - geom[4];
      const int hexadecapole_base =
          spin_atom_cache_index<ChiC * kSpinChiralHReducedCount>(
              atom_stride, atom, c * kSpinChiralHReducedCount);
#pragma unroll
      for (int k = 0; k < kSpinChiralHReducedCount; ++k) {
        h_reduced[k] =
            chiral_hexadecapoles_raw_cache[hexadecapole_base + k];
      }
      add_spin_chiral_qoh_reduced_pull_f32(
          grad_chiral,
          q_reduced,
          octupole_reduced,
          h_reduced,
          grad_geom,
          grad_octupole_reduced,
          grad_hexadecapole_reduced);
    }

#pragma unroll
    for (int b = 0; b < 3; ++b) {
      float raw1_transpose_si = 0.0f;
#pragma unroll
      for (int a = 0; a < 3; ++a) {
        raw1_transpose_si += raw1[3 * a + b] * si[a];
        grad_raw1[3 * a + b] -= alpha_q1 * si[a] * polar[b];
      }
      grad_polar[b] +=
          alpha_q1 * (si[b] * l1_rdot - raw1_transpose_si);
    }
#pragma unroll
    for (int a = 0; a < 3; ++a) {
      float raw1_polar = 0.0f;
#pragma unroll
      for (int b = 0; b < 3; ++b) {
        raw1_polar += raw1[3 * a + b] * polar[b];
      }
      direct_grad_si[a] +=
          alpha_q1 * (l1_rdot * polar[a] - raw1_polar);
    }

    float octupole_raw[kSpinDeg3Count];
    float pseudodev[9];
    reconstruct_spin_rank3_raw_f32(
        polar, octupole_reduced, octupole_raw);
    build_spin_chiral_pseudodev_center_f32(
        geom, polar, octupole_reduced, pseudodev);
    float grad_pseudodev[9] = {};
#pragma unroll
    for (int b = 0; b < 3; ++b) {
      const float raw1_row[3] = {
          raw1[3 * b], raw1[3 * b + 1], raw1[3 * b + 2]};
      float spin_cross_raw1[3];
      cross3f(si, raw1_row, spin_cross_raw1);
      float grad_spin_cross_raw1[3];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
        grad_pseudodev[3 * a + b] +=
            alpha_q2 * spin_cross_raw1[a];
        grad_spin_cross_raw1[a] = alpha_q2 * pseudodev[3 * a + b];
      }
      float grad_si_part[3];
      float grad_raw1_row[3];
      cross3f(raw1_row, grad_spin_cross_raw1, grad_si_part);
      cross3f(grad_spin_cross_raw1, si, grad_raw1_row);
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        direct_grad_si[d] += grad_si_part[d];
        grad_raw1[3 * b + d] += grad_raw1_row[d];
      }
    }

    float grad_octupole_raw[kSpinDeg3Count] = {};
    add_spin_chiral_pseudodev_center_pull_f32(
        geom,
        octupole_raw,
        grad_pseudodev,
        grad_geom,
        grad_octupole_raw);
    add_spin_rank3_reconstruction_pull_f32(
        grad_octupole_raw, grad_polar, grad_octupole_reduced);

    project_rank2_spin_gradientf(
        grad_geom, atom_pulls->geom_terms + c * kSpinDeg2Count);
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      atom_pulls->polar[c * 3 + k] = grad_polar[k];
      atom_pulls->cross[c * 3 + k] = grad_l1_cross[k];
    }
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      atom_pulls->raw1[c * 9 + k] = grad_raw1[k];
    }
    atom_pulls->rdot[c] = grad_l1_rdot;
    float octupole_terms[kSpinDeg3Count] = {};
    add_spin_chiral_o_reduced_terms_f32(
        grad_octupole_reduced, octupole_terms);
#pragma unroll
    for (int k = 0; k < kSpinDeg3Count; ++k) {
      atom_pulls->octupole_terms[c * kSpinDeg3Count + k] =
          octupole_terms[k];
    }
    if (c < ChiC) {
      float hexadecapole_terms[kSpinDeg4Count] = {};
      add_spin_chiral_h_reduced_terms_f32(
          grad_hexadecapole_reduced, hexadecapole_terms);
#pragma unroll
      for (int k = 0; k < kSpinDeg4Count; ++k) {
        atom_pulls->hexadecapole_terms[c * kSpinDeg4Count + k] =
            hexadecapole_terms[k];
      }
    }
  }

#pragma unroll
  for (int offset = EdgesPerAtomBatch / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (int d = 0; d < 3; ++d) {
      direct_grad_si[d] += __shfl_down_sync(
          FullWarpMask, direct_grad_si[d], offset, EdgesPerAtomBatch);
    }
  }
  if (active_atom && edge_lane == 0) {
#pragma unroll
    for (int d = 0; d < 3; ++d) {
      atom_pulls->direct_mforce[d] = -direct_grad_si[d];
    }
  }
  __syncwarp(FullWarpMask);

  float center_force[3] = {};
  float center_mforce[3] = {};
  float center_virial[VirialMode == SpinVirialMode::disabled ? 1 : 9] = {};
  const int radial_count = active_atom ? nn_radial[atom] : 0;
  for (int slot = edge_lane;
       slot < radial_count;
       slot += EdgesPerAtomBatch) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    float rhat[3];
    float dist = 0.0f;
    float edge_si[3];
    float sj[3];
    float weights[C];
    float weight_derivatives[C];
    if (!load_spin_edge_f32<C>(
        atom,
        neighbor,
        atom_stride,
        num_types,
        spin_basis_size,
        spin_cutoff,
        box,
        types,
        positions_soa3,
        spins_soa3,
        descriptor_coefficients,
        spin_coefficient_offset,
        rhat,
        dist,
        edge_si,
        sj,
        weights,
        weight_derivatives)) {
      continue;
    }
    float m2[kSpinDeg2Count];
    float m3[kSpinDeg3Count];
    float m4[kSpinDeg4Count];
    fill_spin_monomials2f(rhat, m2);
    fill_spin_monomialsf(rhat, m3, m4);

    float grad_weight[C] = {};
    float grad_rhat[3] = {};
    float grad_sj[3] = {};
    for (int c = 0; c < C; ++c) {
      const float w = weights[c];
      const float* q_terms =
          atom_pulls->geom_terms + c * kSpinDeg2Count;
      grad_weight[c] +=
          dot_spin_termsf(q_terms, m2, kSpinDeg2Count);
      grad_rhat[0] += w *
          (2.0f * q_terms[0] * rhat[0] + q_terms[3] * rhat[1] +
           q_terms[4] * rhat[2]);
      grad_rhat[1] += w *
          (2.0f * q_terms[1] * rhat[1] + q_terms[3] * rhat[0] +
           q_terms[5] * rhat[2]);
      grad_rhat[2] += w *
          (2.0f * q_terms[2] * rhat[2] + q_terms[4] * rhat[0] +
           q_terms[5] * rhat[1]);

      const float* grad_polar = atom_pulls->polar + c * 3;
      grad_weight[c] += dot3f(grad_polar, rhat);
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        grad_rhat[d] += w * grad_polar[d];
      }

      const float* grad_raw1 = atom_pulls->raw1 + c * 9;
      float grad_raw1_sj[3];
      float grad_raw1_transpose_rhat[3];
#pragma unroll
      for (int a = 0; a < 3; ++a) {
        grad_raw1_sj[a] =
            grad_raw1[3 * a] * sj[0] +
            grad_raw1[3 * a + 1] * sj[1] +
            grad_raw1[3 * a + 2] * sj[2];
      }
#pragma unroll
      for (int b = 0; b < 3; ++b) {
        grad_raw1_transpose_rhat[b] =
            grad_raw1[b] * rhat[0] +
            grad_raw1[3 + b] * rhat[1] +
            grad_raw1[6 + b] * rhat[2];
      }
      grad_weight[c] += dot3f(rhat, grad_raw1_sj);
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        grad_rhat[d] += w * grad_raw1_sj[d];
        grad_sj[d] += w * grad_raw1_transpose_rhat[d];
      }

      const float grad_rdot = atom_pulls->rdot[c];
      const float rhat_dot_sj = dot3f(rhat, sj);
      grad_weight[c] += grad_rdot * rhat_dot_sj;
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        grad_rhat[d] += w * grad_rdot * sj[d];
        grad_sj[d] += w * grad_rdot * rhat[d];
      }

      const float* grad_cross = atom_pulls->cross + c * 3;
      float rhat_cross_sj[3];
      float sj_cross_grad[3];
      float grad_cross_rhat[3];
      cross3f(rhat, sj, rhat_cross_sj);
      cross3f(sj, grad_cross, sj_cross_grad);
      cross3f(grad_cross, rhat, grad_cross_rhat);
      grad_weight[c] += dot3f(grad_cross, rhat_cross_sj);
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        grad_rhat[d] += w * sj_cross_grad[d];
        grad_sj[d] += w * grad_cross_rhat[d];
      }

      const float* octupole_terms =
          atom_pulls->octupole_terms + c * kSpinDeg3Count;
      grad_weight[c] +=
          dot_spin_termsf(octupole_terms, m3, kSpinDeg3Count);
      float octupole_gradient[3];
      evaluate_spin_polynomial_gradientf(
          3, octupole_terms, rhat, octupole_gradient);
#pragma unroll
      for (int d = 0; d < 3; ++d) {
        grad_rhat[d] += w * octupole_gradient[d];
      }

      if (c < ChiC) {
        const float* hexadecapole_terms =
            atom_pulls->hexadecapole_terms + c * kSpinDeg4Count;
        grad_weight[c] +=
            dot_spin_termsf(hexadecapole_terms, m4, kSpinDeg4Count);
        float hexadecapole_gradient[3];
        evaluate_spin_polynomial_gradientf(
            4, hexadecapole_terms, rhat, hexadecapole_gradient);
#pragma unroll
        for (int d = 0; d < 3; ++d) {
          grad_rhat[d] += w * hexadecapole_gradient[d];
        }
      }
    }

    float grad_dist = 0.0f;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      grad_dist += grad_weight[c] * weight_derivatives[c];
    }
    const float dot_r = dot3f(grad_rhat, rhat);
    float grad_rij[3];
#pragma unroll
    for (int d = 0; d < 3; ++d) {
      grad_rij[d] =
          grad_dist * rhat[d] + (grad_rhat[d] - dot_r * rhat[d]) / dist;
      center_force[d] += grad_rij[d];
      atomicAdd(
          force_soa3 + d * atom_stride + neighbor,
          -static_cast<double>(grad_rij[d]));
      atomicAdd(
          mforce_soa3 + d * atom_stride + neighbor,
          -static_cast<double>(grad_sj[d]));
    }
    if constexpr (VirialMode != SpinVirialMode::disabled) {
#pragma unroll
      for (int a = 0; a < 3; ++a) {
        const float rij_a = rhat[a] * dist;
#pragma unroll
        for (int b = 0; b < 3; ++b) {
          center_virial[virial_internal_component(3 * a + b)] -=
              rij_a * grad_rij[b];
        }
      }
    }
  }

#pragma unroll
  for (int offset = EdgesPerAtomBatch / 2; offset > 0; offset >>= 1) {
#pragma unroll
    for (int d = 0; d < 3; ++d) {
      center_force[d] += __shfl_down_sync(
          FullWarpMask, center_force[d], offset, EdgesPerAtomBatch);
      center_mforce[d] += __shfl_down_sync(
          FullWarpMask, center_mforce[d], offset, EdgesPerAtomBatch);
    }
    if constexpr (VirialMode != SpinVirialMode::disabled) {
#pragma unroll
      for (int component = 0; component < 9; ++component) {
        center_virial[component] += __shfl_down_sync(
            FullWarpMask,
            center_virial[component],
            offset,
            EdgesPerAtomBatch);
      }
    }
  }
  if (active_atom && edge_lane == 0) {
#pragma unroll
    for (int d = 0; d < 3; ++d) {
      atomicAdd(
          force_soa3 + d * atom_stride + atom,
          static_cast<double>(center_force[d]));
      atomicAdd(
          mforce_soa3 + d * atom_stride + atom,
          static_cast<double>(center_mforce[d]) +
              static_cast<double>(atom_pulls->direct_mforce[d]));
    }
    if constexpr (VirialMode != SpinVirialMode::disabled) {
#pragma unroll
      for (int component = 0; component < 9; ++component) {
        const double value = static_cast<double>(center_virial[component]);
        if constexpr (VirialMode == SpinVirialMode::cpu_atom_decomposition) {
          atomicAdd(virial_soa9 + component * atom_stride, value);
        } else {
          virial_soa9[component * atom_stride + atom] += value;
        }
      }
    }
  }
}
