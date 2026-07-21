#pragma once

// Internal CUDA implementation fragment for spin_onsite.cu.
// Included inside nep_adapters::cuda_backend's anonymous namespace.
// Contains shared device math and the unified compact spin descriptor core.

template <int ComponentCount>
__device__ __forceinline__ int spin_atom_cache_index(
    int /*atom_stride*/,
    int atom,
    int component) {
  static_assert(ComponentCount > 0, "spin cache component count must be positive");
  return atom * ComponentCount + component;
}

__device__ __forceinline__ int spin_atom_cache_index(
    int /*atom_stride*/,
    int component_count,
    int atom,
    int component) {
  return atom * component_count + component;
}

__device__ __forceinline__ int virial_internal_component(int row_major) {
  return row_major == 0 ? 0 :
         row_major == 1 ? 3 :
         row_major == 2 ? 4 :
         row_major == 3 ? 6 :
         row_major == 4 ? 1 :
         row_major == 5 ? 5 :
         row_major == 6 ? 7 :
         row_major == 7 ? 8 : 2;
}

// Descriptor construction and streaming primitive core.

__device__ __forceinline__ void compute_spin_edge_geometry_f32(
    int atom,
    int neighbor,
    int atom_stride,
    SimulationBox box,
    const double* __restrict__ positions_soa3,
    float* rhat,
    float& dist,
    float* delta = nullptr) {
  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
  minimum_image_delta(
      box,
      positions_soa3[neighbor] - positions_soa3[atom],
      positions_soa3[atom_stride + neighbor] -
          positions_soa3[atom_stride + atom],
      positions_soa3[2 * atom_stride + neighbor] -
          positions_soa3[2 * atom_stride + atom],
      dx,
      dy,
      dz);
  dist = sqrtf(dx * dx + dy * dy + dz * dz);
  const float inv_dist = dist > 0.0f ? 1.0f / dist : 0.0f;
  rhat[0] = dx * inv_dist;
  rhat[1] = dy * inv_dist;
  rhat[2] = dz * inv_dist;
  if (delta != nullptr) {
    delta[0] = dx;
    delta[1] = dy;
    delta[2] = dz;
  }
}

template <int C, bool NeedDerivatives>
__device__ __forceinline__ void evaluate_spin_edge_weights_b4_f32(
    float spin_cutoff,
    float dist,
    int num_types,
    int type_pair,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    float* weights,
    float* weight_derivatives) {
  constexpr int BasisCount = 4;
  constexpr float Pi = 3.14159265358979323846f;
  const float rcinv = 1.0f / spin_cutoff;
  const float r_scaled = dist * rcinv;
  const float cutoff_phase = Pi * r_scaled;
  const float fc = 0.5f * cosf(cutoff_phase) + 0.5f;
  const float shifted = r_scaled - 1.0f;
  const float x = 2.0f * shifted * shifted - 1.0f;
  const float t2 = 2.0f * x * x - 1.0f;
  const float t3 = 2.0f * x * t2 - x;
  const float fn_raw[BasisCount] = {
      1.0f,
      0.5f * (x + 1.0f),
      0.5f * (t2 + 1.0f),
      0.5f * (t3 + 1.0f)};
  const float fn[BasisCount] = {
      fc,
      fn_raw[1] * fc,
      fn_raw[2] * fc,
      fn_raw[3] * fc};
  float fnp[BasisCount] = {};
  if constexpr (NeedDerivatives) {
    const float fcp = -0.5f * Pi * sinf(cutoff_phase) * rcinv;
    const float radial_derivative = 2.0f * shifted * rcinv;
    const float u1 = 2.0f * x;
    const float u2 = 2.0f * x * u1 - 1.0f;
    fnp[0] = fcp;
    fnp[1] = radial_derivative * fc + fn_raw[1] * fcp;
    fnp[2] = 2.0f * u1 * radial_derivative * fc + fn_raw[2] * fcp;
    fnp[3] = 3.0f * u2 * radial_derivative * fc + fn_raw[3] * fcp;
  }

  #pragma unroll
  for (int c = 0; c < C; ++c) {
    float weight = 0.0f;
    float derivative = 0.0f;
    #pragma unroll
    for (int k = 0; k < BasisCount; ++k) {
      const int coefficient_index = spin_coefficient_offset +
          ((c * BasisCount + k) * num_types * num_types + type_pair);
      const float coefficient = descriptor_coefficients[coefficient_index];
      weight += fn[k] * coefficient;
      if constexpr (NeedDerivatives) {
        derivative += fnp[k] * coefficient;
      }
    }
    weights[c] = weight;
    if constexpr (NeedDerivatives) {
      weight_derivatives[c] = derivative;
    }
  }
}

template <int C, bool NeedDerivatives>
__device__ __noinline__ void evaluate_spin_edge_weights_generic_f32(
    int spin_basis_size,
    float spin_cutoff,
    float dist,
    int num_types,
    int type_pair,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    float* weights,
    float* weight_derivatives) {
  constexpr float Pi = 3.14159265358979323846f;
  const int basis_count = spin_basis_size + 1;
  const float rcinv = 1.0f / spin_cutoff;
  const float r_scaled = dist * rcinv;
  const float cutoff_phase = Pi * r_scaled;
  const float fc = 0.5f * cosf(cutoff_phase) + 0.5f;
  const float shifted = r_scaled - 1.0f;
  const float x = 2.0f * shifted * shifted - 1.0f;
  float fn[kMaxSpinBasis] = {};
  float fnp[kMaxSpinBasis] = {};
  fn[0] = fc;
  if constexpr (NeedDerivatives) {
    fnp[0] = -0.5f * Pi * sinf(cutoff_phase) * rcinv;
  }
  if (spin_basis_size >= 1) {
    const float raw = 0.5f * (x + 1.0f);
    fn[1] = raw * fc;
    if constexpr (NeedDerivatives) {
      const float radial_derivative = 2.0f * shifted * rcinv;
      fnp[1] = radial_derivative * fc + raw * fnp[0];
    }
  }
  float t0 = 1.0f;
  float t1 = x;
  float u0 = 1.0f;
  float u1 = 2.0f * x;
  for (int n = 2; n <= spin_basis_size; ++n) {
    const float t2 = 2.0f * x * t1 - t0;
    const float raw = 0.5f * (t2 + 1.0f);
    fn[n] = raw * fc;
    if constexpr (NeedDerivatives) {
      const float radial_derivative = 2.0f * shifted * rcinv;
      fnp[n] =
          static_cast<float>(n) * u1 * radial_derivative * fc +
          raw * fnp[0];
    }
    const float u2 = 2.0f * x * u1 - u0;
    t0 = t1;
    t1 = t2;
    u0 = u1;
    u1 = u2;
  }

#pragma unroll
  for (int c = 0; c < C; ++c) {
    float weight = 0.0f;
    float derivative = 0.0f;
    for (int k = 0; k < basis_count; ++k) {
      const int coefficient_index = spin_coefficient_offset +
          ((c * basis_count + k) * num_types * num_types + type_pair);
      const float coefficient = descriptor_coefficients[coefficient_index];
      weight += fn[k] * coefficient;
      if constexpr (NeedDerivatives) {
        derivative += fnp[k] * coefficient;
      }
    }
    weights[c] = weight;
    if constexpr (NeedDerivatives) {
      weight_derivatives[c] = derivative;
    }
  }
}

template <int C, bool NeedDerivatives>
__device__ __forceinline__ void evaluate_spin_edge_weights_f32(
    int spin_basis_size,
    float spin_cutoff,
    float dist,
    int num_types,
    int type_pair,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    float* weights,
    float* weight_derivatives) {
  if (spin_basis_size == 3) {
    evaluate_spin_edge_weights_b4_f32<C, NeedDerivatives>(
        spin_cutoff,
        dist,
        num_types,
        type_pair,
        descriptor_coefficients,
        spin_coefficient_offset,
        weights,
        weight_derivatives);
    return;
  }
  evaluate_spin_edge_weights_generic_f32<C, NeedDerivatives>(
      spin_basis_size,
      spin_cutoff,
      dist,
      num_types,
      type_pair,
      descriptor_coefficients,
      spin_coefficient_offset,
      weights,
      weight_derivatives);
}

// Compact float moment helpers shared by the descriptor and force cores.

__device__ __forceinline__ void cross3f(
    const float* a,
    const float* b,
    float* out);
__device__ __forceinline__ float dot3f(const float* a, const float* b);
__device__ __forceinline__ void stf_outer3f(
    const float* a,
    const float* b,
    float* out);
__device__ __forceinline__ void fill_spin_monomialsf(
    const float* u,
    float* m3,
    float* m4);
__device__ int real_spherical_harmonics_spinf(
    const float* rhat,
    int ell,
    float* out);

__device__ __forceinline__ void fill_spin_chiral_reduced_momentsf(
    const float* u,
    float* o_reduced,
    float* h_reduced) {
  const float x = u[0];
  const float y = u[1];
  const float z = u[2];
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float xy = x * y;
  const float xz = x * z;
  const float yz = y * z;

  o_reduced[0] = y * (y2 - 3.0f * z2);
  o_reduced[1] = z * (z2 - 3.0f * y2);
  o_reduced[2] = y * (x2 - z2);
  o_reduced[3] = z * (x2 - y2);
  o_reduced[4] = x * (y2 - z2);
  o_reduced[5] = xy * z;
  o_reduced[6] = x * (3.0f * z2 - x2);

  constexpr float OneSeventh = 1.0f / 7.0f;
  h_reduced[0] = xz * (-x2 - z2 + 6.0f * y2) * OneSeventh;
  h_reduced[1] = xy * (x2 + y2 - 6.0f * z2) * OneSeventh;
  h_reduced[2] = xz * (4.0f * x2 - 3.0f * z2 - 3.0f * y2) * OneSeventh;
  h_reduced[3] = xy * (-4.0f * x2 + 3.0f * y2 + 3.0f * z2) * OneSeventh;
  h_reduced[4] = yz * (-2.0f * y2 - 2.0f * z2 + 12.0f * x2) * OneSeventh;
  h_reduced[5] =
      2.0f * (y2 - z2) * (y2 + z2 - 6.0f * x2) * OneSeventh;
  h_reduced[6] = yz * (4.0f * y2 - 3.0f * z2 - 3.0f * x2) * OneSeventh;
  h_reduced[7] = (y2 - x2) * (x2 + y2 - 6.0f * z2) * OneSeventh;
  h_reduced[8] =
      (4.0f * x2 * x2 + y2 * y2 + 2.0f * z2 * z2 - 9.0f * x2 * y2 -
       15.0f * x2 * z2 + 3.0f * y2 * z2) * OneSeventh;
}

__device__ __forceinline__ float spin_raw1_tracef(const float* raw1) {
  return raw1[0] + raw1[4] + raw1[8];
}

__device__ __forceinline__ void spin_raw1_crossf(
    const float* raw1,
    float* cross) {
  cross[0] = raw1[5] - raw1[7];
  cross[1] = raw1[6] - raw1[2];
  cross[2] = raw1[1] - raw1[3];
}

__device__ __forceinline__ float spin_raw1_stf_componentf(
    const float* raw1,
    int a,
    int b,
    float trace_third) {
  float value = 0.5f * (raw1[3 * a + b] + raw1[3 * b + a]);
  if (a == b) {
    value -= trace_third;
  }
  return value;
}

__device__ __forceinline__ void unpack_spin_symmetric6f(
    const float* packed,
    float* full) {
  full[0] = packed[0];
  full[1] = packed[1];
  full[2] = packed[2];
  full[3] = packed[1];
  full[4] = packed[3];
  full[5] = packed[4];
  full[6] = packed[2];
  full[7] = packed[4];
  full[8] = packed[5];
}

constexpr int kSpinPrimitiveTileSlots = 64;
constexpr int kSpinPrimitiveTapeRows = 54;

template <int C, int LMax, bool Chiral, int TileSlots>
__device__ __forceinline__ void fill_spin_primitive_tape(
    int atom,
    int global_slot,
    int local_slot,
    int atom_stride,
    int num_types,
    int spin_basis_size,
    float spin_cutoff,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nl_radial,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    float (*tape)[TileSlots + 1],
    float (*weights)[TileSlots + 1]) {
  float rhat[3];
  float dist = 0.0f;
  float si[3];
  float sj[3];
  const int neighbor = nl_radial[atom + atom_stride * global_slot];
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
    for (int row = 0; row < kSpinPrimitiveTapeRows; ++row) {
      tape[row][local_slot] = 0.0f;
    }
    for (int c = 0; c < C; ++c) {
      weights[c][local_slot] = 0.0f;
    }
    return;
  }

  float edge_weights[C];
  const int type_pair = types[atom] * num_types + types[neighbor];
  evaluate_spin_edge_weights_f32<C, false>(
      spin_basis_size,
      spin_cutoff,
      dist,
      num_types,
      type_pair,
      descriptor_coefficients,
      spin_coefficient_offset,
      edge_weights,
      nullptr);
  #pragma unroll
  for (int c = 0; c < C; ++c) {
    weights[c][local_slot] = edge_weights[c];
  }
  tape[0][local_slot] = sj[0];
  tape[1][local_slot] = sj[1];
  tape[2][local_slot] = sj[2];
  tape[3][local_slot] = rhat[0];
  tape[4][local_slot] = rhat[1];
  tape[5][local_slot] = rhat[2];
  tape[6][local_slot] = dot3f(si, sj);

  float ylm[9];
  if constexpr (LMax >= 2) {
    const int width = real_spherical_harmonics_spinf(rhat, 2, ylm);
    for (int m = 0; m < width; ++m) {
      tape[7 + m][local_slot] = ylm[m];
    }
  }
  if constexpr (LMax >= 3) {
    const int width = real_spherical_harmonics_spinf(rhat, 3, ylm);
    for (int m = 0; m < width; ++m) {
      tape[12 + m][local_slot] = ylm[m];
    }
  }
  if constexpr (LMax >= 4) {
    const int width = real_spherical_harmonics_spinf(rhat, 4, ylm);
    for (int m = 0; m < width; ++m) {
      tape[19 + m][local_slot] = ylm[m];
    }
  }

  if constexpr (Chiral) {
    float o_reduced[kSpinChiralOReducedCount];
    float h_reduced[kSpinChiralHReducedCount];
    fill_spin_chiral_reduced_momentsf(rhat, o_reduced, h_reduced);
    for (int k = 0; k < kSpinChiralOReducedCount; ++k) {
      tape[28 + k][local_slot] = o_reduced[k];
    }
    for (int k = 0; k < kSpinChiralHReducedCount; ++k) {
      tape[35 + k][local_slot] = h_reduced[k];
    }
  }

  tape[44][local_slot] = 1.0f;
  tape[45][local_slot] = dot3f(sj, sj);
  tape[46][local_slot] = dot3f(rhat, si);
  tape[47][local_slot] = dot3f(rhat, sj);
  float geom[9];
  stf_outer3f(rhat, rhat, geom);
  tape[48][local_slot] = geom[0];
  tape[49][local_slot] = geom[1];
  tape[50][local_slot] = geom[2];
  tape[51][local_slot] = geom[4];
  tape[52][local_slot] = geom[5];
  tape[53][local_slot] = geom[8];
}

template <int C, int LMax, bool Chiral>
__global__ void __launch_bounds__(128, 1)
build_spin_descriptor_core_streaming(
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
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    float* __restrict__ density_rho0_cache,
    float* __restrict__ density_raw1_cache,
    float* __restrict__ density_angular2_cache,
    float* __restrict__ density_angular3_cache,
    float* __restrict__ density_angular4_cache,
    float* __restrict__ density_geom_cache,
    float* __restrict__ density_rho0_dot_cache,
    float* __restrict__ density_raw1_dot_cache,
    float* __restrict__ chiral_polar_cache,
    float* __restrict__ chiral_octupoles_raw_cache,
    float* __restrict__ chiral_hexadecapoles_raw_cache,
    float* __restrict__ descriptors) {
  constexpr int ChiC = C < 2 ? C : 2;
  constexpr int Rho0Base = 0;
  constexpr int Raw1Base = Rho0Base + C * 3;
  constexpr int Angular2Base = Raw1Base + C * 9;
  constexpr int Angular3Base = Angular2Base + C * 15;
  constexpr int Angular4Base = Angular3Base + C * 21;
  constexpr int GeomBase = Angular4Base + C * 27;
  constexpr int Rho0DotBase = GeomBase + C * 6;
  constexpr int Raw1DotBase = Rho0DotBase + C * 3;
  constexpr int DensityComponentCount = Raw1DotBase + C * 9;
  static_assert(DensityComponentCount == 93 * C);
  __shared__ float tape
      [kSpinPrimitiveTapeRows][kSpinPrimitiveTileSlots + 1];
  __shared__ float weights[C][kSpinPrimitiveTileSlots + 1];
  __shared__ float density_components[DensityComponentCount];

  const int lane = threadIdx.x;
  const int atom = blockIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const float si[3] = {
      static_cast<float>(spins_soa3[atom]),
      static_cast<float>(spins_soa3[atom_stride + atom]),
      static_cast<float>(spins_soa3[2 * atom_stride + atom])};
  if (lane == 0) {
    const float s2 = dot3f(si, si);
    descriptors[atom + atom_stride * struct_dim] = s2;
    descriptors[atom + atom_stride * (struct_dim + 1)] =
        s2 * s2;
  }

  constexpr int ScalarTaskBase = 0;
  constexpr int Rho0TaskBase = ScalarTaskBase + 4;
  constexpr int Raw1TaskBase = Rho0TaskBase + 3;
  constexpr int GeomTaskBase = Raw1TaskBase + 9;
  constexpr int PolarTaskBase = GeomTaskBase + 6;
  constexpr int OctTaskBase = PolarTaskBase + 3;
  constexpr int Angular2TaskBase = OctTaskBase + kSpinChiralOReducedCount;
  constexpr int HexTaskBase = Angular2TaskBase + 15;
  constexpr int Angular3TaskBase = HexTaskBase + kSpinChiralHReducedCount;
  constexpr int Angular4TaskBase = Angular3TaskBase + 21;
  constexpr int Rho0DotTaskBase = Angular4TaskBase + 27;
  constexpr int Raw1DotTaskBase = Rho0DotTaskBase + 3;
  constexpr int TaskCount = Raw1DotTaskBase + 9;
  static_assert(TaskCount == 116);

  const int task = lane;
  bool active_task = task < TaskCount;
  int row0 = 44;
  int row1 = 44;
  int row2 = -1;
  int width = 0;
  int component = 0;
  int component_count = 0;
  int channel_count = C;
  int moment_base = -1;
  int scalar_term = -1;
  float* output = nullptr;
  if (task < Rho0TaskBase) {
    scalar_term = task - ScalarTaskBase;
    if (scalar_term == 0) {
      row0 = 6;
    } else if (scalar_term == 1) {
      row0 = 6;
      row1 = 6;
    } else if (scalar_term == 2) {
      row0 = 45;
    } else {
      row0 = 46;
      row1 = 47;
    }
  } else if (task < Raw1TaskBase) {
    component = task - Rho0TaskBase;
    row0 = component;
    width = 3;
    component_count = C * width;
    moment_base = Rho0Base;
    output = density_rho0_cache;
  } else if (task < GeomTaskBase) {
    component = task - Raw1TaskBase;
    const int a = component / 3;
    row0 = 3 + a;
    row1 = component - 3 * a;
    width = 9;
    component_count = C * width;
    moment_base = Raw1Base;
    output = density_raw1_cache;
  } else if (task < PolarTaskBase) {
    component = task - GeomTaskBase;
    row0 = 48 + component;
    width = 6;
    component_count = C * width;
    moment_base = GeomBase;
    output = density_geom_cache;
  } else if (task < OctTaskBase) {
    component = task - PolarTaskBase;
    row0 = 3 + component;
    width = 3;
    component_count = C * width;
    output = chiral_polar_cache;
  } else if (task < Angular2TaskBase) {
    component = task - OctTaskBase;
    row0 = 28 + component;
    width = kSpinChiralOReducedCount;
    component_count = C * width;
    output = chiral_octupoles_raw_cache;
  } else if (task < HexTaskBase) {
    component = task - Angular2TaskBase;
    const int m = component / 3;
    row0 = 7 + m;
    row1 = component - 3 * m;
    width = 15;
    component_count = C * width;
    moment_base = Angular2Base;
    output = density_angular2_cache;
  } else if (task < Angular3TaskBase) {
    component = task - HexTaskBase;
    row0 = 35 + component;
    width = kSpinChiralHReducedCount;
    channel_count = ChiC;
    component_count = channel_count * width;
    output = chiral_hexadecapoles_raw_cache;
  } else if (task < Angular4TaskBase) {
    component = task - Angular3TaskBase;
    const int m = component / 3;
    row0 = 12 + m;
    row1 = component - 3 * m;
    width = 21;
    component_count = C * width;
    moment_base = Angular3Base;
    output = density_angular3_cache;
  } else if (task < Rho0DotTaskBase) {
    component = task - Angular4TaskBase;
    const int m = component / 3;
    row0 = 19 + m;
    row1 = component - 3 * m;
    width = 27;
    component_count = C * width;
    moment_base = Angular4Base;
    output = density_angular4_cache;
  } else if (task < Raw1DotTaskBase) {
    component = task - Rho0DotTaskBase;
    row0 = component;
    row1 = 6;
    width = 3;
    component_count = C * width;
    moment_base = Rho0DotBase;
    output = density_rho0_dot_cache;
  } else if (active_task) {
    component = task - Raw1DotTaskBase;
    const int a = component / 3;
    row0 = 3 + a;
    row1 = component - 3 * a;
    row2 = 6;
    width = 9;
    component_count = C * width;
    moment_base = Raw1DotBase;
    output = density_raw1_dot_cache;
  }
  if (task >= Raw1TaskBase && task < GeomTaskBase &&
      LMax < 1 && !Chiral) {
    active_task = false;
  }
  if (task >= PolarTaskBase && task < Angular2TaskBase && !Chiral) {
    active_task = false;
  }
  if (task >= Angular2TaskBase && task < HexTaskBase && LMax < 2) {
    active_task = false;
  }
  if (task >= HexTaskBase && task < Angular3TaskBase && !Chiral) {
    active_task = false;
  }
  if (task >= Angular3TaskBase && task < Angular4TaskBase && LMax < 3) {
    active_task = false;
  }
  if (task >= Angular4TaskBase && task < Rho0DotTaskBase && LMax < 4) {
    active_task = false;
  }
  if (task >= Raw1DotTaskBase && LMax < 1 && !Chiral) {
    active_task = false;
  }

  float channel_acc[C] = {};
  const int count = nn_radial[atom];
  int chunk_count =
      (count + kSpinPrimitiveTileSlots - 1) / kSpinPrimitiveTileSlots;
  if (chunk_count == 0) {
    chunk_count = 1;
  }
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    const int slot_base = chunk * kSpinPrimitiveTileSlots;
    int tile_count = count - slot_base;
    if (tile_count > kSpinPrimitiveTileSlots) {
      tile_count = kSpinPrimitiveTileSlots;
    }
    if (tile_count < 0) {
      tile_count = 0;
    }
    if (lane < tile_count) {
      fill_spin_primitive_tape<
          C,
          LMax,
          Chiral,
          kSpinPrimitiveTileSlots>(
          atom,
          slot_base + lane,
          lane,
          atom_stride,
          num_types,
          spin_basis_size,
          spin_cutoff,
          box,
          types,
          positions_soa3,
          spins_soa3,
          nl_radial,
          descriptor_coefficients,
          spin_coefficient_offset,
          tape,
          weights);
    }
    __syncthreads();

    if (active_task) {
      for (int slot = 0; slot < tile_count; ++slot) {
        float value = tape[row0][slot] * tape[row1][slot];
        if (row2 >= 0) {
          value *= tape[row2][slot];
        }
        for (int c = 0; c < channel_count; ++c) {
          channel_acc[c] += weights[c][slot] * value;
        }
      }
    }

    if (chunk + 1 == chunk_count) {
      if (scalar_term >= 0) {
        for (int c = 0; c < C; ++c) {
          descriptors[
              atom + atom_stride * (struct_dim + 2 + scalar_term * C + c)] =
              channel_acc[c];
        }
      } else if (active_task) {
        for (int c = 0; c < channel_count; ++c) {
          const int cache_component = c * width + component;
          output[spin_atom_cache_index(
              atom_stride, component_count, atom, cache_component)] =
              channel_acc[c];
          if (moment_base >= 0) {
            density_components[moment_base + cache_component] =
                channel_acc[c];
          }
        }
      }
    }
    __syncthreads();
  }

  if (lane < C) {
    const int channel = lane;
    int offset = 2 + 4 * C;
    float value = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const float rho0 = density_components[Rho0Base + channel * 3 + k];
      value += rho0 * rho0;
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        value;
    offset += C;

    const float* raw1 = density_components + Raw1Base + channel * 9;
    if constexpr (LMax >= 1) {
      const float rdot = spin_raw1_tracef(raw1);
      descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
          rdot * rdot;
      offset += C;

      float l1_cross[3];
      spin_raw1_crossf(raw1, l1_cross);
      value = dot3f(l1_cross, l1_cross);
      descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
          value;
      offset += C;

      value = 0.0f;
      const float trace_third = rdot / 3.0f;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          const float stf =
              spin_raw1_stf_componentf(raw1, a, b, trace_third);
          value += stf * stf;
        }
      }
      descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
          value;
      offset += C;
    }

    constexpr int AngularBases[3] = {
        Angular2Base,
        Angular3Base,
        Angular4Base};
    constexpr int AngularWidths[3] = {15, 21, 27};
    for (int ell_index = 0; ell_index < LMax - 1; ++ell_index) {
      value = 0.0f;
      for (int k = 0; k < AngularWidths[ell_index]; ++k) {
        const float x = density_components[
            AngularBases[ell_index] + channel * AngularWidths[ell_index] + k];
        value += x * x;
      }
      descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
          value;
      offset += C;
    }

    value = 0.0f;
    const float* geom = density_components + GeomBase + channel * 6;
    value = si[0] * geom[0] * si[0] + si[1] * geom[3] * si[1] +
        si[2] * geom[5] * si[2] +
        2.0f * (si[0] * geom[1] * si[1] +
                si[0] * geom[2] * si[2] +
                si[1] * geom[4] * si[2]);
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        value;
    offset += C;

    value = 0.0f;
    for (int k = 0; k < 3; ++k) {
      value += density_components[Rho0Base + channel * 3 + k] *
               density_components[Rho0DotBase + channel * 3 + k];
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        value;
    offset += C;

    if constexpr (LMax >= 1) {
      value = 0.0f;
      for (int k = 0; k < 9; ++k) {
        value += density_components[Raw1Base + channel * 9 + k] *
                 density_components[Raw1DotBase + channel * 9 + k];
      }
      descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
          value;
    }
  }
}
