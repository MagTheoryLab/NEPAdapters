#pragma once

// Internal CUDA implementation fragment for spin_onsite.cu.
// Included inside nep_adapters::cuda_backend's anonymous namespace.
// Contains shared device math and spin descriptor construction kernels.

__device__ __forceinline__ int idx2(int c, int k, int width) {
  return c * width + k;
}
template <bool AtomMajor, int ComponentCount>
__device__ __forceinline__ int spin_component_cache_index(
    int atom_stride,
    int atom,
    int component) {
  static_assert(ComponentCount > 0, "spin cache component count must be positive");
  if constexpr (AtomMajor) {
    return atom * ComponentCount + component;
  }
  return atom + atom_stride * component;
}

template <bool AtomMajor>
__device__ __forceinline__ int spin_component_cache_index(
    int atom_stride,
    int component_count,
    int atom,
    int component) {
  if constexpr (AtomMajor) {
    return atom * component_count + component;
  }
  return atom + atom_stride * component;
}

template <bool AtomMajor>
__device__ __forceinline__ int spin_component_cache_stride(int atom_stride) {
  if constexpr (AtomMajor) {
    return 1;
  }
  return atom_stride;
}

__device__ __forceinline__ int tensor3(int a, int b, int c) {
  return (a * 3 + b) * 3 + c;
}

__device__ __forceinline__ int tensor4(int a, int b, int c, int d) {
  return ((a * 3 + b) * 3 + c) * 3 + d;
}

__device__ __forceinline__ void cross3(
    const double* a,
    const double* b,
    double* out) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ __forceinline__ double dot3(const double* a, const double* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ __forceinline__ void stf_outer3(
    const double* a,
    const double* b,
    double* out) {
  const double trace = dot3(a, b) / 3.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double value = 0.5 * (a[i] * b[j] + a[j] * b[i]);
      if (i == j) {
        value -= trace;
      }
      out[3 * i + j] = value;
    }
  }
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

__device__ __forceinline__ int spin_edge_cache_index(
    int atom_stride,
    int slot,
    int atom,
    int c) {
  return ((slot * atom_stride + atom) * 4) + c;
}

__device__ __forceinline__ void load_spin_edge_weight_cache(
    int atom_stride,
    int slot,
    int atom,
    const float* __restrict__ spin_edge_weights,
    double* weights) {
  for (int c = 0; c < 4; ++c) {
    weights[c] = static_cast<double>(
        spin_edge_weights[spin_edge_cache_index(atom_stride, slot, atom, c)]);
  }
}

__device__ __forceinline__ void load_spin_edge_weight_derivative_cache(
    int atom_stride,
    int slot,
    int atom,
    const float* __restrict__ spin_edge_weights,
    const float* __restrict__ spin_edge_weight_derivatives,
    double* weights,
    double* weight_derivatives) {
  for (int c = 0; c < 4; ++c) {
    const int index = spin_edge_cache_index(atom_stride, slot, atom, c);
    weights[c] = static_cast<double>(spin_edge_weights[index]);
    weight_derivatives[c] =
        static_cast<double>(spin_edge_weight_derivatives[index]);
  }
}

__device__ __forceinline__ int levi_civita(int a, int b, int c) {
  if (a == b || b == c || a == c) {
    return 0;
  }
  return ((a == 0 && b == 1 && c == 2) ||
          (a == 1 && b == 2 && c == 0) ||
          (a == 2 && b == 0 && c == 1)) ? 1 : -1;
}

__device__ __forceinline__ void find_fc_and_fcp(
    double rc,
    double rcinv,
    double r,
    double& fc,
    double& fcp) {
  if (r < rc) {
    const double x = r * rcinv;
    fc = 0.5 * cos(kPi * x) + 0.5;
    fcp = -0.5 * kPi * sin(kPi * x) * rcinv;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

__device__ void find_fn_and_fnp(
    int basis_size,
    double rcinv,
    double r,
    double fc,
    double fcp,
    double* fn,
    double* fnp) {
  fn[0] = fc;
  fnp[0] = fcp;
  if (basis_size == 0) {
    return;
  }
  const double r_scaled = r * rcinv;
  const double x = 2.0 * (r_scaled - 1.0) * (r_scaled - 1.0) - 1.0;
  fn[1] = 0.5 * (x + 1.0);
  fnp[1] = 2.0 * (r_scaled - 1.0) * rcinv * fc + fn[1] * fcp;
  fn[1] *= fc;
  double t0 = 1.0;
  double t1 = x;
  double u0 = 1.0;
  double u1 = 2.0 * x;
  for (int n = 2; n <= basis_size; ++n) {
    const double t2 = 2.0 * x * t1 - t0;
    fn[n] = 0.5 * (t2 + 1.0);
    fnp[n] = n * u1 * 2.0 * (r_scaled - 1.0) * rcinv;
    fnp[n] = fnp[n] * fc + fn[n] * fcp;
    fn[n] *= fc;
    const double u2 = 2.0 * x * u1 - u0;
    t0 = t1;
    t1 = t2;
    u0 = u1;
    u1 = u2;
  }
}

__device__ int real_spherical_harmonics_spin(
    const double* rhat,
    int ell,
    double* out) {
  const double x = rhat[0];
  const double y = rhat[1];
  const double z = rhat[2];
  if (ell == 2) {
    out[0] = sqrt(15.0 / (4.0 * kPi)) * x * y;
    out[1] = sqrt(15.0 / (4.0 * kPi)) * y * z;
    out[2] = sqrt(5.0 / (16.0 * kPi)) * (2.0 * z * z - x * x - y * y);
    out[3] = sqrt(15.0 / (4.0 * kPi)) * x * z;
    out[4] = sqrt(15.0 / (16.0 * kPi)) * (x * x - y * y);
    return 5;
  }
  if (ell == 3) {
    const double rho2 = x * x + y * y;
    out[0] = sqrt(35.0 / (32.0 * kPi)) * y * (3.0 * x * x - y * y);
    out[1] = sqrt(105.0 / (4.0 * kPi)) * x * y * z;
    out[2] = sqrt(21.0 / (32.0 * kPi)) * y * (4.0 * z * z - rho2);
    out[3] = sqrt(7.0 / (16.0 * kPi)) * z * (2.0 * z * z - 3.0 * rho2);
    out[4] = sqrt(21.0 / (32.0 * kPi)) * x * (4.0 * z * z - rho2);
    out[5] = sqrt(105.0 / (16.0 * kPi)) * z * (x * x - y * y);
    out[6] = sqrt(35.0 / (32.0 * kPi)) * x * (x * x - 3.0 * y * y);
    return 7;
  }
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  out[0] = 0.75 * sqrt(35.0 / kPi) * x * y * (x2 - y2);
  out[1] = 0.75 * sqrt(35.0 / (2.0 * kPi)) * y * z * (3.0 * x2 - y2);
  out[2] = 0.75 * sqrt(5.0 / kPi) * x * y * (7.0 * z2 - 1.0);
  out[3] = 0.75 * sqrt(5.0 / (2.0 * kPi)) * y * z * (7.0 * z2 - 3.0);
  out[4] = (3.0 / 16.0) * sqrt(1.0 / kPi) * (35.0 * z2 * z2 - 30.0 * z2 + 3.0);
  out[5] = 0.75 * sqrt(5.0 / (2.0 * kPi)) * x * z * (7.0 * z2 - 3.0);
  out[6] = 0.375 * sqrt(5.0 / kPi) * (x2 - y2) * (7.0 * z2 - 1.0);
  out[7] = 0.75 * sqrt(35.0 / (2.0 * kPi)) * x * z * (x2 - 3.0 * y2);
  out[8] = (3.0 / 16.0) * sqrt(35.0 / kPi) * (x2 * x2 - 6.0 * x2 * y2 + y2 * y2);
  return 9;
}

__device__ void add_real_spherical_harmonics_gradient(
    const double* r,
    int ell,
    const double* grad_y,
    double* grad_r) {
  const double x = r[0];
  const double y = r[1];
  const double z = r[2];
  if (ell == 2) {
    const double a = sqrt(15.0 / (4.0 * kPi));
    const double b = sqrt(5.0 / (16.0 * kPi));
    const double c = sqrt(15.0 / (16.0 * kPi));
    grad_r[0] += grad_y[0] * a * y - grad_y[2] * 2.0 * b * x +
                 grad_y[3] * a * z + grad_y[4] * 2.0 * c * x;
    grad_r[1] += grad_y[0] * a * x + grad_y[1] * a * z -
                 grad_y[2] * 2.0 * b * y - grad_y[4] * 2.0 * c * y;
    grad_r[2] += grad_y[1] * a * y + grad_y[2] * 4.0 * b * z +
                 grad_y[3] * a * x;
    return;
  }
  if (ell == 3) {
    const double x2 = x * x;
    const double y2 = y * y;
    const double z2 = z * z;
    const double rho2 = x2 + y2;
    const double a = sqrt(35.0 / (32.0 * kPi));
    const double b = sqrt(105.0 / (4.0 * kPi));
    const double c = sqrt(21.0 / (32.0 * kPi));
    const double d = sqrt(7.0 / (16.0 * kPi));
    const double e = sqrt(105.0 / (16.0 * kPi));
    grad_r[0] += grad_y[0] * 6.0 * a * x * y + grad_y[1] * b * y * z -
                 grad_y[2] * 2.0 * c * x * y - grad_y[3] * 6.0 * d * x * z +
                 grad_y[4] * c * (4.0 * z2 - 3.0 * x2 - y2) +
                 grad_y[5] * 2.0 * e * x * z + grad_y[6] * 3.0 * a * (x2 - y2);
    grad_r[1] += grad_y[0] * 3.0 * a * (x2 - y2) + grad_y[1] * b * x * z +
                 grad_y[2] * c * (4.0 * z2 - x2 - 3.0 * y2) -
                 grad_y[3] * 6.0 * d * y * z - grad_y[4] * 2.0 * c * x * y -
                 grad_y[5] * 2.0 * e * y * z - grad_y[6] * 6.0 * a * x * y;
    grad_r[2] += grad_y[1] * b * x * y + grad_y[2] * 8.0 * c * y * z +
                 grad_y[3] * d * (6.0 * z2 - 3.0 * rho2) +
                 grad_y[4] * 8.0 * c * x * z + grad_y[5] * e * (x2 - y2);
    return;
  }
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double a = 0.75 * sqrt(35.0 / kPi);
  const double b = 0.75 * sqrt(35.0 / (2.0 * kPi));
  const double c = 0.75 * sqrt(5.0 / kPi);
  const double d = 0.75 * sqrt(5.0 / (2.0 * kPi));
  const double e = (3.0 / 16.0) * sqrt(1.0 / kPi);
  const double f = 0.375 * sqrt(5.0 / kPi);
  const double g = (3.0 / 16.0) * sqrt(35.0 / kPi);
  grad_r[0] += grad_y[0] * a * y * (3.0 * x2 - y2) +
               grad_y[1] * b * 6.0 * x * y * z +
               grad_y[2] * c * y * (7.0 * z2 - 1.0) +
               grad_y[5] * d * z * (7.0 * z2 - 3.0) +
               grad_y[6] * 2.0 * f * x * (7.0 * z2 - 1.0) +
               grad_y[7] * b * z * (3.0 * x2 - 3.0 * y2) +
               grad_y[8] * g * (4.0 * x * x2 - 12.0 * x * y2);
  grad_r[1] += grad_y[0] * a * x * (x2 - 3.0 * y2) +
               grad_y[1] * b * z * (3.0 * x2 - 3.0 * y2) +
               grad_y[2] * c * x * (7.0 * z2 - 1.0) +
               grad_y[3] * d * z * (7.0 * z2 - 3.0) -
               grad_y[6] * 2.0 * f * y * (7.0 * z2 - 1.0) -
               grad_y[7] * b * 6.0 * x * y * z +
               grad_y[8] * g * (-12.0 * x2 * y + 4.0 * y * y2);
  grad_r[2] += grad_y[1] * b * y * (3.0 * x2 - y2) +
               grad_y[2] * c * 14.0 * x * y * z +
               grad_y[3] * d * y * (21.0 * z2 - 3.0) +
               grad_y[4] * e * (140.0 * z2 * z - 60.0 * z) +
               grad_y[5] * d * x * (21.0 * z2 - 3.0) +
               grad_y[6] * f * 14.0 * z * (x2 - y2) +
               grad_y[7] * b * x * (x2 - 3.0 * y2);
}

__device__ void fill_spin_monomials(const double* u, double* m3, double* m4) {
  const double x = u[0];
  const double y = u[1];
  const double z = u[2];
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double x3 = x2 * x;
  const double y3 = y2 * y;
  const double z3 = z2 * z;
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

__device__ __forceinline__ void fill_spin_chiral_reduced_moments(
    const double* u,
    double* o_reduced,
    double* h_reduced) {
  const double x = u[0];
  const double y = u[1];
  const double z = u[2];
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;

  o_reduced[0] = y * (y2 - 3.0 * z2);
  o_reduced[1] = z * (z2 - 3.0 * y2);
  o_reduced[2] = y * (x2 - z2);
  o_reduced[3] = z * (x2 - y2);
  o_reduced[4] = x * (y2 - z2);
  o_reduced[5] = xy * z;
  o_reduced[6] = x * (3.0 * z2 - x2);

  constexpr double OneSeventh = 1.0 / 7.0;
  h_reduced[0] = xz * (-x2 - z2 + 6.0 * y2) * OneSeventh;
  h_reduced[1] = xy * (x2 + y2 - 6.0 * z2) * OneSeventh;
  h_reduced[2] = xz * (4.0 * x2 - 3.0 * z2 - 3.0 * y2) * OneSeventh;
  h_reduced[3] = xy * (-4.0 * x2 + 3.0 * y2 + 3.0 * z2) * OneSeventh;
  h_reduced[4] = yz * (-2.0 * y2 - 2.0 * z2 + 12.0 * x2) * OneSeventh;
  h_reduced[5] =
      2.0 * (y2 - z2) * (y2 + z2 - 6.0 * x2) * OneSeventh;
  h_reduced[6] = yz * (4.0 * y2 - 3.0 * z2 - 3.0 * x2) * OneSeventh;
  h_reduced[7] = (y2 - x2) * (x2 + y2 - 6.0 * z2) * OneSeventh;
  h_reduced[8] =
      (4.0 * x2 * x2 + y2 * y2 + 2.0 * z2 * z2 - 9.0 * x2 * y2 -
       15.0 * x2 * z2 + 3.0 * y2 * z2) * OneSeventh;
}

__device__ __forceinline__ int spin_monomial_index(
    int degree,
    int nx,
    int ny,
    int nz) {
  if (degree == 2) {
    const int exp2[kSpinDeg2Count][3] = {
        {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
        {1, 1, 0}, {1, 0, 1}, {0, 1, 1}};
    for (int k = 0; k < kSpinDeg2Count; ++k) {
      if (exp2[k][0] == nx && exp2[k][1] == ny && exp2[k][2] == nz) {
        return k;
      }
    }
  } else if (degree == 3) {
    const int exp3[kSpinDeg3Count][3] = {
        {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {2, 1, 0}, {2, 0, 1},
        {1, 2, 0}, {0, 2, 1}, {1, 0, 2}, {0, 1, 2}, {1, 1, 1}};
    for (int k = 0; k < kSpinDeg3Count; ++k) {
      if (exp3[k][0] == nx && exp3[k][1] == ny && exp3[k][2] == nz) {
        return k;
      }
    }
  } else {
    const int exp4[kSpinDeg4Count][3] = {
        {4, 0, 0}, {0, 4, 0}, {0, 0, 4}, {3, 1, 0}, {3, 0, 1},
        {1, 3, 0}, {0, 3, 1}, {1, 0, 3}, {0, 1, 3}, {2, 2, 0},
        {2, 0, 2}, {0, 2, 2}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}};
    for (int k = 0; k < kSpinDeg4Count; ++k) {
      if (exp4[k][0] == nx && exp4[k][1] == ny && exp4[k][2] == nz) {
        return k;
      }
    }
  }
  return 0;
}

__device__ double packed_value(const double* packed, int degree, const int* counts) {
  return packed[spin_monomial_index(degree, counts[0], counts[1], counts[2])];
}

__device__ void fill_spin_monomials2(const double* u, double* m2) {
  const double x = u[0];
  const double y = u[1];
  const double z = u[2];
  m2[0] = x * x;
  m2[1] = y * y;
  m2[2] = z * z;
  m2[3] = x * y;
  m2[4] = x * z;
  m2[5] = y * z;
}

__device__ double dot_spin_terms(const double* lhs, const double* rhs, int count) {
  double out = 0.0;
  for (int k = 0; k < count; ++k) {
    out += lhs[k] * rhs[k];
  }
  return out;
}

__device__ void unpack_rank3_spin_stf(const double* raw, double* out) {
  double trace[3] = {0.0, 0.0, 0.0};
  for (int c = 0; c < 3; ++c) {
    for (int e = 0; e < 3; ++e) {
      int counts[3] = {0, 0, 0};
      counts[e] += 2;
      ++counts[c];
      trace[c] += packed_value(raw, 3, counts);
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        int counts[3] = {0, 0, 0};
        ++counts[a];
        ++counts[b];
        ++counts[c];
        const double traced =
            ((a == b) ? trace[c] : 0.0) +
            ((a == c) ? trace[b] : 0.0) +
            ((b == c) ? trace[a] : 0.0);
        out[tensor3(a, b, c)] = packed_value(raw, 3, counts) - traced / 5.0;
      }
    }
  }
}

__device__ void unpack_rank4_spin_stf(const double* raw, double* out) {
  double trace[9] = {0.0};
  for (int c = 0; c < 3; ++c) {
    for (int d = 0; d < 3; ++d) {
      for (int e = 0; e < 3; ++e) {
        int counts[3] = {0, 0, 0};
        counts[e] += 2;
        ++counts[c];
        ++counts[d];
        trace[3 * c + d] += packed_value(raw, 4, counts);
      }
    }
  }
  double double_trace = 0.0;
  for (int e = 0; e < 3; ++e) {
    double_trace += trace[3 * e + e];
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          int counts[3] = {0, 0, 0};
          ++counts[a];
          ++counts[b];
          ++counts[c];
          ++counts[d];
          const double six =
              ((a == b) ? trace[3 * c + d] : 0.0) +
              ((a == c) ? trace[3 * b + d] : 0.0) +
              ((a == d) ? trace[3 * b + c] : 0.0) +
              ((b == c) ? trace[3 * a + d] : 0.0) +
              ((b == d) ? trace[3 * a + c] : 0.0) +
              ((c == d) ? trace[3 * a + b] : 0.0);
          const double three =
              ((a == b && c == d) ? double_trace : 0.0) +
              ((a == c && b == d) ? double_trace : 0.0) +
              ((a == d && b == c) ? double_trace : 0.0);
          out[tensor4(a, b, c, d)] =
              packed_value(raw, 4, counts) - six / 7.0 + three / 35.0;
        }
      }
    }
  }
}

__device__ void project_rank2_spin_gradient(const double* grad, double* terms) {
  const double trace = (grad[0] + grad[4] + grad[8]) / 3.0;
  terms[0] = grad[0] - trace;
  terms[1] = grad[4] - trace;
  terms[2] = grad[8] - trace;
  terms[3] = grad[1] + grad[3];
  terms[4] = grad[2] + grad[6];
  terms[5] = grad[5] + grad[7];
}

__device__ void fill_spin_term_derivatives(
    int degree,
    int count,
    const double* terms,
    double* derivatives) {
  const int lower_count = degree == 3 ? kSpinDeg2Count : kSpinDeg3Count;
  for (int k = 0; k < 3 * lower_count; ++k) {
    derivatives[k] = 0.0;
  }
  const int exp3[kSpinDeg3Count][3] = {
      {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {2, 1, 0}, {2, 0, 1},
      {1, 2, 0}, {0, 2, 1}, {1, 0, 2}, {0, 1, 2}, {1, 1, 1}};
  const int exp4[kSpinDeg4Count][3] = {
      {4, 0, 0}, {0, 4, 0}, {0, 0, 4}, {3, 1, 0}, {3, 0, 1},
      {1, 3, 0}, {0, 3, 1}, {1, 0, 3}, {0, 1, 3}, {2, 2, 0},
      {2, 0, 2}, {0, 2, 2}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}};
  for (int k = 0; k < count; ++k) {
    const int* exps = degree == 3 ? exp3[k] : exp4[k];
    for (int axis = 0; axis < 3; ++axis) {
      const int power = exps[axis];
      if (power == 0) {
        continue;
      }
      int lower[3] = {exps[0], exps[1], exps[2]};
      --lower[axis];
      const int lower_index =
          spin_monomial_index(degree - 1, lower[0], lower[1], lower[2]);
      derivatives[axis * lower_count + lower_index] += power * terms[k];
    }
  }
}

__device__ void add_stf_outer_gradient(
    const double* grad,
    const double* a,
    const double* b,
    double* grad_a,
    double* grad_b) {
  const double trace_grad = (grad[0] + grad[4] + grad[8]) / 3.0;
  for (int p = 0; p < 3; ++p) {
    double ga = -trace_grad * b[p];
    double gb = -trace_grad * a[p];
    for (int q = 0; q < 3; ++q) {
      ga += 0.5 * (grad[3 * p + q] + grad[3 * q + p]) * b[q];
      gb += 0.5 * (grad[3 * p + q] + grad[3 * q + p]) * a[q];
    }
    grad_a[p] += ga;
    grad_b[p] += gb;
  }
}

__device__ void add_density(
    double* density,
    int c_count,
    int width,
    const double* values,
    const double* weights,
    double scale = 1.0) {
  for (int c = 0; c < c_count; ++c) {
    const double weight = scale * weights[c];
    for (int k = 0; k < width; ++k) {
      density[c * width + k] += weight * values[k];
    }
  }
}

__device__ void load_spin_edge(
    int atom,
    int neighbor,
    int atom_stride,
    SimulationBox box,
    const double* positions_soa3,
    const double* spins_soa3,
    double* rhat,
    double& dist,
    double* si,
    double* sj) {
  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
  minimum_image_delta(
      box,
      positions_soa3[neighbor] - positions_soa3[atom],
      positions_soa3[atom_stride + neighbor] - positions_soa3[atom_stride + atom],
      positions_soa3[2 * atom_stride + neighbor] - positions_soa3[2 * atom_stride + atom],
      dx,
      dy,
      dz);
  dist = sqrt(static_cast<double>(dx) * dx + static_cast<double>(dy) * dy +
              static_cast<double>(dz) * dz);
  if (dist > 0.0) {
    rhat[0] = dx / dist;
    rhat[1] = dy / dist;
    rhat[2] = dz / dist;
  } else {
    rhat[0] = 0.0;
    rhat[1] = 0.0;
    rhat[2] = 0.0;
  }
  si[0] = spins_soa3[atom];
  si[1] = spins_soa3[atom_stride + atom];
  si[2] = spins_soa3[2 * atom_stride + atom];
  sj[0] = spins_soa3[neighbor];
  sj[1] = spins_soa3[atom_stride + neighbor];
  sj[2] = spins_soa3[2 * atom_stride + neighbor];
}

__device__ void load_spin_edge_cached(
    int atom,
    int neighbor,
    int atom_stride,
    int slot,
    const double* spins_soa3,
    const float* spin_edge_dx,
    const float* spin_edge_dy,
    const float* spin_edge_dz,
    const float* spin_edge_dist,
    double* rhat,
    double& dist,
    double* si,
    double* sj) {
  const int offset = atom + atom_stride * slot;
  const double dx = static_cast<double>(spin_edge_dx[offset]);
  const double dy = static_cast<double>(spin_edge_dy[offset]);
  const double dz = static_cast<double>(spin_edge_dz[offset]);
  dist = static_cast<double>(spin_edge_dist[offset]);
  if (dist > 0.0) {
    rhat[0] = dx / dist;
    rhat[1] = dy / dist;
    rhat[2] = dz / dist;
  } else {
    rhat[0] = 0.0;
    rhat[1] = 0.0;
    rhat[2] = 0.0;
  }
  si[0] = spins_soa3[atom];
  si[1] = spins_soa3[atom_stride + atom];
  si[2] = spins_soa3[2 * atom_stride + atom];
  sj[0] = spins_soa3[neighbor];
  sj[1] = spins_soa3[atom_stride + neighbor];
  sj[2] = spins_soa3[2 * atom_stride + neighbor];
}


// Descriptor construction and primitive-cache kernels.

__global__ void precompute_spin_edge_weights_c4(
    int atom_count,
    int atom_stride,
    int radial_capacity,
    int num_types,
    float spin_cutoff,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ descriptor_coefficients,
    int spin_coefficient_offset,
    float* __restrict__ spin_edge_dx,
    float* __restrict__ spin_edge_dy,
    float* __restrict__ spin_edge_dz,
    float* __restrict__ spin_edge_dist,
    float* __restrict__ spin_edge_weights,
    float* __restrict__ spin_edge_weight_derivatives) {
  constexpr int C = 4;
  constexpr int BasisCount = 4;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = atom_count * radial_capacity;
  if (index >= total) {
    return;
  }
  const int atom = index % atom_count;
  const int slot = index / atom_count;
  if (slot >= nn_radial[atom]) {
    return;
  }

  const int neighbor = nl_radial[atom + atom_stride * slot];
  float dx = 0.0f;
  float dy = 0.0f;
  float dz = 0.0f;
  minimum_image_delta(
      box,
      positions_soa3[neighbor] - positions_soa3[atom],
      positions_soa3[atom_stride + neighbor] - positions_soa3[atom_stride + atom],
      positions_soa3[2 * atom_stride + neighbor] - positions_soa3[2 * atom_stride + atom],
      dx,
      dy,
      dz);
  const float dist = sqrtf(dx * dx + dy * dy + dz * dz);
  const int edge_offset = atom + atom_stride * slot;
  spin_edge_dx[edge_offset] = dx;
  spin_edge_dy[edge_offset] = dy;
  spin_edge_dz[edge_offset] = dz;
  spin_edge_dist[edge_offset] = dist;
  if (dist <= 1.0e-12f || dist >= spin_cutoff) {
    return;
  }

  constexpr float Pi = 3.14159265358979323846f;
  const float rcinv = 1.0f / spin_cutoff;
  const float r_scaled = dist * rcinv;
  const float cutoff_phase = Pi * r_scaled;
  const float fc = 0.5f * cosf(cutoff_phase) + 0.5f;
  const float fcp = -0.5f * Pi * sinf(cutoff_phase) * rcinv;
  const float shifted = r_scaled - 1.0f;
  const float x = 2.0f * shifted * shifted - 1.0f;
  const float radial_derivative = 2.0f * shifted * rcinv;
  const float t2 = 2.0f * x * x - 1.0f;
  const float t3 = 2.0f * x * t2 - x;
  const float fn_raw[BasisCount] = {
      1.0f,
      0.5f * (x + 1.0f),
      0.5f * (t2 + 1.0f),
      0.5f * (t3 + 1.0f)};
  const float u1 = 2.0f * x;
  const float u2 = 2.0f * x * u1 - 1.0f;
  const float fn[BasisCount] = {
      fc,
      fn_raw[1] * fc,
      fn_raw[2] * fc,
      fn_raw[3] * fc};
  const float fnp[BasisCount] = {
      fcp,
      radial_derivative * fc + fn_raw[1] * fcp,
      2.0f * u1 * radial_derivative * fc + fn_raw[2] * fcp,
      3.0f * u2 * radial_derivative * fc + fn_raw[3] * fcp};

  const int type_pair = types[atom] * num_types + types[neighbor];
  for (int c = 0; c < C; ++c) {
    float weight = 0.0f;
    float derivative = 0.0f;
    #pragma unroll
    for (int k = 0; k < BasisCount; ++k) {
      const int coefficient_index = spin_coefficient_offset +
          ((c * BasisCount + k) * num_types * num_types + type_pair);
      const float coefficient = descriptor_coefficients[coefficient_index];
      weight += fn[k] * coefficient;
      derivative += fnp[k] * coefficient;
    }
    const int cache_index = spin_edge_cache_index(atom_stride, slot, atom, c);
    spin_edge_weights[cache_index] = weight;
    spin_edge_weight_derivatives[cache_index] = derivative;
  }
}

__global__ void build_spin_descriptors(
    int atom_count,
    int atom_stride,
    int struct_dim,
    int spin_dim,
    int num_types,
    int spin_compress,
    int spin_basis_size,
    int spin_l_max,
    int spin_chiral,
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
    float* __restrict__ density_l1_rdot_cache,
    float* __restrict__ density_l1_cross_cache,
    float* __restrict__ density_l1_stf_cache,
    float* __restrict__ density_angular2_cache,
    float* __restrict__ density_angular3_cache,
    float* __restrict__ density_angular4_cache,
    float* __restrict__ density_geom_cache,
    float* __restrict__ density_rho0_dot_cache,
    float* __restrict__ density_raw1_dot_cache,
    float* __restrict__ chiral_polar_cache,
    float* __restrict__ chiral_octupoles_raw_cache,
    float* __restrict__ chiral_hexadecapoles_raw_cache,
    float* __restrict__ chiral_chirals_cache,
    float* __restrict__ chiral_pseudodevs_cache,
    float* __restrict__ descriptors) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const double sx = spins_soa3[atom];
  const double sy = spins_soa3[atom_stride + atom];
  const double sz = spins_soa3[2 * atom_stride + atom];
  const double spin_i[3] = {sx, sy, sz};
  const double s2 = sx * sx + sy * sy + sz * sz;
  descriptors[atom + atom_stride * struct_dim] = static_cast<float>(s2);
  descriptors[atom + atom_stride * (struct_dim + 1)] =
      static_cast<float>(s2 * s2);

  double scalar_q[4 * kMaxSpinCompress] = {};
  double rho0[kMaxSpinCompress * 3] = {};
  double raw1[kMaxSpinCompress * 9] = {};
  double l1_rdot[kMaxSpinCompress] = {};
  double l1_cross[kMaxSpinCompress * 3] = {};
  double l1_stf[kMaxSpinCompress * 9] = {};
  double angular2[kMaxSpinCompress * 15] = {};
  double angular3[kMaxSpinCompress * 21] = {};
  double angular4[kMaxSpinCompress * 27] = {};
  double geom[kMaxSpinCompress * 9] = {};
  double rho0_dot[kMaxSpinCompress * 3] = {};
  double raw1_dot[kMaxSpinCompress * 9] = {};
  double polars[kMaxSpinCompress * 3] = {};
  double octupoles_raw[2 * kSpinDeg3Count] = {};
  double hexadecapoles_raw[2 * kSpinDeg4Count] = {};
  double octupoles[2 * 27] = {};
  double hexadecapoles[2 * 81] = {};
  double chirals[2] = {};
  double pseudodevs[kMaxSpinCompress * 9] = {};

  const int type1 = types[atom];
  const int type_pair_base = type1 * num_types;
  const double rcinv = 1.0 / static_cast<double>(spin_cutoff);
  const int basis_count = spin_basis_size + 1;
  const int radial_count = nn_radial[atom];
  const int chi_c = spin_compress < 2 ? spin_compress : 2;

  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    double rhat[3];
    double dist = 0.0;
    double si[3];
    double sj[3];
    load_spin_edge(atom, neighbor, atom_stride, box, positions_soa3, spins_soa3,
                   rhat, dist, si, sj);
    if (dist <= 1.0e-12 || dist >= spin_cutoff) {
      continue;
    }

    const int type_pair = type_pair_base + types[neighbor];
    double fc = 0.0;
    double fcp = 0.0;
    double fn[kMaxSpinBasis] = {};
    double fnp[kMaxSpinBasis] = {};
    find_fc_and_fcp(spin_cutoff, rcinv, dist, fc, fcp);
    find_fn_and_fnp(spin_basis_size, rcinv, dist, fc, fcp, fn, fnp);
    double weights[kMaxSpinCompress] = {};
    for (int c = 0; c < spin_compress; ++c) {
      double w = 0.0;
      for (int k = 0; k < basis_count; ++k) {
        const int index = spin_coefficient_offset +
            ((c * basis_count + k) * num_types * num_types + type_pair);
        w += fn[k] * static_cast<double>(descriptor_coefficients[index]);
      }
      weights[c] = w;
    }

    const double dot = dot3(si, sj);
    const double sj2 = dot3(sj, sj);
    const double ri_dot_si = dot3(rhat, si);
    const double ri_dot_sj = dot3(rhat, sj);
    const double bond_axis = ri_dot_si * ri_dot_sj;

    int offset = 2;
    const double scalars[4] = {dot, dot * dot, sj2, bond_axis};
    for (int term = 0; term < 4; ++term) {
      for (int c = 0; c < spin_compress; ++c) {
        scalar_q[term * kMaxSpinCompress + c] += weights[c] * scalars[term];
      }
      offset += spin_compress;
    }

    add_density(rho0, spin_compress, 3, sj, weights);
    double raw_value[9];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        raw_value[3 * a + b] = rhat[a] * sj[b];
      }
    }
    add_density(raw1, spin_compress, 9, raw_value, weights);

    if (spin_l_max >= 1) {
      const double rdot = ri_dot_sj;
      add_density(l1_rdot, spin_compress, 1, &rdot, weights);
      double cross_value[3];
      cross3(rhat, sj, cross_value);
      add_density(l1_cross, spin_compress, 3, cross_value, weights);
      double stf[9];
      stf_outer3(rhat, sj, stf);
      add_density(l1_stf, spin_compress, 9, stf, weights);
    }
    for (int ell = 2; ell <= spin_l_max; ++ell) {
      double ylm[9];
      const int ylm_width = real_spherical_harmonics_spin(rhat, ell, ylm);
      double values[27];
      int width = 0;
      for (int m = 0; m < ylm_width; ++m) {
        values[width++] = ylm[m] * sj[0];
        values[width++] = ylm[m] * sj[1];
        values[width++] = ylm[m] * sj[2];
      }
      add_density(
          ell == 2 ? angular2 : ell == 3 ? angular3 : angular4,
          spin_compress,
          width,
          values,
          weights);
    }
    double rr[9];
    stf_outer3(rhat, rhat, rr);
    add_density(geom, spin_compress, 9, rr, weights);
    if (spin_chiral) {
      add_density(polars, spin_compress, 3, rhat, weights);
      double m3[kSpinDeg3Count];
      double m4[kSpinDeg4Count];
      fill_spin_monomials(rhat, m3, m4);
      add_density(octupoles_raw, chi_c, kSpinDeg3Count, m3, weights);
      add_density(hexadecapoles_raw, chi_c, kSpinDeg4Count, m4, weights);
    }
    add_density(rho0_dot, spin_compress, 3, sj, weights, dot);
    add_density(raw1_dot, spin_compress, 9, raw_value, weights, dot);
  }

  for (int c = 0; c < spin_compress; ++c) {
    for (int k = 0; k < 3; ++k) {
      density_rho0_cache[atom + atom_stride * (c * 3 + k)] =
          rho0[idx2(c, k, 3)];
      density_rho0_dot_cache[atom + atom_stride * (c * 3 + k)] =
          rho0_dot[idx2(c, k, 3)];
    }
    for (int k = 0; k < 9; ++k) {
      density_raw1_cache[atom + atom_stride * (c * 9 + k)] =
          raw1[idx2(c, k, 9)];
      density_l1_stf_cache[atom + atom_stride * (c * 9 + k)] =
          l1_stf[idx2(c, k, 9)];
      density_geom_cache[atom + atom_stride * (c * 9 + k)] =
          geom[idx2(c, k, 9)];
      density_raw1_dot_cache[atom + atom_stride * (c * 9 + k)] =
          raw1_dot[idx2(c, k, 9)];
    }
    density_l1_rdot_cache[atom + atom_stride * c] = l1_rdot[c];
    for (int k = 0; k < 3; ++k) {
      density_l1_cross_cache[atom + atom_stride * (c * 3 + k)] =
          l1_cross[idx2(c, k, 3)];
    }
    if (density_angular2_cache != nullptr) {
      for (int k = 0; k < 15; ++k) {
        density_angular2_cache[atom + atom_stride * (c * 15 + k)] =
            angular2[idx2(c, k, 15)];
      }
    }
    if (density_angular3_cache != nullptr) {
      for (int k = 0; k < 21; ++k) {
        density_angular3_cache[atom + atom_stride * (c * 21 + k)] =
            angular3[idx2(c, k, 21)];
      }
    }
    if (density_angular4_cache != nullptr) {
      for (int k = 0; k < 27; ++k) {
        density_angular4_cache[atom + atom_stride * (c * 27 + k)] =
            angular4[idx2(c, k, 27)];
      }
    }
  }

  for (int term = 0; term < 4; ++term) {
    for (int c = 0; c < spin_compress; ++c) {
      descriptors[
          atom + atom_stride *
              (struct_dim + 2 + term * spin_compress + c)] =
          static_cast<float>(scalar_q[term * kMaxSpinCompress + c]);
    }
  }

  int offset = 2 + 4 * spin_compress;
  for (int c = 0; c < spin_compress; ++c) {
    double value = 0.0;
    for (int k = 0; k < 3; ++k) {
      value += rho0[idx2(c, k, 3)] * rho0[idx2(c, k, 3)];
    }
    descriptors[atom + atom_stride * (struct_dim + offset + c)] =
        static_cast<float>(value);
  }
  offset += spin_compress;
  if (spin_l_max >= 1) {
    for (int c = 0; c < spin_compress; ++c) {
      descriptors[atom + atom_stride * (struct_dim + offset + c)] =
          static_cast<float>(l1_rdot[c] * l1_rdot[c]);
    }
    offset += spin_compress;
    for (int c = 0; c < spin_compress; ++c) {
      double value = 0.0;
      for (int k = 0; k < 3; ++k) {
        value += l1_cross[idx2(c, k, 3)] * l1_cross[idx2(c, k, 3)];
      }
      descriptors[atom + atom_stride * (struct_dim + offset + c)] =
          static_cast<float>(value);
    }
    offset += spin_compress;
    for (int c = 0; c < spin_compress; ++c) {
      double value = 0.0;
      for (int k = 0; k < 9; ++k) {
        value += l1_stf[idx2(c, k, 9)] * l1_stf[idx2(c, k, 9)];
      }
      descriptors[atom + atom_stride * (struct_dim + offset + c)] =
          static_cast<float>(value);
    }
    offset += spin_compress;
  }
  for (int ell = 2; ell <= spin_l_max; ++ell) {
    const int width = (2 * ell + 1) * 3;
    const double* angular = ell == 2 ? angular2 : ell == 3 ? angular3 : angular4;
    for (int c = 0; c < spin_compress; ++c) {
      double value = 0.0;
      for (int k = 0; k < width; ++k) {
        value += angular[idx2(c, k, width)] * angular[idx2(c, k, width)];
      }
      descriptors[atom + atom_stride * (struct_dim + offset + c)] =
          static_cast<float>(value);
    }
    offset += spin_compress;
  }
  for (int c = 0; c < spin_compress; ++c) {
    double value = 0.0;
    const double* g = geom + c * 9;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        value += spin_i[a] * g[3 * a + b] * spin_i[b];
      }
    }
    descriptors[atom + atom_stride * (struct_dim + offset + c)] =
        static_cast<float>(value);
  }
  offset += spin_compress;
  for (int c = 0; c < spin_compress; ++c) {
    double value = 0.0;
    for (int k = 0; k < 3; ++k) {
      value += rho0[idx2(c, k, 3)] * rho0_dot[idx2(c, k, 3)];
    }
    descriptors[atom + atom_stride * (struct_dim + offset + c)] =
        static_cast<float>(value);
  }
  offset += spin_compress;
  if (spin_l_max >= 1) {
    for (int c = 0; c < spin_compress; ++c) {
      double value = 0.0;
      for (int k = 0; k < 9; ++k) {
        value += raw1[idx2(c, k, 9)] * raw1_dot[idx2(c, k, 9)];
      }
      descriptors[atom + atom_stride * (struct_dim + offset + c)] =
          static_cast<float>(value);
    }
    offset += spin_compress;
  }

  if (spin_chiral) {
    double chiral_q[2 + 2 * kMaxSpinCompress] = {};
    const int chiral_base_offset = offset;
    for (int c = 0; c < chi_c; ++c) {
      unpack_rank3_spin_stf(octupoles_raw + c * kSpinDeg3Count, octupoles + c * 27);
      unpack_rank4_spin_stf(hexadecapoles_raw + c * kSpinDeg4Count, hexadecapoles + c * 81);
      const double* qmat = geom + c * 9;
      const double* oct = octupoles + c * 27;
      const double* hex = hexadecapoles + c * 81;
      double value = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          for (int cc = 0; cc < 3; ++cc) {
            const int eps = levi_civita(a, b, cc);
            if (eps == 0) {
              continue;
            }
            for (int d = 0; d < 3; ++d) {
              for (int e = 0; e < 3; ++e) {
                for (int f = 0; f < 3; ++f) {
                  value += eps * qmat[3 * a + d] * oct[tensor3(b, e, f)] *
                           hex[tensor4(cc, d, e, f)];
                }
              }
            }
          }
        }
      }
      chirals[c] = value;
    }

    for (int slot = 0; slot < radial_count; ++slot) {
      const int neighbor = nl_radial[atom + atom_stride * slot];
      double rhat[3];
      double dist = 0.0;
      double si[3];
      double sj[3];
      load_spin_edge(atom, neighbor, atom_stride, box, positions_soa3, spins_soa3,
                     rhat, dist, si, sj);
      if (dist <= 1.0e-12 || dist >= spin_cutoff) {
        continue;
      }
      const int type_pair = type_pair_base + types[neighbor];
      double fc = 0.0;
      double fcp = 0.0;
      double fn[kMaxSpinBasis] = {};
      double fnp[kMaxSpinBasis] = {};
      find_fc_and_fcp(spin_cutoff, rcinv, dist, fc, fcp);
      find_fn_and_fnp(spin_basis_size, rcinv, dist, fc, fcp, fn, fnp);
      double weights[kMaxSpinCompress] = {};
      for (int c = 0; c < spin_compress; ++c) {
        for (int k = 0; k < basis_count; ++k) {
          const int index = spin_coefficient_offset +
              ((c * basis_count + k) * num_types * num_types + type_pair);
          weights[c] += fn[k] * static_cast<double>(descriptor_coefficients[index]);
        }
      }
      for (int c = 0; c < spin_compress; ++c) {
        const double* qmat = geom + c * 9;
        double qu[3] = {0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            qu[a] += qmat[3 * a + b] * rhat[b];
          }
        }
        double axis[3];
        cross3(rhat, qu, axis);
        double pseudo[9];
        stf_outer3(axis, rhat, pseudo);
        for (int k = 0; k < 9; ++k) {
          pseudodevs[c * 9 + k] += weights[c] * pseudo[k];
        }
      }
    }

    for (int c = 0; c < spin_compress; ++c) {
      for (int k = 0; k < 3; ++k) {
        chiral_polar_cache[atom + atom_stride * (c * 3 + k)] =
            static_cast<float>(polars[idx2(c, k, 3)]);
      }
      for (int k = 0; k < 9; ++k) {
        chiral_pseudodevs_cache[atom + atom_stride * (c * 9 + k)] =
            static_cast<float>(pseudodevs[idx2(c, k, 9)]);
      }
    }
    for (int c = 0; c < chi_c; ++c) {
      for (int k = 0; k < kSpinDeg3Count; ++k) {
        chiral_octupoles_raw_cache[atom + atom_stride * (c * kSpinDeg3Count + k)] =
            static_cast<float>(octupoles_raw[c * kSpinDeg3Count + k]);
      }
      for (int k = 0; k < kSpinDeg4Count; ++k) {
        chiral_hexadecapoles_raw_cache[
            atom + atom_stride * (c * kSpinDeg4Count + k)] =
            static_cast<float>(hexadecapoles_raw[c * kSpinDeg4Count + k]);
      }
      chiral_chirals_cache[atom + atom_stride * c] =
          static_cast<float>(chirals[c]);
    }

    for (int slot = 0; slot < radial_count; ++slot) {
      const int neighbor = nl_radial[atom + atom_stride * slot];
      double rhat[3];
      double dist = 0.0;
      double si[3];
      double sj[3];
      load_spin_edge(atom, neighbor, atom_stride, box, positions_soa3, spins_soa3,
                     rhat, dist, si, sj);
      if (dist <= 1.0e-12 || dist >= spin_cutoff) {
        continue;
      }
      const int type_pair = type_pair_base + types[neighbor];
      double fc = 0.0;
      double fcp = 0.0;
      double fn[kMaxSpinBasis] = {};
      double fnp[kMaxSpinBasis] = {};
      find_fc_and_fcp(spin_cutoff, rcinv, dist, fc, fcp);
      find_fn_and_fnp(spin_basis_size, rcinv, dist, fc, fcp, fn, fnp);
      double weights[kMaxSpinCompress] = {};
      for (int c = 0; c < spin_compress; ++c) {
        for (int k = 0; k < basis_count; ++k) {
          const int index = spin_coefficient_offset +
              ((c * basis_count + k) * num_types * num_types + type_pair);
          weights[c] += fn[k] * static_cast<double>(descriptor_coefficients[index]);
        }
      }
      double spin_cross[3];
      cross3(si, sj, spin_cross);
      for (int c = 0; c < chi_c; ++c) {
        chiral_q[c] += weights[c] * dot3(spin_cross, rhat) * chirals[c];
      }
      int chiral_offset = chi_c;
      for (int c = 0; c < spin_compress; ++c) {
        const double* polar = polars + c * 3;
        double axis[3];
        cross3(polar, rhat, axis);
        chiral_q[chiral_offset + c] += weights[c] * dot3(spin_cross, axis);
      }
      chiral_offset += spin_compress;
      for (int c = 0; c < spin_compress; ++c) {
        const double* p = pseudodevs + c * 9;
        double axis[3] = {0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            axis[a] += p[3 * a + b] * rhat[b];
          }
        }
        chiral_q[chiral_offset + c] += weights[c] * dot3(spin_cross, axis);
      }
    }
    const int chiral_dim = chi_c + 2 * spin_compress;
    for (int d = 0; d < chiral_dim; ++d) {
      descriptors[atom + atom_stride * (struct_dim + chiral_base_offset + d)] =
          static_cast<float>(chiral_q[d]);
    }
  }
}

__global__ void build_spin_descriptors_c4_l4_basic(
    int atom_count,
    int atom_stride,
    int struct_dim,
    float spin_cutoff,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ spin_edge_dx,
    const float* __restrict__ spin_edge_dy,
    const float* __restrict__ spin_edge_dz,
    const float* __restrict__ spin_edge_dist,
    const float* __restrict__ spin_edge_weights,
    float* __restrict__ density_rho0_cache,
    float* __restrict__ density_raw1_cache,
    float* __restrict__ density_l1_rdot_cache,
    float* __restrict__ density_l1_cross_cache,
    float* __restrict__ density_l1_stf_cache,
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
  constexpr int C = 4;
  constexpr int ChiC = 2;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int atom = tid / C;
  const int c = tid - atom * C;
  if (atom >= atom_count) {
    return;
  }

  const double sx = spins_soa3[atom];
  const double sy = spins_soa3[atom_stride + atom];
  const double sz = spins_soa3[2 * atom_stride + atom];
  const double spin_i[3] = {sx, sy, sz};
  if (c == 0) {
    const double s2 = sx * sx + sy * sy + sz * sz;
    descriptors[atom + atom_stride * struct_dim] = static_cast<float>(s2);
    descriptors[atom + atom_stride * (struct_dim + 1)] =
        static_cast<float>(s2 * s2);
  }

  double scalar_q[4] = {};
  double rho0[3] = {};
  double raw1[9] = {};
  double l1_rdot = 0.0;
  double l1_cross[3] = {};
  double l1_stf[9] = {};
  double angular2[15] = {};
  double angular3[21] = {};
  double angular4[27] = {};
  double geom[9] = {};
  double rho0_dot[3] = {};
  double raw1_dot[9] = {};
  double polar[3] = {};
  double octupoles_reduced[kSpinChiralOReducedCount] = {};
  double hexadecapoles_reduced[kSpinChiralHReducedCount] = {};

  const int radial_count = nn_radial[atom];
  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    double rhat[3];
    double dist = 0.0;
    double si[3];
    double sj[3];
    load_spin_edge_cached(
        atom,
        neighbor,
        atom_stride,
        slot,
        spins_soa3,
        spin_edge_dx,
        spin_edge_dy,
        spin_edge_dz,
        spin_edge_dist,
        rhat,
        dist,
        si,
        sj);
    if (dist <= 1.0e-12 || dist >= spin_cutoff) {
      continue;
    }

    const double weight = static_cast<double>(
        spin_edge_weights[spin_edge_cache_index(atom_stride, slot, atom, c)]);
    const double dot = dot3(si, sj);
    const double sj2 = dot3(sj, sj);
    const double ri_dot_si = dot3(rhat, si);
    const double ri_dot_sj = dot3(rhat, sj);
    const double scalars[4] = {
        dot, dot * dot, sj2, ri_dot_si * ri_dot_sj};
    for (int term = 0; term < 4; ++term) {
      scalar_q[term] += weight * scalars[term];
    }
    for (int k = 0; k < 3; ++k) {
      rho0[k] += weight * sj[k];
      rho0_dot[k] += weight * dot * sj[k];
      polar[k] += weight * rhat[k];
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        const double value = rhat[a] * sj[b];
        raw1[3 * a + b] += weight * value;
        raw1_dot[3 * a + b] += weight * dot * value;
      }
    }
    l1_rdot += weight * ri_dot_sj;
    double cross_value[3];
    cross3(rhat, sj, cross_value);
    for (int k = 0; k < 3; ++k) {
      l1_cross[k] += weight * cross_value[k];
    }
    double stf[9];
    stf_outer3(rhat, sj, stf);
    for (int k = 0; k < 9; ++k) {
      l1_stf[k] += weight * stf[k];
    }
    for (int ell = 2; ell <= 4; ++ell) {
      double ylm[9];
      const int ylm_width = real_spherical_harmonics_spin(rhat, ell, ylm);
      double* angular = ell == 2 ? angular2 : ell == 3 ? angular3 : angular4;
      int width = 0;
      for (int m = 0; m < ylm_width; ++m) {
        angular[width++] += weight * ylm[m] * sj[0];
        angular[width++] += weight * ylm[m] * sj[1];
        angular[width++] += weight * ylm[m] * sj[2];
      }
    }
    double rr[9];
    stf_outer3(rhat, rhat, rr);
    for (int k = 0; k < 9; ++k) {
      geom[k] += weight * rr[k];
    }
    if (c < ChiC) {
      double o_reduced[kSpinChiralOReducedCount];
      double h_reduced[kSpinChiralHReducedCount];
      fill_spin_chiral_reduced_moments(rhat, o_reduced, h_reduced);
      for (int k = 0; k < kSpinChiralOReducedCount; ++k) {
        octupoles_reduced[k] += weight * o_reduced[k];
      }
      for (int k = 0; k < kSpinChiralHReducedCount; ++k) {
        hexadecapoles_reduced[k] += weight * h_reduced[k];
      }
    }
  }

  for (int k = 0; k < 3; ++k) {
    density_rho0_cache[atom + atom_stride * (c * 3 + k)] =
        static_cast<float>(rho0[k]);
    density_rho0_dot_cache[atom + atom_stride * (c * 3 + k)] =
        static_cast<float>(rho0_dot[k]);
    density_l1_cross_cache[atom + atom_stride * (c * 3 + k)] =
        static_cast<float>(l1_cross[k]);
    chiral_polar_cache[atom + atom_stride * (c * 3 + k)] =
        static_cast<float>(polar[k]);
  }
  for (int k = 0; k < 9; ++k) {
    density_raw1_cache[atom + atom_stride * (c * 9 + k)] =
        static_cast<float>(raw1[k]);
    density_l1_stf_cache[atom + atom_stride * (c * 9 + k)] =
        static_cast<float>(l1_stf[k]);
    density_geom_cache[atom + atom_stride * (c * 9 + k)] =
        static_cast<float>(geom[k]);
    density_raw1_dot_cache[atom + atom_stride * (c * 9 + k)] =
        static_cast<float>(raw1_dot[k]);
  }
  density_l1_rdot_cache[atom + atom_stride * c] =
      static_cast<float>(l1_rdot);
  for (int k = 0; k < 15; ++k) {
    density_angular2_cache[atom + atom_stride * (c * 15 + k)] =
        static_cast<float>(angular2[k]);
  }
  for (int k = 0; k < 21; ++k) {
    density_angular3_cache[atom + atom_stride * (c * 21 + k)] =
        static_cast<float>(angular3[k]);
  }
  for (int k = 0; k < 27; ++k) {
    density_angular4_cache[atom + atom_stride * (c * 27 + k)] =
        static_cast<float>(angular4[k]);
  }
  if (c < ChiC) {
    // The c4/l4 fast path stores the exact 7- and 9-dimensional images in
    // the leading slots of the existing raw-moment workspaces.  Keeping the
    // original channel stride preserves the workspace layout and fallback ABI.
    for (int k = 0; k < kSpinChiralOReducedCount; ++k) {
      chiral_octupoles_raw_cache[
          atom + atom_stride * (c * kSpinDeg3Count + k)] =
          static_cast<float>(octupoles_reduced[k]);
    }
    for (int k = 0; k < kSpinChiralHReducedCount; ++k) {
      chiral_hexadecapoles_raw_cache[
          atom + atom_stride * (c * kSpinDeg4Count + k)] =
          static_cast<float>(hexadecapoles_reduced[k]);
    }
  }

  for (int term = 0; term < 4; ++term) {
    descriptors[atom + atom_stride * (struct_dim + 2 + term * C + c)] =
        static_cast<float>(scalar_q[term]);
  }
  int offset = 2 + 4 * C;
  double value = 0.0;
  for (int k = 0; k < 3; ++k) {
    value += rho0[k] * rho0[k];
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
  offset += C;
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(l1_rdot * l1_rdot);
  offset += C;
  value = 0.0;
  for (int k = 0; k < 3; ++k) {
    value += l1_cross[k] * l1_cross[k];
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
  offset += C;
  value = 0.0;
  for (int k = 0; k < 9; ++k) {
    value += l1_stf[k] * l1_stf[k];
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
  offset += C;
  const double* angular[3] = {angular2, angular3, angular4};
  const int widths[3] = {15, 21, 27};
  for (int ell_index = 0; ell_index < 3; ++ell_index) {
    value = 0.0;
    for (int k = 0; k < widths[ell_index]; ++k) {
      value += angular[ell_index][k] * angular[ell_index][k];
    }
    descriptors[atom + atom_stride * (struct_dim + offset + c)] =
        static_cast<float>(value);
    offset += C;
  }
  value = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      value += spin_i[a] * geom[3 * a + b] * spin_i[b];
    }
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
  offset += C;
  value = 0.0;
  for (int k = 0; k < 3; ++k) {
    value += rho0[k] * rho0_dot[k];
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
  offset += C;
  value = 0.0;
  for (int k = 0; k < 9; ++k) {
    value += raw1[k] * raw1_dot[k];
  }
  descriptors[atom + atom_stride * (struct_dim + offset + c)] =
      static_cast<float>(value);
}

__device__ __forceinline__ void load_spin_edge_cached_f32(
    int atom,
    int neighbor,
    int atom_stride,
    int slot,
    const double* spins_soa3,
    const float* spin_edge_dx,
    const float* spin_edge_dy,
    const float* spin_edge_dz,
    const float* spin_edge_dist,
    float* rhat,
    float& dist,
    float* si,
    float* sj);
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

template <int SlotCapacity, bool AtomMajor>
__global__ void __launch_bounds__(128, 1) build_spin_primitive_cache_c4_l4_warp(
    int atom_count,
    int atom_stride,
    int struct_dim,
    float spin_cutoff,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ spin_edge_dx,
    const float* __restrict__ spin_edge_dy,
    const float* __restrict__ spin_edge_dz,
    const float* __restrict__ spin_edge_dist,
    const float* __restrict__ spin_edge_weights,
    float* __restrict__ density_rho0_cache,
    float* __restrict__ density_raw1_cache,
    float* __restrict__ density_l1_rdot_cache,
    float* __restrict__ density_l1_cross_cache,
    float* __restrict__ density_l1_stf_cache,
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
  constexpr int C = 4;
  constexpr int ChiC = 2;
  constexpr int ScalarComponents = 16;
  constexpr int Rho0Base = ScalarComponents;
  constexpr int Raw1Base = Rho0Base + C * 3;
  constexpr int L1RdotBase = Raw1Base + C * 9;
  constexpr int L1CrossBase = L1RdotBase + C;
  constexpr int L1StfBase = L1CrossBase + C * 3;
  constexpr int Angular2Base = L1StfBase + C * 9;
  constexpr int Angular3Base = Angular2Base + C * 15;
  constexpr int Angular4Base = Angular3Base + C * 21;
  constexpr int GeomBase = Angular4Base + C * 27;
  constexpr int Rho0DotBase = GeomBase + C * 9;
  constexpr int Raw1DotBase = Rho0DotBase + C * 3;
  constexpr int PolarBase = Raw1DotBase + C * 9;
  constexpr int DensityComponentCount = PolarBase - Rho0Base;
  __shared__ float prim[kSpinPrimitiveCount][SlotCapacity + 1];
  __shared__ float weights[C][SlotCapacity + 1];
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

  const int radial_count = nn_radial[atom];
  const int count = radial_count < SlotCapacity ? radial_count : SlotCapacity;
  for (int slot = lane; slot < count; slot += blockDim.x) {
    float rhat[3];
    float dist = 0.0f;
    float si_edge[3];
    float sj[3];
    load_spin_edge_cached_f32(
        atom,
        nl_radial[atom + atom_stride * slot],
        atom_stride,
        slot,
        spins_soa3,
        spin_edge_dx,
        spin_edge_dy,
        spin_edge_dz,
        spin_edge_dist,
        rhat,
        dist,
        si_edge,
        sj);

    for (int p = 0; p < kSpinPrimitiveCount; ++p) {
      prim[p][slot] = 0.0f;
    }
    for (int c = 0; c < C; ++c) {
      weights[c][slot] = 0.0f;
    }
    if (dist > 1.0e-12f && dist < spin_cutoff) {
      for (int c = 0; c < C; ++c) {
        weights[c][slot] = spin_edge_weights[
            spin_edge_cache_index(atom_stride, slot, atom, c)];
      }
      const float dot = dot3f(si, sj);
      const float sj2 = dot3f(sj, sj);
      const float ri_dot_si = dot3f(rhat, si);
      const float ri_dot_sj = dot3f(rhat, sj);
      prim[0][slot] = sj[0];
      prim[1][slot] = sj[1];
      prim[2][slot] = sj[2];
      prim[3][slot] = dot;
      prim[4][slot] = sj2;
      prim[5][slot] = ri_dot_si;
      prim[6][slot] = ri_dot_sj;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          prim[7 + 3 * a + b][slot] = rhat[a] * sj[b];
        }
      }
      float cross_value[3];
      cross3f(rhat, sj, cross_value);
      for (int k = 0; k < 3; ++k) {
        prim[16 + k][slot] = cross_value[k];
      }
      float stf[9];
      stf_outer3f(rhat, sj, stf);
      for (int k = 0; k < 9; ++k) {
        prim[19 + k][slot] = stf[k];
      }
      float ylm[9];
      int width = real_spherical_harmonics_spinf(rhat, 2, ylm);
      for (int m = 0; m < width; ++m) {
        prim[28 + m][slot] = ylm[m];
      }
      width = real_spherical_harmonics_spinf(rhat, 3, ylm);
      for (int m = 0; m < width; ++m) {
        prim[33 + m][slot] = ylm[m];
      }
      width = real_spherical_harmonics_spinf(rhat, 4, ylm);
      for (int m = 0; m < width; ++m) {
        prim[40 + m][slot] = ylm[m];
      }
      float rr[9];
      stf_outer3f(rhat, rhat, rr);
      for (int k = 0; k < 9; ++k) {
        prim[49 + k][slot] = rr[k];
      }
      prim[58][slot] = rhat[0];
      prim[59][slot] = rhat[1];
      prim[60][slot] = rhat[2];
      float o_reduced[kSpinChiralOReducedCount];
      float h_reduced[kSpinChiralHReducedCount];
      fill_spin_chiral_reduced_momentsf(rhat, o_reduced, h_reduced);
      for (int k = 0; k < kSpinChiralOReducedCount; ++k) {
        prim[61 + k][slot] = o_reduced[k];
      }
      for (int k = 0; k < kSpinChiralHReducedCount; ++k) {
        prim[61 + kSpinChiralOReducedCount + k][slot] = h_reduced[k];
      }
    }
  }
  __syncthreads();

  constexpr int ScalarTaskBase = 0;
  constexpr int Rho0TaskBase = ScalarTaskBase + 4;
  constexpr int Raw1TaskBase = Rho0TaskBase + 3;
  constexpr int L1RdotTaskBase = Raw1TaskBase + 9;
  constexpr int L1CrossTaskBase = L1RdotTaskBase + 1;
  constexpr int L1StfTaskBase = L1CrossTaskBase + 3;
  constexpr int Angular2TaskBase = L1StfTaskBase + 9;
  constexpr int Angular3TaskBase = Angular2TaskBase + 15;
  constexpr int Angular4TaskBase = Angular3TaskBase + 21;
  constexpr int GeomTaskBase = Angular4TaskBase + 27;
  constexpr int Rho0DotTaskBase = GeomTaskBase + 9;
  constexpr int Raw1DotTaskBase = Rho0DotTaskBase + 3;
  constexpr int PolarTaskBase = Raw1DotTaskBase + 9;
  constexpr int ChannelTaskCount = PolarTaskBase + 3;

  for (int task = lane; task < ChannelTaskCount; task += blockDim.x) {
    float channel_acc[C] = {};
    if (task < Rho0TaskBase) {
      const int term = task - ScalarTaskBase;
      for (int slot = 0; slot < count; ++slot) {
        const float dot = prim[3][slot];
        const float value =
            term == 0 ? dot :
            term == 1 ? dot * dot :
            term == 2 ? prim[4][slot] : prim[5][slot] * prim[6][slot];
        for (int c = 0; c < C; ++c) {
          channel_acc[c] += weights[c][slot] * value;
        }
      }
      for (int c = 0; c < C; ++c) {
        descriptors[
            atom + atom_stride * (struct_dim + 2 + term * C + c)] =
            channel_acc[c];
      }
      continue;
    }

    int width = 0;
    int component_base = 0;
    int component_count = 0;
    int primitive_base = 0;
    int angular_ylm_base = -1;
    int k = 0;
    bool multiply_dot = false;
    float* output = nullptr;
    if (task < Raw1TaskBase) {
      width = 3;
      component_base = Rho0Base;
      component_count = C * 3;
      primitive_base = 0;
      k = task - Rho0TaskBase;
      output = density_rho0_cache;
    } else if (task < L1RdotTaskBase) {
      width = 9;
      component_base = Raw1Base;
      component_count = C * 9;
      primitive_base = 7;
      k = task - Raw1TaskBase;
      output = density_raw1_cache;
    } else if (task < L1CrossTaskBase) {
      width = 1;
      component_base = L1RdotBase;
      component_count = C;
      primitive_base = 6;
      k = task - L1RdotTaskBase;
      output = density_l1_rdot_cache;
    } else if (task < L1StfTaskBase) {
      width = 3;
      component_base = L1CrossBase;
      component_count = C * 3;
      primitive_base = 16;
      k = task - L1CrossTaskBase;
      output = density_l1_cross_cache;
    } else if (task < Angular2TaskBase) {
      width = 9;
      component_base = L1StfBase;
      component_count = C * 9;
      primitive_base = 19;
      k = task - L1StfTaskBase;
      output = density_l1_stf_cache;
    } else if (task < Angular3TaskBase) {
      width = 15;
      component_base = Angular2Base;
      component_count = C * 15;
      primitive_base = 28;
      angular_ylm_base = 28;
      k = task - Angular2TaskBase;
      output = density_angular2_cache;
    } else if (task < Angular4TaskBase) {
      width = 21;
      component_base = Angular3Base;
      component_count = C * 21;
      primitive_base = 33;
      angular_ylm_base = 33;
      k = task - Angular3TaskBase;
      output = density_angular3_cache;
    } else if (task < GeomTaskBase) {
      width = 27;
      component_base = Angular4Base;
      component_count = C * 27;
      primitive_base = 40;
      angular_ylm_base = 40;
      k = task - Angular4TaskBase;
      output = density_angular4_cache;
    } else if (task < Rho0DotTaskBase) {
      width = 9;
      component_base = GeomBase;
      component_count = C * 9;
      primitive_base = 49;
      k = task - GeomTaskBase;
      output = density_geom_cache;
    } else if (task < Raw1DotTaskBase) {
      width = 3;
      component_base = Rho0DotBase;
      component_count = C * 3;
      primitive_base = 0;
      k = task - Rho0DotTaskBase;
      multiply_dot = true;
      output = density_rho0_dot_cache;
    } else if (task < PolarTaskBase) {
      width = 9;
      component_base = Raw1DotBase;
      component_count = C * 9;
      primitive_base = 7;
      k = task - Raw1DotTaskBase;
      multiply_dot = true;
      output = density_raw1_dot_cache;
    } else {
      width = 3;
      component_base = PolarBase;
      component_count = C * 3;
      primitive_base = 58;
      k = task - PolarTaskBase;
      output = chiral_polar_cache;
    }

    for (int slot = 0; slot < count; ++slot) {
      float value;
      if (angular_ylm_base >= 0) {
        const int m = k / 3;
        const int d = k - 3 * m;
        value = prim[angular_ylm_base + m][slot] * prim[d][slot];
      } else {
        value = prim[primitive_base + k][slot];
      }
      if (multiply_dot) {
        value *= prim[3][slot];
      }
      for (int c = 0; c < C; ++c) {
        channel_acc[c] += weights[c][slot] * value;
      }
    }
    for (int c = 0; c < C; ++c) {
      const int component = c * width + k;
      output[spin_component_cache_index<AtomMajor>(
          atom_stride, component_count, atom, component)] = channel_acc[c];
      if (task < PolarTaskBase) {
        density_components[
            component_base - Rho0Base + component] = channel_acc[c];
      }
    }
  }

  constexpr int OctTaskBase = 0;
  constexpr int HexTaskBase = OctTaskBase + kSpinChiralOReducedCount;
  constexpr int ChiralTaskCount = HexTaskBase + kSpinChiralHReducedCount;
  for (int task = lane; task < ChiralTaskCount; task += blockDim.x) {
    const bool is_octupole = task < HexTaskBase;
    const int cache_width = is_octupole ? kSpinDeg3Count : kSpinDeg4Count;
    const int k = is_octupole ? task - OctTaskBase : task - HexTaskBase;
    const int primitive_base =
        is_octupole ? 61 : 61 + kSpinChiralOReducedCount;
    const int component_count = ChiC * cache_width;
    float channel_acc[ChiC] = {};
    for (int slot = 0; slot < count; ++slot) {
      const float value = prim[primitive_base + k][slot];
      for (int c = 0; c < ChiC; ++c) {
        channel_acc[c] += weights[c][slot] * value;
      }
    }
    float* output = is_octupole
        ? chiral_octupoles_raw_cache
        : chiral_hexadecapoles_raw_cache;
    for (int c = 0; c < ChiC; ++c) {
      output[spin_component_cache_index<AtomMajor>(
          atom_stride, component_count, atom, c * cache_width + k)] =
          channel_acc[c];
    }
  }
  __syncthreads();
  if (lane < C) {
    const int channel = lane;
    int offset = 2 + 4 * C;
    float value = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const float rho0 = density_components[
          Rho0Base - Rho0Base + channel * 3 + k];
      value += rho0 * rho0;
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        value;
    offset += C;

    const float rdot = density_components[
        L1RdotBase - Rho0Base + channel];
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        rdot * rdot;
    offset += C;

    value = 0.0f;
    for (int k = 0; k < 3; ++k) {
      const float x = density_components[
          L1CrossBase - Rho0Base + channel * 3 + k];
      value += x * x;
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        value;
    offset += C;

    value = 0.0f;
    for (int k = 0; k < 9; ++k) {
      const float x = density_components[
          L1StfBase - Rho0Base + channel * 9 + k];
      value += x * x;
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        value;
    offset += C;

    constexpr int AngularBases[3] = {
        Angular2Base,
        Angular3Base,
        Angular4Base};
    constexpr int AngularWidths[3] = {15, 21, 27};
    for (int ell_index = 0; ell_index < 3; ++ell_index) {
      value = 0.0f;
      for (int k = 0; k < AngularWidths[ell_index]; ++k) {
        const float x = density_components[
            AngularBases[ell_index] - Rho0Base +
            channel * AngularWidths[ell_index] + k];
        value += x * x;
      }
      descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
          value;
      offset += C;
    }

    value = 0.0f;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        value += si[a] * density_components[
            GeomBase - Rho0Base + channel * 9 + 3 * a + b] * si[b];
      }
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        value;
    offset += C;

    value = 0.0f;
    for (int k = 0; k < 3; ++k) {
      value += density_components[
                   Rho0Base - Rho0Base + channel * 3 + k] *
               density_components[
                   Rho0DotBase - Rho0Base + channel * 3 + k];
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        value;
    offset += C;

    value = 0.0f;
    for (int k = 0; k < 9; ++k) {
      value += density_components[
                   Raw1Base - Rho0Base + channel * 9 + k] *
               density_components[
                   Raw1DotBase - Rho0Base + channel * 9 + k];
    }
    descriptors[atom + atom_stride * (struct_dim + offset + channel)] =
        value;
  }
}
