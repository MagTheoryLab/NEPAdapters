#pragma once

#include "../../common/spin_polynomial_layout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace nep_adapters::cpu_spin2 {

using nep_adapters::common::SpinPolynomialLayout;

constexpr int kMaxChannels = 9;
constexpr int kM = 0;
constexpr int kP = 3;
constexpr int kL = 6;
constexpr int kX = 7;
constexpr int kT = 10;
constexpr int kQ = 15;
constexpr int kQP = 20;
constexpr int kDM = 35;
constexpr int kDensityStride = 38;

inline int channel_offset(int channel) { return channel * kDensityStride; }
inline int same_offset(int channels) { return channels * kDensityStride; }
inline int pair_index(int channels, int left, int right) {
  return left * channels - left * (left - 1) / 2 + right - left;
}

struct Edge {
  int neighbor = -1;
  double displacement[3] = {};
  double distance = 0.0;
  double weights[kMaxChannels] = {};
  double derivatives[kMaxChannels] = {};
};

struct EdgeGradient {
  int neighbor = -1;
  double displacement[3] = {};
  double position[3] = {};
  double center_spin[3] = {};
  double neighbor_spin[3] = {};
};

namespace detail {

template <typename Scalar>
Scalar dot3(const Scalar* left, const Scalar* right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

template <typename Scalar>
void cross3(const Scalar* left, const Scalar* right, Scalar* out) {
  out[0] = left[1] * right[2] - left[2] * right[1];
  out[1] = left[2] * right[0] - left[0] * right[2];
  out[2] = left[0] * right[1] - left[1] * right[0];
}

template <typename Scalar>
void stf5_outer(const Scalar* left, const Scalar* right, Scalar* out) {
  constexpr double inv_sqrt6 = 0.40824829046386301637;
  constexpr double inv_sqrt2 = 0.70710678118654752440;
  const Scalar xx = left[0] * right[0];
  const Scalar yy = left[1] * right[1];
  const Scalar zz = left[2] * right[2];
  out[0] = (2.0 * zz - xx - yy) * inv_sqrt6;
  out[1] = (xx - yy) * inv_sqrt2;
  out[2] = (left[0] * right[1] + left[1] * right[0]) * inv_sqrt2;
  out[3] = (left[1] * right[2] + left[2] * right[1]) * inv_sqrt2;
  out[4] = (left[2] * right[0] + left[0] * right[2]) * inv_sqrt2;
}

template <typename Scalar>
void expand_stf5(const Scalar* value, Scalar* out) {
  constexpr double inv_sqrt6 = 0.40824829046386301637;
  constexpr double inv_sqrt2 = 0.70710678118654752440;
  const Scalar xx = -value[0] * inv_sqrt6 + value[1] * inv_sqrt2;
  const Scalar yy = -value[0] * inv_sqrt6 - value[1] * inv_sqrt2;
  const Scalar zz = 2.0 * value[0] * inv_sqrt6;
  const Scalar xy = value[2] * inv_sqrt2;
  const Scalar yz = value[3] * inv_sqrt2;
  const Scalar zx = value[4] * inv_sqrt2;
  out[0] = xx; out[1] = xy; out[2] = zx;
  out[3] = xy; out[4] = yy; out[5] = yz;
  out[6] = zx; out[7] = yz; out[8] = zz;
}

template <typename Scalar>
void stf5_matvec(const Scalar* matrix, const Scalar* vector, Scalar* out) {
  Scalar full[9];
  expand_stf5(matrix, full);
  for (int a = 0; a < 3; ++a) {
    out[a] = Scalar(0.0);
    for (int b = 0; b < 3; ++b) out[a] = out[a] + full[3 * a + b] * vector[b];
  }
}

template <int Width, typename Scalar>
Scalar dotn(const Scalar* left, const Scalar* right) {
  Scalar value(0.0);
  for (int k = 0; k < Width; ++k) value = value + left[k] * right[k];
  return value;
}

template <int Width, typename Scalar>
void project_density(
    const Scalar* center, const double* projection, int channels,
    int density_offset, int leg, int row, Scalar* out) {
  for (int k = 0; k < Width; ++k) {
    out[k] = Scalar(0.0);
    for (int source = 0; source < channels; ++source) {
      out[k] = out[k] + projection[(leg * channels + row) * channels + source] *
          center[channel_offset(source) + density_offset + k];
    }
  }
}

template <typename Scalar>
void qp_matrix(const Scalar* qp, int spin_component, Scalar* matrix) {
  Scalar coefficients[5];
  for (int k = 0; k < 5; ++k) coefficients[k] = qp[3 * k + spin_component];
  expand_stf5(coefficients, matrix);
}

template <typename Scalar>
void qp_vector(const Scalar* qp, Scalar* vector) {
  for (int d = 0; d < 3; ++d) vector[d] = Scalar(0.0);
  for (int spin_component = 0; spin_component < 3; ++spin_component) {
    Scalar matrix[9];
    qp_matrix(qp, spin_component, matrix);
    for (int a = 0; a < 3; ++a) {
      vector[a] = vector[a] + matrix[3 * a + spin_component];
    }
  }
}

template <typename Scalar>
void axial_commutator(const Scalar* left, const Scalar* right, Scalar* out) {
  Scalar l[9], r[9], comm[9];
  expand_stf5(left, l);
  expand_stf5(right, r);
  for (auto& value : comm) value = Scalar(0.0);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int k = 0; k < 3; ++k) {
        comm[3 * a + b] = comm[3 * a + b] +
            l[3 * a + k] * r[3 * k + b] - r[3 * a + k] * l[3 * k + b];
      }
    }
  }
  out[0] = comm[7]; out[1] = comm[2]; out[2] = comm[3];
}

template <typename Scalar>
void l11_axis(
    const Scalar* si, const Scalar& longitudinal, const Scalar* axial,
    const Scalar* stf_first, Scalar* axis) {
  Scalar matrix[9], matrix_on_spin[3], axial_cross_spin[3];
  expand_stf5(stf_first, matrix);
  for (int a = 0; a < 3; ++a) {
    matrix_on_spin[a] = matrix[3 * a] * si[0] +
        matrix[3 * a + 1] * si[1] + matrix[3 * a + 2] * si[2];
  }
  cross3(axial, si, axial_cross_spin);
  for (int d = 0; d < 3; ++d) {
    axis[d] = (2.0 / 3.0) * longitudinal * si[d] -
        matrix_on_spin[d] - 0.5 * axial_cross_spin[d];
  }
}

template <typename Scalar>
void qp_raw_matrix(
    const Scalar* moment, const Scalar* qp, int spin_component, Scalar* matrix) {
  qp_matrix(qp, spin_component, matrix);
  for (int a = 0; a < 3; ++a) matrix[3 * a + a] = matrix[3 * a + a] + moment[spin_component] / 3.0;
}

template <typename Scalar>
Scalar l22_scalar(
    const Scalar* si, const Scalar* moment, const Scalar* projected_q,
    const Scalar* projected_qp) {
  Scalar q_matrix[9], q_on_spin[3];
  expand_stf5(projected_q, q_matrix);
  for (int a = 0; a < 3; ++a) {
    q_on_spin[a] = Scalar(0.0);
    for (int b = 0; b < 3; ++b) q_on_spin[a] = q_on_spin[a] + q_matrix[3 * a + b] * si[b];
  }
  Scalar value(0.0);
  for (int spin_component = 0; spin_component < 3; ++spin_component) {
    Scalar raw_second[9];
    qp_raw_matrix(moment, projected_qp, spin_component, raw_second);
    for (int a = 0; a < 3; ++a) {
      value = value - raw_second[3 * spin_component + a] * q_on_spin[a];
      for (int b = 0; b < 3; ++b) {
        value = value + q_matrix[3 * spin_component + a] *
            raw_second[3 * a + b] * si[b];
      }
    }
  }
  return value;
}

inline void add_dot_gradient(
    const double* left, const double* right, double output_gradient,
    double* left_gradient, double* right_gradient) {
  for (int d = 0; d < 3; ++d) {
    left_gradient[d] += output_gradient * right[d];
    right_gradient[d] += output_gradient * left[d];
  }
}

inline void add_cross_gradient(
    const double* left, const double* right, const double* output_gradient,
    double* left_gradient, double* right_gradient) {
  double left_delta[3], right_delta[3];
  cross3(right, output_gradient, left_delta);
  cross3(output_gradient, left, right_delta);
  for (int d = 0; d < 3; ++d) {
    left_gradient[d] += left_delta[d];
    right_gradient[d] += right_delta[d];
  }
}

inline void add_stf5_outer_gradient(
    const double* left, const double* right, const double* output_gradient,
    double* left_gradient, double* right_gradient) {
  constexpr double inv_sqrt6 = 0.40824829046386301637;
  constexpr double inv_sqrt2 = 0.70710678118654752440;
  const double diagonal[3] = {
      -output_gradient[0] * inv_sqrt6 + output_gradient[1] * inv_sqrt2,
      -output_gradient[0] * inv_sqrt6 - output_gradient[1] * inv_sqrt2,
      2.0 * output_gradient[0] * inv_sqrt6};
  left_gradient[0] += diagonal[0] * right[0] +
      output_gradient[2] * inv_sqrt2 * right[1] + output_gradient[4] * inv_sqrt2 * right[2];
  left_gradient[1] += diagonal[1] * right[1] +
      output_gradient[2] * inv_sqrt2 * right[0] + output_gradient[3] * inv_sqrt2 * right[2];
  left_gradient[2] += diagonal[2] * right[2] +
      output_gradient[3] * inv_sqrt2 * right[1] + output_gradient[4] * inv_sqrt2 * right[0];
  right_gradient[0] += diagonal[0] * left[0] +
      output_gradient[2] * inv_sqrt2 * left[1] + output_gradient[4] * inv_sqrt2 * left[2];
  right_gradient[1] += diagonal[1] * left[1] +
      output_gradient[2] * inv_sqrt2 * left[0] + output_gradient[3] * inv_sqrt2 * left[2];
  right_gradient[2] += diagonal[2] * left[2] +
      output_gradient[3] * inv_sqrt2 * left[1] + output_gradient[4] * inv_sqrt2 * left[0];
}

inline void add_expanded_stf5_gradient(const double* matrix_gradient, double* value_gradient) {
  constexpr double inv_sqrt6 = 0.40824829046386301637;
  constexpr double inv_sqrt2 = 0.70710678118654752440;
  value_gradient[0] += inv_sqrt6 *
      (-matrix_gradient[0] - matrix_gradient[4] + 2.0 * matrix_gradient[8]);
  value_gradient[1] += inv_sqrt2 * (matrix_gradient[0] - matrix_gradient[4]);
  value_gradient[2] += inv_sqrt2 * (matrix_gradient[1] + matrix_gradient[3]);
  value_gradient[3] += inv_sqrt2 * (matrix_gradient[5] + matrix_gradient[7]);
  value_gradient[4] += inv_sqrt2 * (matrix_gradient[2] + matrix_gradient[6]);
}

inline void add_stf5_matvec_gradient(
    const double* matrix, const double* vector, const double* output_gradient,
    double* matrix_gradient, double* vector_gradient) {
  double full[9];
  expand_stf5(matrix, full);
  double full_gradient[9] = {};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      full_gradient[3 * a + b] += output_gradient[a] * vector[b];
      vector_gradient[b] += full[3 * a + b] * output_gradient[a];
    }
  }
  add_expanded_stf5_gradient(full_gradient, matrix_gradient);
}

template <int Width>
inline void add_projected_density_gradient(
    const double* projection, int channels, int density_offset,
    int leg, int row, const double* output_gradient, double* center_gradient) {
  for (int source = 0; source < channels; ++source) {
    const double coefficient = projection[(leg * channels + row) * channels + source];
    for (int k = 0; k < Width; ++k) {
      center_gradient[channel_offset(source) + density_offset + k] +=
          coefficient * output_gradient[k];
    }
  }
}

inline void add_qp_matrix_gradient(
    int spin_component, const double* matrix_gradient, double* qp_gradient) {
  double coefficient_gradient[5] = {};
  add_expanded_stf5_gradient(matrix_gradient, coefficient_gradient);
  for (int k = 0; k < 5; ++k) {
    qp_gradient[3 * k + spin_component] += coefficient_gradient[k];
  }
}

inline void add_qp_vector_gradient(
    const double* qp, const double* output_gradient, double* qp_gradient) {
  for (int spin_component = 0; spin_component < 3; ++spin_component) {
    double matrix_gradient[9] = {};
    for (int a = 0; a < 3; ++a) {
      matrix_gradient[3 * a + spin_component] = output_gradient[a];
    }
    add_qp_matrix_gradient(spin_component, matrix_gradient, qp_gradient);
  }
}

inline void add_axial_commutator_gradient(
    const double* left, const double* right, const double* output_gradient,
    double* left_gradient, double* right_gradient) {
  double l[9], r[9], comm_gradient[9] = {}, l_gradient[9] = {}, r_gradient[9] = {};
  expand_stf5(left, l);
  expand_stf5(right, r);
  comm_gradient[7] = output_gradient[0];
  comm_gradient[2] = output_gradient[1];
  comm_gradient[3] = output_gradient[2];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const double g = comm_gradient[3 * a + b];
      for (int k = 0; k < 3; ++k) {
        l_gradient[3 * a + k] += g * r[3 * k + b];
        r_gradient[3 * k + b] += g * l[3 * a + k];
        r_gradient[3 * a + k] -= g * l[3 * k + b];
        l_gradient[3 * k + b] -= g * r[3 * a + k];
      }
    }
  }
  add_expanded_stf5_gradient(l_gradient, left_gradient);
  add_expanded_stf5_gradient(r_gradient, right_gradient);
}

inline void add_l11_axis_gradient(
    const double* si, double longitudinal, const double* axial,
    const double* stf_first, const double* output_gradient,
    double* si_gradient, double& longitudinal_gradient,
    double* axial_gradient, double* stf_gradient) {
  double matrix[9];
  expand_stf5(stf_first, matrix);
  longitudinal_gradient += (2.0 / 3.0) * dot3(output_gradient, si);
  double matrix_gradient[9] = {};
  for (int a = 0; a < 3; ++a) {
    si_gradient[a] += (2.0 / 3.0) * longitudinal * output_gradient[a];
    for (int b = 0; b < 3; ++b) {
      matrix_gradient[3 * a + b] -= output_gradient[a] * si[b];
      si_gradient[b] -= matrix[3 * a + b] * output_gradient[a];
    }
  }
  double cross_gradient[3] = {
      -0.5 * output_gradient[0],
      -0.5 * output_gradient[1],
      -0.5 * output_gradient[2]};
  add_cross_gradient(axial, si, cross_gradient, axial_gradient, si_gradient);
  add_expanded_stf5_gradient(matrix_gradient, stf_gradient);
}

inline void add_qp_raw_matrix_gradient(
    int spin_component, const double* matrix_gradient,
    double* moment_gradient, double* qp_gradient) {
  add_qp_matrix_gradient(spin_component, matrix_gradient, qp_gradient);
  moment_gradient[spin_component] +=
      (matrix_gradient[0] + matrix_gradient[4] + matrix_gradient[8]) / 3.0;
}

inline void add_l22_scalar_gradient(
    const double* si, const double* moment, const double* projected_q,
    const double* projected_qp, double output_gradient,
    double* si_gradient, double* moment_gradient,
    double* q_gradient, double* qp_gradient) {
  double q_matrix[9], q_on_spin[3] = {};
  expand_stf5(projected_q, q_matrix);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      q_on_spin[a] += q_matrix[3 * a + b] * si[b];
    }
  }
  double q_matrix_gradient[9] = {}, q_on_spin_gradient[3] = {};
  for (int spin_component = 0; spin_component < 3; ++spin_component) {
    double raw_second[9], raw_gradient[9] = {};
    qp_raw_matrix(moment, projected_qp, spin_component, raw_second);
    for (int a = 0; a < 3; ++a) {
      raw_gradient[3 * spin_component + a] -= output_gradient * q_on_spin[a];
      q_on_spin_gradient[a] -= output_gradient * raw_second[3 * spin_component + a];
      for (int b = 0; b < 3; ++b) {
        q_matrix_gradient[3 * spin_component + a] +=
            output_gradient * raw_second[3 * a + b] * si[b];
        raw_gradient[3 * a + b] +=
            output_gradient * q_matrix[3 * spin_component + a] * si[b];
        si_gradient[b] +=
            output_gradient * q_matrix[3 * spin_component + a] * raw_second[3 * a + b];
      }
    }
    add_qp_raw_matrix_gradient(spin_component, raw_gradient, moment_gradient, qp_gradient);
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      q_matrix_gradient[3 * a + b] += q_on_spin_gradient[a] * si[b];
      si_gradient[b] += q_matrix[3 * a + b] * q_on_spin_gradient[a];
    }
  }
  add_expanded_stf5_gradient(q_matrix_gradient, q_gradient);
}

template <typename Scalar, typename Emit>
void contract(
    const SpinPolynomialLayout& layout, const Scalar* center,
    const Scalar* direct, const Scalar* si, const double* projection, Emit emit) {
  const int channels = layout.channels;
  emit(layout.local_s2, dot3(si, si));
  Scalar local_q[5];
  Scalar first_dot[kMaxChannels];
  stf5_outer(si, si, local_q);
  for (int c = 0; c < channels; ++c) {
    const int base = channel_offset(c);
    const Scalar* m = center + base + kM;
    first_dot[c] = dot3(si, m);
    emit(layout.edge_l0_dot + c, first_dot[c]);
    emit(layout.edge_l0_neighbor_s2 + c, direct[c]);
    if (layout.edge_l2_pair >= 0) {
      Scalar q_on_spin[3];
      qp_vector(center + base + kQP, q_on_spin);
      emit(layout.edge_l2_pair + c, dot3(si, q_on_spin));
      emit(layout.center_l2_environment + c, dotn<5>(center + base + kQ, local_q));
    }
    if (layout.edge_l0_dot2 >= 0) emit(layout.edge_l0_dot2 + c, direct[channels + c]);
    if (layout.density_l0_self >= 0) {
      emit(layout.density_l0_self + c, dotn<3>(m, m));
      if (layout.density_l1_longitudinal_self >= 0) {
        emit(layout.density_l1_longitudinal_self + c,
             center[base + kL] * center[base + kL]);
        emit(layout.density_l1_axial_self + c,
             dotn<3>(center + base + kX, center + base + kX));
        emit(layout.density_l1_stf_self + c,
             dotn<5>(center + base + kT, center + base + kT));
      }
      if (layout.density_l1_product_self >= 0) {
        emit(layout.density_l1_product_self + c,
             center[base + kL] * center[base + kL] / 3.0 +
             dotn<3>(center + base + kX, center + base + kX) * 0.5 +
             dotn<5>(center + base + kT, center + base + kT));
      }
      if (layout.density_l2_product_self >= 0) {
        emit(layout.density_l2_product_self + c,
             dotn<15>(center + base + kQP, center + base + kQP));
      }
      emit(layout.density_l0_dot_response + c,
           dotn<3>(m, center + base + kDM));
    }
    if (layout.edge_l0_moment_gate >= 0) {
      emit(layout.edge_l0_moment_gate + c, direct[2 * channels + c]);
    }
  }
  if (layout.correlation_same_edge >= 0) {
    const int offset = same_offset(channels);
    for (int left = 0; left < channels; ++left) {
      for (int right = left; right < channels; ++right) {
        const int pair = pair_index(channels, left, right);
        const Scalar same = center[offset + pair];
        emit(layout.correlation_same_edge + pair, same);
        emit(layout.correlation_distinct_neighbor + pair,
             first_dot[left] * first_dot[right] - same);
      }
    }
  }
  for (int row = 0; row < channels; ++row) {
    if (layout.edge_l11_axial >= 0) {
      Scalar m0[3], p0[3], p1[3], x0[3], l0[1], t0[5], w0[3];
      project_density<3>(center, projection, channels, kM, 0, row, m0);
      project_density<3>(center, projection, channels, kP, 0, row, p0);
      project_density<3>(center, projection, channels, kP, 1, row, p1);
      project_density<1>(center, projection, channels, kL, 0, row, l0);
      project_density<3>(center, projection, channels, kX, 0, row, x0);
      project_density<5>(center, projection, channels, kT, 0, row, t0);
      cross3(si, m0, w0);
      Scalar p_axis[3], axis[3];
      cross3(p0, p1, p_axis);
      l11_axis(si, l0[0], x0, t0, axis);
      if (layout.coupling_l11_axial >= 0) {
        emit(layout.coupling_l11_axial + row, dot3(p_axis, w0));
      }
      emit(layout.edge_l11_axial + row, dot3(p0, axis));
      if (layout.coupling_l11_dot_response >= 0) {
        Scalar dm0[3], dw0[3];
        project_density<3>(center, projection, channels, kDM, 0, row, dm0);
        cross3(si, dm0, dw0);
        emit(layout.coupling_l11_dot_response + row, dot3(p_axis, dw0));
      }
      if (layout.coupling_l111_p_m_x >= 0) {
        Scalar m1[3], x2[3], cross_value[3];
        project_density<3>(center, projection, channels, kM, 1, row, m1);
        project_density<3>(center, projection, channels, kX, 2, row, x2);
        cross3(m1, x2, cross_value);
        emit(layout.coupling_l111_p_m_x + row, dot3(p0, cross_value));
      }
      if (layout.coupling_l111_bulk >= 0) {
        Scalar p2[3], x3[3], cross12[3];
        project_density<3>(center, projection, channels, kP, 2, row, p2);
        project_density<3>(center, projection, channels, kX, 3, row, x3);
        cross3(p1, p2, cross12);
        emit(layout.coupling_l111_bulk + row,
             -dot3(p0, cross12) * dot3(si, x3));
      }
    }
    if (layout.edge_l22_axial >= 0) {
      Scalar m0[3], p0[3], p1[3], x0[3], q0[5], q1[5], qp0[15];
      project_density<3>(center, projection, channels, kM, 0, row, m0);
      project_density<3>(center, projection, channels, kP, 0, row, p0);
      project_density<3>(center, projection, channels, kP, 1, row, p1);
      project_density<3>(center, projection, channels, kX, 0, row, x0);
      project_density<5>(center, projection, channels, kQ, 0, row, q0);
      project_density<5>(center, projection, channels, kQ, 1, row, q1);
      project_density<15>(center, projection, channels, kQP, 0, row, qp0);
      Scalar q_axis[3], w0[3];
      axial_commutator(q0, q1, q_axis);
      cross3(si, m0, w0);
      if (layout.coupling_l22_axial >= 0) {
        emit(layout.coupling_l22_axial + row, dot3(q_axis, w0));
      }
      emit(layout.edge_l22_axial + row, l22_scalar(si, m0, q0, qp0));
      if (layout.coupling_l22_dot_response >= 0) {
        Scalar dm0[3], dw0[3];
        project_density<3>(center, projection, channels, kDM, 0, row, dm0);
        cross3(si, dm0, dw0);
        emit(layout.coupling_l22_dot_response + row, dot3(q_axis, dw0));
      }
      if (layout.coupling_l111_p_qs_x >= 0) {
        Scalar qp1[15], qs1[3], x2[3], cross_value[3];
        project_density<15>(center, projection, channels, kQP, 1, row, qp1);
        project_density<3>(center, projection, channels, kX, 2, row, x2);
        qp_vector(qp1, qs1);
        cross3(qs1, x2, cross_value);
        emit(layout.coupling_l111_p_qs_x + row, dot3(p0, cross_value));
      }
      if (layout.coupling_l112_edge_response >= 0) {
        Scalar q_on_p1[3], mixed_axis[3];
        stf5_matvec(q0, p1, q_on_p1);
        cross3(p1, q_on_p1, mixed_axis);
        emit(layout.coupling_l112_edge_response + row,
             -dot3(p0, mixed_axis) * dot3(si, x0));
      }
    }
  }
}

inline void accumulate_descriptor_gradients(
    const SpinPolynomialLayout& layout, const double* center,
    const double* si, const double* projection, const double* descriptor_gradient,
    double* center_gradient, double* direct_gradient, double* si_gradient) {
  const int channels = layout.channels;
  auto g = [&](int index) { return index < 0 ? 0.0 : descriptor_gradient[index]; };
  for (int d = 0; d < 3; ++d) si_gradient[d] += 2.0 * g(layout.local_s2) * si[d];

  double local_q[5], local_q_gradient[5] = {}, first_dot[kMaxChannels] = {};
  double first_dot_gradient[kMaxChannels] = {};
  stf5_outer(si, si, local_q);
  for (int c = 0; c < channels; ++c) {
    const int base = channel_offset(c);
    const double* m = center + base + kM;
    double* m_gradient = center_gradient + base + kM;
    first_dot[c] = dot3(si, m);
    first_dot_gradient[c] += g(layout.edge_l0_dot + c);
    direct_gradient[c] += g(layout.edge_l0_neighbor_s2 + c);
    if (layout.edge_l2_pair >= 0) {
      const double edge_gradient = g(layout.edge_l2_pair + c);
      double q_on_spin[3], q_on_spin_gradient[3] = {};
      qp_vector(center + base + kQP, q_on_spin);
      add_dot_gradient(si, q_on_spin, edge_gradient, si_gradient, q_on_spin_gradient);
      add_qp_vector_gradient(center + base + kQP, q_on_spin_gradient, center_gradient + base + kQP);
      const double env_gradient = g(layout.center_l2_environment + c);
      for (int k = 0; k < 5; ++k) {
        center_gradient[base + kQ + k] += env_gradient * local_q[k];
        local_q_gradient[k] += env_gradient * center[base + kQ + k];
      }
    }
    if (layout.edge_l0_dot2 >= 0) {
      direct_gradient[channels + c] += g(layout.edge_l0_dot2 + c);
    }
    if (layout.density_l0_self >= 0) {
      const double m2_gradient = 2.0 * g(layout.density_l0_self + c);
      for (int d = 0; d < 3; ++d) m_gradient[d] += m2_gradient * m[d];
      if (layout.density_l1_longitudinal_self >= 0) {
        center_gradient[base + kL] +=
            2.0 * g(layout.density_l1_longitudinal_self + c) * center[base + kL];
        const double x_gradient = 2.0 * g(layout.density_l1_axial_self + c);
        const double t_gradient = 2.0 * g(layout.density_l1_stf_self + c);
        for (int d = 0; d < 3; ++d) {
          center_gradient[base + kX + d] += x_gradient * center[base + kX + d];
        }
        for (int k = 0; k < 5; ++k) {
          center_gradient[base + kT + k] += t_gradient * center[base + kT + k];
        }
      }
      if (layout.density_l1_product_self >= 0) {
        const double output_gradient = g(layout.density_l1_product_self + c);
        center_gradient[base + kL] +=
            (2.0 / 3.0) * output_gradient * center[base + kL];
        for (int d = 0; d < 3; ++d) {
          center_gradient[base + kX + d] +=
              output_gradient * center[base + kX + d];
        }
        for (int k = 0; k < 5; ++k) {
          center_gradient[base + kT + k] +=
              2.0 * output_gradient * center[base + kT + k];
        }
      }
      if (layout.density_l2_product_self >= 0) {
        const double output_gradient =
            2.0 * g(layout.density_l2_product_self + c);
        for (int k = 0; k < 15; ++k) {
          center_gradient[base + kQP + k] +=
              output_gradient * center[base + kQP + k];
        }
      }
      add_dot_gradient(
          m, center + base + kDM, g(layout.density_l0_dot_response + c),
          m_gradient, center_gradient + base + kDM);
    }
    if (layout.edge_l0_moment_gate >= 0) {
      direct_gradient[2 * channels + c] += g(layout.edge_l0_moment_gate + c);
    }
  }

  if (layout.correlation_same_edge >= 0) {
    const int offset = same_offset(channels);
    for (int left = 0; left < channels; ++left) {
      for (int right = left; right < channels; ++right) {
        const int pair = pair_index(channels, left, right);
        const double distinct_gradient = g(layout.correlation_distinct_neighbor + pair);
        center_gradient[offset + pair] +=
            g(layout.correlation_same_edge + pair) - distinct_gradient;
        first_dot_gradient[left] += distinct_gradient * first_dot[right];
        first_dot_gradient[right] += distinct_gradient * first_dot[left];
      }
    }
  }
  for (int c = 0; c < channels; ++c) {
    add_dot_gradient(
        si, center + channel_offset(c) + kM, first_dot_gradient[c],
        si_gradient, center_gradient + channel_offset(c) + kM);
  }
  double local_q_si_gradient[3] = {};
  add_stf5_outer_gradient(si, si, local_q_gradient, si_gradient, local_q_si_gradient);
  for (int d = 0; d < 3; ++d) si_gradient[d] += local_q_si_gradient[d];

  for (int row = 0; row < channels; ++row) {
    if (layout.edge_l11_axial >= 0) {
      double m0[3], p0[3], p1[3], x0[3], l0[1], t0[5];
      project_density<3>(center, projection, channels, kM, 0, row, m0);
      project_density<3>(center, projection, channels, kP, 0, row, p0);
      project_density<3>(center, projection, channels, kP, 1, row, p1);
      project_density<1>(center, projection, channels, kL, 0, row, l0);
      project_density<3>(center, projection, channels, kX, 0, row, x0);
      project_density<5>(center, projection, channels, kT, 0, row, t0);

      double p_axis[3], w0[3];
      cross3(p0, p1, p_axis);
      cross3(si, m0, w0);
      double p_axis_gradient[3] = {}, w0_gradient[3] = {};
      if (layout.coupling_l11_axial >= 0) {
        add_dot_gradient(
            p_axis, w0, g(layout.coupling_l11_axial + row),
            p_axis_gradient, w0_gradient);
      }
      double p0_gradient[3] = {}, p1_gradient[3] = {}, m0_gradient[3] = {};
      add_cross_gradient(si, m0, w0_gradient, si_gradient, m0_gradient);

      double axis[3], axis_gradient[3] = {};
      l11_axis(si, l0[0], x0, t0, axis);
      add_dot_gradient(p0, axis, g(layout.edge_l11_axial + row), p0_gradient, axis_gradient);
      double l0_gradient = 0.0, x0_gradient[3] = {}, t0_gradient[5] = {};
      add_l11_axis_gradient(
          si, l0[0], x0, t0, axis_gradient,
          si_gradient, l0_gradient, x0_gradient, t0_gradient);

      if (layout.coupling_l11_dot_response >= 0) {
        double dm0[3];
        project_density<3>(center, projection, channels, kDM, 0, row, dm0);
        double dw0[3], dw0_gradient[3] = {};
        cross3(si, dm0, dw0);
        add_dot_gradient(
            p_axis, dw0, g(layout.coupling_l11_dot_response + row),
            p_axis_gradient, dw0_gradient);
        double dm0_gradient[3] = {};
        add_cross_gradient(si, dm0, dw0_gradient, si_gradient, dm0_gradient);
        add_projected_density_gradient<3>(projection, channels, kDM, 0, row, dm0_gradient, center_gradient);
      }
      if (layout.coupling_l111_p_m_x >= 0) {
        double m1[3], x2[3];
        project_density<3>(center, projection, channels, kM, 1, row, m1);
        project_density<3>(center, projection, channels, kX, 2, row, x2);
        double cross_value[3], cross_gradient[3] = {};
        cross3(m1, x2, cross_value);
        add_dot_gradient(
            p0, cross_value, g(layout.coupling_l111_p_m_x + row),
            p0_gradient, cross_gradient);
        double m1_gradient[3] = {}, x2_gradient[3] = {};
        add_cross_gradient(m1, x2, cross_gradient, m1_gradient, x2_gradient);
        add_projected_density_gradient<3>(projection, channels, kM, 1, row, m1_gradient, center_gradient);
        add_projected_density_gradient<3>(projection, channels, kX, 2, row, x2_gradient, center_gradient);
      }
      if (layout.coupling_l111_bulk >= 0) {
        double p2[3], x3[3], cross12[3];
        project_density<3>(center, projection, channels, kP, 2, row, p2);
        project_density<3>(center, projection, channels, kX, 3, row, x3);
        cross3(p1, p2, cross12);
        const double a = dot3(p0, cross12);
        const double b = dot3(si, x3);
        const double output_gradient = g(layout.coupling_l111_bulk + row);
        double cross12_gradient[3] = {}, p2_gradient[3] = {}, x3_gradient[3] = {};
        add_dot_gradient(
            p0, cross12, -output_gradient * b,
            p0_gradient, cross12_gradient);
        add_dot_gradient(
            si, x3, -output_gradient * a,
            si_gradient, x3_gradient);
        add_cross_gradient(p1, p2, cross12_gradient, p1_gradient, p2_gradient);
        add_projected_density_gradient<3>(projection, channels, kP, 2, row, p2_gradient, center_gradient);
        add_projected_density_gradient<3>(projection, channels, kX, 3, row, x3_gradient, center_gradient);
      }
      add_cross_gradient(p0, p1, p_axis_gradient, p0_gradient, p1_gradient);
      add_projected_density_gradient<3>(projection, channels, kM, 0, row, m0_gradient, center_gradient);
      add_projected_density_gradient<3>(projection, channels, kP, 0, row, p0_gradient, center_gradient);
      add_projected_density_gradient<3>(projection, channels, kP, 1, row, p1_gradient, center_gradient);
      add_projected_density_gradient<1>(projection, channels, kL, 0, row, &l0_gradient, center_gradient);
      add_projected_density_gradient<3>(projection, channels, kX, 0, row, x0_gradient, center_gradient);
      add_projected_density_gradient<5>(projection, channels, kT, 0, row, t0_gradient, center_gradient);
    }

    if (layout.edge_l22_axial >= 0) {
      double m0[3], p0[3], p1[3], x0[3], q0[5], q1[5], qp0[15];
      project_density<3>(center, projection, channels, kM, 0, row, m0);
      project_density<3>(center, projection, channels, kP, 0, row, p0);
      project_density<3>(center, projection, channels, kP, 1, row, p1);
      project_density<3>(center, projection, channels, kX, 0, row, x0);
      project_density<5>(center, projection, channels, kQ, 0, row, q0);
      project_density<5>(center, projection, channels, kQ, 1, row, q1);
      project_density<15>(center, projection, channels, kQP, 0, row, qp0);
      double q_axis[3], w0[3];
      axial_commutator(q0, q1, q_axis);
      cross3(si, m0, w0);
      double q_axis_gradient[3] = {}, w0_gradient[3] = {};
      if (layout.coupling_l22_axial >= 0) {
        add_dot_gradient(
            q_axis, w0, g(layout.coupling_l22_axial + row),
            q_axis_gradient, w0_gradient);
      }
      double q0_gradient[5] = {}, q1_gradient[5] = {}, m0_gradient[3] = {};
      add_cross_gradient(si, m0, w0_gradient, si_gradient, m0_gradient);
      double qp0_gradient[15] = {};
      add_l22_scalar_gradient(
          si, m0, q0, qp0, g(layout.edge_l22_axial + row),
          si_gradient, m0_gradient, q0_gradient, qp0_gradient);

      double p0_gradient[3] = {}, p1_gradient[3] = {}, x0_gradient[3] = {};
      if (layout.coupling_l22_dot_response >= 0) {
        double dm0[3], dw0[3];
        project_density<3>(center, projection, channels, kDM, 0, row, dm0);
        cross3(si, dm0, dw0);
        double dw0_gradient[3] = {};
        add_dot_gradient(
            q_axis, dw0, g(layout.coupling_l22_dot_response + row),
            q_axis_gradient, dw0_gradient);
        double dm0_gradient[3] = {};
        add_cross_gradient(si, dm0, dw0_gradient, si_gradient, dm0_gradient);
        add_projected_density_gradient<3>(projection, channels, kDM, 0, row, dm0_gradient, center_gradient);
      }
      if (layout.coupling_l111_p_qs_x >= 0) {
        double qp1[15], x2[3], qs1[3];
        project_density<15>(center, projection, channels, kQP, 1, row, qp1);
        project_density<3>(center, projection, channels, kX, 2, row, x2);
        qp_vector(qp1, qs1);
        double cross_value[3], cross_gradient[3] = {}, qs1_gradient[3] = {}, x2_gradient[3] = {};
        cross3(qs1, x2, cross_value);
        add_dot_gradient(
            p0, cross_value, g(layout.coupling_l111_p_qs_x + row),
            p0_gradient, cross_gradient);
        add_cross_gradient(qs1, x2, cross_gradient, qs1_gradient, x2_gradient);
        double qp1_gradient[15] = {};
        add_qp_vector_gradient(qp1, qs1_gradient, qp1_gradient);
        add_projected_density_gradient<15>(projection, channels, kQP, 1, row, qp1_gradient, center_gradient);
        add_projected_density_gradient<3>(projection, channels, kX, 2, row, x2_gradient, center_gradient);
      }
      if (layout.coupling_l112_edge_response >= 0) {
        double q_on_p1[3], mixed_axis[3];
        stf5_matvec(q0, p1, q_on_p1);
        cross3(p1, q_on_p1, mixed_axis);
        const double a = dot3(p0, mixed_axis);
        const double b = dot3(si, x0);
        const double output_gradient =
            g(layout.coupling_l112_edge_response + row);
        double mixed_gradient[3] = {}, q_on_p1_gradient[3] = {};
        add_dot_gradient(
            p0, mixed_axis, -output_gradient * b,
            p0_gradient, mixed_gradient);
        add_dot_gradient(
            si, x0, -output_gradient * a,
            si_gradient, x0_gradient);
        add_cross_gradient(p1, q_on_p1, mixed_gradient, p1_gradient, q_on_p1_gradient);
        add_stf5_matvec_gradient(q0, p1, q_on_p1_gradient, q0_gradient, p1_gradient);

      }
      add_axial_commutator_gradient(q0, q1, q_axis_gradient, q0_gradient, q1_gradient);
      add_projected_density_gradient<3>(projection, channels, kM, 0, row, m0_gradient, center_gradient);
      add_projected_density_gradient<3>(projection, channels, kP, 0, row, p0_gradient, center_gradient);
      add_projected_density_gradient<3>(projection, channels, kP, 1, row, p1_gradient, center_gradient);
      add_projected_density_gradient<3>(projection, channels, kX, 0, row, x0_gradient, center_gradient);
      add_projected_density_gradient<5>(projection, channels, kQ, 0, row, q0_gradient, center_gradient);
      add_projected_density_gradient<5>(projection, channels, kQ, 1, row, q1_gradient, center_gradient);
      add_projected_density_gradient<15>(projection, channels, kQP, 0, row, qp0_gradient, center_gradient);
    }
  }
}

inline bool uses_dense_edge_primitives(const SpinPolynomialLayout& layout) {
  return layout.edge_l0_moment_gate >= 0 &&
      layout.edge_l11_axial >= 0 &&
      layout.edge_l2_pair >= 0;
}

template <bool Dense>
inline void add_edge_state(
    const SpinPolynomialLayout& layout,
    const double* rhat, const double* si, const double* sj,
    const double* weights, double* center, double* direct) {
  const int channels = layout.channels;
  const bool has_order2 = Dense || layout.density_l0_self >= 0;
  const bool has_order3 = Dense || layout.edge_l0_moment_gate >= 0;
  const bool needs_l1 = Dense || layout.density_l1_product_self >= 0 ||
      layout.edge_l11_axial >= 0;
  const bool needs_p = Dense || layout.edge_l11_axial >= 0;
  const bool needs_q = Dense || layout.edge_l2_pair >= 0;
  const bool needs_qp = Dense || needs_q || layout.density_l2_product_self >= 0;
  const double si2 = has_order3 ? dot3(si, si) : 0.0;
  const double sj2 = dot3(sj, sj);
  const double dot = has_order2 ? dot3(si, sj) : 0.0;
  const double longitudinal = needs_l1 ? dot3(rhat, sj) : 0.0;
  double axial[3] = {}, qrr[5] = {}, edge_stf[5] = {};
  if (needs_l1) {
    cross3(rhat, sj, axial);
    stf5_outer(rhat, sj, edge_stf);
  }
  if (needs_q || needs_qp) stf5_outer(rhat, rhat, qrr);
  for (int c = 0; c < channels; ++c) {
    const int base = channel_offset(c);
    const double weight = weights[c];
    for (int d = 0; d < 3; ++d) {
      center[base + kM + d] = center[base + kM + d] + weight * sj[d];
      if (needs_p) {
        center[base + kP + d] = center[base + kP + d] + weight * rhat[d];
      }
      if (has_order2) {
        center[base + kDM + d] = center[base + kDM + d] + weight * dot * sj[d];
      }
      if (needs_l1) {
        center[base + kX + d] = center[base + kX + d] + weight * axial[d];
      }
    }
    if (needs_l1) {
      center[base + kL] = center[base + kL] + weight * longitudinal;
    }
    if (needs_l1 || needs_q || needs_qp) {
      for (int k = 0; k < 5; ++k) {
        if (needs_l1) {
          center[base + kT + k] = center[base + kT + k] + weight * edge_stf[k];
        }
        if (needs_q) {
          center[base + kQ + k] = center[base + kQ + k] + weight * qrr[k];
        }
        if (needs_qp) {
          for (int d = 0; d < 3; ++d) {
            center[base + kQP + 3 * k + d] =
                center[base + kQP + 3 * k + d] + weight * qrr[k] * sj[d];
          }
        }
      }
    }
    direct[c] = direct[c] + weight * sj2;
    if (has_order2) {
      direct[channels + c] = direct[channels + c] + weight * dot * dot;
    }
    if (has_order3) {
      direct[2 * channels + c] =
          direct[2 * channels + c] + weight * dot * (si2 + sj2);
    }
  }
  if (layout.correlation_same_edge >= 0) {
    const int offset = same_offset(channels);
    for (int left = 0; left < channels; ++left) {
      for (int right = left; right < channels; ++right) {
        center[offset + pair_index(channels, left, right)] =
            center[offset + pair_index(channels, left, right)] +
            weights[left] * weights[right] * dot * dot;
      }
    }
  }
}

template <bool Dense>
inline void accumulate_edge_gradients(
    const SpinPolynomialLayout& layout,
    const double* rhat, const double* si, const double* sj,
    const double* weights, const double* center_gradient, const double* direct_gradient,
    double* rhat_gradient, double* si_gradient, double* sj_gradient, double* weight_gradient) {
  const int channels = layout.channels;
  const bool has_order2 = Dense || layout.density_l0_self >= 0;
  const bool has_order3 = Dense || layout.edge_l0_moment_gate >= 0;
  const bool needs_l1 = Dense || layout.density_l1_product_self >= 0 ||
      layout.edge_l11_axial >= 0;
  const bool needs_p = Dense || layout.edge_l11_axial >= 0;
  const bool needs_q = Dense || layout.edge_l2_pair >= 0;
  const bool needs_qp = Dense || needs_q || layout.density_l2_product_self >= 0;
  const double si2 = has_order3 ? dot3(si, si) : 0.0;
  const double sj2 = dot3(sj, sj);
  const double spin_dot = has_order2 ? dot3(si, sj) : 0.0;
  const double longitudinal = needs_l1 ? dot3(rhat, sj) : 0.0;
  double axial[3] = {}, qrr[5] = {}, edge_stf[5] = {};
  if (needs_l1) {
    cross3(rhat, sj, axial);
    stf5_outer(rhat, sj, edge_stf);
  }
  if (needs_q || needs_qp) stf5_outer(rhat, rhat, qrr);
  double si2_gradient = 0.0, sj2_gradient = 0.0, dot_gradient = 0.0;
  double longitudinal_gradient = 0.0, axial_gradient[3] = {};
  double qrr_gradient[5] = {}, edge_stf_gradient[5] = {};
  for (int c = 0; c < channels; ++c) {
    const int base = channel_offset(c);
    const double weight = weights[c];
    double weight_pull = weight_gradient[c];
    for (int d = 0; d < 3; ++d) {
      const double gm = center_gradient[base + kM + d];
      weight_pull += gm * sj[d];
      sj_gradient[d] += weight * gm;
      if (needs_p) {
        const double gp = center_gradient[base + kP + d];
        weight_pull += gp * rhat[d];
        rhat_gradient[d] += weight * gp;
      }
      if (has_order2) {
        const double gdm = center_gradient[base + kDM + d];
        weight_pull += gdm * spin_dot * sj[d];
        dot_gradient += weight * gdm * sj[d];
        sj_gradient[d] += weight * spin_dot * gdm;
      }
      if (needs_l1) {
        const double gx = center_gradient[base + kX + d];
        weight_pull += gx * axial[d];
        axial_gradient[d] += weight * gx;
      }
    }
    if (needs_l1) {
      const double gl = center_gradient[base + kL];
      weight_pull += gl * longitudinal;
      longitudinal_gradient += weight * gl;
    }
    if (needs_l1 || needs_q || needs_qp) {
      for (int k = 0; k < 5; ++k) {
        double edge_stf_pull = edge_stf_gradient[k];
        double qrr_pull = qrr_gradient[k];
        if (needs_l1) {
          const double gt = center_gradient[base + kT + k];
          weight_pull += gt * edge_stf[k];
          edge_stf_pull += weight * gt;
        }
        if (needs_q) {
          const double gq = center_gradient[base + kQ + k];
          weight_pull += gq * qrr[k];
          qrr_pull += weight * gq;
        }
        if (needs_qp) {
          const double* gqp = center_gradient + base + kQP + 3 * k;
          const double qp_spin_pull =
              gqp[0] * sj[0] + gqp[1] * sj[1] + gqp[2] * sj[2];
          weight_pull += qrr[k] * qp_spin_pull;
          qrr_pull += weight * qp_spin_pull;
          const double weighted_qrr = weight * qrr[k];
          for (int d = 0; d < 3; ++d) {
            sj_gradient[d] += weighted_qrr * gqp[d];
          }
        }
        if (needs_l1) edge_stf_gradient[k] = edge_stf_pull;
        if (needs_q || needs_qp) qrr_gradient[k] = qrr_pull;
      }
    }
    const double gd0 = direct_gradient[c];
    weight_pull += gd0 * sj2;
    sj2_gradient += gd0 * weight;
    if (has_order2) {
      const double gd1 = direct_gradient[channels + c];
      weight_pull += gd1 * spin_dot * spin_dot;
      dot_gradient += 2.0 * gd1 * weight * spin_dot;
    }
    if (has_order3) {
      const double gd2 = direct_gradient[2 * channels + c];
      weight_pull += gd2 * spin_dot * (si2 + sj2);
      dot_gradient += gd2 * weight * (si2 + sj2);
      si2_gradient += gd2 * weight * spin_dot;
      sj2_gradient += gd2 * weight * spin_dot;
    }
    weight_gradient[c] = weight_pull;
  }
  if (layout.correlation_same_edge >= 0) {
    const int offset = same_offset(channels);
    for (int left = 0; left < channels; ++left) {
      for (int right = left; right < channels; ++right) {
        const double pair_gradient =
            center_gradient[offset + pair_index(channels, left, right)];
        const double dot2 = spin_dot * spin_dot;
        weight_gradient[left] += pair_gradient * weights[right] * dot2;
        weight_gradient[right] += pair_gradient * weights[left] * dot2;
        dot_gradient +=
            2.0 * pair_gradient * weights[left] * weights[right] * spin_dot;
      }
    }
  }
  for (int d = 0; d < 3; ++d) {
    si_gradient[d] += 2.0 * si2_gradient * si[d] + dot_gradient * sj[d];
    sj_gradient[d] += 2.0 * sj2_gradient * sj[d] + dot_gradient * si[d];
  }
  if (needs_l1) {
    add_dot_gradient(rhat, sj, longitudinal_gradient, rhat_gradient, sj_gradient);
    add_cross_gradient(rhat, sj, axial_gradient, rhat_gradient, sj_gradient);
  }
  if (needs_q || needs_qp) {
    double qrr_right_gradient[3] = {};
    add_stf5_outer_gradient(
        rhat, rhat, qrr_gradient, rhat_gradient, qrr_right_gradient);
    for (int d = 0; d < 3; ++d) rhat_gradient[d] += qrr_right_gradient[d];
  }
  if (needs_l1) {
    add_stf5_outer_gradient(
        rhat, sj, edge_stf_gradient, rhat_gradient, sj_gradient);
  }
}

}  // namespace detail

struct CenterScratch {
  std::vector<double> state;
  std::vector<double> state_gradient;
  std::vector<EdgeGradient> edge_gradients;
};

inline int state_size(const SpinPolynomialLayout& layout) {
  return layout.moment_count + 3 * layout.channels;
}

inline void build_state(
    const SpinPolynomialLayout& layout, const double* center_spin,
    const double* all_spins_aos3, const std::vector<Edge>& edges,
    std::vector<double>& state) {
  state.assign(static_cast<std::size_t>(state_size(layout)), 0.0);
  double* direct = state.data() + layout.moment_count;
  const bool dense = detail::uses_dense_edge_primitives(layout);
  for (const Edge& edge : edges) {
    const double inv_distance = 1.0 / edge.distance;
    double rhat[3] = {
        edge.displacement[0] * inv_distance,
        edge.displacement[1] * inv_distance,
        edge.displacement[2] * inv_distance};
    const double* neighbor_spin = all_spins_aos3 + static_cast<std::size_t>(edge.neighbor) * 3;
    if (dense) {
      detail::add_edge_state<true>(
          layout, rhat, center_spin, neighbor_spin, edge.weights,
          state.data(), direct);
    } else {
      detail::add_edge_state<false>(
          layout, rhat, center_spin, neighbor_spin, edge.weights,
          state.data(), direct);
    }
  }
}

inline void descriptors(
    const SpinPolynomialLayout& layout, const double* center_spin,
    const double* projection, const std::vector<double>& state,
    double* output) {
  const double* direct = state.data() + layout.moment_count;
  detail::contract(
      layout, state.data(), direct, center_spin, projection,
      [&](int index, double value) { if (index >= 0) output[index] = value; });
}

inline void gradients(
    const SpinPolynomialLayout& layout, const double* center_spin,
    const double* all_spins_aos3, const double* projection,
    const std::vector<Edge>& edges, const double* descriptor_gradient,
    CenterScratch& scratch, double center_spin_gradient[3]) {
  scratch.state_gradient.assign(scratch.state.size(), 0.0);
  detail::accumulate_descriptor_gradients(
      layout, scratch.state.data(), center_spin, projection, descriptor_gradient,
      scratch.state_gradient.data(), scratch.state_gradient.data() + layout.moment_count,
      center_spin_gradient);
  scratch.edge_gradients.resize(edges.size());
  const bool dense = detail::uses_dense_edge_primitives(layout);
  for (std::size_t edge_index = 0; edge_index < edges.size(); ++edge_index) {
    const Edge& edge = edges[edge_index];
    const double inv_distance = 1.0 / edge.distance;
    double rhat[3] = {
        edge.displacement[0] * inv_distance,
        edge.displacement[1] * inv_distance,
        edge.displacement[2] * inv_distance};
    const double* neighbor_spin =
        all_spins_aos3 + static_cast<std::size_t>(edge.neighbor) * 3;
    double rhat_gradient[3] = {}, si_gradient[3] = {}, sj_gradient[3] = {};
    double weight_gradient[kMaxChannels] = {};
    if (dense) {
      detail::accumulate_edge_gradients<true>(
          layout, rhat, center_spin, neighbor_spin, edge.weights,
          scratch.state_gradient.data(),
          scratch.state_gradient.data() + layout.moment_count,
          rhat_gradient, si_gradient, sj_gradient, weight_gradient);
    } else {
      detail::accumulate_edge_gradients<false>(
          layout, rhat, center_spin, neighbor_spin, edge.weights,
          scratch.state_gradient.data(),
          scratch.state_gradient.data() + layout.moment_count,
          rhat_gradient, si_gradient, sj_gradient, weight_gradient);
    }
    double radial_gradient = 0.0;
    for (int c = 0; c < layout.channels; ++c) {
      radial_gradient += weight_gradient[c] * edge.derivatives[c];
    }
    const double tangential = detail::dot3(rhat, rhat_gradient);
    EdgeGradient& result = scratch.edge_gradients[edge_index];
    result.neighbor = edge.neighbor;
    for (int d = 0; d < 3; ++d) {
      result.displacement[d] = edge.displacement[d];
      result.position[d] =
          (rhat_gradient[d] - tangential * rhat[d]) * inv_distance +
          radial_gradient * rhat[d];
      result.center_spin[d] = si_gradient[d];
      result.neighbor_spin[d] = sj_gradient[d];
      center_spin_gradient[d] += result.center_spin[d];
    }
  }
}

}  // namespace nep_adapters::cpu_spin2
