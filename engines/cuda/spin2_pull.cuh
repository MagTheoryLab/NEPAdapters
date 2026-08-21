#pragma once

template <bool AtomicPull>
__device__ __forceinline__ void spin2_oc_accumulate_pull(
    float* target,
    float value) {
  if constexpr (AtomicPull) {
    atomicAdd(target, value);
  } else {
    *target += value;
  }
}

template <int C, int Width, bool AtomicPull = true>
__device__ __forceinline__ void spin2_oc_add_projected_pull(
    float* __restrict__ pull,
    const float* __restrict__ projection,
    int density_offset,
    int leg,
    int row,
    const float* __restrict__ projected_pull) {
  for (int source = 0; source < C; ++source) {
    const float scale = spin2_oc_mix(projection, C, leg, row, source);
    for (int k = 0; k < Width; ++k) {
      float* target =
          pull + spin2_oc_channel_offset(source) + density_offset + k;
      if constexpr (AtomicPull) {
        atomicAdd(target, scale * projected_pull[k]);
      } else {
        *target += scale * projected_pull[k];
      }
    }
  }
}

__device__ __forceinline__ void spin2_oc_reverse_mixed_chirality(
    const float* p0,
    const float* p1,
    const float* q0,
    float scale,
    float* gp0,
    float* gp1,
    float* gq0) {
  float q_on_p1[3], cross_value[3];
  spin2_oc_stf5_matvec(q0, p1, q_on_p1);
  spin2_cross3(p1, q_on_p1, cross_value);
  float cross_pull[3] = {
      scale * p0[0], scale * p0[1], scale * p0[2]};
  for (int d = 0; d < 3; ++d) {
    gp0[d] += scale * cross_value[d];
  }
  float gy[3] = {};
  spin2_add_cross_pull(p1, q_on_p1, cross_pull, gp1, gy);
  float matrix_pull[9];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      matrix_pull[3 * a + b] = gy[a] * p1[b];
    }
  }
  spin2_oc_add_full_matrix_pull_to_stf5(matrix_pull, gq0);
  float q_times_gy[3];
  spin2_oc_stf5_matvec(q0, gy, q_times_gy);
  for (int d = 0; d < 3; ++d) {
    gp1[d] += q_times_gy[d];
  }
}

__device__ __forceinline__ void spin2_oc_reverse_triple_product(
    const float* left,
    const float* middle,
    const float* right,
    float alpha,
    float* gleft,
    float* gmiddle,
    float* gright) {
  float cross_value[3], cross_pull[3];
  spin2_cross3(middle, right, cross_value);
  for (int d = 0; d < 3; ++d) {
    gleft[d] += alpha * cross_value[d];
    cross_pull[d] = alpha * left[d];
  }
  spin2_add_cross_pull(middle, right, cross_pull, gmiddle, gright);
}

__device__ __forceinline__ void spin2_oc_add_qp_vector_pull(
    const float* vector_pull,
    float* qp_pull) {
  for (int spin_component = 0; spin_component < 3; ++spin_component) {
    float matrix_pull[9] = {}, component_pull[5] = {};
    for (int a = 0; a < 3; ++a) {
      matrix_pull[3 * a + spin_component] = vector_pull[a];
    }
    spin2_oc_add_full_matrix_pull_to_stf5(matrix_pull, component_pull);
    for (int k = 0; k < 5; ++k) {
      qp_pull[3 * k + spin_component] += component_pull[k];
    }
  }
}

__device__ __forceinline__ void spin2_oc_reverse_l11_axis_from_canonical(
    const float* si,
    float longitudinal,
    const float* axial,
    const float* stf_first,
    const float* axis_pull,
    float* spin_pull,
    float* longitudinal_pull,
    float* axial_pull,
    float* stf_first_pull) {
  float matrix[9], matrix_on_pull[3];
  spin2_oc_expand_stf5(stf_first, matrix);
  for (int a = 0; a < 3; ++a) {
    matrix_on_pull[a] = matrix[3 * a] * axis_pull[0] +
        matrix[3 * a + 1] * axis_pull[1] +
        matrix[3 * a + 2] * axis_pull[2];
    spin_pull[a] +=
        (2.0f / 3.0f) * longitudinal * axis_pull[a] - matrix_on_pull[a];
  }
  longitudinal_pull[0] +=
      (2.0f / 3.0f) * spin2_dot3(axis_pull, si);
  float matrix_pull[9];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      matrix_pull[3 * a + b] = -axis_pull[a] * si[b];
    }
  }
  spin2_oc_add_full_matrix_pull_to_stf5(matrix_pull, stf_first_pull);
  float cross_pull[3] = {
      -0.5f * axis_pull[0],
      -0.5f * axis_pull[1],
      -0.5f * axis_pull[2]};
  spin2_add_cross_pull(axial, si, cross_pull, axial_pull, spin_pull);
}

__device__ __forceinline__ void spin2_oc_reverse_l22_scalar_from_canonical(
    const float* si,
    const float* moment,
    const float* projected_q,
    const float* projected_qp,
    float scale,
    float* spin_pull,
    float* moment_pull,
    float* q_pull,
    float* qp_pull) {
  float q_matrix[9], q_matrix_pull[9] = {};
  float q_on_spin[3], q_on_spin_pull[3] = {};
  spin2_oc_expand_stf5(projected_q, q_matrix);
  for (int a = 0; a < 3; ++a) {
    q_on_spin[a] = 0.0f;
    for (int b = 0; b < 3; ++b) {
      q_on_spin[a] += q_matrix[3 * a + b] * si[b];
    }
  }
  for (int spin_component = 0; spin_component < 3; ++spin_component) {
    float raw_second[9], raw_second_pull[9] = {};
    spin2_oc_qp_raw_matrix(
        moment, projected_qp, spin_component, raw_second);
    for (int a = 0; a < 3; ++a) {
      raw_second_pull[3 * spin_component + a] -=
          scale * q_on_spin[a];
      q_on_spin_pull[a] -=
          scale * raw_second[3 * spin_component + a];
      for (int b = 0; b < 3; ++b) {
        q_matrix_pull[3 * spin_component + a] +=
            scale * raw_second[3 * a + b] * si[b];
        raw_second_pull[3 * a + b] +=
            scale * q_matrix[3 * spin_component + a] * si[b];
        spin_pull[b] +=
            scale * q_matrix[3 * spin_component + a] *
            raw_second[3 * a + b];
      }
    }
    moment_pull[spin_component] +=
        (raw_second_pull[0] + raw_second_pull[4] +
         raw_second_pull[8]) / 3.0f;
    float component_pull[5] = {};
    spin2_oc_add_full_matrix_pull_to_stf5(
        raw_second_pull, component_pull);
    for (int k = 0; k < 5; ++k) {
      qp_pull[3 * k + spin_component] += component_pull[k];
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      q_matrix_pull[3 * a + b] += q_on_spin_pull[a] * si[b];
      spin_pull[b] += q_on_spin_pull[a] * q_matrix[3 * a + b];
    }
  }
  spin2_oc_add_full_matrix_pull_to_stf5(q_matrix_pull, q_pull);
}


__global__ void clear_spin2_oc_pulls(
    int value_count,
    float* __restrict__ pulls) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < value_count) pulls[index] = 0.0f;
}

__global__ void accumulate_spin2_oc_onsite_mforces(
    SpinPolynomialLayout layout,
    int atom_count,
    int atom_stride,
    int struct_dim,
    const int* __restrict__ types,
    const int* __restrict__ spin_dof_type_active,
    const double* __restrict__ spins_soa3,
    const float* __restrict__ fp,
    double* __restrict__ mforce_soa3) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count || spin_dof_type_active[types[atom]] == 0) return;
  const double scale = 2.0 * static_cast<double>(
      fp[atom + atom_stride * (struct_dim + layout.local_s2)]);
  for (int d = 0; d < 3; ++d) {
    mforce_soa3[d * atom_stride + atom] -=
        scale * spins_soa3[d * atom_stride + atom];
  }
}


template <int C, bool AtomicPull = true>
__device__ __noinline__ void build_spin2_oc_center_pull_row(
    SpinPolynomialLayout layout,
    int atom,
    int row,
    unsigned active_mask,
    int atom_stride,
    int struct_dim,
    const double* __restrict__ spins_soa3,
    const float* __restrict__ fp,
    const float* __restrict__ projection,
    const float* __restrict__ moments,
    float* __restrict__ pull,
    double* __restrict__ mforce_soa3) {
  const float* center = moments + atom * layout.moment_count;
  float si[3] = {
      static_cast<float>(spins_soa3[atom]),
      static_cast<float>(spins_soa3[atom_stride + atom]),
      static_cast<float>(spins_soa3[2 * atom_stride + atom])};
  float direct_spin_pull[3] = {};
#define NEP_SPIN2_OC_FP_DYNAMIC(index)                                  \
  fp[atom + atom_stride * (struct_dim + (index))]

  if (row == 0) {
    float first_dot[C], first_dot_pull[C] = {};
    float local_q[5];
    spin2_oc_stf5_outer(si, si, local_q);
    for (int c = 0; c < C; ++c) {
      const int base = spin2_oc_channel_offset(c);
      const float* moment = center + base + kSpin2OcM;
      first_dot[c] = spin2_dot3(si, moment);
      const float a_dot = NEP_SPIN2_OC_FP_DYNAMIC(layout.edge_l0_dot + c);
      first_dot_pull[c] += a_dot;
      if (layout.edge_l2_pair >= 0) {
        float qp_vector[3], qp_vector_pull[3];
        spin2_oc_qp_vector(center + base + kSpin2OcQP, qp_vector);
        const float a_pair =
            NEP_SPIN2_OC_FP_DYNAMIC(layout.edge_l2_pair + c);
        for (int d = 0; d < 3; ++d) {
          qp_vector_pull[d] = a_pair * si[d];
          direct_spin_pull[d] += a_pair * qp_vector[d];
        }
        float qp_pull[15] = {};
        spin2_oc_add_qp_vector_pull(qp_vector_pull, qp_pull);
        for (int k = 0; k < 15; ++k) {
          spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcQP + k, qp_pull[k]);
        }
        const float a_env =
            NEP_SPIN2_OC_FP_DYNAMIC(layout.center_l2_environment + c);
        for (int k = 0; k < 5; ++k) {
          spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcQ + k, a_env * local_q[k]);
        }
        float left_pull[3] = {}, right_pull[3] = {};
        spin2_oc_add_stf5_outer_pull(
            si, si, center + base + kSpin2OcQ, a_env,
            left_pull, right_pull);
        for (int d = 0; d < 3; ++d) {
          direct_spin_pull[d] += left_pull[d] + right_pull[d];
        }
      }
      if (layout.density_l0_self >= 0) {
        const float a_m =
            NEP_SPIN2_OC_FP_DYNAMIC(layout.density_l0_self + c);
        const float a_dm =
            NEP_SPIN2_OC_FP_DYNAMIC(layout.density_l0_dot_response + c);
        for (int d = 0; d < 3; ++d) {
          spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcM + d,
              2.0f * a_m * center[base + kSpin2OcM + d] +
              a_dm * center[base + kSpin2OcDM + d]);
          spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcDM + d,
              a_dm * center[base + kSpin2OcM + d]);
        }
        if (layout.density_l1_longitudinal_self >= 0) {
          spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcL,
              2.0f * NEP_SPIN2_OC_FP_DYNAMIC(
                  layout.density_l1_longitudinal_self + c) *
              center[base + kSpin2OcL]);
          const float a_x = NEP_SPIN2_OC_FP_DYNAMIC(
              layout.density_l1_axial_self + c);
          const float a_t = NEP_SPIN2_OC_FP_DYNAMIC(
              layout.density_l1_stf_self + c);
          for (int d = 0; d < 3; ++d) {
            spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcX + d,
                2.0f * a_x * center[base + kSpin2OcX + d]);
          }
          for (int k = 0; k < 5; ++k) {
            spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcT + k,
                2.0f * a_t * center[base + kSpin2OcT + k]);
          }
        }
        if (layout.density_l1_product_self >= 0) {
          const float a = NEP_SPIN2_OC_FP_DYNAMIC(
              layout.density_l1_product_self + c);
          spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcL,
              (2.0f / 3.0f) * a * center[base + kSpin2OcL]);
          for (int d = 0; d < 3; ++d) {
            spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcX + d,
                a * center[base + kSpin2OcX + d]);
          }
          for (int k = 0; k < 5; ++k) {
            spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcT + k,
                2.0f * a * center[base + kSpin2OcT + k]);
          }
        }
        if (layout.density_l2_product_self >= 0) {
          const float a = NEP_SPIN2_OC_FP_DYNAMIC(
              layout.density_l2_product_self + c);
          for (int k = 0; k < 15; ++k) {
            spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcQP + k,
                2.0f * a * center[base + kSpin2OcQP + k]);
          }
        }
      }
    }
    if (layout.correlation_same_edge >= 0) {
      const int same_offset = spin2_oc_same_offset(C);
      for (int left = 0; left < C; ++left) {
        for (int right = left; right < C; ++right) {
          const int pair = spin2_oc_pair_index(C, left, right);
          const float a_same = NEP_SPIN2_OC_FP_DYNAMIC(
              layout.correlation_same_edge + pair);
          const float a_distinct = NEP_SPIN2_OC_FP_DYNAMIC(
              layout.correlation_distinct_neighbor + pair);
          spin2_oc_accumulate_pull<AtomicPull>(pull + same_offset + pair, a_same - a_distinct);
          first_dot_pull[left] += a_distinct * first_dot[right];
          first_dot_pull[right] += a_distinct * first_dot[left];
        }
      }
    }
    for (int c = 0; c < C; ++c) {
      const int base = spin2_oc_channel_offset(c);
      for (int d = 0; d < 3; ++d) {
        spin2_oc_accumulate_pull<AtomicPull>(pull + base + kSpin2OcM + d,
            first_dot_pull[c] * si[d]);
        direct_spin_pull[d] +=
            first_dot_pull[c] * center[base + kSpin2OcM + d];
      }
    }
  }

  __syncwarp(active_mask);

  if (layout.coupling_l11_axial >= 0) {
    float m0[3], p0[3], p1[3], x0[3], l0[1], t0[5], w0[3];
    spin2_oc_project_density<3>(center, projection, C, kSpin2OcM, 0, row, m0);
    spin2_oc_project_density<3>(center, projection, C, kSpin2OcP, 0, row, p0);
    spin2_oc_project_density<3>(center, projection, C, kSpin2OcP, 1, row, p1);
    spin2_oc_project_density<1>(center, projection, C, kSpin2OcL, 0, row, l0);
    spin2_oc_project_density<3>(center, projection, C, kSpin2OcX, 0, row, x0);
    spin2_oc_project_density<5>(center, projection, C, kSpin2OcT, 0, row, t0);
    spin2_cross3(si, m0, w0);
    float gm0[3] = {}, gp0[3] = {}, gp1[3] = {}, gx0[3] = {};
    float gl0[1] = {}, gt0[5] = {}, gw0[3] = {};
    float p_axis[3], p_axis_pull[3] = {};
    spin2_cross3(p0, p1, p_axis);
    const float a_l11 = NEP_SPIN2_OC_FP_DYNAMIC(
        layout.coupling_l11_axial + row);
    for (int d = 0; d < 3; ++d) {
      p_axis_pull[d] += a_l11 * w0[d];
      gw0[d] += a_l11 * p_axis[d];
    }
    float l11_axis[3], l11_axis_pull[3];
    spin2_oc_l11_axis_from_canonical(si, l0[0], x0, t0, l11_axis);
    const float a_edge = NEP_SPIN2_OC_FP_DYNAMIC(layout.edge_l11_axial + row);
    for (int d = 0; d < 3; ++d) {
      gp0[d] += a_edge * l11_axis[d];
      l11_axis_pull[d] = a_edge * p0[d];
    }
    spin2_oc_reverse_l11_axis_from_canonical(
        si, l0[0], x0, t0, l11_axis_pull, direct_spin_pull,
        gl0, gx0, gt0);
    if (layout.coupling_l11_dot_response >= 0) {
      float dm0[3], dw0[3], m1[3], x2[3];
      float gdm0[3] = {}, gdw0[3] = {}, gm1[3] = {}, gx2[3] = {};
      spin2_oc_project_density<3>(center, projection, C, kSpin2OcDM, 0, row, dm0);
      spin2_oc_project_density<3>(center, projection, C, kSpin2OcM, 1, row, m1);
      spin2_oc_project_density<3>(center, projection, C, kSpin2OcX, 2, row, x2);
      spin2_cross3(si, dm0, dw0);
      const float a_dot = NEP_SPIN2_OC_FP_DYNAMIC(
          layout.coupling_l11_dot_response + row);
      for (int d = 0; d < 3; ++d) {
        p_axis_pull[d] += a_dot * dw0[d];
        gdw0[d] += a_dot * p_axis[d];
      }
      spin2_oc_reverse_triple_product(
          p0, m1, x2,
          NEP_SPIN2_OC_FP_DYNAMIC(layout.coupling_l111_p_m_x + row),
          gp0, gm1, gx2);
      spin2_add_cross_pull(si, dm0, gdw0, direct_spin_pull, gdm0);
      spin2_oc_add_projected_pull<C, 3, AtomicPull>(
          pull, projection, kSpin2OcDM, 0, row, gdm0);
      spin2_oc_add_projected_pull<C, 3, AtomicPull>(
          pull, projection, kSpin2OcM, 1, row, gm1);
      spin2_oc_add_projected_pull<C, 3, AtomicPull>(
          pull, projection, kSpin2OcX, 2, row, gx2);
    }
    if (layout.coupling_l111_bulk >= 0) {
      float p2[3], x3[3], gp2[3] = {}, gx3[3] = {};
      spin2_oc_project_density<3>(center, projection, C, kSpin2OcP, 2, row, p2);
      spin2_oc_project_density<3>(center, projection, C, kSpin2OcX, 3, row, x3);
      float cross12[3];
      spin2_cross3(p1, p2, cross12);
      const float chirality = spin2_dot3(p0, cross12);
      const float edge_response = -spin2_dot3(si, x3);
      const float alpha = NEP_SPIN2_OC_FP_DYNAMIC(
          layout.coupling_l111_bulk + row);
      spin2_oc_reverse_triple_product(
          p0, p1, p2, alpha * edge_response, gp0, gp1, gp2);
      for (int d = 0; d < 3; ++d) {
        gx3[d] -= alpha * chirality * si[d];
        direct_spin_pull[d] -= alpha * chirality * x3[d];
      }
      spin2_oc_add_projected_pull<C, 3, AtomicPull>(
          pull, projection, kSpin2OcP, 2, row, gp2);
      spin2_oc_add_projected_pull<C, 3, AtomicPull>(
          pull, projection, kSpin2OcX, 3, row, gx3);
    }
    spin2_add_cross_pull(p0, p1, p_axis_pull, gp0, gp1);
    spin2_add_cross_pull(si, m0, gw0, direct_spin_pull, gm0);
    spin2_oc_add_projected_pull<C, 3, AtomicPull>(pull, projection, kSpin2OcM, 0, row, gm0);
    spin2_oc_add_projected_pull<C, 3, AtomicPull>(pull, projection, kSpin2OcP, 0, row, gp0);
    spin2_oc_add_projected_pull<C, 3, AtomicPull>(pull, projection, kSpin2OcP, 1, row, gp1);
    spin2_oc_add_projected_pull<C, 1, AtomicPull>(pull, projection, kSpin2OcL, 0, row, gl0);
    spin2_oc_add_projected_pull<C, 3, AtomicPull>(pull, projection, kSpin2OcX, 0, row, gx0);
    spin2_oc_add_projected_pull<C, 5, AtomicPull>(pull, projection, kSpin2OcT, 0, row, gt0);
  }

  if (layout.coupling_l22_axial >= 0) {
    float m0[3], p0[3], p1[3], x0[3], q0[5], q1[5], qp0[15], w0[3];
    spin2_oc_project_density<3>(center, projection, C, kSpin2OcM, 0, row, m0);
    spin2_oc_project_density<3>(center, projection, C, kSpin2OcP, 0, row, p0);
    spin2_oc_project_density<3>(center, projection, C, kSpin2OcP, 1, row, p1);
    spin2_oc_project_density<3>(center, projection, C, kSpin2OcX, 0, row, x0);
    spin2_oc_project_density<5>(center, projection, C, kSpin2OcQ, 0, row, q0);
    spin2_oc_project_density<5>(center, projection, C, kSpin2OcQ, 1, row, q1);
    spin2_oc_project_density<15>(center, projection, C, kSpin2OcQP, 0, row, qp0);
    spin2_cross3(si, m0, w0);
    float gm0[3] = {}, gp0[3] = {}, gp1[3] = {}, gx0[3] = {};
    float gq0[5] = {}, gq1[5] = {}, gqp0[15] = {}, gw0[3] = {};
    float q_axis[3], q_axis_pull[3] = {};
    spin2_oc_axial_commutator(q0, q1, q_axis);
    const float a_l22 = NEP_SPIN2_OC_FP_DYNAMIC(
        layout.coupling_l22_axial + row);
    for (int d = 0; d < 3; ++d) {
      q_axis_pull[d] += a_l22 * w0[d];
      gw0[d] += a_l22 * q_axis[d];
    }
    spin2_oc_reverse_l22_scalar_from_canonical(
        si, m0, q0, qp0,
        NEP_SPIN2_OC_FP_DYNAMIC(layout.edge_l22_axial + row),
        direct_spin_pull, gm0, gq0, gqp0);
    if (layout.coupling_l22_dot_response >= 0) {
      float dm0[3], dw0[3], qp1[15], qs1[3], x2[3];
      float gdm0[3] = {}, gdw0[3] = {}, gqp1[15] = {}, gqs1[3] = {}, gx2[3] = {};
      spin2_oc_project_density<3>(center, projection, C, kSpin2OcDM, 0, row, dm0);
      spin2_oc_project_density<15>(center, projection, C, kSpin2OcQP, 1, row, qp1);
      spin2_oc_project_density<3>(center, projection, C, kSpin2OcX, 2, row, x2);
      spin2_cross3(si, dm0, dw0);
      spin2_oc_qp_vector(qp1, qs1);
      const float a_dot = NEP_SPIN2_OC_FP_DYNAMIC(
          layout.coupling_l22_dot_response + row);
      for (int d = 0; d < 3; ++d) {
        q_axis_pull[d] += a_dot * dw0[d];
        gdw0[d] += a_dot * q_axis[d];
      }
      spin2_oc_reverse_triple_product(
          p0, qs1, x2,
          NEP_SPIN2_OC_FP_DYNAMIC(layout.coupling_l111_p_qs_x + row),
          gp0, gqs1, gx2);
      float q_on_p1[3], mixed_axis[3];
      spin2_oc_stf5_matvec(q0, p1, q_on_p1);
      spin2_cross3(p1, q_on_p1, mixed_axis);
      const float mixed = spin2_dot3(p0, mixed_axis);
      const float edge_sum = -spin2_dot3(si, x0);
      const float a_edge = NEP_SPIN2_OC_FP_DYNAMIC(
          layout.coupling_l112_edge_response + row);
      spin2_oc_reverse_mixed_chirality(
          p0, p1, q0, a_edge * edge_sum, gp0, gp1, gq0);
      const float edge_pull = a_edge * mixed;
      for (int d = 0; d < 3; ++d) {
        gx0[d] -= edge_pull * si[d];
        direct_spin_pull[d] -= edge_pull * x0[d];
      }
      spin2_oc_add_qp_vector_pull(gqs1, gqp1);
      spin2_add_cross_pull(si, dm0, gdw0, direct_spin_pull, gdm0);
      spin2_oc_add_projected_pull<C, 3, AtomicPull>(pull, projection, kSpin2OcDM, 0, row, gdm0);
      spin2_oc_add_projected_pull<C, 15, AtomicPull>(pull, projection, kSpin2OcQP, 1, row, gqp1);
      spin2_oc_add_projected_pull<C, 3, AtomicPull>(pull, projection, kSpin2OcX, 2, row, gx2);
    }
    spin2_oc_add_commutator_axial_pull(q0, q1, q_axis_pull, gq0, gq1);
    spin2_add_cross_pull(si, m0, gw0, direct_spin_pull, gm0);
    spin2_oc_add_projected_pull<C, 3, AtomicPull>(pull, projection, kSpin2OcM, 0, row, gm0);
    spin2_oc_add_projected_pull<C, 3, AtomicPull>(pull, projection, kSpin2OcP, 0, row, gp0);
    spin2_oc_add_projected_pull<C, 3, AtomicPull>(pull, projection, kSpin2OcP, 1, row, gp1);
    spin2_oc_add_projected_pull<C, 3, AtomicPull>(pull, projection, kSpin2OcX, 0, row, gx0);
    spin2_oc_add_projected_pull<C, 5, AtomicPull>(pull, projection, kSpin2OcQ, 0, row, gq0);
    spin2_oc_add_projected_pull<C, 5, AtomicPull>(pull, projection, kSpin2OcQ, 1, row, gq1);
    spin2_oc_add_projected_pull<C, 15, AtomicPull>(pull, projection, kSpin2OcQP, 0, row, gqp0);
  }
  for (int d = 0; d < 3; ++d) {
    atomicAdd(mforce_soa3 + d * atom_stride + atom,
        -static_cast<double>(direct_spin_pull[d]));
  }
#undef NEP_SPIN2_OC_FP_DYNAMIC
}

template <int C>
__global__ void build_spin2_oc_center_pulls(
    SpinPolynomialLayout layout,
    int atom_count,
    int atom_stride,
    int struct_dim,
    const int* __restrict__ types,
    const int* __restrict__ spin_dof_type_active,
    const double* __restrict__ spins_soa3,
    const float* __restrict__ fp,
    const float* __restrict__ projection,
    const float* __restrict__ moments,
    float* __restrict__ pulls,
    double* __restrict__ mforce_soa3) {
  const int work = blockIdx.x * blockDim.x + threadIdx.x;
  const int atom = work / C;
  const int row = work - atom * C;
  const bool active =
      atom < atom_count && spin_dof_type_active[types[atom]] != 0;
  const unsigned active_mask = __ballot_sync(0xffffffffu, active);
  if (!active) return;
  build_spin2_oc_center_pull_row<C>(
      layout,
      atom,
      row,
      active_mask,
      atom_stride,
      struct_dim,
      spins_soa3,
      fp,
      projection,
      moments,
      pulls + atom * layout.moment_count,
      mforce_soa3);
}


template <int C, SpinVirialMode VirialMode, bool AccumulateSpinTransfer>
__global__ void accumulate_spin2_oc_native_forces(
    SpinPolynomialLayout layout,
    int atom_count,
    int atom_stride,
    int struct_dim,
    int num_types,
    int spin_basis_size,
    float spin_cutoff,
    SimulationBox box,
    const int* __restrict__ types,
    const int* __restrict__ spin_dof_type_active,
    const int* __restrict__ spin_env_type_active,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ spins_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const float* __restrict__ fp,
    const float* __restrict__ descriptor_coefficients,
    const float* __restrict__ pulls,
    int spin_coefficient_offset,
    double* __restrict__ force_soa3,
    double* __restrict__ mforce_soa3,
    double* __restrict__ virial_soa9,
    float* __restrict__ virial_float_soa9,
    float* __restrict__ spin_transfer_soa9) {
  constexpr bool AccumulateCenterVirial =
      VirialMode == SpinVirialMode::center_owned ||
      VirialMode == SpinVirialMode::center_and_neighbor_float_sink;
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count || spin_dof_type_active[types[atom]] == 0) return;
  const float* pull = pulls + atom * layout.moment_count;
  float center_force[3] = {};
  float center_mforce[3] = {};
  float center_virial[AccumulateCenterVirial ? 9 : 1] = {};
#define NEP_SPIN2_OC_FP_EDGE(index)                                     \
  fp[atom + atom_stride * (struct_dim + (index))]
  const int neighbor_count = nn_radial[atom];
  for (int slot = 0; slot < neighbor_count; ++slot) {
    const int neighbor = nl_radial[slot * atom_stride + atom];
    if (neighbor < 0 || spin_env_type_active[types[neighbor]] == 0) continue;
    float r[3], dist, si[3], sj[3], weights[C], derivatives[C];
    if (!load_spin_edge_f32<C>(
            atom, neighbor, atom_stride, num_types, spin_basis_size,
            spin_cutoff, box, types, positions_soa3, spins_soa3,
            descriptor_coefficients, spin_coefficient_offset, r, dist, si,
            sj, weights, derivatives)) continue;
    const float si2 = spin2_dot3(si, si);
    const float sj2 = spin2_dot3(sj, sj);
    const float dot = spin2_dot3(si, sj);
    const float longitudinal = spin2_dot3(r, sj);
    float axial[3], qrr[5], edge_stf[5];
    spin2_cross3(r, sj, axial);
    spin2_oc_stf5_outer(r, r, qrr);
    spin2_oc_stf5_outer(r, sj, edge_stf);
    float grad_weight[C] = {};
    float grad_r[3] = {}, grad_si[3] = {}, grad_sj[3] = {};
    float grad_q[5] = {};
    float grad_dot = 0.0f;
    float grad_longitudinal = 0.0f;
    for (int c = 0; c < C; ++c) {
      const int base = spin2_oc_channel_offset(c);
      const float w = weights[c];
      const float a_sj2 =
          NEP_SPIN2_OC_FP_EDGE(layout.edge_l0_neighbor_s2 + c);
      grad_weight[c] += a_sj2 * sj2;
      for (int d = 0; d < 3; ++d) grad_sj[d] += 2.0f * w * a_sj2 * sj[d];
      if (layout.edge_l0_dot2 >= 0) {
        const float a = NEP_SPIN2_OC_FP_EDGE(layout.edge_l0_dot2 + c);
        grad_weight[c] += a * dot * dot;
        grad_dot += 2.0f * w * a * dot;
      }
      if (layout.edge_l0_moment_gate >= 0) {
        const float a = NEP_SPIN2_OC_FP_EDGE(layout.edge_l0_moment_gate + c);
        const float gate = dot * (si2 + sj2);
        grad_weight[c] += a * gate;
        grad_dot += w * a * (si2 + sj2);
        for (int d = 0; d < 3; ++d) {
          grad_si[d] += 2.0f * w * a * dot * si[d];
          grad_sj[d] += 2.0f * w * a * dot * sj[d];
        }
      }
      const float* gm = pull + base + kSpin2OcM;
      const float* gp = pull + base + kSpin2OcP;
      const float gl = pull[base + kSpin2OcL];
      const float* gx = pull + base + kSpin2OcX;
      const float* gt = pull + base + kSpin2OcT;
      const float* gq = pull + base + kSpin2OcQ;
      const float* gqp = pull + base + kSpin2OcQP;
      const float* gdm = pull + base + kSpin2OcDM;
      grad_weight[c] += spin2_dot3(gm, sj) + spin2_dot3(gp, r) +
          gl * longitudinal + spin2_dot3(gx, axial) +
          spin2_oc_dotn<5>(gt, edge_stf) + spin2_oc_dotn<5>(gq, qrr) +
          dot * spin2_dot3(gdm, sj);
      for (int k = 0; k < 5; ++k) {
        for (int d = 0; d < 3; ++d) {
          grad_weight[c] += gqp[3 * k + d] * qrr[k] * sj[d];
        }
      }
      for (int d = 0; d < 3; ++d) {
        grad_sj[d] += w * gm[d];
        grad_r[d] += w * gp[d];
      }
      grad_longitudinal += w * gl;
      float gx_r[3] = {}, gx_s[3] = {};
      spin2_add_cross_pull(r, sj, gx, gx_r, gx_s);
      for (int d = 0; d < 3; ++d) {
        grad_r[d] += w * gx_r[d];
        grad_sj[d] += w * gx_s[d];
      }
      float gt_r[3] = {}, gt_s[3] = {};
      spin2_oc_add_stf5_outer_pull(r, sj, gt, w, gt_r, gt_s);
      for (int d = 0; d < 3; ++d) {
        grad_r[d] += gt_r[d];
        grad_sj[d] += gt_s[d];
      }
      for (int k = 0; k < 5; ++k) {
        grad_q[k] += w * gq[k];
        for (int d = 0; d < 3; ++d) {
          grad_q[k] += w * gqp[3 * k + d] * sj[d];
          grad_sj[d] += w * gqp[3 * k + d] * qrr[k];
        }
      }
      const float dm_contract = spin2_dot3(gdm, sj);
      grad_dot += w * dm_contract;
      for (int d = 0; d < 3; ++d) grad_sj[d] += w * dot * gdm[d];
    }
    if (layout.correlation_same_edge >= 0) {
      const int same_offset = spin2_oc_same_offset(C);
      const float dot2 = dot * dot;
      for (int left = 0; left < C; ++left) {
        for (int right = left; right < C; ++right) {
          const float a = pull[same_offset +
              spin2_oc_pair_index(C, left, right)];
          if (left == right) {
            grad_weight[left] += 2.0f * a * weights[left] * dot2;
          } else {
            grad_weight[left] += a * weights[right] * dot2;
            grad_weight[right] += a * weights[left] * dot2;
          }
          grad_dot += 2.0f * a * weights[left] * weights[right] * dot;
        }
      }
    }
    float qr_left[3] = {}, qr_right[3] = {};
    spin2_oc_add_stf5_outer_pull(r, r, grad_q, 1.0f, qr_left, qr_right);
    for (int d = 0; d < 3; ++d) {
      grad_r[d] += qr_left[d] + qr_right[d] + grad_longitudinal * sj[d];
      grad_sj[d] += grad_longitudinal * r[d];
      grad_si[d] += grad_dot * sj[d];
      grad_sj[d] += grad_dot * si[d];
    }
    float grad_dist = 0.0f;
    for (int c = 0; c < C; ++c) grad_dist += grad_weight[c] * derivatives[c];
    const float dot_r = spin2_dot3(grad_r, r);
    float grad_rij[3];
    for (int d = 0; d < 3; ++d) {
      grad_rij[d] = grad_dist * r[d] + (grad_r[d] - dot_r * r[d]) / dist;
      center_force[d] += grad_rij[d];
      center_mforce[d] -= grad_si[d];
      atomicAdd(force_soa3 + d * atom_stride + neighbor,
          -static_cast<double>(grad_rij[d]));
      atomicAdd(mforce_soa3 + d * atom_stride + neighbor,
          -static_cast<double>(grad_sj[d]));
    }
    if constexpr (AccumulateSpinTransfer) {
      atomic_add_spin_transfer_float(
          atom_stride, neighbor, r[0] * dist, r[1] * dist, r[2] * dist,
          grad_sj, spin_transfer_soa9);
    }
    if constexpr (VirialMode == SpinVirialMode::neighbor_owned) {
      atomic_add_per_atom_virial_double(
          atom_stride, neighbor, r[0] * dist, r[1] * dist, r[2] * dist,
          grad_rij[0], grad_rij[1], grad_rij[2], virial_soa9);
    } else if constexpr (
        VirialMode == SpinVirialMode::center_and_neighbor_float_sink) {
      atomic_add_per_atom_virial_float(
          atom_stride, neighbor, r[0] * dist, r[1] * dist, r[2] * dist,
          grad_rij[0], grad_rij[1], grad_rij[2], virial_float_soa9);
    }
    if constexpr (AccumulateCenterVirial) {
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          center_virial[virial_internal_component(3 * a + b)] -=
              r[a] * dist * grad_rij[b];
        }
      }
    }
  }
  for (int d = 0; d < 3; ++d) {
    atomicAdd(force_soa3 + d * atom_stride + atom,
        static_cast<double>(center_force[d]));
    atomicAdd(mforce_soa3 + d * atom_stride + atom,
        static_cast<double>(center_mforce[d]));
  }
  if constexpr (AccumulateCenterVirial) {
    for (int component = 0; component < 9; ++component) {
      atomicAdd(virial_soa9 + component * atom_stride + atom,
          static_cast<double>(center_virial[component]));
    }
  }
#undef NEP_SPIN2_OC_FP_EDGE
}
