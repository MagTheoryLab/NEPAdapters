#pragma once

// Cooperative steady-state force path. AtomsPerWarp centers share a warp and
// the remaining lanes process independent edges for each center. Lane ordering
// keeps equal edge slots adjacent across the center tile.
template <int C, int MomentCount, int AtomsPerWarp, bool InlinePull,
          SpinVirialMode VirialMode,
          bool FuseStructuralRadial = false>
__global__ __launch_bounds__(128, 3)
void accumulate_spin2_oc_native_forces_cooperative_o3(
    SpinPolynomialLayout layout,
    int atom_count,
    int atom_stride,
    int struct_dim,
    int num_types,
    int spin_basis_size,
    float spin_cutoff,
    const float* __restrict__ spin_cutoff_pair,
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
    const float* __restrict__ structural_radial_coefficients,
    const float* __restrict__ projection,
    const float* __restrict__ moments,
    const float* __restrict__ materialized_pulls,
    int spin_coefficient_offset,
    double* __restrict__ force_soa3,
    double* __restrict__ mforce_soa3,
    double* __restrict__ virial_soa9) {
  constexpr bool AccumulateCenterVirial =
      VirialMode == SpinVirialMode::center_owned;
  constexpr int EdgeLanes = 32 / AtomsPerWarp;
  constexpr int WarpsPerBlock = 4;
  static_assert(C <= EdgeLanes, "each O/C row needs one pull-builder lane");
  __shared__ float pull_storage[WarpsPerBlock][AtomsPerWarp]
      [InlinePull ? C : 1][InlinePull ? MomentCount : 1];
  extern __shared__ float structural_radial_pull_storage[];
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warp = global_thread >> 5;
  const int atom_lane = lane & (AtomsPerWarp - 1);
  const int edge_lane = lane / AtomsPerWarp;
  const int atom = warp * AtomsPerWarp + atom_lane;
  const bool valid = atom < atom_count;
  const bool spin_active =
      valid && spin_dof_type_active[types[atom]] != 0;
  const float* pull = valid
      ? materialized_pulls + atom * layout.moment_count
      : nullptr;
  if constexpr (InlinePull) {
    float* pull_banks = &pull_storage[warp_in_block][atom_lane][0][0];
    float* pull_row0 = pull_storage[warp_in_block][atom_lane][0];
    for (int component = edge_lane; component < C * MomentCount;
         component += EdgeLanes) {
      pull_banks[component] = 0.0f;
    }
    __syncwarp();
    const bool builds_pull = spin_active && edge_lane < C;
    const unsigned pull_builder_mask =
        __ballot_sync(0xffffffffu, builds_pull);
    if (builds_pull) {
      float* row_pull = pull_storage[warp_in_block][atom_lane][edge_lane];
      build_spin2_oc_center_pull_row<C, false>(
          layout,
          atom,
          edge_lane,
          pull_builder_mask,
          atom_stride,
          struct_dim,
          spins_soa3,
          fp,
          projection,
          moments,
          row_pull,
          mforce_soa3);
    }
    __syncwarp();
    for (int component = edge_lane; component < MomentCount;
         component += EdgeLanes) {
      float value = pull_row0[component];
#pragma unroll
      for (int row = 1; row < C; ++row) {
        value += pull_storage[warp_in_block][atom_lane][row][component];
      }
      pull_row0[component] = value;
    }
    __syncwarp();
    pull = pull_row0;
  }
  float* structural_radial_pulls = nullptr;
  if constexpr (FuseStructuralRadial) {
    constexpr int StructuralBasisCount = 9;
    constexpr int StructuralTypeCapacity = 2;
    constexpr int StructuralPullCount =
        StructuralBasisCount * StructuralTypeCapacity;
    structural_radial_pulls = structural_radial_pull_storage +
        (warp_in_block * AtomsPerWarp + atom_lane) * StructuralPullCount;
    for (int flat = edge_lane; flat < StructuralPullCount;
         flat += EdgeLanes) {
      const int neighbor_type = flat / StructuralBasisCount;
      const int k = flat - neighbor_type * StructuralBasisCount;
      float value = 0.0f;
      if (valid && neighbor_type < num_types) {
        constexpr int StructuralNCount = 5;
        constexpr int StructuralRadialBasisCount =
            StructuralNCount * StructuralBasisCount;
        const int type_pair = types[atom] * num_types + neighbor_type;
#pragma unroll
        for (int n = 0; n < StructuralNCount; ++n) {
          value += fp[atom + atom_stride * n] *
              structural_radial_coefficients[
                  type_pair * StructuralRadialBasisCount +
                  n * StructuralBasisCount + k];
        }
      }
      structural_radial_pulls[flat] = value;
    }
    __syncwarp();
  }
  float center_force[3] = {};
  float center_mforce[3] = {};
  float center_virial[AccumulateCenterVirial ? 9 : 1] = {};
  const int neighbor_count =
      (FuseStructuralRadial ? valid : spin_active) ? nn_radial[atom] : 0;
#define NEP_SPIN2_OC_COOP_FP(index) \
  fp[atom + atom_stride * (struct_dim + (index))]
  for (int slot = edge_lane; slot < neighbor_count; slot += EdgeLanes) {
    const int neighbor = nl_radial[slot * atom_stride + atom];
    if (neighbor < 0) {
      continue;
    }
    const bool spin_neighbor_active =
        spin_env_type_active[types[neighbor]] != 0;
    if (!FuseStructuralRadial && !spin_neighbor_active) continue;
    float r[3], dist, si[3], sj[3], weights[C], weight_derivatives[C];
    float structural_radial_scale = 0.0f;
    if (!load_spin_edge_f32<C, true, FuseStructuralRadial>(
            atom,
            neighbor,
            atom_stride,
            num_types,
            spin_basis_size,
            spin_cutoff,
            spin_cutoff_pair,
            box,
            types,
            positions_soa3,
            spins_soa3,
            descriptor_coefficients,
            spin_coefficient_offset,
            r,
            dist,
            si,
            sj,
            weights,
            weight_derivatives,
            structural_radial_pulls,
            &structural_radial_scale)) {
      continue;
    }
    if constexpr (FuseStructuralRadial) {
      if (!spin_active || !spin_neighbor_active) {
        float radial_force[3];
#pragma unroll
        for (int d = 0; d < 3; ++d) {
          radial_force[d] = structural_radial_scale * r[d];
          center_force[d] += radial_force[d];
          atomicAdd(
              force_soa3 + d * atom_stride + neighbor,
              -static_cast<double>(radial_force[d]));
        }
        if constexpr (AccumulateCenterVirial) {
#pragma unroll
          for (int d = 0; d < 3; ++d) {
#pragma unroll
            for (int a = 0; a < 3; ++a) {
              center_virial[virial_internal_component(3 * a + d)] -=
                  r[a] * dist * radial_force[d];
            }
          }
        }
        continue;
      }
    }
    const float si2 = spin2_dot3(si, si);
    const float sj2 = spin2_dot3(sj, sj);
    const float dot = spin2_dot3(si, sj);
    const float longitudinal = spin2_dot3(r, sj);
    float axial[3], qrr[5], edge_stf[5];
    spin2_cross3(r, sj, axial);
    spin2_oc_stf5_outer(r, r, qrr);
    spin2_oc_stf5_outer(r, sj, edge_stf);

    float grad_weight[C] = {};
    float grad_r[3] = {};
    float grad_si[3] = {};
    float grad_sj[3] = {};
    float grad_q[5] = {};
    float weighted_m[3] = {};
    float weighted_p[3] = {};
    float weighted_x[3] = {};
    float weighted_t[5] = {};
    float weighted_q[5] = {};
    float weighted_qp[15] = {};
    float weighted_dm[3] = {};
    float grad_dot = 0.0f;
    float grad_longitudinal = 0.0f;

    for (int c = 0; c < C; ++c) {
      const int base = spin2_oc_channel_offset(c);
      const float w = weights[c];
      const float a_sj2 =
          NEP_SPIN2_OC_COOP_FP(layout.edge_l0_neighbor_s2 + c);
      const float a_dot2 =
          NEP_SPIN2_OC_COOP_FP(layout.edge_l0_dot2 + c);
      const float a_gate =
          NEP_SPIN2_OC_COOP_FP(layout.edge_l0_moment_gate + c);
      grad_weight[c] += a_sj2 * sj2 + a_dot2 * dot * dot;
      grad_dot += 2.0f * w * a_dot2 * dot;
      for (int d = 0; d < 3; ++d) {
        grad_sj[d] += 2.0f * w * a_sj2 * sj[d];
      }
      const float gate = dot * (si2 + sj2);
      grad_weight[c] += a_gate * gate;
      grad_dot += w * a_gate * (si2 + sj2);
      for (int d = 0; d < 3; ++d) {
        grad_si[d] += 2.0f * w * a_gate * dot * si[d];
        grad_sj[d] += 2.0f * w * a_gate * dot * sj[d];
      }
      const float* gm = pull + base + kSpin2OcM;
      const float* gp = pull + base + kSpin2OcP;
      const float gl = pull[base + kSpin2OcL];
      const float* gx = pull + base + kSpin2OcX;
      const float* gt = pull + base + kSpin2OcT;
      const float* gq = pull + base + kSpin2OcQ;
      const float* gqp = pull + base + kSpin2OcQP;
      const float* gdm = pull + base + kSpin2OcDM;
      const float* ga1 = layout.angular_l1_moment_offset >= 0
          ? pull + layout.angular_l1_moment_offset + 3 * c
          : nullptr;
      const float* ga2 = layout.angular_l2_moment_offset >= 0
          ? pull + layout.angular_l2_moment_offset + 5 * c
          : nullptr;
      grad_weight[c] += spin2_dot3(gm, sj) + spin2_dot3(gp, r) +
          gl * longitudinal + spin2_dot3(gx, axial) +
          spin2_oc_dotn<5>(gt, edge_stf) +
          spin2_oc_dotn<5>(gq, qrr) +
          dot * spin2_dot3(gdm, sj);
      for (int k = 0; k < 5; ++k) {
        for (int d = 0; d < 3; ++d) {
          grad_weight[c] += gqp[3 * k + d] * qrr[k] * sj[d];
        }
      }
      for (int d = 0; d < 3; ++d) {
        weighted_m[d] += w * gm[d];
        weighted_p[d] += w * gp[d];
        weighted_x[d] += w * gx[d];
        weighted_dm[d] += w * gdm[d];
        if (ga1 != nullptr) {
          grad_weight[c] += ga1[d] * dot * r[d];
          grad_dot += w * ga1[d] * r[d];
          grad_r[d] += w * dot * ga1[d];
        }
      }
      grad_longitudinal += w * gl;
      for (int k = 0; k < 5; ++k) {
        weighted_t[k] += w * gt[k];
        weighted_q[k] += w * gq[k];
        if (ga2 != nullptr) {
          grad_weight[c] += ga2[k] * dot * qrr[k];
          grad_dot += w * ga2[k] * qrr[k];
          grad_q[k] += w * dot * ga2[k];
        }
        for (int d = 0; d < 3; ++d) {
          weighted_qp[3 * k + d] += w * gqp[3 * k + d];
        }
      }
    }

    for (int d = 0; d < 3; ++d) {
      grad_sj[d] += weighted_m[d];
      grad_r[d] += weighted_p[d];
    }
    float gx_r[3] = {}, gx_s[3] = {};
    spin2_add_cross_pull(r, sj, weighted_x, gx_r, gx_s);
    for (int d = 0; d < 3; ++d) {
      grad_r[d] += gx_r[d];
      grad_sj[d] += gx_s[d];
    }
    float gt_r[3] = {}, gt_s[3] = {};
    spin2_oc_add_stf5_outer_pull(
        r, sj, weighted_t, 1.0f, gt_r, gt_s);
    for (int d = 0; d < 3; ++d) {
      grad_r[d] += gt_r[d];
      grad_sj[d] += gt_s[d];
    }
    for (int k = 0; k < 5; ++k) {
      grad_q[k] += weighted_q[k];
      for (int d = 0; d < 3; ++d) {
        grad_q[k] += weighted_qp[3 * k + d] * sj[d];
        grad_sj[d] += weighted_qp[3 * k + d] * qrr[k];
      }
    }
    grad_dot += spin2_dot3(weighted_dm, sj);
    for (int d = 0; d < 3; ++d) {
      grad_sj[d] += dot * weighted_dm[d];
    }

    const float dot2 = dot * dot;
    const int same_offset = spin2_oc_same_offset(C);
    for (int left = 0; left < C; ++left) {
      for (int right = left; right < C; ++right) {
        const float a =
            pull[same_offset + spin2_oc_pair_index(C, left, right)];
        if (left == right) {
          grad_weight[left] += 2.0f * a * weights[left] * dot2;
        } else {
          grad_weight[left] += a * weights[right] * dot2;
          grad_weight[right] += a * weights[left] * dot2;
        }
        grad_dot += 2.0f * a * weights[left] * weights[right] * dot;
      }
    }

    float qr_left[3] = {}, qr_right[3] = {};
    spin2_oc_add_stf5_outer_pull(r, r, grad_q, 1.0f, qr_left, qr_right);
    for (int d = 0; d < 3; ++d) {
      grad_r[d] += qr_left[d] + qr_right[d];
      grad_r[d] += grad_longitudinal * sj[d];
      grad_sj[d] += grad_longitudinal * r[d];
    }
    for (int d = 0; d < 3; ++d) {
      grad_si[d] += grad_dot * sj[d];
      grad_sj[d] += grad_dot * si[d];
    }

    float grad_dist = 0.0f;
    for (int c = 0; c < C; ++c) {
      grad_dist += grad_weight[c] * weight_derivatives[c];
    }
    const float dot_r = spin2_dot3(grad_r, r);
    for (int d = 0; d < 3; ++d) {
      const float grad_rij =
          grad_dist * r[d] + (grad_r[d] - dot_r * r[d]) / dist +
          (FuseStructuralRadial ? structural_radial_scale * r[d] : 0.0f);
      center_force[d] += grad_rij;
      center_mforce[d] -= grad_si[d];
      if constexpr (AccumulateCenterVirial) {
        for (int a = 0; a < 3; ++a) {
          center_virial[virial_internal_component(3 * a + d)] -=
              r[a] * dist * grad_rij;
        }
      }
      atomicAdd(
          force_soa3 + d * atom_stride + neighbor,
          -static_cast<double>(grad_rij));
      atomicAdd(
          mforce_soa3 + d * atom_stride + neighbor,
          -static_cast<double>(grad_sj[d]));
    }
  }

#pragma unroll
  for (int offset = AtomsPerWarp; offset < 32; offset <<= 1) {
    for (int d = 0; d < 3; ++d) {
      center_force[d] += __shfl_xor_sync(0xffffffffu, center_force[d], offset);
      center_mforce[d] +=
          __shfl_xor_sync(0xffffffffu, center_mforce[d], offset);
    }
    if constexpr (AccumulateCenterVirial) {
      for (int component = 0; component < 9; ++component) {
        center_virial[component] += __shfl_xor_sync(
            0xffffffffu, center_virial[component], offset);
      }
    }
  }
  const bool writes_center = FuseStructuralRadial ? valid : spin_active;
  if (writes_center && edge_lane == 0) {
    for (int d = 0; d < 3; ++d) {
      atomicAdd(
          force_soa3 + d * atom_stride + atom,
          static_cast<double>(center_force[d]));
      if (spin_active) {
        atomicAdd(
            mforce_soa3 + d * atom_stride + atom,
            static_cast<double>(center_mforce[d]));
      }
    }
    if constexpr (AccumulateCenterVirial) {
      for (int component = 0; component < 9; ++component) {
        atomicAdd(
            virial_soa9 + component * atom_stride + atom,
            static_cast<double>(center_virial[component]));
      }
    }
  }
#undef NEP_SPIN2_OC_COOP_FP
}
