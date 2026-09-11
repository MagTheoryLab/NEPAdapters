#pragma once

namespace nep_adapters::common {

struct SpinPolynomialLayout {
  int channels = 0;
  int pair_count = 0;
  int density_stride = 38;
  int moment_count = 0;

  int local_s2 = -1;
  int edge_l0_dot = -1;
  int edge_l0_neighbor_s2 = -1;
  int edge_l2_pair = -1;
  int center_l2_environment = -1;
  int edge_l0_dot2 = -1;
  int density_l0_self = -1;
  int density_l1_longitudinal_self = -1;
  int density_l1_axial_self = -1;
  int density_l1_stf_self = -1;
  int density_l1_product_self = -1;
  int density_l2_product_self = -1;
  int density_l0_dot_response = -1;
  int correlation_same_edge = -1;
  int correlation_distinct_neighbor = -1;
  int correlation_distinct_l1 = -1;
  int correlation_distinct_l2 = -1;
  int angular_l1_moment_offset = -1;
  int angular_l2_moment_offset = -1;
  int coupling_l11_axial = -1;
  int edge_l11_axial = -1;
  int coupling_l22_axial = -1;
  int edge_l22_axial = -1;
  int edge_l0_moment_gate = -1;
  int coupling_l11_dot_response = -1;
  int coupling_l111_p_m_x = -1;
  int coupling_l22_dot_response = -1;
  int coupling_l111_p_qs_x = -1;
  int coupling_l112_edge_response = -1;
  int coupling_l111_bulk = -1;
  int descriptor_dim = 0;
};

inline SpinPolynomialLayout make_spin_polynomial_layout(
    int channels, int l_max, int order, int soc,
    bool angular_exchange = false) noexcept {
  SpinPolynomialLayout layout;
  layout.channels = channels;
  layout.pair_count = channels * (channels + 1) / 2;
  layout.moment_count = layout.density_stride * channels + layout.pair_count;
  if (angular_exchange && order >= 2 && l_max >= 1) {
    layout.angular_l1_moment_offset = layout.moment_count;
    layout.moment_count += 3 * channels;
  }
  if (angular_exchange && order >= 2 && l_max >= 2) {
    layout.angular_l2_moment_offset = layout.moment_count;
    layout.moment_count += 5 * channels;
  }

  int offset = 0;
  layout.local_s2 = offset++;
  layout.edge_l0_dot = offset; offset += channels;
  layout.edge_l0_neighbor_s2 = offset; offset += channels;
  if (soc != 0 && l_max >= 2) {
    layout.edge_l2_pair = offset; offset += channels;
    layout.center_l2_environment = offset; offset += channels;
  }
  if (order >= 2) {
    layout.edge_l0_dot2 = offset; offset += channels;
    layout.density_l0_self = offset; offset += channels;
    if (l_max >= 1) {
      if (soc != 0) {
        layout.density_l1_longitudinal_self = offset; offset += channels;
        layout.density_l1_axial_self = offset; offset += channels;
        layout.density_l1_stf_self = offset; offset += channels;
      } else {
        layout.density_l1_product_self = offset; offset += channels;
      }
    }
    if (l_max >= 2) {
      layout.density_l2_product_self = offset; offset += channels;
    }
    layout.density_l0_dot_response = offset; offset += channels;
    layout.correlation_same_edge = offset; offset += layout.pair_count;
    layout.correlation_distinct_neighbor = offset; offset += layout.pair_count;
    if (angular_exchange && l_max >= 1) {
      layout.correlation_distinct_l1 = offset; offset += layout.pair_count;
    }
    if (angular_exchange && l_max >= 2) {
      layout.correlation_distinct_l2 = offset; offset += layout.pair_count;
    }
    if (soc != 0 && l_max >= 1) {
      if (channels >= 2) {
        layout.coupling_l11_axial = offset; offset += channels;
      }
      layout.edge_l11_axial = offset; offset += channels;
    }
    if (soc != 0 && l_max >= 2) {
      if (channels >= 2) {
        layout.coupling_l22_axial = offset; offset += channels;
      }
      layout.edge_l22_axial = offset; offset += channels;
    }
  }
  if (order >= 3) {
    layout.edge_l0_moment_gate = offset; offset += channels;
    if (soc != 0 && l_max >= 1) {
      if (channels >= 2) {
        layout.coupling_l11_dot_response = offset; offset += channels;
      }
      layout.coupling_l111_p_m_x = offset; offset += channels;
    }
    if (soc != 0 && l_max >= 2) {
      if (channels >= 2) {
        layout.coupling_l22_dot_response = offset; offset += channels;
      }
      layout.coupling_l111_p_qs_x = offset; offset += channels;
      if (channels >= 2) {
        layout.coupling_l112_edge_response = offset; offset += channels;
      }
    }
    if (soc != 0 && l_max >= 1 && channels >= 3) {
      layout.coupling_l111_bulk = offset; offset += channels;
    }
  }
  layout.descriptor_dim = offset;
  return layout;
}

}  // namespace nep_adapters::common
