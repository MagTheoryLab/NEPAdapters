#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace nep_adapters::cuda_backend {

class UnsupportedModelProtocol : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct BodyChannelConfig {
  int l_max_3body = 0;
  bool has_q_222 = false;
  bool has_q_1111 = false;
  bool has_q_112 = false;
  bool has_q_123 = false;
  bool has_q_233 = false;
  bool has_q_134 = false;

  int channel_count() const;
  int abc_count() const;
};

struct ModelProtocol {
  int version = 0;
  int charge_mode = 0;
  int spin_mode = 0;
  int num_types = 0;
  int n_max_radial = 0;
  int n_max_angular = 0;
  int basis_size_radial = 0;
  int basis_size_angular = 0;
  int descriptor_dim = 0;
  int struct_descriptor_dim = 0;
  int spin_descriptor_dim = 0;
  int spin_compress = 0;
  int spin_basis_size = 0;
  int spin_basis_size_angular = 0;
  int spin_n_max_radial = 0;
  int spin_n_max_angular = 0;
  int spin_l_max = 0;
  int spin_chiral = 0;
  int hidden_neurons = 0;
  int hidden_neurons2 = 0;
  int max_neighbors_radial = 0;
  int max_neighbors_angular = 0;
  int neighbor_capacity_radial = 0;
  int neighbor_capacity_angular = 0;
  bool has_zbl = false;
  bool flexible_zbl = false;
  bool use_typewise_cutoff = false;
  bool use_typewise_cutoff_zbl = false;
  double zbl_inner = 0.0;
  double zbl_outer = 0.0;
  double typewise_cutoff_zbl_factor = 0.0;
  double cutoff_radial = 0.0;
  double cutoff_neighbor = 0.0;
  double cutoff_angular = 0.0;
  double cutoff_max = 0.0;
  double spin_cutoff_radial = 0.0;
  std::size_t ann_parameter_count = 0;
  std::size_t descriptor_parameter_count = 0;
  std::size_t ordinary_descriptor_parameter_count = 0;
  std::size_t spin_descriptor_parameter_count = 0;
  std::size_t model_parameter_count = 0;
  std::size_t q_scaler_count = 0;
  std::vector<std::string> elements;
  std::vector<int> atomic_numbers;
  std::vector<double> cutoff_radial_by_type;
  std::vector<double> cutoff_angular_by_type;
  std::vector<double> spin_baseline;
  std::vector<int> spin_dof_type_active;
  std::vector<int> spin_env_type_active;
  BodyChannelConfig body_channels;
};

struct SpinCoreLayout {
  int channels = 0;
  int basis_count = 0;
  int l_max = 0;
  int chi_channels = 0;
  int rho0_offset = -1;
  int l1_rdot_offset = -1;
  int l1_cross_offset = -1;
  int l1_stf_offset = -1;
  int angular2_offset = -1;
  int angular3_offset = -1;
  int angular4_offset = -1;
  int geom_offset = -1;
  int rho0_dot_offset = -1;
  int raw1_dot_offset = -1;
  int chiral_offset = -1;
  int descriptor_dim = 0;
};

inline SpinCoreLayout make_spin_core_layout(
    const ModelProtocol& protocol) noexcept {
  SpinCoreLayout layout;
  layout.channels = protocol.spin_compress;
  layout.basis_count = protocol.spin_basis_size + 1;
  layout.l_max = protocol.spin_l_max;
  layout.chi_channels = layout.channels < 2 ? layout.channels : 2;

  int offset = 2 + 4 * layout.channels;
  layout.rho0_offset = offset;
  offset += layout.channels;
  if (layout.l_max >= 1) {
    layout.l1_rdot_offset = offset;
    offset += layout.channels;
    layout.l1_cross_offset = offset;
    offset += layout.channels;
    layout.l1_stf_offset = offset;
    offset += layout.channels;
  }
  if (layout.l_max >= 2) {
    layout.angular2_offset = offset;
    offset += layout.channels;
  }
  if (layout.l_max >= 3) {
    layout.angular3_offset = offset;
    offset += layout.channels;
  }
  if (layout.l_max >= 4) {
    layout.angular4_offset = offset;
    offset += layout.channels;
  }
  layout.geom_offset = offset;
  offset += layout.channels;
  layout.rho0_dot_offset = offset;
  offset += layout.channels;
  if (layout.l_max >= 1) {
    layout.raw1_dot_offset = offset;
    offset += layout.channels;
  }
  if (protocol.spin_chiral != 0) {
    layout.chiral_offset = offset;
    offset += layout.chi_channels + 2 * layout.channels;
  }
  layout.descriptor_dim = offset;
  return layout;
}

inline bool supports_cuda_spin_shape(const ModelProtocol& protocol) noexcept {
  if (protocol.spin_mode == 0) {
    return true;
  }
  const int basis_count = protocol.spin_basis_size + 1;
  return protocol.spin_compress >= 1 && protocol.spin_compress <= 4 &&
         basis_count >= protocol.spin_compress && basis_count <= 8 &&
         protocol.spin_l_max >= 0 && protocol.spin_l_max <= 4;
}

struct ParsedModelFile {
  ModelProtocol protocol;
  std::vector<float> parameters_and_q_scaler;
  std::vector<float> flexible_zbl_parameters;
};

ModelProtocol parse_model_protocol(const std::string& model_path);
ParsedModelFile parse_model_file(const std::string& model_path);

}  // namespace nep_adapters::cuda_backend
