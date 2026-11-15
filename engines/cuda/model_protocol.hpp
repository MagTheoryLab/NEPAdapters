#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>
#include "../common/spin_density_layout.hpp"

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
  bool spin_type_semantics = false;
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
  int spin_l_max = 0;
  int spin_order = 0;
  int spin_soc = 0;
  int spin_scaler = 0;
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
  std::size_t spin_parameter_count = 0;
  std::size_t model_parameter_count = 0;
  std::size_t q_scaler_count = 0;
  std::vector<std::string> elements;
  std::vector<int> atomic_numbers;
  std::vector<double> cutoff_radial_by_type;
  std::vector<double> cutoff_angular_by_type;
  std::vector<double> spin_cutoff_by_type;
  std::vector<double> spin_baseline;
  std::vector<double> energy_reference;
  std::vector<int> spin_dof_type_active;
  BodyChannelConfig body_channels;
};

inline spin_density::Layout make_spin_density_layout(const ModelProtocol& p) {
  return spin_density::Layout(p.spin_compress, p.spin_l_max, p.spin_order, p.spin_soc);
}
inline bool supports_cuda_spin_shape(const ModelProtocol& p) noexcept {
  return !p.spin_mode || (p.spin_mode == 1 && p.spin_compress >= 1 && p.spin_compress <= 9 &&
    p.spin_l_max >= 0 && p.spin_l_max <= 4 && p.spin_order >= 1 && p.spin_order <= 4 &&
    p.spin_basis_size == p.basis_size_angular && p.spin_basis_size <= 12 &&
    (p.spin_soc == 0 || p.spin_soc == 1));
}

struct ParsedModelFile {
  ModelProtocol protocol;
  std::vector<float> parameters_and_q_scaler;
  std::vector<float> flexible_zbl_parameters;
};

ModelProtocol parse_model_protocol(const std::string& model_path);
ParsedModelFile parse_model_file(const std::string& model_path);

}  // namespace nep_adapters::cuda_backend
