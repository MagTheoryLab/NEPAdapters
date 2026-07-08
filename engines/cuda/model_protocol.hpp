#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace nep_adapters::cuda_backend {

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
  int num_types = 0;
  int n_max_radial = 0;
  int n_max_angular = 0;
  int basis_size_radial = 0;
  int basis_size_angular = 0;
  int descriptor_dim = 0;
  int hidden_neurons = 0;
  int max_neighbors_radial = 0;
  int max_neighbors_angular = 0;
  int neighbor_capacity_radial = 0;
  int neighbor_capacity_angular = 0;
  bool has_zbl = false;
  bool flexible_zbl = false;
  double zbl_inner = 0.0;
  double zbl_outer = 0.0;
  double cutoff_radial = 0.0;
  double cutoff_angular = 0.0;
  double cutoff_max = 0.0;
  std::size_t ann_parameter_count = 0;
  std::size_t descriptor_parameter_count = 0;
  std::size_t model_parameter_count = 0;
  std::size_t q_scaler_count = 0;
  std::vector<std::string> elements;
  std::vector<int> atomic_numbers;
  BodyChannelConfig body_channels;
};

struct ParsedModelFile {
  ModelProtocol protocol;
  std::vector<float> parameters_and_q_scaler;
  std::vector<float> flexible_zbl_parameters;
};

ModelProtocol parse_model_protocol(const std::string& model_path);
ParsedModelFile parse_model_file(const std::string& model_path);

}  // namespace nep_adapters::cuda_backend
