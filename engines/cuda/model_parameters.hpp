#pragma once

#include "model_protocol.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace nep_adapters::cuda_backend {

struct AnnTypeBlock {
  std::size_t w0_offset = 0;
  std::size_t b0_offset = 0;
  std::size_t w1_offset = 0;
  std::size_t charge_w1_offset = 0;
  std::size_t extra_bias_offset = 0;
  bool has_extra_bias = false;
  bool has_charge_w1 = false;
};

struct DescriptorCoefficientLayout {
  std::size_t radial_offset = 0;
  std::size_t angular_offset = 0;
  std::size_t spin_offset = 0;
  std::size_t radial_count = 0;
  std::size_t angular_count = 0;
  std::size_t spin_count = 0;
};

struct HostModelParameters {
  ModelProtocol protocol;
  std::vector<float> ann_type_major;
  std::vector<float> ann_type_major_qscaled;
  std::vector<float> descriptor_coefficients;
  std::vector<float> descriptor_coefficients_type_pair_major;
  std::vector<float> q_scaler;
  std::vector<float> spin_baseline;
  std::vector<int> atomic_numbers;
  std::vector<float> flexible_zbl_parameters;
  std::vector<AnnTypeBlock> ann_blocks;
  DescriptorCoefficientLayout descriptor_layout;
  std::size_t b1_offset = 0;
  std::size_t sqrt_epsilon_inf_offset = 0;
  bool has_charge = false;
};

HostModelParameters load_host_model_parameters(const std::string& model_path);

}  // namespace nep_adapters::cuda_backend
