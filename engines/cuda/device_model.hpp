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
  std::vector<float> angular_coefficients_center_type_major;
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

struct DeviceModelUploadSummary {
  std::size_t ann_type_major_bytes = 0;
  std::size_t ann_type_major_qscaled_bytes = 0;
  std::size_t descriptor_coefficients_bytes = 0;
  std::size_t descriptor_coefficients_type_pair_major_bytes = 0;
  std::size_t angular_coefficients_center_type_major_bytes = 0;
  std::size_t q_scaler_bytes = 0;
  std::size_t spin_baseline_bytes = 0;
  std::size_t atomic_numbers_bytes = 0;
  std::size_t total_bytes = 0;
};

struct DeviceModelView {
  const float* ann_type_major = nullptr;
  const float* ann_type_major_qscaled = nullptr;
  const float* descriptor_coefficients = nullptr;
  const float* descriptor_coefficients_type_pair_major = nullptr;
  const float* angular_coefficients_center_type_major = nullptr;
  const float* q_scaler = nullptr;
  const float* spin_baseline = nullptr;
  const int* atomic_numbers = nullptr;
  std::size_t ann_type_major_count = 0;
  std::size_t ann_type_major_qscaled_count = 0;
  std::size_t descriptor_coefficients_count = 0;
  std::size_t descriptor_coefficients_type_pair_major_count = 0;
  std::size_t angular_coefficients_center_type_major_count = 0;
  std::size_t q_scaler_count = 0;
  std::size_t spin_baseline_count = 0;
  std::size_t atomic_numbers_count = 0;
};

class DeviceModel {
 public:
  explicit DeviceModel(const HostModelParameters& host);
  ~DeviceModel();

  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;
  DeviceModel(DeviceModel&& other) noexcept;
  DeviceModel& operator=(DeviceModel&& other) noexcept;

  const float* ann_type_major_device() const;
  const float* descriptor_coefficients_device() const;
  const float* q_scaler_device() const;
  DeviceModelView view() const;
  DeviceModelUploadSummary upload_summary() const;

 private:
  void release();

  float* ann_type_major_device_ = nullptr;
  float* ann_type_major_qscaled_device_ = nullptr;
  float* descriptor_coefficients_device_ = nullptr;
  float* descriptor_coefficients_type_pair_major_device_ = nullptr;
  float* angular_coefficients_center_type_major_device_ = nullptr;
  float* q_scaler_device_ = nullptr;
  float* spin_baseline_device_ = nullptr;
  int* atomic_numbers_device_ = nullptr;
  DeviceModelView view_{};
  DeviceModelUploadSummary summary_{};
};

}  // namespace nep_adapters::cuda_backend
