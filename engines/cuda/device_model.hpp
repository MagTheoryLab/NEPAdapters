#pragma once

#include "model_parameters.hpp"

#include <cstddef>

namespace nep_adapters::cuda_backend {

struct DeviceModelUploadSummary {
  std::size_t ann_type_major_bytes = 0;
  std::size_t ann_type_major_qscaled_bytes = 0;
  std::size_t descriptor_coefficients_bytes = 0;
  std::size_t descriptor_coefficients_type_pair_major_bytes = 0;
  std::size_t q_scaler_bytes = 0;
  std::size_t atomic_numbers_bytes = 0;
  std::size_t total_bytes = 0;
};

struct DeviceModelView {
  const float* ann_type_major = nullptr;
  const float* ann_type_major_qscaled = nullptr;
  const float* descriptor_coefficients = nullptr;
  const float* descriptor_coefficients_type_pair_major = nullptr;
  const float* q_scaler = nullptr;
  const int* atomic_numbers = nullptr;
  std::size_t ann_type_major_count = 0;
  std::size_t ann_type_major_qscaled_count = 0;
  std::size_t descriptor_coefficients_count = 0;
  std::size_t descriptor_coefficients_type_pair_major_count = 0;
  std::size_t q_scaler_count = 0;
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
  float* q_scaler_device_ = nullptr;
  int* atomic_numbers_device_ = nullptr;
  DeviceModelView view_{};
  DeviceModelUploadSummary summary_{};
};

}  // namespace nep_adapters::cuda_backend
