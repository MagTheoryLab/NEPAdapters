#include "device_model.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace nep_adapters::cuda_backend {
namespace {

void check_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(action) + ": " + cudaGetErrorString(status));
  }
}

void upload_float_array(
    const std::vector<float>& host,
    float*& device,
    std::size_t& bytes,
    const char* name) {
  bytes = host.size() * sizeof(float);
  if (host.empty()) {
    device = nullptr;
    return;
  }

  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device), bytes), name);
  check_cuda(
      cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice),
      name);
}

void upload_double_array(
    const std::vector<double>& host,
    double*& device,
    std::size_t& bytes,
    const char* name) {
  bytes = host.size() * sizeof(double);
  if (host.empty()) {
    device = nullptr;
    return;
  }

  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device), bytes), name);
  check_cuda(
      cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice),
      name);
}

void upload_int_array(
    const std::vector<int>& host,
    int*& device,
    std::size_t& bytes,
    const char* name) {
  bytes = host.size() * sizeof(int);
  if (host.empty()) {
    device = nullptr;
    return;
  }

  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device), bytes), name);
  check_cuda(
      cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice),
      name);
}

void free_device(float*& ptr) noexcept {
  if (ptr != nullptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

void free_device(double*& ptr) noexcept {
  if (ptr != nullptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

void free_device(int*& ptr) noexcept {
  if (ptr != nullptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

}  // namespace

DeviceModel::DeviceModel(const HostModelParameters& host) {
  try {
    upload_float_array(
        host.ann_type_major,
        ann_type_major_device_,
        summary_.ann_type_major_bytes,
        "upload ann_type_major");
    view_.ann_type_major_count = host.ann_type_major.size();
    upload_float_array(
        host.ann_type_major_qscaled,
        ann_type_major_qscaled_device_,
        summary_.ann_type_major_qscaled_bytes,
        "upload ann_type_major_qscaled");
    view_.ann_type_major_qscaled_count = host.ann_type_major_qscaled.size();
    upload_float_array(
        host.descriptor_coefficients,
        descriptor_coefficients_device_,
        summary_.descriptor_coefficients_bytes,
        "upload descriptor_coefficients");
    view_.descriptor_coefficients_count = host.descriptor_coefficients.size();
    upload_float_array(
        host.descriptor_coefficients_type_pair_major,
        descriptor_coefficients_type_pair_major_device_,
        summary_.descriptor_coefficients_type_pair_major_bytes,
        "upload type-pair-major descriptor_coefficients");
    view_.descriptor_coefficients_type_pair_major_count =
        host.descriptor_coefficients_type_pair_major.size();
    upload_float_array(
        host.angular_coefficients_center_type_major,
        angular_coefficients_center_type_major_device_,
        summary_.angular_coefficients_center_type_major_bytes,
        "upload center-type-major angular coefficients");
    view_.angular_coefficients_center_type_major_count =
        host.angular_coefficients_center_type_major.size();
    upload_float_array(
        host.spin_projection_parameters,
        spin_projection_parameters_device_,
        summary_.spin_projection_parameters_bytes,
        "upload spin projection parameters");
    view_.spin_projection_parameters_count =
        host.spin_projection_parameters.size();
    upload_float_array(
        host.q_scaler,
        q_scaler_device_,
        summary_.q_scaler_bytes,
        "upload q_scaler");
    view_.q_scaler_count = host.q_scaler.size();
    upload_float_array(
        host.cutoff_radial_pair,
        cutoff_radial_pair_device_,
        summary_.cutoff_radial_pair_bytes,
        "upload radial pair cutoffs");
    view_.cutoff_radial_pair_count = host.cutoff_radial_pair.size();
    upload_float_array(
        host.cutoff_angular_pair,
        cutoff_angular_pair_device_,
        summary_.cutoff_angular_pair_bytes,
        "upload angular pair cutoffs");
    view_.cutoff_angular_pair_count = host.cutoff_angular_pair.size();
    upload_float_array(
        host.spin_cutoff_pair,
        spin_cutoff_pair_device_,
        summary_.spin_cutoff_pair_bytes,
        "upload spin pair cutoffs");
    view_.spin_cutoff_pair_count = host.spin_cutoff_pair.size();
    upload_float_array(
        host.zbl_parameters_pair,
        zbl_parameters_pair_device_,
        summary_.zbl_parameters_pair_bytes,
        "upload ZBL pair parameters");
    view_.zbl_parameters_pair_count = host.zbl_parameters_pair.size();
    upload_double_array(
        host.spin_baseline,
        spin_baseline_device_,
        summary_.spin_baseline_bytes,
        "upload spin_baseline");
    view_.spin_baseline_count = host.spin_baseline.size();
    upload_int_array(
        host.protocol.spin_dof_type_active,
        spin_dof_type_active_device_,
        summary_.spin_dof_type_active_bytes,
        "upload spin_dof_type_active");
    view_.spin_dof_type_active_count =
        host.protocol.spin_dof_type_active.size();
    upload_int_array(
        host.protocol.spin_env_type_active,
        spin_env_type_active_device_,
        summary_.spin_env_type_active_bytes,
        "upload spin_env_type_active");
    view_.spin_env_type_active_count =
        host.protocol.spin_env_type_active.size();
    upload_int_array(
        host.atomic_numbers,
        atomic_numbers_device_,
        summary_.atomic_numbers_bytes,
        "upload atomic_numbers");
    view_.atomic_numbers_count = host.atomic_numbers.size();
    view_.ann_type_major = ann_type_major_device_;
    view_.ann_type_major_qscaled = ann_type_major_qscaled_device_;
    view_.descriptor_coefficients = descriptor_coefficients_device_;
    view_.descriptor_coefficients_type_pair_major =
        descriptor_coefficients_type_pair_major_device_;
    view_.angular_coefficients_center_type_major =
        angular_coefficients_center_type_major_device_;
    view_.spin_projection_parameters = spin_projection_parameters_device_;
    view_.q_scaler = q_scaler_device_;
    view_.cutoff_radial_pair = cutoff_radial_pair_device_;
    view_.cutoff_angular_pair = cutoff_angular_pair_device_;
    view_.spin_cutoff_pair = spin_cutoff_pair_device_;
    view_.zbl_parameters_pair = zbl_parameters_pair_device_;
    view_.spin_baseline = spin_baseline_device_;
    view_.spin_dof_type_active = spin_dof_type_active_device_;
    view_.spin_env_type_active = spin_env_type_active_device_;
    view_.atomic_numbers = atomic_numbers_device_;
    summary_.total_bytes =
        summary_.ann_type_major_bytes +
        summary_.ann_type_major_qscaled_bytes +
        summary_.descriptor_coefficients_bytes +
        summary_.descriptor_coefficients_type_pair_major_bytes +
        summary_.angular_coefficients_center_type_major_bytes +
        summary_.spin_projection_parameters_bytes +
        summary_.q_scaler_bytes +
        summary_.cutoff_radial_pair_bytes +
        summary_.cutoff_angular_pair_bytes +
        summary_.spin_cutoff_pair_bytes +
        summary_.zbl_parameters_pair_bytes +
        summary_.spin_baseline_bytes +
        summary_.spin_dof_type_active_bytes +
        summary_.spin_env_type_active_bytes +
        summary_.atomic_numbers_bytes;
  } catch (...) {
    release();
    throw;
  }
}

DeviceModel::~DeviceModel() {
  release();
}

DeviceModel::DeviceModel(DeviceModel&& other) noexcept {
  *this = std::move(other);
}

DeviceModel& DeviceModel::operator=(DeviceModel&& other) noexcept {
  if (this == &other) {
    return *this;
  }

  release();
  ann_type_major_device_ = other.ann_type_major_device_;
  ann_type_major_qscaled_device_ = other.ann_type_major_qscaled_device_;
  descriptor_coefficients_device_ = other.descriptor_coefficients_device_;
  descriptor_coefficients_type_pair_major_device_ =
      other.descriptor_coefficients_type_pair_major_device_;
  angular_coefficients_center_type_major_device_ =
      other.angular_coefficients_center_type_major_device_;
  spin_projection_parameters_device_ =
      other.spin_projection_parameters_device_;
  q_scaler_device_ = other.q_scaler_device_;
  cutoff_radial_pair_device_ = other.cutoff_radial_pair_device_;
  cutoff_angular_pair_device_ = other.cutoff_angular_pair_device_;
  spin_cutoff_pair_device_ = other.spin_cutoff_pair_device_;
  zbl_parameters_pair_device_ = other.zbl_parameters_pair_device_;
  spin_baseline_device_ = other.spin_baseline_device_;
  spin_dof_type_active_device_ = other.spin_dof_type_active_device_;
  spin_env_type_active_device_ = other.spin_env_type_active_device_;
  atomic_numbers_device_ = other.atomic_numbers_device_;
  view_ = other.view_;
  summary_ = other.summary_;

  other.ann_type_major_device_ = nullptr;
  other.ann_type_major_qscaled_device_ = nullptr;
  other.descriptor_coefficients_device_ = nullptr;
  other.descriptor_coefficients_type_pair_major_device_ = nullptr;
  other.angular_coefficients_center_type_major_device_ = nullptr;
  other.spin_projection_parameters_device_ = nullptr;
  other.q_scaler_device_ = nullptr;
  other.cutoff_radial_pair_device_ = nullptr;
  other.cutoff_angular_pair_device_ = nullptr;
  other.spin_cutoff_pair_device_ = nullptr;
  other.zbl_parameters_pair_device_ = nullptr;
  other.spin_baseline_device_ = nullptr;
  other.spin_dof_type_active_device_ = nullptr;
  other.spin_env_type_active_device_ = nullptr;
  other.atomic_numbers_device_ = nullptr;
  other.view_ = {};
  other.summary_ = {};
  return *this;
}

DeviceModelView DeviceModel::view() const {
  return view_;
}

DeviceModelUploadSummary DeviceModel::upload_summary() const {
  return summary_;
}

void DeviceModel::release() {
  free_device(ann_type_major_device_);
  free_device(ann_type_major_qscaled_device_);
  free_device(descriptor_coefficients_device_);
  free_device(descriptor_coefficients_type_pair_major_device_);
  free_device(angular_coefficients_center_type_major_device_);
  free_device(spin_projection_parameters_device_);
  free_device(q_scaler_device_);
  free_device(cutoff_radial_pair_device_);
  free_device(cutoff_angular_pair_device_);
  free_device(spin_cutoff_pair_device_);
  free_device(zbl_parameters_pair_device_);
  free_device(spin_baseline_device_);
  free_device(spin_dof_type_active_device_);
  free_device(spin_env_type_active_device_);
  free_device(atomic_numbers_device_);
  view_ = {};
  summary_ = {};
}

}  // namespace nep_adapters::cuda_backend
