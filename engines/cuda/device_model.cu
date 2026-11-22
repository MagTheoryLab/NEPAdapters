#include "device_model.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <cstring>
#include <cstdint>
#include <type_traits>
#include <iterator>

namespace nep_adapters::cuda_backend {
namespace {

void check_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(action) + ": " + cudaGetErrorString(status));
  }
}

template<class T>
void upload_array(
    const std::vector<T>& host,
    T*& device,
    std::size_t& bytes,
    const char* name) {
  bytes = host.size() * sizeof(T);
  if (host.empty()) {
    device = nullptr;
    return;
  }

  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device), bytes), name);
  check_cuda(
      cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice),
      name);
}

template<class T>
void free_device(T*& ptr) noexcept {
  if (ptr != nullptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

}  // namespace

DeviceModel::DeviceModel(const HostModelParameters& host) {
  try {
    upload_array(
        host.ann_type_major,
        ann_type_major_device_,
        summary_.ann_type_major_bytes,
        "upload ann_type_major");
    view_.ann_type_major_count = host.ann_type_major.size();
    upload_array(
        host.ann_type_major_qscaled,
        ann_type_major_qscaled_device_,
        summary_.ann_type_major_qscaled_bytes,
        "upload ann_type_major_qscaled");
    view_.ann_type_major_qscaled_count = host.ann_type_major_qscaled.size();
    upload_array(
        host.descriptor_coefficients,
        descriptor_coefficients_device_,
        summary_.descriptor_coefficients_bytes,
        "upload descriptor_coefficients");
    view_.descriptor_coefficients_count = host.descriptor_coefficients.size();
    upload_array(
        host.descriptor_coefficients_type_pair_major,
        descriptor_coefficients_type_pair_major_device_,
        summary_.descriptor_coefficients_type_pair_major_bytes,
        "upload type-pair-major descriptor_coefficients");
    view_.descriptor_coefficients_type_pair_major_count =
        host.descriptor_coefficients_type_pair_major.size();
    upload_array(
        host.angular_coefficients_center_type_major,
        angular_coefficients_center_type_major_device_,
        summary_.angular_coefficients_center_type_major_bytes,
        "upload center-type-major angular coefficients");
    view_.angular_coefficients_center_type_major_count =
        host.angular_coefficients_center_type_major.size();
    upload_array(
        host.spin_parameters,
        spin_parameters_device_,
        summary_.spin_parameters_bytes,
        "upload density spin parameters");
    view_.spin_parameters_count =
        host.spin_parameters.size();
    if (host.protocol.spin_mode) {
      const auto layout = make_spin_density_layout(host.protocol);
      // One aligned, model-owned allocation holds immutable contraction tables.
      std::vector<unsigned char> bytes;
      auto append = [&](const auto* values, std::size_t count) {
        using T = std::remove_cv_t<std::remove_reference_t<decltype(*values)>>;
        const std::size_t offset = (bytes.size() + alignof(T) - 1) / alignof(T) * alignof(T);
        bytes.resize(offset + count * sizeof(T));
        if (count) std::memcpy(bytes.data() + offset, values, count * sizeof(T));
        return offset;
      };
      const auto p = append(layout.primitives.data(), layout.primitives.size());
      const auto s = append(layout.sectors.data(), layout.sectors.size());
      const auto t = append(layout.terms.data(), layout.terms.size());
      const auto c = append(layout.centers.data(), layout.centers.size());
      const auto h = append(spin_density::harmonics, std::size(spin_density::harmonics));
      const int harmonics = (layout.lmax+1)*(layout.lmax+1);
      const int lanes = spin_density::md_force_lanes(layout.lmax,layout.order);
      const bool wide = spin_density::md_wide(layout.lmax,layout.order,host.protocol.spin_soc);
      const bool forward_projected = spin_density::md_warp_density(harmonics,layout.spin_components);
      const bool paired = spin_density::md_paired(layout,host.protocol.spin_soc);
      const float* weights = host.spin_parameters.data() + (host.protocol.n_max_angular+1)*layout.channels;
      std::size_t forward_rows = 0, forward_terms = 0, transpose_rows = 0;
      std::size_t transpose_terms = 0, transpose_neighbors = 0;
      if (forward_projected) {
        const spin_density::SpinProjection forward(layout,weights,false);
        forward_rows = append(forward.rows.data(),forward.rows.size());
        forward_terms = append(forward.packed.data(),forward.packed.size());
      }
      const bool has_transpose = paired || lanes == spin_density::SpinProjection::width;
      bool single_term_rows = false;
      if (has_transpose) {
        const spin_density::SpinProjection transpose(layout,weights,true,paired);
        single_term_rows = paired && std::all_of(transpose.rows.begin(),transpose.rows.end(),
          [](const spin_density::ProjectionRow& row) { return row.count == 1; });
        if (!single_term_rows)
          transpose_rows = append(transpose.rows.data(),transpose.rows.size());
        transpose_terms = append(transpose.packed.data(),transpose.packed.size());
        transpose_neighbors = append(transpose.neighbor_weights.data(),transpose.neighbor_weights.size());
      }
      std::vector<int> adjoint_order;
      if (!paired && lanes > 1 && lanes != spin_density::SpinProjection::width) {
        adjoint_order.resize(lanes+1);
        for (int lane = 0; lane < lanes; ++lane) {
          adjoint_order[lane] = static_cast<int>(adjoint_order.size());
          for (int term = 0; term < static_cast<int>(layout.terms.size()); ++term)
            if (layout.terms[term].geometry % lanes == lane) adjoint_order.push_back(term);
        }
        adjoint_order[lanes] = static_cast<int>(adjoint_order.size());
      }
      const auto md_order = append(adjoint_order.data(),adjoint_order.size());
      std::size_t wide_radial = 0;
      if (wide) {
        const int channels = layout.channels, types = host.protocol.num_types;
        const int basis = host.protocol.spin_basis_size+1, na = host.protocol.n_max_angular+1;
        const float* angular = host.descriptor_coefficients_type_pair_major.data() +
          host.descriptor_layout.angular_offset;
        std::vector<double> coeffs(static_cast<std::size_t>(types)*types*channels*basis,0);
        for (int pair = 0; pair < types*types; ++pair)
          for (int c = 0; c < channels; ++c)
            for (int k = 0; k < basis; ++k)
              for (int a = 0; a < na; ++a)
                coeffs[(pair*channels+c)*basis+k] +=
                  static_cast<double>(angular[(pair*na+a)*basis+k])*host.spin_parameters[a*channels+c];
        wide_radial = append(coeffs.data(),coeffs.size());
      }
      check_cuda(cudaMalloc(&spin_layout_device_, bytes.size()), "allocate density layout");
      check_cuda(cudaMemcpy(spin_layout_device_, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "upload density layout");
      const auto* base = static_cast<const unsigned char*>(spin_layout_device_);
      if (forward_projected)
        view_.spin_forward = {reinterpret_cast<const spin_density::ProjectionRow*>(base+forward_rows),
          reinterpret_cast<const spin_density::ProjectionTerm*>(base+forward_terms), nullptr};
      if (has_transpose)
        view_.spin_transpose = {
          single_term_rows ? nullptr : reinterpret_cast<const spin_density::ProjectionRow*>(base+transpose_rows),
          reinterpret_cast<const spin_density::ProjectionTerm*>(base+transpose_terms),
          paired ? reinterpret_cast<const float*>(base+transpose_neighbors) : nullptr};
      view_.spin_md_adjoint_order = adjoint_order.empty() ? nullptr :
        reinterpret_cast<const int*>(base+md_order);
      view_.spin_radial_coefficients_wide = wide ? reinterpret_cast<const double*>(base+wide_radial) : nullptr;
      view_.spin_layout = {layout.channels, layout.order, layout.lmax,
        static_cast<int>(layout.primitives.size()), layout.spin_components, layout.components,
        static_cast<int>(layout.sectors.size()), static_cast<int>(layout.terms.size()),
        static_cast<int>(layout.centers.size()), layout.path_count, layout.local_count, layout.dimension,
        reinterpret_cast<const spin_density::Primitive*>(base+p),
        reinterpret_cast<const spin_density::Sector*>(base+s),
        reinterpret_cast<const spin_density::Term*>(base+t),
        reinterpret_cast<const spin_density::Center*>(base+c),
        reinterpret_cast<const spin_density::Harmonic*>(base+h),
        static_cast<int>(std::size(spin_density::harmonics))};
      std::size_t radial_bytes = 0;
      upload_array(host.spin_radial_coefficients, spin_radial_coefficients_device_, radial_bytes, "upload density radial coefficients");
      view_.spin_radial_coefficients = spin_radial_coefficients_device_;
      summary_.total_bytes += bytes.size() + radial_bytes;
    }
    upload_array(
        host.q_scaler,
        q_scaler_device_,
        summary_.q_scaler_bytes,
        "upload q_scaler");
    view_.q_scaler_count = host.q_scaler.size();
    upload_array(
        host.cutoff_radial_pair,
        cutoff_radial_pair_device_,
        summary_.cutoff_radial_pair_bytes,
        "upload radial pair cutoffs");
    view_.cutoff_radial_pair_count = host.cutoff_radial_pair.size();
    upload_array(
        host.cutoff_angular_pair,
        cutoff_angular_pair_device_,
        summary_.cutoff_angular_pair_bytes,
        "upload angular pair cutoffs");
    view_.cutoff_angular_pair_count = host.cutoff_angular_pair.size();
    upload_array(
        host.spin_cutoff_pair,
        spin_cutoff_pair_device_,
        summary_.spin_cutoff_pair_bytes,
        "upload spin pair cutoffs");
    view_.spin_cutoff_pair_count = host.spin_cutoff_pair.size();
    upload_array(
        host.zbl_parameters_pair,
        zbl_parameters_pair_device_,
        summary_.zbl_parameters_pair_bytes,
        "upload ZBL pair parameters");
    view_.zbl_parameters_pair_count = host.zbl_parameters_pair.size();
    upload_array(
        host.spin_baseline,
        spin_baseline_device_,
        summary_.spin_baseline_bytes,
        "upload spin_baseline");
    view_.spin_baseline_count = host.spin_baseline.size();
    upload_array(host.protocol.spin_walls, spin_walls_device_,
                 summary_.spin_wall_bytes, "upload spin_wall");
    upload_array(
        host.protocol.spin_dof_type_active,
        spin_dof_type_active_device_,
        summary_.spin_dof_type_active_bytes,
        "upload spin_dof_type_active");
    view_.spin_dof_type_active_count =
        host.protocol.spin_dof_type_active.size();
    upload_array(
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
    view_.spin_parameters = spin_parameters_device_;
    view_.q_scaler = q_scaler_device_;
    view_.cutoff_radial_pair = cutoff_radial_pair_device_;
    view_.cutoff_angular_pair = cutoff_angular_pair_device_;
    view_.spin_cutoff_pair = spin_cutoff_pair_device_;
    view_.zbl_parameters_pair = zbl_parameters_pair_device_;
    view_.spin_baseline = spin_baseline_device_;
    view_.spin_walls = spin_walls_device_;
    view_.spin_dof_type_active = spin_dof_type_active_device_;
    view_.atomic_numbers = atomic_numbers_device_;
    summary_.total_bytes +=
        summary_.ann_type_major_bytes +
        summary_.ann_type_major_qscaled_bytes +
        summary_.descriptor_coefficients_bytes +
        summary_.descriptor_coefficients_type_pair_major_bytes +
        summary_.angular_coefficients_center_type_major_bytes +
        summary_.spin_parameters_bytes +
        summary_.q_scaler_bytes +
        summary_.cutoff_radial_pair_bytes +
        summary_.cutoff_angular_pair_bytes +
        summary_.spin_cutoff_pair_bytes +
        summary_.zbl_parameters_pair_bytes +
        summary_.spin_baseline_bytes +
        summary_.spin_wall_bytes +
        summary_.spin_dof_type_active_bytes +
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
  spin_parameters_device_ =
      other.spin_parameters_device_;
  spin_layout_device_ = other.spin_layout_device_;
  spin_radial_coefficients_device_ = other.spin_radial_coefficients_device_;
  other.spin_layout_device_ = nullptr;
  other.spin_radial_coefficients_device_ = nullptr;
  q_scaler_device_ = other.q_scaler_device_;
  cutoff_radial_pair_device_ = other.cutoff_radial_pair_device_;
  cutoff_angular_pair_device_ = other.cutoff_angular_pair_device_;
  spin_cutoff_pair_device_ = other.spin_cutoff_pair_device_;
  zbl_parameters_pair_device_ = other.zbl_parameters_pair_device_;
  spin_baseline_device_ = other.spin_baseline_device_;
  spin_walls_device_ = other.spin_walls_device_;
  spin_dof_type_active_device_ = other.spin_dof_type_active_device_;
  atomic_numbers_device_ = other.atomic_numbers_device_;
  view_ = other.view_;
  summary_ = other.summary_;

  other.ann_type_major_device_ = nullptr;
  other.ann_type_major_qscaled_device_ = nullptr;
  other.descriptor_coefficients_device_ = nullptr;
  other.descriptor_coefficients_type_pair_major_device_ = nullptr;
  other.angular_coefficients_center_type_major_device_ = nullptr;
  other.spin_parameters_device_ = nullptr;
  other.q_scaler_device_ = nullptr;
  other.cutoff_radial_pair_device_ = nullptr;
  other.cutoff_angular_pair_device_ = nullptr;
  other.spin_cutoff_pair_device_ = nullptr;
  other.zbl_parameters_pair_device_ = nullptr;
  other.spin_baseline_device_ = nullptr;
  other.spin_walls_device_ = nullptr;
  other.spin_dof_type_active_device_ = nullptr;
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
  free_device(spin_parameters_device_);
  if (spin_layout_device_) cudaFree(spin_layout_device_);
  spin_layout_device_ = nullptr;
  free_device(spin_radial_coefficients_device_);
  free_device(q_scaler_device_);
  free_device(cutoff_radial_pair_device_);
  free_device(cutoff_angular_pair_device_);
  free_device(spin_cutoff_pair_device_);
  free_device(zbl_parameters_pair_device_);
  free_device(spin_baseline_device_);
  free_device(spin_walls_device_);
  free_device(spin_dof_type_active_device_);
  free_device(atomic_numbers_device_);
  view_ = {};
  summary_ = {};
}

}  // namespace nep_adapters::cuda_backend
