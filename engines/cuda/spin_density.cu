#include "device_operations.hpp"
#include "spin_density_kernels.cuh"
#include "spin_density_md_layouts.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace nep_adapters::cuda_backend {
namespace {
void check_launch() {
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(std::string("density spin kernel: ") + cudaGetErrorString(status));
}
template<class Layout>
void launch(Layout layout, const ModelProtocol& p, int count, const SimulationBox& box,
            const DeviceModelView& m, const DeviceWorkspaceView& w, bool backward,
            VirialTarget target = VirialTarget::none, bool transfer = false) {
  const int stride = static_cast<int>(w.atom_capacity);
  const int plane = stride * p.neighbor_capacity_radial;
  spin_density::Geometry geometry{box, w.neighbor_source, w.atom_to_structure,
    w.boxes_row_major9, w.box_inverse_row_major9, w.pbc_flags3};
  const int blocks = (count * p.spin_compress * Layout::tile + 127) / 128;
  if (!blocks) return;
  if (!backward) {
    spin_density::descriptors<<<blocks, 128>>>(layout, count, stride, p.struct_descriptor_dim,
      p.num_types, p.n_max_angular, p.spin_basis_size, geometry,
      w.types, m.spin_dof_type_active, w.positions_soa3, w.spins_soa3,
      w.nn_radial, w.nl_radial_slot_major, nullptr, plane, m.spin_cutoff_pair,
      m.spin_radial_coefficients, m.spin_parameters, w.spin_moments, w.descriptors);
  } else {
    const int mode = target == VirialTarget::none ? 0 : target == VirialTarget::center_atom ? 1 :
      target == VirialTarget::neighbor_atom ? 2 : 3;
    spin_density::forces<<<blocks, 128>>>(layout, count, stride, p.struct_descriptor_dim,
      p.num_types, p.n_max_angular, p.spin_basis_size, geometry,
      w.types, m.spin_dof_type_active, w.positions_soa3, w.spins_soa3,
      w.nn_radial, w.nl_radial_slot_major, nullptr, plane, m.spin_cutoff_pair,
      m.spin_radial_coefficients, m.spin_parameters, w.spin_moments, w.fp,
      w.force_soa3, w.mforce_soa3, w.virial_soa9, mode, w.per_atom_virial_float_soa9,
      transfer ? w.spin_transfer_soa9 : nullptr);
  }
  check_launch();
}
void dispatch(const ModelProtocol& p, int count, const SimulationBox& box,
              const DeviceModel& model, DeviceWorkspace& workspace, bool backward,
              VirialTarget target = VirialTarget::none, bool transfer = false) {
  const auto m = model.view();
  const auto w = workspace.view();
  if (!p.spin_mode || !supports_cuda_spin_shape(p) || !w.spin_moments ||
      count < 0 || static_cast<std::size_t>(count) > w.atom_capacity)
    throw std::runtime_error("invalid density spin workspace or protocol");
  if (p.spin_compress == 3 && p.spin_l_max == 2 && p.spin_order == 2) {
    if (p.spin_soc) launch(spin_density::MDLayoutSOC<4>{}, p, count, box, m, w, backward, target, transfer);
    else launch(spin_density::MDLayoutNoSOC<4>{}, p, count, box, m, w, backward, target, transfer);
  } else if (m.spin_layout.components <= 100) {
    launch(spin_density::RuntimeLayout<100, true>(m.spin_layout), p, count, box, m, w, backward, target, transfer);
  } else {
    launch(spin_density::RuntimeLayout<625, true>(m.spin_layout), p, count, box, m, w, backward, target, transfer);
  }
}
}  // namespace
void build_spin_descriptors_on_device(const ModelProtocol& p, int count,
  const SimulationBox& box, const DeviceModel& m, DeviceWorkspace& w) {
  dispatch(p, count, box, m, w, false);
}
void accumulate_spin_forces_on_device(const ModelProtocol& p, int count,
  const SimulationBox& box, const DeviceModel& m, DeviceWorkspace& w,
  VirialTarget target, bool transfer, bool, SpinForceTimings* timings) {
  if (transfer) {
    const auto view = w.view();
    if (!view.spin_transfer_soa9)
      throw std::runtime_error("density spin workspace is missing spin-transfer storage");
    const auto status = cudaMemsetAsync(
      view.spin_transfer_soa9, 0, view.atom_capacity * 9 * sizeof(float));
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("clear density spin transfer: ") + cudaGetErrorString(status));
  }
  cudaEvent_t start = nullptr, end = nullptr;
  if (timings) { cudaEventCreate(&start); cudaEventCreate(&end); cudaEventRecord(start); }
  dispatch(p, count, box, m, w, true, target, transfer);
  if (timings) {
    cudaEventRecord(end); cudaEventSynchronize(end);
    cudaEventElapsedTime(&timings->density_ms, start, end);
    cudaEventDestroy(start); cudaEventDestroy(end);
  }
}
}  // namespace nep_adapters::cuda_backend
