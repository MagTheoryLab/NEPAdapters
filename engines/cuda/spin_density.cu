#include "device_operations.hpp"
#include "simulation_box_device.cuh"
#include "spin_md_projection.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace nep_adapters::spin_density::md {
bool prefer_ghost_coordinates(const View& layout) {
  const int ranks = layout.lmax + 1;
  // Keep the existing launch for scalar single-channel work and large matrices.
  return (ranks > 1 || layout.channels > 1) &&
    ranks * ranks * layout.spin_components <= 10 * SpinProjection::width;
}
}  // namespace nep_adapters::spin_density::md

namespace nep_adapters::cuda_backend {
void launch_spin_md_L0(const ModelProtocol&, int, const SimulationBox&,
    const DeviceModelView&, const DeviceWorkspaceView&, bool, int, bool);
void launch_spin_md_L1(const ModelProtocol&, int, const SimulationBox&,
    const DeviceModelView&, const DeviceWorkspaceView&, bool, int, bool);
void launch_spin_md_L2(const ModelProtocol&, int, const SimulationBox&,
    const DeviceModelView&, const DeviceWorkspaceView&, bool, int, bool);
void launch_spin_md_L3(const ModelProtocol&, int, const SimulationBox&,
    const DeviceModelView&, const DeviceWorkspaceView&, bool, int, bool);
void launch_spin_md_L4(const ModelProtocol&, int, const SimulationBox&,
    const DeviceModelView&, const DeviceWorkspaceView&, bool, int, bool);

namespace {
void check_launch() {
  const auto status = cudaGetLastError();
  if (status != cudaSuccess) throw std::runtime_error(std::string("density spin kernel: ") + cudaGetErrorString(status));
}
// The structural radial list includes edges between the spin and structural cutoffs.
template<class Real>
__global__ void compact_spin_neighbors(int count, int n, int types, SimulationBox default_box,
    int neighbor_source, const int* atom_to_structure, const double* boxes, const double* inverses,
    const int* pbc, const double* x, const int* type, const float* cutoff, const int* nn,
    const int* nl, int* spin_nn, int* spin_nl) {
  constexpr unsigned mask = 0xffffffffu;
  const int thread = blockIdx.x*blockDim.x+threadIdx.x;
  const int i = thread/32, lane = thread%32;
  if (i >= count) return;

  SimulationBox box = default_box;
  if (neighbor_source == 0 && atom_to_structure && boxes)
    box = load_structure_box(atom_to_structure[i],boxes,inverses,pbc);
  const int num = nn[i], pair = type[i]*types;
  const double xi = x[i], yi = x[n+i], zi = x[2*n+i];
  int written = 0;
  for (int start = 0; start < num; start += 32) {
    int j = 0;
    bool keep = false;
    if (start+lane < num) {
      j = nl[(start+lane)*n+i];
      const double dx = x[j]-xi, dy = x[n+j]-yi, dz = x[2*n+j]-zi;
      Real rx = dx, ry = dy, rz = dz;
      if (neighbor_source == 0) minimum_image_delta(box,dx,dy,dz,rx,ry,rz);
      const Real r = sqrt(rx*rx+ry*ry+rz*rz);
      keep = r > Real(1e-12) && r < cutoff[pair+type[j]];
    }
    const unsigned accepted = __ballot_sync(mask,keep);
    if (keep) {
      const unsigned lower = (1u<<lane)-1u;
      spin_nl[(written+__popc(accepted & lower))*n+i] = j;
    }
    written += __popc(accepted);
  }
  if (lane == 0) spin_nn[i] = written;
}

void dispatch(const ModelProtocol& p, int count, const SimulationBox& box,
              const DeviceModel& model, DeviceWorkspace& workspace, bool backward,
              VirialTarget target = VirialTarget::none, bool transfer = false) {
  const auto m = model.view();
  auto w = workspace.view();
  if (!p.spin_mode || !supports_cuda_spin_shape(p) || !w.spin_moments ||
      count < 0 || static_cast<std::size_t>(count) > w.atom_capacity)
    throw std::runtime_error("invalid density spin workspace or protocol");
  if (!count) return;
  if (w.spin_nn) {
    if (!backward) {
      const int stride = static_cast<int>(w.atom_capacity);
      auto compact = [&](auto precision) {
        compact_spin_neighbors<decltype(precision)><<<(count+3)/4,128>>>(count,stride,p.num_types,box,
          w.neighbor_source,w.atom_to_structure,w.boxes_row_major9,w.box_inverse_row_major9,w.pbc_flags3,
          w.positions_soa3,w.types,m.spin_cutoff_pair,w.nn_radial,w.nl_radial_slot_major,w.spin_nn,w.spin_nl);
      };
      if (m.spin_radial_coefficients_wide) compact(double{});
      else compact(float{});
      check_launch();
    }
    w.nn_radial = w.spin_nn;
    w.nl_radial_slot_major = w.spin_nl;
  }
  const int mode = target == VirialTarget::none ? 0 : target == VirialTarget::center_atom ? 1 :
    target == VirialTarget::neighbor_atom ? 2 : 3;
  constexpr decltype(&launch_spin_md_L0) launchers[] = {
    launch_spin_md_L0, launch_spin_md_L1, launch_spin_md_L2, launch_spin_md_L3, launch_spin_md_L4};
  launchers[p.spin_l_max](p,count,box,m,w,backward,mode,transfer);
  check_launch();
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
