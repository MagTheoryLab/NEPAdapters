/* Adapted from GPUMD 73f6b09. SPDX-License-Identifier: LGPL-3.0-or-later */
#include "spin_md_quadratic_kernels.cuh"
#include "spin_md_quadratic.cuh"

namespace nep_adapters::cuda_backend {
void launch_spin_md_L2(const ModelProtocol& p, int count, const SimulationBox& box,
    const DeviceModelView& m, const DeviceWorkspaceView& w, bool backward, int mode, bool transfer) {
  if (p.spin_order == 2) {
    auto quadratic = [&](auto layout) {
      using Layout = decltype(layout);
      constexpr bool factor_center = Layout::soc == 1;
      if (p.spin_compress == Layout::channels)
        spin_density::md::launch_quadratic_l2<factor_center>(
          layout,p,count,box,m,w,backward,mode,transfer);
      else
        spin_density::md::launch_quadratic_l2<factor_center>(
          spin_density::md::MDQuadraticChannels<Layout>(p.spin_compress),p,count,box,m,w,
          backward,mode,transfer);
    };
    if (p.spin_soc) quadratic(spin_density::MDProjectedQuadraticSOC<4>{});
    else quadratic(spin_density::MDProjectedQuadraticNoSOC<4>{});
    return;
  }
  spin_density::md::launch_order<2>(p,count,box,m,w,backward,mode,transfer);
}
} // namespace nep_adapters::cuda_backend
