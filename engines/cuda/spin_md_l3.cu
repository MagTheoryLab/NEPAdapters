/* Adapted from GPUMD 73f6b09. SPDX-License-Identifier: LGPL-3.0-or-later */
#include "spin_md_launch.cuh"

namespace nep_adapters::cuda_backend {
void launch_spin_md_L3(const ModelProtocol& p, int count, const SimulationBox& box,
    const DeviceModelView& m, const DeviceWorkspaceView& w, bool backward, int mode, bool transfer) {
  spin_density::md::launch_order<3>(p,count,box,m,w,backward,mode,transfer);
}
} // namespace nep_adapters::cuda_backend
