#pragma once

#include "device_workspace.hpp"
#include "nep_adapters/api.h"

namespace nep_adapters::cuda_backend {

void write_lammps_device_outputs(
    const NepaLammpsDeviceNeighborInput& input,
    const NepaLammpsDeviceNeighborResult& result,
    DeviceWorkspace& workspace);

}  // namespace nep_adapters::cuda_backend
