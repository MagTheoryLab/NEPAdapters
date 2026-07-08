#pragma once

#include "device_workspace.hpp"
#include "model_protocol.hpp"

#include "nep_adapters/api.h"

namespace nep_adapters::cuda_backend {

struct LammpsDeviceNeighborCounts {
  int max_radial = 0;
  int max_angular = 0;
};

void stage_batch_on_device(
    const NepaStructureBatch& batch,
    DeviceWorkspace& workspace);

void stage_lammps_external_neighbors_on_device(
    const NepaLammpsNeighborInput& input,
    const ModelProtocol& protocol,
    DeviceWorkspace& workspace);

LammpsDeviceNeighborCounts stage_lammps_device_neighbors_on_device(
    const NepaLammpsDeviceNeighborInput& input,
    const ModelProtocol& protocol,
    DeviceWorkspace& workspace,
    bool check_overflow = true);

LammpsDeviceNeighborCounts count_lammps_device_neighbors_on_device(
    const NepaLammpsDeviceNeighborInput& input,
    const ModelProtocol& protocol);

}  // namespace nep_adapters::cuda_backend
