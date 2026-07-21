#pragma once

#include "nep_adapters/api.h"

#include <cstdint>

namespace nep_adapters {

using CapabilityMask = std::uint64_t;

enum class Capability : CapabilityMask {
  batch_find_force = NEPA_CAPABILITY_BATCH_FIND_FORCE,
  external_neighbors = NEPA_CAPABILITY_EXTERNAL_NEIGHBORS,
  device_input = NEPA_CAPABILITY_DEVICE_INPUT,
  spin = NEPA_CAPABILITY_SPIN,
  charge = NEPA_CAPABILITY_CHARGE,
  virial = NEPA_CAPABILITY_VIRIAL,
  descriptors = NEPA_CAPABILITY_DESCRIPTORS,
  spin_energy_transfer = NEPA_CAPABILITY_SPIN_ENERGY_TRANSFER,
  dipole = NEPA_CAPABILITY_DIPOLE,
  polarizability = NEPA_CAPABILITY_POLARIZABILITY,
  dftd3 = NEPA_CAPABILITY_DFTD3,
  evaluate_with_descriptors = NEPA_CAPABILITY_EVALUATE_WITH_DESCRIPTORS,
};

constexpr CapabilityMask to_mask(Capability capability) {
  return static_cast<CapabilityMask>(capability);
}

constexpr bool has_capability(CapabilityMask mask, Capability capability) {
  return (mask & to_mask(capability)) != 0;
}

}  // namespace nep_adapters
