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
};

constexpr CapabilityMask to_mask(Capability capability) {
  return static_cast<CapabilityMask>(capability);
}

constexpr bool has_capability(CapabilityMask mask, Capability capability) {
  return (mask & to_mask(capability)) != 0;
}

}  // namespace nep_adapters
