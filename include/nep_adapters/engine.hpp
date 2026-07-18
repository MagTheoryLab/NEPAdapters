#pragma once

#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#include "nep_adapters/views.hpp"

#include <memory>
#include <atomic>
#include <string>

namespace nep_adapters {

struct EngineInfo {
  std::string name;
  std::string version;
  CapabilityMask capabilities = 0;
};

class Model {
 public:
  virtual ~Model() = default;

  virtual NepaStatus model_info(NepaModelInfo& out) const {
    out = {};
    return NEPA_STATUS_UNSUPPORTED;
  }

  virtual NepaModelKind model_kind() const {
    return NEPA_MODEL_KIND_ORDINARY;
  }

  virtual NepaStatus find_force_batch(
      const NepaStructureBatch& batch,
      NepaFindForceResult& result) = 0;

  virtual NepaStatus find_charge_batch(
      const NepaStructureBatch& batch,
      NepaFindForceResult& result) {
    (void)batch;
    (void)result;
    return NEPA_STATUS_UNSUPPORTED;
  }

  virtual NepaStatus find_descriptors(
      const NepaStructureBatch& batch,
      NepaFindDescriptorResult& result) {
    (void)batch;
    (void)result;
    return NEPA_STATUS_UNSUPPORTED;
  }

  virtual NepaStatus find_dipoles(
      const NepaStructureBatch& batch,
      NepaDipoleResult& result) {
    (void)batch;
    (void)result;
    return NEPA_STATUS_UNSUPPORTED;
  }

  virtual NepaStatus find_polarizabilities(
      const NepaStructureBatch& batch,
      NepaPolarizabilityResult& result) {
    (void)batch;
    (void)result;
    return NEPA_STATUS_UNSUPPORTED;
  }

  virtual NepaStatus compute_dftd3_batch(
      const NepaStructureBatch& batch,
      const NepaDftd3Parameters& parameters,
      NepaDftd3Result& result,
      bool include_nep) {
    (void)batch;
    (void)parameters;
    (void)result;
    (void)include_nep;
    return NEPA_STATUS_UNSUPPORTED;
  }

  void cancel() noexcept { cancelled_.store(true, std::memory_order_release); }
  void reset_cancel() noexcept { cancelled_.store(false, std::memory_order_release); }
  bool is_cancelled() const noexcept {
    return cancelled_.load(std::memory_order_acquire);
  }

  virtual NepaStatus find_force_lammps_neighbors(
      const NepaLammpsNeighborInput& input,
      NepaLammpsNeighborResult& result) {
    (void)input;
    (void)result;
    return NEPA_STATUS_UNSUPPORTED;
  }

  virtual NepaStatus find_force_lammps_device_neighbors(
      const NepaLammpsDeviceNeighborInput& input,
      NepaLammpsDeviceNeighborResult& result) {
    (void)input;
    (void)result;
    return NEPA_STATUS_UNSUPPORTED;
  }

 private:
  std::atomic<bool> cancelled_{false};
};

class Engine {
 public:
  virtual ~Engine() = default;

  virtual EngineInfo info() const = 0;
  virtual NepaStatus load_model(
      const std::string& model_path,
      std::unique_ptr<Model>& out) = 0;
};

bool register_engine(Engine* engine);
void clear_last_error();
void set_last_error(const std::string& message);

}  // namespace nep_adapters
