#pragma once

#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#include "nep_adapters/views.hpp"

#include <memory>
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

  virtual NepaStatus find_force_batch(
      const NepaStructureBatch& batch,
      NepaFindForceResult& result) = 0;

  virtual NepaStatus find_force_lammps_neighbors(
      const NepaLammpsNeighborInput& input,
      NepaLammpsNeighborResult& result) {
    (void)input;
    (void)result;
    return NEPA_STATUS_UNSUPPORTED;
  }
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

// Transitional names kept while the public C ABI still says "backend".
using BackendInfo = EngineInfo;
using Backend = Engine;

bool register_backend(Backend* backend);

}  // namespace nep_adapters
