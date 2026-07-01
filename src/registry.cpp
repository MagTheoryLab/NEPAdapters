#include "nep_adapters/engine.hpp"

#include <exception>
#include <memory>
#include <string>
#include <vector>

struct NepaModel {
  std::unique_ptr<nep_adapters::Model> impl;
};

namespace nep_adapters {
namespace {

std::vector<Engine*>& registry() {
  static std::vector<Engine*> engines;
  return engines;
}

Engine* find_engine(const char* engine_name) {
  for (Engine* engine : registry()) {
    if (engine == nullptr) {
      continue;
    }

    EngineInfo info = engine->info();
    if (info.name == engine_name) {
      return engine;
    }
  }

  return nullptr;
}

}  // namespace

bool register_engine(Engine* engine) {
  if (engine == nullptr) {
    return false;
  }

  EngineInfo candidate = engine->info();
  if (candidate.name.empty()) {
    return false;
  }

  for (Engine* existing : registry()) {
    if (existing != nullptr && existing->info().name == candidate.name) {
      return false;
    }
  }

  registry().push_back(engine);
  return true;
}

bool register_backend(Backend* backend) {
  return register_engine(backend);
}

}  // namespace nep_adapters

int nepa_api_version(void) {
  return NEP_ADAPTERS_API_VERSION_MAJOR * 10000 +
         NEP_ADAPTERS_API_VERSION_MINOR * 100 +
         NEP_ADAPTERS_API_VERSION_PATCH;
}

int nepa_backend_count(void) {
  return static_cast<int>(nep_adapters::registry().size());
}

NepaStatus nepa_backend_info(int index, NepaBackendInfo* out) {
  if (out == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }

  if (index < 0 || index >= nepa_backend_count()) {
    out->name = nullptr;
    out->version = nullptr;
    out->capabilities = 0;
    return NEPA_STATUS_BACKEND_UNAVAILABLE;
  }

  static thread_local nep_adapters::EngineInfo info;
  info = nep_adapters::registry()[index]->info();
  out->name = info.name.c_str();
  out->version = info.version.c_str();
  out->capabilities = info.capabilities;
  return NEPA_STATUS_OK;
}

NepaStatus nepa_load_model(
    const char* backend_name,
    const char* model_path,
    NepaModel** out) {
  if (backend_name == nullptr || model_path == nullptr || out == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }

  *out = nullptr;
  nep_adapters::Engine* engine = nep_adapters::find_engine(backend_name);
  if (engine == nullptr) {
    return NEPA_STATUS_BACKEND_UNAVAILABLE;
  }

  try {
    std::unique_ptr<nep_adapters::Model> model;
    NepaStatus status = engine->load_model(model_path, model);
    if (status != NEPA_STATUS_OK) {
      return status;
    }
    if (model == nullptr) {
      return NEPA_STATUS_RUNTIME_ERROR;
    }

    *out = new NepaModel{std::move(model)};
    return NEPA_STATUS_OK;
  } catch (const std::exception&) {
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_model_info(NepaModel* model, NepaModelInfo* out) {
  if (model == nullptr || out == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }

  try {
    return model->impl->model_info(*out);
  } catch (const std::exception&) {
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_find_force_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaFindForceResult* result) {
  if (model == nullptr || batch == nullptr || result == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }

  try {
    return model->impl->find_force_batch(*batch, *result);
  } catch (const std::exception&) {
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_find_descriptors(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaFindDescriptorResult* result) {
  if (model == nullptr || batch == nullptr || result == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }

  try {
    return model->impl->find_descriptors(*batch, *result);
  } catch (const std::exception&) {
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_find_force_lammps_neighbors(
    NepaModel* model,
    const NepaLammpsNeighborInput* input,
    NepaLammpsNeighborResult* result) {
  if (model == nullptr || input == nullptr || result == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }

  try {
    return model->impl->find_force_lammps_neighbors(*input, *result);
  } catch (const std::exception&) {
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

void nepa_free_model(NepaModel* model) {
  delete model;
}

const char* nepa_status_message(NepaStatus status) {
  switch (status) {
    case NEPA_STATUS_OK:
      return "ok";
    case NEPA_STATUS_INVALID_ARGUMENT:
      return "invalid argument";
    case NEPA_STATUS_UNSUPPORTED:
      return "unsupported";
    case NEPA_STATUS_BACKEND_UNAVAILABLE:
      return "backend unavailable";
    case NEPA_STATUS_RUNTIME_ERROR:
      return "runtime error";
  }

  return "unknown status";
}
