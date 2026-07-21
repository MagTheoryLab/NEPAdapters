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

std::string& last_error() {
  static thread_local std::string message;
  return message;
}

bool is_fully_periodic(const NepaStructureBatch& batch) {
  if (batch.num_structures <= 0 || batch.pbc_flags3 == nullptr) {
    return false;
  }
  const std::size_t count = static_cast<std::size_t>(batch.num_structures) * 3;
  for (std::size_t component = 0; component < count; ++component) {
    if (batch.pbc_flags3[component] != 1) {
      return false;
    }
  }
  return true;
}

}  // namespace

void clear_last_error() {
  last_error().clear();
}

void set_last_error(const std::string& message) {
  last_error() = message;
}

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
  nep_adapters::clear_last_error();
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
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_model_info(NepaModel* model, NepaModelInfo* out) {
  nep_adapters::clear_last_error();
  if (model == nullptr || out == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }

  try {
    return model->impl->model_info(*out);
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_model_kind(NepaModel* model, NepaModelKind* out) {
  nep_adapters::clear_last_error();
  if (model == nullptr || out == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }
  try {
    *out = model->impl->model_kind();
    return NEPA_STATUS_OK;
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_estimate_workspace(
    NepaModel* model,
    int32_t atom_capacity,
    int32_t structure_capacity,
    NepaWorkspaceEstimate* out) {
  nep_adapters::clear_last_error();
  if (model == nullptr || out == nullptr || atom_capacity <= 0 ||
      structure_capacity <= 0) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }
  try {
    return model->impl->estimate_workspace(
        atom_capacity, structure_capacity, *out);
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_find_force_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaFindForceResult* result) {
  nep_adapters::clear_last_error();
  if (model == nullptr || batch == nullptr || result == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }
  if (!nep_adapters::is_fully_periodic(*batch)) {
    nep_adapters::set_last_error(
        "NEPAdapters supports fully periodic structures only (pbc=[1,1,1])");
    return NEPA_STATUS_UNSUPPORTED;
  }

  const NepaModelKind kind = model->impl->model_kind();
  if (kind == NEPA_MODEL_KIND_CHARGE) {
    nep_adapters::set_last_error(
        "charge models require nepa_find_charge_batch");
    return NEPA_STATUS_UNSUPPORTED;
  }
  if (kind == NEPA_MODEL_KIND_DIPOLE ||
      kind == NEPA_MODEL_KIND_POLARIZABILITY) {
    nep_adapters::set_last_error(
        "response models require their explicit dipole or polarizability API");
    return NEPA_STATUS_UNSUPPORTED;
  }

  try {
    return model->impl->find_force_batch(*batch, *result);
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_evaluate_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaEvaluateResult* result) {
  nep_adapters::clear_last_error();
  if (model == nullptr || batch == nullptr || result == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }
  if (!nep_adapters::is_fully_periodic(*batch)) {
    nep_adapters::set_last_error(
        "NEPAdapters supports fully periodic structures only (pbc=[1,1,1])");
    return NEPA_STATUS_UNSUPPORTED;
  }
  if (model->impl->model_kind() != NEPA_MODEL_KIND_ORDINARY) {
    nep_adapters::set_last_error(
        "evaluate_batch currently supports ordinary NEP models only");
    return NEPA_STATUS_UNSUPPORTED;
  }
  try {
    return model->impl->evaluate_batch(*batch, *result);
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_find_charge_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaFindForceResult* result) {
  nep_adapters::clear_last_error();
  if (model == nullptr || batch == nullptr || result == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }
  if (!nep_adapters::is_fully_periodic(*batch)) {
    nep_adapters::set_last_error(
        "NEPAdapters supports fully periodic structures only (pbc=[1,1,1])");
    return NEPA_STATUS_UNSUPPORTED;
  }
  if (model->impl->model_kind() != NEPA_MODEL_KIND_CHARGE) {
    nep_adapters::set_last_error("calculate_charge requires a charge NEP model");
    return NEPA_STATUS_UNSUPPORTED;
  }
  try {
    return model->impl->find_charge_batch(*batch, *result);
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_find_descriptors(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaFindDescriptorResult* result) {
  nep_adapters::clear_last_error();
  if (model == nullptr || batch == nullptr || result == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }
  if (!nep_adapters::is_fully_periodic(*batch)) {
    nep_adapters::set_last_error(
        "NEPAdapters supports fully periodic structures only (pbc=[1,1,1])");
    return NEPA_STATUS_UNSUPPORTED;
  }

  try {
    return model->impl->find_descriptors(*batch, *result);
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_find_dipoles(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaDipoleResult* result) {
  nep_adapters::clear_last_error();
  if (model == nullptr || batch == nullptr || result == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }
  if (!nep_adapters::is_fully_periodic(*batch)) {
    nep_adapters::set_last_error(
        "NEPAdapters supports fully periodic structures only (pbc=[1,1,1])");
    return NEPA_STATUS_UNSUPPORTED;
  }
  if (model->impl->model_kind() != NEPA_MODEL_KIND_DIPOLE) {
    nep_adapters::set_last_error("dipoles require a dipole NEP model");
    return NEPA_STATUS_UNSUPPORTED;
  }
  try {
    return model->impl->find_dipoles(*batch, *result);
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_find_polarizabilities(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaPolarizabilityResult* result) {
  nep_adapters::clear_last_error();
  if (model == nullptr || batch == nullptr || result == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }
  if (!nep_adapters::is_fully_periodic(*batch)) {
    nep_adapters::set_last_error(
        "NEPAdapters supports fully periodic structures only (pbc=[1,1,1])");
    return NEPA_STATUS_UNSUPPORTED;
  }
  if (model->impl->model_kind() != NEPA_MODEL_KIND_POLARIZABILITY) {
    nep_adapters::set_last_error(
        "polarizabilities require a polarizability NEP model");
    return NEPA_STATUS_UNSUPPORTED;
  }
  try {
    return model->impl->find_polarizabilities(*batch, *result);
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

namespace {

NepaStatus compute_dftd3(
    NepaModel* model,
    const NepaStructureBatch* batch,
    const NepaDftd3Parameters* parameters,
    NepaDftd3Result* result,
    bool include_nep) {
  nep_adapters::clear_last_error();
  if (model == nullptr || batch == nullptr || parameters == nullptr ||
      result == nullptr || parameters->functional == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }
  if (!nep_adapters::is_fully_periodic(*batch)) {
    nep_adapters::set_last_error(
        "NEPAdapters supports fully periodic structures only (pbc=[1,1,1])");
    return NEPA_STATUS_UNSUPPORTED;
  }
  if (model->impl->model_kind() != NEPA_MODEL_KIND_ORDINARY) {
    nep_adapters::set_last_error(
        "DFT-D3 is supported only for ordinary non-spin, non-charge NEP models");
    return NEPA_STATUS_UNSUPPORTED;
  }
  try {
    return model->impl->compute_dftd3_batch(
        *batch, *parameters, *result, include_nep);
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

}  // namespace

NepaStatus nepa_compute_dftd3_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    const NepaDftd3Parameters* parameters,
    NepaDftd3Result* result) {
  return compute_dftd3(model, batch, parameters, result, false);
}

NepaStatus nepa_compute_with_dftd3_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    const NepaDftd3Parameters* parameters,
    NepaDftd3Result* result) {
  return compute_dftd3(model, batch, parameters, result, true);
}

NepaStatus nepa_find_force_lammps_neighbors(
    NepaModel* model,
    const NepaLammpsNeighborInput* input,
    NepaLammpsNeighborResult* result) {
  nep_adapters::clear_last_error();
  if (model == nullptr || input == nullptr || result == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }

  try {
    return model->impl->find_force_lammps_neighbors(*input, *result);
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_find_force_lammps_device_neighbors(
    NepaModel* model,
    const NepaLammpsDeviceNeighborInput* input,
    NepaLammpsDeviceNeighborResult* result) {
  nep_adapters::clear_last_error();
  if (model == nullptr || input == nullptr || result == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }

  try {
    return model->impl->find_force_lammps_device_neighbors(*input, *result);
  } catch (const std::exception& error) {
    nep_adapters::set_last_error(error.what());
    return NEPA_STATUS_RUNTIME_ERROR;
  }
}

NepaStatus nepa_cancel_model(NepaModel* model) {
  nep_adapters::clear_last_error();
  if (model == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }
  model->impl->cancel();
  return NEPA_STATUS_OK;
}

NepaStatus nepa_reset_cancel(NepaModel* model) {
  nep_adapters::clear_last_error();
  if (model == nullptr) {
    return NEPA_STATUS_INVALID_ARGUMENT;
  }
  model->impl->reset_cancel();
  return NEPA_STATUS_OK;
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
    case NEPA_STATUS_CANCELLED:
      return "cancelled";
  }

  return "unknown status";
}

const char* nepa_last_error_message(void) {
  return nep_adapters::last_error().c_str();
}
