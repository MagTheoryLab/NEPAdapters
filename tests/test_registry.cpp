#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#include "nep_adapters/engine.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace {

class FakeModel : public nep_adapters::Model {
 public:
  NepaStatus find_force_batch(
      const NepaStructureBatch& batch,
      NepaFindForceResult& result) override {
    if (batch.num_structures == 1 && result.energy_per_structure != nullptr) {
      result.energy_per_structure[0] = 42.0;
    }
    return NEPA_STATUS_OK;
  }
};

class FakeEngine : public nep_adapters::Engine {
 public:
  nep_adapters::EngineInfo info() const override {
    return {
        "fake",
        "0.0.0",
        nep_adapters::to_mask(nep_adapters::Capability::batch_find_force)};
  }

  NepaStatus load_model(
      const std::string& model_path,
      std::unique_ptr<nep_adapters::Model>& out) override {
    if (model_path.empty()) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }

    out = std::make_unique<FakeModel>();
    return NEPA_STATUS_OK;
  }
};

}  // namespace

int main() {
  if (nepa_api_version() != 100) {
    return EXIT_FAILURE;
  }

  if (nepa_backend_count() != 0) {
    return EXIT_FAILURE;
  }

  static FakeEngine fake_engine;
  if (!nep_adapters::register_engine(&fake_engine)) {
    return EXIT_FAILURE;
  }

  if (nep_adapters::register_engine(&fake_engine)) {
    return EXIT_FAILURE;
  }

  if (nepa_backend_count() != 1) {
    return EXIT_FAILURE;
  }

  NepaBackendInfo info{};
  if (nepa_backend_info(0, &info) != NEPA_STATUS_OK) {
    return EXIT_FAILURE;
  }

  if (std::strcmp(info.name, "fake") != 0) {
    return EXIT_FAILURE;
  }

  if (!nep_adapters::has_capability(
          info.capabilities, nep_adapters::Capability::batch_find_force)) {
    return EXIT_FAILURE;
  }

  if (nepa_backend_info(1, &info) != NEPA_STATUS_BACKEND_UNAVAILABLE) {
    return EXIT_FAILURE;
  }

  NepaModel* model = nullptr;
  if (nepa_load_model("missing", "nep.txt", &model) != NEPA_STATUS_BACKEND_UNAVAILABLE) {
    return EXIT_FAILURE;
  }

  if (nepa_load_model("fake", "nep.txt", &model) != NEPA_STATUS_OK || model == nullptr) {
    return EXIT_FAILURE;
  }

  NepaModelInfo model_info{};
  if (nepa_model_info(model, &model_info) != NEPA_STATUS_UNSUPPORTED) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  int32_t atom_counts[] = {1};
  int32_t atom_offsets[] = {0};
  int32_t types[] = {0};
  double positions[] = {0.0, 0.0, 0.0};
  double boxes[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  int32_t pbc[] = {1, 1, 1};
  double energy[] = {0.0};
  double forces[] = {0.0, 0.0, 0.0};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = 1;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions;
  batch.boxes_row_major9 = boxes;
  batch.pbc_flags3 = pbc;

  NepaFindForceResult result{};
  result.energy_per_structure = energy;
  result.forces_aos3 = forces;

  if (nepa_find_force_batch(model, &batch, &result) != NEPA_STATUS_OK) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  if (energy[0] != 42.0) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  nepa_free_model(model);

  if (std::strcmp(nepa_status_message(NEPA_STATUS_OK), "ok") != 0) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
