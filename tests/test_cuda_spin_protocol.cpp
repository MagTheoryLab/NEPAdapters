#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
  if (!nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }

  NepaModel* model = nullptr;
  const NepaStatus load_status =
      nepa_load_model("cuda", NEP_ADAPTERS_SPIN_CHIRAL_FIXTURE, &model);
  if (load_status != NEPA_STATUS_OK || model == nullptr) {
    std::cerr << "failed to load CUDA spin model, status=" << load_status << "\n";
    return EXIT_FAILURE;
  }

  NepaModelInfo info{};
  const NepaStatus info_status = nepa_model_info(model, &info);
  nepa_free_model(model);
  if (info_status != NEPA_STATUS_OK) {
    std::cerr << "model_info failed, status=" << info_status << "\n";
    return EXIT_FAILURE;
  }
  if (!nep_adapters::has_capability(
          info.capabilities,
          nep_adapters::Capability::spin)) {
    std::cerr << "CUDA spin model did not advertise spin capability\n";
    return EXIT_FAILURE;
  }
  if (!nep_adapters::has_capability(
          info.capabilities,
          nep_adapters::Capability::spin_energy_transfer)) {
    std::cerr << "CUDA spin model did not advertise spin-energy-transfer capability\n";
    return EXIT_FAILURE;
  }
  if (info.descriptor_dim != 88) {
    std::cerr << "unexpected descriptor_dim=" << info.descriptor_dim << "\n";
    return EXIT_FAILURE;
  }
  if (std::abs(info.cutoff_max - 6.0) > 1.0e-12) {
    std::cerr << "unexpected cutoff_max=" << info.cutoff_max << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
