#pragma once

namespace nep_adapters::cuda_backend {

struct SimulationBox {
  double frac_to_cart[9] = {};
  double cart_to_frac[9] = {};
  int pbc[3] = {};
};

}  // namespace nep_adapters::cuda_backend
