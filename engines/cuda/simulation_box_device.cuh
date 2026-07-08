#pragma once

#include "simulation_box.hpp"

#include <math.h>

namespace nep_adapters::cuda_backend {

__device__ __forceinline__ SimulationBox load_structure_box(
    int structure,
    const double* boxes_row_major9,
    const double* box_inverse_row_major9,
    const int* pbc_flags3) {
  SimulationBox box;
  const int box_offset = 9 * structure;
  const int pbc_offset = 3 * structure;
  for (int component = 0; component < 9; ++component) {
    box.frac_to_cart[component] = boxes_row_major9[box_offset + component];
    box.cart_to_frac[component] = box_inverse_row_major9[box_offset + component];
  }
  box.pbc[0] = pbc_flags3[pbc_offset + 0];
  box.pbc[1] = pbc_flags3[pbc_offset + 1];
  box.pbc[2] = pbc_flags3[pbc_offset + 2];
  return box;
}

__device__ __forceinline__ double fractional_component(
    const SimulationBox& box,
    int row,
    double x,
    double y,
    double z) {
  return box.cart_to_frac[3 * row + 0] * x +
         box.cart_to_frac[3 * row + 1] * y +
         box.cart_to_frac[3 * row + 2] * z;
}

__device__ __forceinline__ double wrap_fractional_component(
    double value,
    int pbc) {
  if (pbc) {
    return value - floor(value);
  }
  return value;
}

__device__ __forceinline__ int fractional_component_to_cell(
    double value,
    int dim,
    int pbc) {
  const double wrapped = wrap_fractional_component(value, pbc);
  int cell = static_cast<int>(floor(wrapped * dim));
  if (cell < 0) {
    cell = 0;
  }
  if (cell >= dim) {
    cell = dim - 1;
  }
  return cell;
}

__device__ __forceinline__ void minimum_image_delta(
    const SimulationBox& box,
    double dx,
    double dy,
    double dz,
    float& x12,
    float& y12,
    float& z12) {
  if (!box.pbc[0] && !box.pbc[1] && !box.pbc[2]) {
    x12 = static_cast<float>(dx);
    y12 = static_cast<float>(dy);
    z12 = static_cast<float>(dz);
    return;
  }
  double sx = fractional_component(box, 0, dx, dy, dz);
  double sy = fractional_component(box, 1, dx, dy, dz);
  double sz = fractional_component(box, 2, dx, dy, dz);
  if (box.pbc[0]) {
    sx -= nearbyint(sx);
  }
  if (box.pbc[1]) {
    sy -= nearbyint(sy);
  }
  if (box.pbc[2]) {
    sz -= nearbyint(sz);
  }
  x12 = static_cast<float>(
      box.frac_to_cart[0] * sx +
      box.frac_to_cart[1] * sy +
      box.frac_to_cart[2] * sz);
  y12 = static_cast<float>(
      box.frac_to_cart[3] * sx +
      box.frac_to_cart[4] * sy +
      box.frac_to_cart[5] * sz);
  z12 = static_cast<float>(
      box.frac_to_cart[6] * sx +
      box.frac_to_cart[7] * sy +
      box.frac_to_cart[8] * sz);
}

}  // namespace nep_adapters::cuda_backend
