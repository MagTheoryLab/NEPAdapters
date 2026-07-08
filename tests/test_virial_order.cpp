#include "nep_adapters/virial_order.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

bool nearly_equal(double lhs, double rhs) {
  return std::abs(lhs - rhs) < 1.0e-12;
}

}  // namespace

int main() {
  // Same tensor, two public raw9 contracts:
  // batch/compute raw9: xx, xy, xz, yx, yy, yz, zx, zy, zz
  // LAMMPS raw9:        xx, yy, zz, xy, xz, yz, yx, zx, zy
  const std::array<double, 9> nep_compute_raw9 = {
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0};
  const std::array<double, 9> lammps_raw9 = {
      1.0, 5.0, 9.0,
      2.0, 3.0, 6.0,
      4.0, 7.0, 8.0};
  const std::array<double, 6> expected_from_compute = {
      1.0, 5.0, 9.0, 3.0, 5.0, 7.0};
  const std::array<double, 6> expected_from_lammps = {
      1.0, 5.0, 9.0, 3.0, 5.0, 7.0};

  for (int component = 0; component < 6; ++component) {
    const double from_compute =
        nep_adapters::lammps_voigt6_from_nep_compute_raw9(
            nep_compute_raw9.data(),
            component);
    const double from_lammps =
        nep_adapters::lammps_voigt6_from_lammps_raw9(
            lammps_raw9.data(),
            component);
    if (!nearly_equal(from_compute, expected_from_compute[component]) ||
        !nearly_equal(from_lammps, expected_from_lammps[component]) ||
        !nearly_equal(from_compute, from_lammps)) {
      std::cerr << "virial order contract failed at component "
                << component << '\n';
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
