#pragma once

namespace nep_adapters {

inline double lammps_voigt6_from_nep_compute_raw9(
    const double raw9[9],
    int component) {
  // NEP::compute raw9 order: xx, xy, xz, yx, yy, yz, zx, zy, zz.
  switch (component) {
    case 0:
      return raw9[0];
    case 1:
      return raw9[4];
    case 2:
      return raw9[8];
    case 3:
      return 0.5 * (raw9[1] + raw9[3]);
    case 4:
      return 0.5 * (raw9[2] + raw9[6]);
    case 5:
      return 0.5 * (raw9[5] + raw9[7]);
  }
  return 0.0;
}

inline double lammps_voigt6_from_lammps_raw9(
    const double raw9[9],
    int component) {
  // NEP::compute_for_lammps raw9 order: xx, yy, zz, xy, xz, yz, yx, zx, zy.
  switch (component) {
    case 0:
      return raw9[0];
    case 1:
      return raw9[1];
    case 2:
      return raw9[2];
    case 3:
      return 0.5 * (raw9[3] + raw9[6]);
    case 4:
      return 0.5 * (raw9[4] + raw9[7]);
    case 5:
      return 0.5 * (raw9[5] + raw9[8]);
  }
  return 0.0;
}

}  // namespace nep_adapters
