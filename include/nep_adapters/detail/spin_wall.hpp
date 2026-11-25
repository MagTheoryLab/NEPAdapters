/* Adapted from GPUMD c98b46e.
 * Copyright 2017 Zheyong Fan and GPUMD development team.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
#pragma once

#include <cmath>

namespace nep_adapters::detail {

struct SpinWall {
  double low_threshold, low_stiffness, high_threshold, high_stiffness;

#if defined(__CUDACC__) || defined(__HIPCC__)
  __host__ __device__
#endif
  bool enabled() const { return low_stiffness > 0.0 || high_stiffness > 0.0; }
};

// GPUMD c98b46e: scaled norm for the upper wall, squared-norm deficit
// for the lower wall. The latter retains its energy at zero moment.
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
inline double evaluate_spin_wall(const SpinWall& wall, const double* spin, double (&field)[3]) {
  const double scale = fmax(fabs(spin[0]), fmax(fabs(spin[1]), fabs(spin[2])));
  const double divisor = scale > 0.0 ? scale : 1.0;
  const double x = spin[0] / divisor, y = spin[1] / divisor, z = spin[2] / divisor;
  const double root = sqrt(x * x + y * y + z * z);
  const double norm = scale * root;
  if (wall.high_stiffness > 0.0 && norm > wall.high_threshold) {
    const double excess = norm - wall.high_threshold;
    const double f = (wall.high_stiffness * excess) * excess;
    field[0] = -f * (x / root);
    field[1] = -f * (y / root);
    field[2] = -f * (z / root);
    return f * (excess / 3.0);
  }
  if (wall.low_stiffness > 0.0 && norm < wall.low_threshold) {
    const double a = spin[0] / wall.low_threshold;
    const double b = spin[1] / wall.low_threshold;
    const double c = spin[2] / wall.low_threshold;
    const double deficit = fmax(0.0, 1.0 - (a * a + b * b + c * c));
    const double f = wall.low_stiffness * deficit * deficit;
    field[0] = 2.0 * (f * a);
    field[1] = 2.0 * (f * b);
    field[2] = 2.0 * (f * c);
    return f * (wall.low_threshold / 3.0) * deficit;
  }
  field[0] = field[1] = field[2] = 0.0;
  return 0.0;
}

}  // namespace nep_adapters::detail
