#include "angular_force.hpp"
#include "simulation_box_device.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace nep_adapters::cuda_backend {
namespace {

constexpr float kPi = 3.1415927f;
constexpr int kMaxCachedAngularBasisDerivatives = 16;
constexpr int kAngularForceThreads = 32;

void check_cuda(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(message) + ": " + cudaGetErrorString(status));
  }
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

__device__ __forceinline__ void find_fc_and_fcp(
    float rc,
    float rcinv,
    float r,
    float& fc,
    float& fcp) {
  if (r < rc) {
    const float x = r * rcinv;
    fc = 0.5f * cosf(kPi * x) + 0.5f;
    fcp = -1.5707963f * sinf(kPi * x) * rcinv;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

__device__ __forceinline__ void find_fn_and_fnp(
    int n,
    float rcinv,
    float r,
    float fc,
    float fcp,
    float& fn,
    float& fnp) {
  if (n == 0) {
    fn = fc;
    fnp = fcp;
    return;
  }

  const float r_scaled = r * rcinv;
  const float x = 2.0f * (r_scaled - 1.0f) * (r_scaled - 1.0f) - 1.0f;
  if (n == 1) {
    fn = (x + 1.0f) * 0.5f;
    fnp = 2.0f * (r_scaled - 1.0f) * rcinv * fc + fn * fcp;
    fn *= fc;
    return;
  }

  float t0 = 1.0f;
  float t1 = x;
  float t2 = x;
  float u0 = 1.0f;
  float u1 = 2.0f * x;
  for (int m = 2; m <= n; ++m) {
    t2 = 2.0f * x * t1 - t0;
    t0 = t1;
    t1 = t2;
    const float u2 = 2.0f * x * u1 - u0;
    u0 = u1;
    u1 = u2;
  }
  fn = (t2 + 1.0f) * 0.5f;
  fnp = n * u0 * 2.0f * (r_scaled - 1.0f) * rcinv;
  fnp = fnp * fc + fn * fcp;
  fn *= fc;
}

__device__ __forceinline__ void add_unit_spherical_derivative(
    float scale,
    float y_value,
    const float* grad_y,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  float projection = 0.0f;
  for (int d = 0; d < 3; ++d) {
    projection += grad_y[d] * unit[d];
  }
  for (int d = 0; d < 3; ++d) {
    const float d_y_dr =
        (grad_y[d] - unit[d] * projection) * rinv;
    force[d] += scale * (gnp * y_value * unit[d] + gn * d_y_dr);
  }
}

__device__ __forceinline__ void add_unit_spherical_derivative_accumulated(
    float gn_scale,
    float gnp_scale,
    float y_value,
    float grad_x,
    float grad_y,
    float grad_z,
    float rinv,
    const float* unit,
    float* force) {
  const float ux = unit[0];
  const float uy = unit[1];
  const float uz = unit[2];
  const float projection = grad_x * ux + grad_y * uy + grad_z * uz;
  const float radial = gnp_scale * y_value;
  const float angular = gn_scale * rinv;
  force[0] += radial * ux + angular * (grad_x - ux * projection);
  force[1] += radial * uy + angular * (grad_y - uy * projection);
  force[2] += radial * uz + angular * (grad_z - uz * projection);
}

__device__ __forceinline__ void accumulate_l1_scaled_force(
    const float* scales,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  const float y0 = unit[2];
  const float gy0[3] = {0.0f, 0.0f, 1.0f};
  add_unit_spherical_derivative(scales[0], y0, gy0, gn, gnp, rinv, unit, force);
  const float y1 = unit[0];
  const float gy1[3] = {1.0f, 0.0f, 0.0f};
  add_unit_spherical_derivative(scales[1], y1, gy1, gn, gnp, rinv, unit, force);
  const float y2 = unit[1];
  const float gy2[3] = {0.0f, 1.0f, 0.0f};
  add_unit_spherical_derivative(scales[2], y2, gy2, gn, gnp, rinv, unit, force);
}

__device__ __forceinline__ void accumulate_l1_force(
    float fp,
    const float* sum,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  const float scales[3] = {
      2.0f * 0.238732414637843f * sum[0] * fp,
      4.0f * 0.119366207318922f * sum[1] * fp,
      4.0f * 0.119366207318922f * sum[2] * fp,
  };
  accumulate_l1_scaled_force(scales, gn, gnp, rinv, unit, force);
}

__device__ __forceinline__ void accumulate_l2_scaled_force(
    const float* scales,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  const float x = unit[0];
  const float y = unit[1];
  const float z = unit[2];
  const float gy0[3] = {0.0f, 0.0f, 6.0f * z};
  add_unit_spherical_derivative(
      scales[0],
      -1.0f + 3.0f * z * z,
      gy0,
      gn,
      gnp,
      rinv,
      unit,
      force);
  const float gy1[3] = {z, 0.0f, x};
  add_unit_spherical_derivative(scales[1], z * x, gy1, gn, gnp, rinv, unit, force);
  const float gy2[3] = {0.0f, z, y};
  add_unit_spherical_derivative(scales[2], z * y, gy2, gn, gnp, rinv, unit, force);
  const float gy3[3] = {2.0f * x, -2.0f * y, 0.0f};
  add_unit_spherical_derivative(
      scales[3],
      x * x - y * y,
      gy3,
      gn,
      gnp,
      rinv,
      unit,
      force);
  const float gy4[3] = {2.0f * y, 2.0f * x, 0.0f};
  add_unit_spherical_derivative(
      scales[4],
      2.0f * x * y,
      gy4,
      gn,
      gnp,
      rinv,
      unit,
      force);
}

__device__ __forceinline__ void accumulate_l2_force(
    float fp,
    const float* sum,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  const float scales[5] = {
      2.0f * 0.099471839432435f * sum[3] * fp,
      4.0f * 0.596831036594608f * sum[4] * fp,
      4.0f * 0.596831036594608f * sum[5] * fp,
      4.0f * 0.149207759148652f * sum[6] * fp,
      4.0f * 0.149207759148652f * sum[7] * fp,
  };
  accumulate_l2_scaled_force(scales, gn, gnp, rinv, unit, force);
}

__device__ __forceinline__ float regular_scale(
    float coefficient,
    float sum_value,
    float fp,
    bool real_m0) {
  return (real_m0 ? 2.0f : 4.0f) * coefficient * sum_value * fp;
}

__device__ __forceinline__ void complex_product(
    float a_real,
    float a_imag,
    float& b_real,
    float& b_imag) {
  const float real = a_real * b_real - a_imag * b_imag;
  const float imag = a_real * b_imag + a_imag * b_real;
  b_real = real;
  b_imag = imag;
}

__device__ __forceinline__ void accumulate_l3_scaled_force(
    const float* scales,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  const float x = unit[0];
  const float y = unit[1];
  const float z = unit[2];
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float x2_minus_y2 = x2 - y2;
  const float two_xy = 2.0f * x * y;
  const float x3_minus_3xy2 = x * x2_minus_y2 - y * two_xy;
  const float three_x2y_minus_y3 = x * two_xy + y * x2_minus_y2;

  const float gy0[3] = {0.0f, 0.0f, -3.0f + 15.0f * z2};
  add_unit_spherical_derivative(
      scales[0],
      -3.0f * z + 5.0f * z * z2,
      gy0,
      gn,
      gnp,
      rinv,
      unit,
      force);
  const float gy1[3] = {-1.0f + 5.0f * z2, 0.0f, 10.0f * x * z};
  add_unit_spherical_derivative(
      scales[1],
      (-1.0f + 5.0f * z2) * x,
      gy1,
      gn,
      gnp,
      rinv,
      unit,
      force);
  const float gy2[3] = {0.0f, -1.0f + 5.0f * z2, 10.0f * y * z};
  add_unit_spherical_derivative(
      scales[2],
      (-1.0f + 5.0f * z2) * y,
      gy2,
      gn,
      gnp,
      rinv,
      unit,
      force);
  const float gy3[3] = {2.0f * x * z, -2.0f * y * z, x2_minus_y2};
  add_unit_spherical_derivative(
      scales[3], z * x2_minus_y2, gy3, gn, gnp, rinv, unit, force);
  const float gy4[3] = {2.0f * y * z, 2.0f * x * z, two_xy};
  add_unit_spherical_derivative(
      scales[4], z * two_xy, gy4, gn, gnp, rinv, unit, force);
  const float gy5[3] = {3.0f * x2 - 3.0f * y2, -6.0f * x * y, 0.0f};
  add_unit_spherical_derivative(
      scales[5],
      x3_minus_3xy2,
      gy5,
      gn,
      gnp,
      rinv,
      unit,
      force);
  const float gy6[3] = {6.0f * x * y, 3.0f * x2 - 3.0f * y2, 0.0f};
  add_unit_spherical_derivative(
      scales[6],
      three_x2y_minus_y3,
      gy6,
      gn,
      gnp,
      rinv,
      unit,
      force);
}

__device__ __forceinline__ void accumulate_l3_force(
    float fp,
    const float* sum,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  const float scales[7] = {
      regular_scale(0.139260575205408f, sum[8], fp, true),
      regular_scale(0.104445431404056f, sum[9], fp, false),
      regular_scale(0.104445431404056f, sum[10], fp, false),
      regular_scale(1.044454314040563f, sum[11], fp, false),
      regular_scale(1.044454314040563f, sum[12], fp, false),
      regular_scale(0.174075719006761f, sum[13], fp, false),
      regular_scale(0.174075719006761f, sum[14], fp, false),
  };
  accumulate_l3_scaled_force(scales, gn, gnp, rinv, unit, force);
}

__device__ __forceinline__ void accumulate_l4_scaled_force(
    const float* scales,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  const float x = unit[0];
  const float y = unit[1];
  const float z = unit[2];
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float x2_minus_y2 = x2 - y2;
  const float two_xy = 2.0f * x * y;
  const float x3_minus_3xy2 = x * x2_minus_y2 - y * two_xy;
  const float three_x2y_minus_y3 = x * two_xy + y * x2_minus_y2;
  const float x4_minus_6x2y2_plus_y4 =
      x * x3_minus_3xy2 - y * three_x2y_minus_y3;
  const float four_x3y_minus_4xy3 =
      x * three_x2y_minus_y3 + y * x3_minus_3xy2;

  const float gy0[3] = {0.0f, 0.0f, -60.0f * z + 140.0f * z * z2};
  add_unit_spherical_derivative(
      scales[0],
      3.0f - 30.0f * z2 + 35.0f * z2 * z2,
      gy0,
      gn,
      gnp,
      rinv,
      unit,
      force);
  const float z_factor = -3.0f * z + 7.0f * z * z2;
  const float gy1[3] = {z_factor, 0.0f, (-3.0f + 21.0f * z2) * x};
  add_unit_spherical_derivative(
      scales[1], z_factor * x, gy1, gn, gnp, rinv, unit, force);
  const float gy2[3] = {0.0f, z_factor, (-3.0f + 21.0f * z2) * y};
  add_unit_spherical_derivative(
      scales[2], z_factor * y, gy2, gn, gnp, rinv, unit, force);
  const float a = -1.0f + 7.0f * z2;
  const float gy3[3] = {2.0f * x * a, -2.0f * y * a, 14.0f * z * x2_minus_y2};
  add_unit_spherical_derivative(
      scales[3], a * x2_minus_y2, gy3, gn, gnp, rinv, unit, force);
  const float gy4[3] = {2.0f * y * a, 2.0f * x * a, 14.0f * z * two_xy};
  add_unit_spherical_derivative(
      scales[4], a * two_xy, gy4, gn, gnp, rinv, unit, force);
  const float gy5[3] = {
      z * (3.0f * x2 - 3.0f * y2),
      z * (-6.0f * x * y),
      x3_minus_3xy2};
  add_unit_spherical_derivative(
      scales[5], z * x3_minus_3xy2, gy5, gn, gnp, rinv, unit, force);
  const float gy6[3] = {
      z * 6.0f * x * y,
      z * (3.0f * x2 - 3.0f * y2),
      three_x2y_minus_y3};
  add_unit_spherical_derivative(
      scales[6],
      z * three_x2y_minus_y3,
      gy6,
      gn,
      gnp,
      rinv,
      unit,
      force);
  const float gy7[3] = {
      4.0f * x * x2 - 12.0f * x * y2,
      -12.0f * x2 * y + 4.0f * y * y2,
      0.0f};
  add_unit_spherical_derivative(
      scales[7],
      x4_minus_6x2y2_plus_y4,
      gy7,
      gn,
      gnp,
      rinv,
      unit,
      force);
  const float gy8[3] = {
      12.0f * x2 * y - 4.0f * y * y2,
      4.0f * x * x2 - 12.0f * x * y2,
      0.0f};
  add_unit_spherical_derivative(
      scales[8],
      four_x3y_minus_4xy3,
      gy8,
      gn,
      gnp,
      rinv,
      unit,
      force);
}

__device__ __forceinline__ void accumulate_l4_force(
    float fp,
    const float* sum,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  const float s[9] = {
      regular_scale(0.011190581936149f, sum[15], fp, true),
      regular_scale(0.223811638722978f, sum[16], fp, false),
      regular_scale(0.223811638722978f, sum[17], fp, false),
      regular_scale(0.111905819361489f, sum[18], fp, false),
      regular_scale(0.111905819361489f, sum[19], fp, false),
      regular_scale(1.566681471060845f, sum[20], fp, false),
      regular_scale(1.566681471060845f, sum[21], fp, false),
      regular_scale(0.195835183882606f, sum[22], fp, false),
      regular_scale(0.195835183882606f, sum[23], fp, false),
  };
  const float dx[3] = {
      (1.0f - unit[0] * unit[0]) * rinv,
      -unit[0] * unit[1] * rinv,
      -unit[0] * unit[2] * rinv};
  const float dy[3] = {
      -unit[0] * unit[1] * rinv,
      (1.0f - unit[1] * unit[1]) * rinv,
      -unit[1] * unit[2] * rinv};
  const float dz[3] = {
      -unit[0] * unit[2] * rinv,
      -unit[1] * unit[2] * rinv,
      (1.0f - unit[2] * unit[2]) * rinv};
  const float z = unit[2];
  const float z_pow[5] = {1.0f, z, z * z, z * z * z, z * z * z * z};
  constexpr float z_coeff[5][5] = {
      {3.0f, 0.0f, -30.0f, 0.0f, 35.0f},
      {0.0f, -3.0f, 0.0f, 7.0f, 0.0f},
      {-1.0f, 0.0f, 7.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
      {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

  float real_part = 1.0f;
  float imag_part = 0.0f;
  for (int n1 = 0; n1 <= 4; ++n1) {
    const int n2_start = (4 + n1) % 2 == 0 ? 0 : 1;
    float z_factor = 0.0f;
    float dz_factor = 0.0f;
    for (int n2 = n2_start; n2 <= 4 - n1; n2 += 2) {
      z_factor += z_coeff[n1][n2] * z_pow[n2];
      if (n2 > 0) {
        dz_factor += z_coeff[n1][n2] * n2 * z_pow[n2 - 1];
      }
    }
    if (n1 == 0) {
      for (int d = 0; d < 3; ++d) {
        force[d] += s[0] * (z_factor * gnp * unit[d] +
                            gn * dz_factor * dz[d]);
      }
    } else {
      float real_part_n1 = n1 * real_part;
      float imag_part_n1 = n1 * imag_part;
      for (int d = 0; d < 3; ++d) {
        float real_part_dx = dx[d];
        float imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        force[d] +=
            (s[2 * n1 - 1] * real_part_dx +
             s[2 * n1] * imag_part_dy) *
            z_factor * gn;
      }
      complex_product(unit[0], unit[1], real_part, imag_part);
      const float xy_temp = s[2 * n1 - 1] * real_part +
                            s[2 * n1] * imag_part;
      for (int d = 0; d < 3; ++d) {
        force[d] += xy_temp * (z_factor * gnp * unit[d] +
                               gn * dz_factor * dz[d]);
      }
    }
  }
}

__device__ __forceinline__ void accumulate_l1_accumulated_force(
    const float* gn_scale,
    const float* gnp_scale,
    float rinv,
    const float* unit,
    float* force) {
  const float y0 = unit[2];
  add_unit_spherical_derivative_accumulated(
      gn_scale[0], gnp_scale[0], y0, 0.0f, 0.0f, 1.0f, rinv, unit, force);
  const float y1 = unit[0];
  add_unit_spherical_derivative_accumulated(
      gn_scale[1], gnp_scale[1], y1, 1.0f, 0.0f, 0.0f, rinv, unit, force);
  const float y2 = unit[1];
  add_unit_spherical_derivative_accumulated(
      gn_scale[2], gnp_scale[2], y2, 0.0f, 1.0f, 0.0f, rinv, unit, force);
}

__device__ __forceinline__ void accumulate_l2_accumulated_force(
    const float* gn_scale,
    const float* gnp_scale,
    float rinv,
    const float* unit,
    float* force) {
  const float x = unit[0];
  const float y = unit[1];
  const float z = unit[2];
  add_unit_spherical_derivative_accumulated(
      gn_scale[0], gnp_scale[0], -1.0f + 3.0f * z * z,
      0.0f, 0.0f, 6.0f * z, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[1], gnp_scale[1], z * x, z, 0.0f, x, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[2], gnp_scale[2], z * y, 0.0f, z, y, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[3], gnp_scale[3], x * x - y * y,
      2.0f * x, -2.0f * y, 0.0f, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[4], gnp_scale[4], 2.0f * x * y,
      2.0f * y, 2.0f * x, 0.0f, rinv, unit, force);
}

__device__ __forceinline__ void accumulate_l3_accumulated_force(
    const float* gn_scale,
    const float* gnp_scale,
    float rinv,
    const float* unit,
    float* force) {
  const float x = unit[0];
  const float y = unit[1];
  const float z = unit[2];
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float x2_minus_y2 = x2 - y2;
  const float two_xy = 2.0f * x * y;
  const float x3_minus_3xy2 = x * x2_minus_y2 - y * two_xy;
  const float three_x2y_minus_y3 = x * two_xy + y * x2_minus_y2;

  add_unit_spherical_derivative_accumulated(
      gn_scale[0], gnp_scale[0], -3.0f * z + 5.0f * z * z2,
      0.0f, 0.0f, -3.0f + 15.0f * z2, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[1], gnp_scale[1], (-1.0f + 5.0f * z2) * x,
      -1.0f + 5.0f * z2, 0.0f, 10.0f * x * z, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[2], gnp_scale[2], (-1.0f + 5.0f * z2) * y,
      0.0f, -1.0f + 5.0f * z2, 10.0f * y * z, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[3], gnp_scale[3], z * x2_minus_y2,
      2.0f * x * z, -2.0f * y * z, x2_minus_y2, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[4], gnp_scale[4], z * two_xy,
      2.0f * y * z, 2.0f * x * z, two_xy, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[5], gnp_scale[5], x3_minus_3xy2,
      3.0f * x2 - 3.0f * y2, -6.0f * x * y, 0.0f, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[6], gnp_scale[6], three_x2y_minus_y3,
      6.0f * x * y, 3.0f * x2 - 3.0f * y2, 0.0f, rinv, unit, force);
}

__device__ __forceinline__ void accumulate_l4_accumulated_force(
    const float* gn_scale,
    const float* gnp_scale,
    float rinv,
    const float* unit,
    float* force) {
  const float x = unit[0];
  const float y = unit[1];
  const float z = unit[2];
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float x2_minus_y2 = x2 - y2;
  const float two_xy = 2.0f * x * y;
  const float x3_minus_3xy2 = x * x2_minus_y2 - y * two_xy;
  const float three_x2y_minus_y3 = x * two_xy + y * x2_minus_y2;
  const float x4_minus_6x2y2_plus_y4 =
      x * x3_minus_3xy2 - y * three_x2y_minus_y3;
  const float four_x3y_minus_4xy3 =
      x * three_x2y_minus_y3 + y * x3_minus_3xy2;

  add_unit_spherical_derivative_accumulated(
      gn_scale[0], gnp_scale[0], 3.0f - 30.0f * z2 + 35.0f * z2 * z2,
      0.0f, 0.0f, -60.0f * z + 140.0f * z * z2, rinv, unit, force);
  const float z_factor = -3.0f * z + 7.0f * z * z2;
  add_unit_spherical_derivative_accumulated(
      gn_scale[1], gnp_scale[1], z_factor * x,
      z_factor, 0.0f, (-3.0f + 21.0f * z2) * x, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[2], gnp_scale[2], z_factor * y,
      0.0f, z_factor, (-3.0f + 21.0f * z2) * y, rinv, unit, force);
  const float a = -1.0f + 7.0f * z2;
  add_unit_spherical_derivative_accumulated(
      gn_scale[3], gnp_scale[3], a * x2_minus_y2,
      2.0f * x * a, -2.0f * y * a, 14.0f * z * x2_minus_y2,
      rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[4], gnp_scale[4], a * two_xy,
      2.0f * y * a, 2.0f * x * a, 14.0f * z * two_xy,
      rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[5], gnp_scale[5], z * x3_minus_3xy2,
      z * (3.0f * x2 - 3.0f * y2), z * (-6.0f * x * y),
      x3_minus_3xy2, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[6], gnp_scale[6], z * three_x2y_minus_y3,
      z * 6.0f * x * y, z * (3.0f * x2 - 3.0f * y2),
      three_x2y_minus_y3, rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[7], gnp_scale[7], x4_minus_6x2y2_plus_y4,
      4.0f * x * x2 - 12.0f * x * y2,
      -12.0f * x2 * y + 4.0f * y * y2, 0.0f,
      rinv, unit, force);
  add_unit_spherical_derivative_accumulated(
      gn_scale[8], gnp_scale[8], four_x3y_minus_4xy3,
      12.0f * x2 * y - 4.0f * y * y2,
      4.0f * x * x2 - 12.0f * x * y2, 0.0f,
      rinv, unit, force);
}

__device__ __forceinline__ void accumulate_q1111_force(
    float fp,
    const float* sum,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  constexpr float c0 = 0.026596810706114f;
  constexpr float c1 = 0.053193621412227f;
  constexpr float c2 = 0.026596810706114f;
  const float s0_sq = sum[0] * sum[0];
  const float s12_sq = sum[1] * sum[1] + sum[2] * sum[2];
  const float scales[3] = {
      fp * (4.0f * c0 * sum[0] * s0_sq + 2.0f * c1 * sum[0] * s12_sq),
      fp * (2.0f * c1 * s0_sq * sum[1] + 4.0f * c2 * s12_sq * sum[1]),
      fp * (2.0f * c1 * s0_sq * sum[2] + 4.0f * c2 * s12_sq * sum[2]),
  };
  const float y0 = unit[2];
  const float gy0[3] = {0.0f, 0.0f, 1.0f};
  add_unit_spherical_derivative(scales[0], y0, gy0, gn, gnp, rinv, unit, force);
  const float y1 = unit[0];
  const float gy1[3] = {1.0f, 0.0f, 0.0f};
  add_unit_spherical_derivative(scales[1], y1, gy1, gn, gnp, rinv, unit, force);
  const float y2 = unit[1];
  const float gy2[3] = {0.0f, 1.0f, 0.0f};
  add_unit_spherical_derivative(scales[2], y2, gy2, gn, gnp, rinv, unit, force);
}

__device__ __forceinline__ void accumulate_q222_force(
    float fp,
    const float* sum,
    float gn,
    float gnp,
    float r,
    float rinv,
    const float* unit,
    float* force) {
  constexpr float c0 = -0.007499480826664f;
  constexpr float c1 = -0.134990654879954f;
  constexpr float c2 = 0.067495327439977f;
  constexpr float c3 = 0.404971964639861f;
  constexpr float c4 = -0.809943929279723f;
  const float s3 = sum[3];
  const float s4 = sum[4];
  const float s5 = sum[5];
  const float s6 = sum[6];
  const float s7 = sum[7];
  float fn = gn * rinv;
  float fnp = gnp * rinv - gn * rinv * rinv;
  const float fn2 = fn * rinv;
  const float fnp2 = fnp * rinv - fn * rinv * rinv;
  const float fn_factor = fp * fn2;
  const float fnp_factor = fp * fnp2 * rinv;
  const float r12[3] = {unit[0] * r, unit[1] * r, unit[2] * r};
  const float y20 = 3.0f * r12[2] * r12[2] - r * r;

  float tmp0 =
      c0 * 3.0f * s3 * s3 + c1 * (s4 * s4 + s5 * s5) +
      c2 * (s6 * s6 + s7 * s7);
  float tmp1 = tmp0 * y20 * fnp_factor;
  float tmp2 = tmp0 * fn_factor;
  force[0] += tmp1 * r12[0] - tmp2 * 2.0f * r12[0];
  force[1] += tmp1 * r12[1] - tmp2 * 2.0f * r12[1];
  force[2] += tmp1 * r12[2] + tmp2 * 4.0f * r12[2];

  tmp0 = c1 * s3 * s4 * 2.0f - c3 * s6 * s4 * 2.0f +
         c4 * s5 * s7;
  tmp1 = tmp0 * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  force[0] += tmp1 * r12[0] + tmp2 * r12[2];
  force[1] += tmp1 * r12[1];
  force[2] += tmp1 * r12[2] + tmp2 * r12[0];

  tmp0 = c1 * s3 * s5 * 2.0f + c3 * s6 * s5 * 2.0f +
         c4 * s4 * s7;
  tmp1 = tmp0 * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  force[0] += tmp1 * r12[0];
  force[1] += tmp1 * r12[1] + tmp2 * r12[2];
  force[2] += tmp1 * r12[2] + tmp2 * r12[1];

  tmp0 = c2 * s3 * s6 * 2.0f + c3 * (s5 * s5 - s4 * s4);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  force[0] += tmp1 * r12[0] + tmp2 * 2.0f * r12[0];
  force[1] += tmp1 * r12[1] - tmp2 * 2.0f * r12[1];
  force[2] += tmp1 * r12[2];

  tmp0 = c2 * s3 * s7 * 2.0f + c4 * s4 * s5;
  tmp1 = tmp0 * (2.0f * r12[0] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  force[0] += tmp1 * r12[0] + tmp2 * 2.0f * r12[1];
  force[1] += tmp1 * r12[1] + tmp2 * 2.0f * r12[0];
  force[2] += tmp1 * r12[2];
}

__device__ __forceinline__ void add_accumulated_scale(
    float scale,
    float gn,
    float gnp,
    float& gn_scale,
    float& gnp_scale) {
  gn_scale += scale * gn;
  gnp_scale += scale * gnp;
}

__device__ __forceinline__ void accumulate_l4_q222_only_accumulated_force(
    const float* sum_cache,
    const float* fp_cache,
    const float* gn_values,
    const float* gnp_values,
    float rinv,
    const float* unit,
    float* force) {
  {
    float l1_gn[3] = {0.0f};
    float l1_gnp[3] = {0.0f};
    #pragma unroll 5
    for (int n = 0; n < 5; ++n) {
      const float gn = gn_values[n];
      const float gnp = gnp_values[n];
      const float* s = sum_cache + n * 24;
      const float fp_l1 = fp_cache[n * 5];
      add_accumulated_scale(
          2.0f * 0.238732414637843f * s[0] * fp_l1,
          gn,
          gnp,
          l1_gn[0],
          l1_gnp[0]);
      add_accumulated_scale(
          4.0f * 0.119366207318922f * s[1] * fp_l1,
          gn,
          gnp,
          l1_gn[1],
          l1_gnp[1]);
      add_accumulated_scale(
          4.0f * 0.119366207318922f * s[2] * fp_l1,
          gn,
          gnp,
          l1_gn[2],
          l1_gnp[2]);
    }
    accumulate_l1_accumulated_force(l1_gn, l1_gnp, rinv, unit, force);
  }

  {
    float l2_gn[5] = {0.0f};
    float l2_gnp[5] = {0.0f};
    #pragma unroll 5
    for (int n = 0; n < 5; ++n) {
      const float gn = gn_values[n];
      const float gnp = gnp_values[n];
      const float* s = sum_cache + n * 24;
      const float fp_l2 = fp_cache[n * 5 + 1];
      add_accumulated_scale(
          2.0f * 0.099471839432435f * s[3] * fp_l2,
          gn,
          gnp,
          l2_gn[0],
          l2_gnp[0]);
      add_accumulated_scale(
          4.0f * 0.596831036594608f * s[4] * fp_l2,
          gn,
          gnp,
          l2_gn[1],
          l2_gnp[1]);
      add_accumulated_scale(
          4.0f * 0.596831036594608f * s[5] * fp_l2,
          gn,
          gnp,
          l2_gn[2],
          l2_gnp[2]);
      add_accumulated_scale(
          4.0f * 0.149207759148652f * s[6] * fp_l2,
          gn,
          gnp,
          l2_gn[3],
          l2_gnp[3]);
      add_accumulated_scale(
          4.0f * 0.149207759148652f * s[7] * fp_l2,
          gn,
          gnp,
          l2_gn[4],
          l2_gnp[4]);

      constexpr float c0 = -0.007499480826664f;
      constexpr float c1 = -0.134990654879954f;
      constexpr float c2 = 0.067495327439977f;
      constexpr float c3 = 0.404971964639861f;
      constexpr float c4 = -0.809943929279723f;
      const float s3 = s[3];
      const float s4 = s[4];
      const float s5 = s[5];
      const float s6 = s[6];
      const float s7 = s[7];
      const float fp_q222 = fp_cache[n * 5 + 4];
      add_accumulated_scale(
          fp_q222 * (3.0f * c0 * s3 * s3 + c1 * (s4 * s4 + s5 * s5) +
                     c2 * (s6 * s6 + s7 * s7)),
          gn,
          gnp,
          l2_gn[0],
          l2_gnp[0]);
      add_accumulated_scale(
          fp_q222 *
              (2.0f * c1 * s3 * s4 - 2.0f * c3 * s6 * s4 + c4 * s5 * s7),
          gn,
          gnp,
          l2_gn[1],
          l2_gnp[1]);
      add_accumulated_scale(
          fp_q222 *
              (2.0f * c1 * s3 * s5 + 2.0f * c3 * s6 * s5 + c4 * s4 * s7),
          gn,
          gnp,
          l2_gn[2],
          l2_gnp[2]);
      add_accumulated_scale(
          fp_q222 * (2.0f * c2 * s3 * s6 + c3 * (s5 * s5 - s4 * s4)),
          gn,
          gnp,
          l2_gn[3],
          l2_gnp[3]);
      add_accumulated_scale(
          fp_q222 * (2.0f * c2 * s3 * s7 + c4 * s4 * s5),
          gn,
          gnp,
          l2_gn[4],
          l2_gnp[4]);
    }
    accumulate_l2_accumulated_force(l2_gn, l2_gnp, rinv, unit, force);
  }

  {
    float l3_gn[7] = {0.0f};
    float l3_gnp[7] = {0.0f};
    #pragma unroll 5
    for (int n = 0; n < 5; ++n) {
      const float gn = gn_values[n];
      const float gnp = gnp_values[n];
      const float* s = sum_cache + n * 24;
      const float fp_l3 = fp_cache[n * 5 + 2];
      add_accumulated_scale(
          regular_scale(0.139260575205408f, s[8], fp_l3, true),
          gn,
          gnp,
          l3_gn[0],
          l3_gnp[0]);
      add_accumulated_scale(
          regular_scale(0.104445431404056f, s[9], fp_l3, false),
          gn,
          gnp,
          l3_gn[1],
          l3_gnp[1]);
      add_accumulated_scale(
          regular_scale(0.104445431404056f, s[10], fp_l3, false),
          gn,
          gnp,
          l3_gn[2],
          l3_gnp[2]);
      add_accumulated_scale(
          regular_scale(1.044454314040563f, s[11], fp_l3, false),
          gn,
          gnp,
          l3_gn[3],
          l3_gnp[3]);
      add_accumulated_scale(
          regular_scale(1.044454314040563f, s[12], fp_l3, false),
          gn,
          gnp,
          l3_gn[4],
          l3_gnp[4]);
      add_accumulated_scale(
          regular_scale(0.174075719006761f, s[13], fp_l3, false),
          gn,
          gnp,
          l3_gn[5],
          l3_gnp[5]);
      add_accumulated_scale(
          regular_scale(0.174075719006761f, s[14], fp_l3, false),
          gn,
          gnp,
          l3_gn[6],
          l3_gnp[6]);
    }
    accumulate_l3_accumulated_force(l3_gn, l3_gnp, rinv, unit, force);
  }

  {
    float l4_gn[9] = {0.0f};
    float l4_gnp[9] = {0.0f};
    #pragma unroll 4
    for (int n = 0; n < 5; ++n) {
      const float gn = gn_values[n];
      const float gnp = gnp_values[n];
      const float* s = sum_cache + n * 24;
      const float fp_l4 = fp_cache[n * 5 + 3];
      add_accumulated_scale(
          regular_scale(0.011190581936149f, s[15], fp_l4, true),
          gn,
          gnp,
          l4_gn[0],
          l4_gnp[0]);
      add_accumulated_scale(
          regular_scale(0.223811638722978f, s[16], fp_l4, false),
          gn,
          gnp,
          l4_gn[1],
          l4_gnp[1]);
      add_accumulated_scale(
          regular_scale(0.223811638722978f, s[17], fp_l4, false),
          gn,
          gnp,
          l4_gn[2],
          l4_gnp[2]);
      add_accumulated_scale(
          regular_scale(0.111905819361489f, s[18], fp_l4, false),
          gn,
          gnp,
          l4_gn[3],
          l4_gnp[3]);
      add_accumulated_scale(
          regular_scale(0.111905819361489f, s[19], fp_l4, false),
          gn,
          gnp,
          l4_gn[4],
          l4_gnp[4]);
      add_accumulated_scale(
          regular_scale(1.566681471060845f, s[20], fp_l4, false),
          gn,
          gnp,
          l4_gn[5],
          l4_gnp[5]);
      add_accumulated_scale(
          regular_scale(1.566681471060845f, s[21], fp_l4, false),
          gn,
          gnp,
          l4_gn[6],
          l4_gnp[6]);
      add_accumulated_scale(
          regular_scale(0.195835183882606f, s[22], fp_l4, false),
          gn,
          gnp,
          l4_gn[7],
          l4_gnp[7]);
      add_accumulated_scale(
          regular_scale(0.195835183882606f, s[23], fp_l4, false),
          gn,
          gnp,
          l4_gn[8],
          l4_gnp[8]);
    }
    accumulate_l4_accumulated_force(l4_gn, l4_gnp, rinv, unit, force);
  }
}

__device__ __forceinline__ void accumulate_q112_force(
    float fp,
    const float* sum,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  constexpr float c0 = 0.027493550848847f;
  constexpr float c1 = 0.164961305093080f;
  constexpr float c2 = -0.013746775424423f;
  constexpr float c3 = 0.041240326273270f;
  constexpr float c4 = 0.082480652546540f;
  const float s0 = sum[0];
  const float s1 = sum[1];
  const float s2 = sum[2];
  const float s3 = sum[3];
  const float s4 = sum[4];
  const float s5 = sum[5];
  const float s6 = sum[6];
  const float s7 = sum[7];
  const float l1_scales[3] = {
      fp * (2.0f * c0 * s0 * s3 + c1 * (s1 * s4 + s2 * s5)),
      fp * (c1 * s0 * s4 + 2.0f * c2 * s3 * s1 +
            2.0f * c3 * s6 * s1 + c4 * s2 * s7),
      fp * (c1 * s0 * s5 + 2.0f * c2 * s3 * s2 -
            2.0f * c3 * s6 * s2 + c4 * s1 * s7),
  };
  const float l2_scales[5] = {
      fp * (c0 * s0 * s0 + c2 * (s1 * s1 + s2 * s2)),
      fp * c1 * s0 * s1,
      fp * c1 * s0 * s2,
      fp * c3 * (s1 * s1 - s2 * s2),
      fp * c4 * s1 * s2,
  };
  accumulate_l1_scaled_force(l1_scales, gn, gnp, rinv, unit, force);
  accumulate_l2_scaled_force(l2_scales, gn, gnp, rinv, unit, force);
}

__device__ __forceinline__ void accumulate_q123_force(
    float fp,
    const float* sum,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  const float s0 = sum[0];
  const float s1 = sum[1];
  const float s2 = sum[2];
  const float s3 = sum[3];
  const float s4 = sum[4];
  const float s5 = sum[5];
  const float s6 = sum[6];
  const float s7 = sum[7];
  const float s8 = sum[8];
  const float s9 = sum[9];
  const float s10 = sum[10];
  const float s11 = sum[11];
  const float s12 = sum[12];
  const float s13 = sum[13];
  const float s14 = sum[14];
  const float l1_scales[3] = {
      fp * (-0.084181463496172f * s11 * s6 -
            0.084181463496172f * s12 * s7 -
            0.067345170796937f * s10 * s5 -
            0.067345170796937f * s4 * s9 -
            0.016836292699234f * s3 * s8),
      fp * (-0.168362926992344f * s11 * s4 -
            0.168362926992344f * s12 * s5 -
            0.042090731748086f * s13 * s6 -
            0.042090731748086f * s14 * s7 -
            0.016836292699234f * s3 * s9 +
            0.008418146349617f * s10 * s7 +
            0.008418146349617f * s6 * s9 +
            0.033672585398469f * s4 * s8),
      fp * (-0.168362926992344f * s12 * s4 +
            0.168362926992344f * s11 * s5 -
            0.042090731748086f * s14 * s6 +
            0.042090731748086f * s13 * s7 -
            0.016836292699234f * s10 * s3 -
            0.008418146349617f * s10 * s6 +
            0.008418146349617f * s7 * s9 +
            0.033672585398469f * s5 * s8),
  };
  const float l2_scales[5] = {
      fp * (-0.016836292699234f * s10 * s2 -
            0.016836292699234f * s0 * s8 -
            0.016836292699234f * s1 * s9),
      fp * (-0.168362926992344f * s12 * s2 -
            0.168362926992344f * s1 * s11 -
            0.067345170796937f * s0 * s9 +
            0.033672585398469f * s1 * s8),
      fp * (0.168362926992344f * s11 * s2 -
            0.168362926992344f * s1 * s12 -
            0.067345170796937f * s10 * s0 +
            0.033672585398469f * s2 * s8),
      fp * (-0.084181463496172f * s0 * s11 -
            0.042090731748086f * s14 * s2 -
            0.042090731748086f * s1 * s13 -
            0.008418146349617f * s10 * s2 +
            0.008418146349617f * s1 * s9),
      fp * (-0.084181463496172f * s0 * s12 +
            0.042090731748086f * s13 * s2 -
            0.042090731748086f * s1 * s14 +
            0.008418146349617f * s10 * s1 +
            0.008418146349617f * s2 * s9),
  };
  const float l3_scales[7] = {
      fp * (-0.016836292699234f * s0 * s3 +
            0.033672585398469f * s2 * s5 +
            0.033672585398469f * s1 * s4),
      fp * (-0.067345170796937f * s0 * s4 -
            0.016836292699234f * s1 * s3 +
            0.008418146349617f * s2 * s7 +
            0.008418146349617f * s1 * s6),
      fp * (-0.067345170796937f * s0 * s5 -
            0.016836292699234f * s2 * s3 -
            0.008418146349617f * s2 * s6 +
            0.008418146349617f * s1 * s7),
      fp * (0.168362926992344f * s2 * s5 -
            0.168362926992344f * s1 * s4 -
            0.084181463496172f * s0 * s6),
      fp * (-0.168362926992344f * s2 * s4 -
            0.168362926992344f * s1 * s5 -
            0.084181463496172f * s0 * s7),
      fp * (0.042090731748086f * s2 * s7 -
            0.042090731748086f * s1 * s6),
      fp * (-0.042090731748086f * s2 * s6 -
            0.042090731748086f * s1 * s7),
  };
  accumulate_l1_scaled_force(l1_scales, gn, gnp, rinv, unit, force);
  accumulate_l2_scaled_force(l2_scales, gn, gnp, rinv, unit, force);
  accumulate_l3_scaled_force(l3_scales, gn, gnp, rinv, unit, force);
}

__device__ __forceinline__ void accumulate_q233_force(
    float fp,
    const float* sum,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  const float s3 = sum[3];
  const float s4 = sum[4];
  const float s5 = sum[5];
  const float s6 = sum[6];
  const float s7 = sum[7];
  const float s8 = sum[8];
  const float s9 = sum[9];
  const float s10 = sum[10];
  const float s11 = sum[11];
  const float s12 = sum[12];
  const float s13 = sum[13];
  const float s14 = sum[14];
  const float l2_scales[5] = {
      fp * (0.008572620635186f * s8 * s8 +
            0.009644198214584f * s10 * s10 +
            0.009644198214584f * s9 * s9 -
            0.026789439484956f * s13 * s13 -
            0.026789439484956f * s14 * s14),
      fp * (0.025717861905558f * s8 * s9 +
            0.192883964291685f * s11 * s9 +
            0.192883964291685f * s10 * s12 +
            0.321473273819474f * s12 * s14 +
            0.321473273819474f * s13 * s11),
      fp * (0.025717861905558f * s10 * s8 +
            0.192883964291685f * s12 * s9 -
            0.192883964291685f * s10 * s11 +
            0.321473273819474f * s11 * s14 -
            0.321473273819474f * s13 * s12),
      fp * (-0.019288396429168f * s10 * s10 +
            0.019288396429168f * s9 * s9 -
            0.032147327381947f * s13 * s9 -
            0.032147327381947f * s10 * s14 -
            0.128589309527790f * s11 * s8),
      fp * (-0.032147327381947f * s14 * s9 +
            0.032147327381947f * s10 * s13 +
            0.038576792858337f * s10 * s9 -
            0.128589309527790f * s12 * s8),
  };
  const float l3_scales[7] = {
      fp * (0.017145241270372f * s3 * s8 +
            0.025717861905558f * s4 * s9 +
            0.025717861905558f * s10 * s5 -
            0.128589309527790f * s11 * s6 -
            0.128589309527790f * s12 * s7),
      fp * (0.019288396429168f * s3 * s9 +
            0.038576792858336f * s6 * s9 +
            0.025717861905558f * s4 * s8 -
            0.032147327381947f * s14 * s7 -
            0.032147327381947f * s13 * s6 +
            0.038576792858337f * s10 * s7 +
            0.192883964291685f * s11 * s4 +
            0.192883964291685f * s12 * s5),
      fp * (0.019288396429168f * s10 * s3 -
            0.038576792858336f * s10 * s6 +
            0.025717861905558f * s5 * s8 -
            0.032147327381947f * s14 * s6 +
            0.032147327381947f * s13 * s7 +
            0.038576792858337f * s7 * s9 +
            0.192883964291685f * s12 * s4 -
            0.192883964291685f * s11 * s5),
      fp * (-0.128589309527790f * s6 * s8 +
            0.192883964291685f * s4 * s9 -
            0.192883964291685f * s10 * s5 +
            0.321473273819474f * s14 * s5 +
            0.321473273819474f * s13 * s4),
      fp * (-0.128589309527790f * s7 * s8 +
            0.192883964291685f * s5 * s9 +
            0.192883964291685f * s10 * s4 +
            0.321473273819474f * s14 * s4 -
            0.321473273819474f * s13 * s5),
      fp * (-0.053578878969912f * s13 * s3 -
            0.032147327381947f * s6 * s9 +
            0.032147327381947f * s10 * s7 +
            0.321473273819474f * s11 * s4 -
            0.321473273819474f * s12 * s5),
      fp * (-0.053578878969912f * s14 * s3 -
            0.032147327381947f * s7 * s9 -
            0.032147327381947f * s10 * s6 +
            0.321473273819474f * s12 * s4 +
            0.321473273819474f * s11 * s5),
  };
  accumulate_l2_scaled_force(l2_scales, gn, gnp, rinv, unit, force);
  accumulate_l3_scaled_force(l3_scales, gn, gnp, rinv, unit, force);
}

__device__ __forceinline__ void accumulate_q134_force(
    float fp,
    const float* sum,
    float gn,
    float gnp,
    float rinv,
    const float* unit,
    float* force) {
  const float s0 = sum[0];
  const float s1 = sum[1];
  const float s2 = sum[2];
  const float s8 = sum[8];
  const float s9 = sum[9];
  const float s10 = sum[10];
  const float s11 = sum[11];
  const float s12 = sum[12];
  const float s13 = sum[13];
  const float s14 = sum[14];
  const float s15 = sum[15];
  const float s16 = sum[16];
  const float s17 = sum[17];
  const float s18 = sum[18];
  const float s19 = sum[19];
  const float s20 = sum[20];
  const float s21 = sum[21];
  const float s22 = sum[22];
  const float s23 = sum[23];
  const float l1_scales[3] = {
      fp * (0.004860219061029f * s15 * s8 +
            0.036451642957719f * s10 * s17 +
            0.036451642957719f * s16 * s9 +
            0.072903285915437f * s11 * s18 +
            0.072903285915437f * s12 * s19 +
            0.085053833568010f * s13 * s20 +
            0.085053833568010f * s14 * s21),
      fp * (-0.003645164295772f * s15 * s9 -
            0.006075273826286f * s13 * s18 -
            0.006075273826286f * s14 * s19 +
            0.018225821478859f * s10 * s19 +
            0.018225821478859f * s18 * s9 +
            0.024301095305146f * s16 * s8 -
            0.036451642957719f * s11 * s16 -
            0.036451642957719f * s12 * s17 +
            0.042526916784005f * s13 * s22 +
            0.042526916784005f * s14 * s23 +
            0.255161500704030f * s11 * s20 +
            0.255161500704030f * s12 * s21),
      fp * (-0.003645164295772f * s10 * s15 -
            0.006075273826286f * s14 * s18 +
            0.006075273826286f * s13 * s19 -
            0.018225821478859f * s10 * s18 +
            0.018225821478859f * s19 * s9 +
            0.024301095305146f * s17 * s8 -
            0.036451642957719f * s12 * s16 +
            0.036451642957719f * s11 * s17 -
            0.042526916784005f * s14 * s22 +
            0.042526916784005f * s13 * s23 -
            0.255161500704030f * s12 * s20 +
            0.255161500704030f * s11 * s21),
  };
  const float l3_scales[7] = {
      fp * (0.004860219061029f * s0 * s15 +
            0.024301095305146f * s1 * s16 +
            0.024301095305146f * s2 * s17),
      fp * (-0.003645164295772f * s1 * s15 +
            0.018225821478859f * s1 * s18 +
            0.018225821478859f * s2 * s19 +
            0.036451642957719f * s0 * s16),
      fp * (-0.003645164295772f * s15 * s2 -
            0.018225821478859f * s18 * s2 +
            0.018225821478859f * s1 * s19 +
            0.036451642957719f * s0 * s17),
      fp * (-0.036451642957719f * s1 * s16 +
            0.036451642957719f * s2 * s17 +
            0.072903285915437f * s0 * s18 +
            0.255161500704030f * s1 * s20 +
            0.255161500704030f * s2 * s21),
      fp * (-0.036451642957719f * s1 * s17 -
            0.036451642957719f * s2 * s16 +
            0.072903285915437f * s0 * s19 +
            0.255161500704030f * s1 * s21 -
            0.255161500704030f * s2 * s20),
      fp * (-0.006075273826286f * s1 * s18 +
            0.006075273826286f * s2 * s19 +
            0.042526916784005f * s1 * s22 +
            0.042526916784005f * s2 * s23 +
            0.085053833568010f * s0 * s20),
      fp * (-0.006075273826286f * s1 * s19 -
            0.006075273826286f * s2 * s18 +
            0.042526916784005f * s1 * s23 -
            0.042526916784005f * s2 * s22 +
            0.085053833568010f * s0 * s21),
  };
  const float l4_scales[9] = {
      fp * (-0.003645164295772f * s10 * s2 -
            0.003645164295772f * s1 * s9 +
            0.004860219061029f * s0 * s8),
      fp * (0.024301095305146f * s1 * s8 +
            0.036451642957719f * s0 * s9 -
            0.036451642957719f * s1 * s11 -
            0.036451642957719f * s2 * s12),
      fp * (0.024301095305146f * s2 * s8 +
            0.036451642957719f * s0 * s10 -
            0.036451642957719f * s1 * s12 +
            0.036451642957719f * s2 * s11),
      fp * (-0.006075273826286f * s1 * s13 -
            0.006075273826286f * s2 * s14 -
            0.018225821478859f * s10 * s2 +
            0.018225821478859f * s1 * s9 +
            0.072903285915437f * s0 * s11),
      fp * (-0.006075273826286f * s1 * s14 +
            0.006075273826286f * s2 * s13 +
            0.018225821478859f * s1 * s10 +
            0.018225821478859f * s2 * s9 +
            0.072903285915437f * s0 * s12),
      fp * (0.085053833568010f * s0 * s13 +
            0.255161500704030f * s1 * s11 -
            0.255161500704030f * s2 * s12),
      fp * (0.085053833568010f * s0 * s14 +
            0.255161500704030f * s1 * s12 +
            0.255161500704030f * s2 * s11),
      fp * (0.042526916784005f * s1 * s13 -
            0.042526916784005f * s2 * s14),
      fp * (0.042526916784005f * s1 * s14 +
            0.042526916784005f * s2 * s13),
  };
  accumulate_l1_scaled_force(l1_scales, gn, gnp, rinv, unit, force);
  accumulate_l3_scaled_force(l3_scales, gn, gnp, rinv, unit, force);
  accumulate_l4_scaled_force(l4_scales, gn, gnp, rinv, unit, force);
}

template <
    bool EdgeParallel,
    bool BlockParallel,
    bool Batched,
    bool AccumulateVirial,
    int LMax,
    int HasQ222,
    int HasQ1111,
    int HasQ112,
    int HasQ123,
    int HasQ233,
    int HasQ134,
    int NMaxAngular,
    int BasisSizeAngular>
__global__ void accumulate_l2_angular_forces(
    int atom_count,
    int atom_stride,
    int num_types,
    int num_type_pairs,
    int n_max_radial,
    int basis_size_radial,
    int n_max_angular,
    int basis_size_angular,
    int l_max_3body,
    int has_q_222,
    int has_q_1111,
    int has_q_112,
    int has_q_123,
    int has_q_233,
    int has_q_134,
    int angular_capacity,
    int abc_count,
    float cutoff_angular,
    SimulationBox single_box,
    const int* __restrict__ atom_to_structure,
    const double* __restrict__ boxes_row_major9,
    const double* __restrict__ box_inverse_row_major9,
    const int* __restrict__ pbc_flags3,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_angular,
    const int* __restrict__ nl_angular,
    const float* __restrict__ r12_angular,
    const float* __restrict__ f12x,
    const float* __restrict__ f12y,
    const float* __restrict__ f12z,
    const float* __restrict__ fp,
    const float* __restrict__ sum_fxyz,
    const float* __restrict__ descriptor_coefficients,
    double* __restrict__ force_soa3,
    double* __restrict__ virial_soa9,
    int virial_to_neighbor) {
  const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  int atom = BlockParallel ? blockIdx.x : thread_index;
  int edge_slot = BlockParallel ? threadIdx.x : (EdgeParallel ? blockIdx.y : 0);
  if (atom >= atom_count) {
    return;
  }
  if constexpr (EdgeParallel || BlockParallel) {
    if (edge_slot >= angular_capacity || edge_slot >= nn_angular[atom]) {
      if constexpr (!BlockParallel) {
        return;
      }
    }
  }

  const int radial_dim = n_max_radial + 1;
  const int n_max_angular_local =
      NMaxAngular >= 0 ? NMaxAngular : n_max_angular;
  const int basis_size_angular_local =
      BasisSizeAngular >= 0 ? BasisSizeAngular : basis_size_angular;
  const int angular_descriptor_stride = n_max_angular_local + 1;
  const int angular_basis_count =
      angular_descriptor_stride * (basis_size_angular_local + 1);
  const int angular_coefficient_offset =
      num_type_pairs * (n_max_radial + 1) * (basis_size_radial + 1);
  const int angular_dim_per_n = LMax >= 0 ? LMax : l_max_3body;
  const int type1 = types[atom];
  SimulationBox atom_box = single_box;
  if constexpr (Batched) {
    atom_box = load_structure_box(
        atom_to_structure[atom],
        boxes_row_major9,
        box_inverse_row_major9,
        pbc_flags3);
  }
  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  const float rcinv = 1.0f / cutoff_angular;
  constexpr bool UseBlockQ222Cache =
      BlockParallel && !Batched && !AccumulateVirial && LMax == 4 &&
      HasQ222 == 1 && HasQ1111 == 0 && HasQ112 == 0 && HasQ123 == 0 &&
      HasQ233 == 0 && HasQ134 == 0 && NMaxAngular == 4;
  constexpr bool UseBlockCompactQ222Cache =
      BlockParallel && !Batched && !AccumulateVirial && LMax == 4 &&
      HasQ222 == 1 && HasQ1111 == 0 && HasQ112 == 0 && HasQ123 == 0 &&
      HasQ233 == 0 && HasQ134 == 0 && NMaxAngular == 2 &&
      BasisSizeAngular == 4;
  constexpr bool UseBlockInputCache =
      UseBlockQ222Cache || UseBlockCompactQ222Cache;
  constexpr int BlockNCount = NMaxAngular >= 0 ? NMaxAngular + 1 : 0;
  __shared__ float block_sum_cache[5 * 24];
  __shared__ float block_fp_cache[5 * 5];
  if constexpr (UseBlockInputCache) {
    for (int i = threadIdx.x; i < BlockNCount * 24; i += blockDim.x) {
      block_sum_cache[i] =
          sum_fxyz[atom + atom_stride * i];
    }
    for (int i = threadIdx.x; i < BlockNCount * 5; i += blockDim.x) {
      const int n = i / 5;
      const int channel = i - n * 5;
      block_fp_cache[i] =
          fp[atom + atom_stride * (radial_dim + channel * 5 + n)];
    }
    __syncthreads();
  }

  float s_fx = 0.0f;
  float s_fy = 0.0f;
  float s_fz = 0.0f;
  float s_sxx = 0.0f;
  float s_sxy = 0.0f;
  float s_sxz = 0.0f;
  float s_syx = 0.0f;
  float s_syy = 0.0f;
  float s_syz = 0.0f;
  float s_szx = 0.0f;
  float s_szy = 0.0f;
  float s_szz = 0.0f;

  const int slot_begin = (EdgeParallel || BlockParallel) ? edge_slot : 0;
  const int slot_end =
      EdgeParallel ? edge_slot + 1 : (BlockParallel ? nn_angular[atom] : nn_angular[atom]);
  const int slot_step = BlockParallel ? blockDim.x : 1;
  for (int slot = slot_begin; slot < slot_end; slot += slot_step) {
    const int pair_offset = atom + atom_stride * slot;
    const int neighbor = nl_angular[pair_offset];
    const int type2 = types[neighbor];
    const int type_pair = type1 * num_types + type2;
    const float* coefficient_pair =
        descriptor_coefficients + angular_coefficient_offset +
        type_pair * angular_basis_count;
    float x12 = 0.0f;
    float y12 = 0.0f;
    float z12 = 0.0f;
    if (f12x != nullptr) {
      x12 = f12x[pair_offset];
      y12 = f12y[pair_offset];
      z12 = f12z[pair_offset];
    } else {
      minimum_image_delta(
          atom_box,
          positions_soa3[neighbor] - xi,
          positions_soa3[atom_stride + neighbor] - yi,
          positions_soa3[2 * atom_stride + neighbor] - zi,
          x12,
          y12,
          z12);
    }
    const float r = r12_angular[pair_offset];
    if (r <= 0.0f) {
      continue;
    }
    const float rinv = 1.0f / r;
    const float unit[3] = {x12 * rinv, y12 * rinv, z12 * rinv};

    float fc = 0.0f;
    float fcp = 0.0f;
    find_fc_and_fcp(cutoff_angular, rcinv, r, fc, fcp);
    float fn_cache[kMaxCachedAngularBasisDerivatives];
    float fnp_cache[kMaxCachedAngularBasisDerivatives];
    const bool cache_basis =
        basis_size_angular_local + 1 <= kMaxCachedAngularBasisDerivatives;
    if (cache_basis) {
      for (int k = 0; k <= basis_size_angular_local; ++k) {
        float fn = 0.0f;
        float fnp = 0.0f;
        find_fn_and_fnp(k, rcinv, r, fc, fcp, fn, fnp);
        fn_cache[k] = fn;
        fnp_cache[k] = fnp;
      }
    }
    float f12[3] = {0.0f, 0.0f, 0.0f};
    if constexpr (UseBlockQ222Cache) {
      float gn_values[5] = {0.0f};
      float gnp_values[5] = {0.0f};
      #pragma unroll
      for (int n = 0; n < 5; ++n) {
        float gn = 0.0f;
        float gnp = 0.0f;
        #pragma unroll
        for (int k = 0; k <= 8; ++k) {
          const float coefficient = coefficient_pair[n * 9 + k];
          gn += fn_cache[k] * coefficient;
          gnp += fnp_cache[k] * coefficient;
        }
        gn_values[n] = gn;
        gnp_values[n] = gnp;
      }
      accumulate_l4_q222_only_accumulated_force(
          block_sum_cache,
          block_fp_cache,
          gn_values,
          gnp_values,
          rinv,
          unit,
          f12);
    } else {
      for (int n = 0; n <= n_max_angular_local; ++n) {
      float gn = 0.0f;
      float gnp = 0.0f;
      for (int k = 0; k <= basis_size_angular_local; ++k) {
        float fn = 0.0f;
        float fnp = 0.0f;
        if (cache_basis) {
          fn = fn_cache[k];
          fnp = fnp_cache[k];
        } else {
          find_fn_and_fnp(k, rcinv, r, fc, fcp, fn, fnp);
        }
        const float coefficient =
            coefficient_pair[n * (basis_size_angular_local + 1) + k];
        gn += fn * coefficient;
        gnp += fnp * coefficient;
      }

      float local_sum_storage[24] = {0.0f};
      const float* local_sum = local_sum_storage;
      if constexpr (UseBlockCompactQ222Cache) {
        local_sum = block_sum_cache + n * 24;
      } else {
        for (int abc = 0; abc < abc_count && abc < 24; ++abc) {
          local_sum_storage[abc] =
              sum_fxyz[atom + atom_stride * (n * abc_count + abc)];
        }
      }
      const float fp_l1 = UseBlockInputCache
          ? block_fp_cache[n * 5]
          : fp[atom + atom_stride * (radial_dim + n)];
      accumulate_l1_force(fp_l1, local_sum, gn, gnp, rinv, unit, f12);
      if constexpr (LMax >= 0) {
        if constexpr (LMax >= 2) {
          const float fp_l2 = UseBlockInputCache
              ? block_fp_cache[n * 5 + 1]
              : fp[atom + atom_stride *
                                (radial_dim + angular_descriptor_stride + n)];
          accumulate_l2_force(fp_l2, local_sum, gn, gnp, rinv, unit, f12);
        }
      } else if (angular_dim_per_n >= 2) {
        const float fp_l2 =
            fp[atom + atom_stride *
                          (radial_dim + angular_descriptor_stride + n)];
        accumulate_l2_force(fp_l2, local_sum, gn, gnp, rinv, unit, f12);
      }
      if constexpr (LMax >= 0) {
        if constexpr (LMax >= 3) {
          const float fp_l3 = UseBlockInputCache
              ? block_fp_cache[n * 5 + 2]
              : fp[atom + atom_stride *
                                (radial_dim + 2 * angular_descriptor_stride + n)];
          accumulate_l3_force(fp_l3, local_sum, gn, gnp, rinv, unit, f12);
        }
      } else if (angular_dim_per_n >= 3) {
        const float fp_l3 =
            fp[atom + atom_stride *
                          (radial_dim + 2 * angular_descriptor_stride + n)];
        accumulate_l3_force(fp_l3, local_sum, gn, gnp, rinv, unit, f12);
      }
      if constexpr (LMax >= 0) {
        if constexpr (LMax >= 4) {
          const float fp_l4 = UseBlockInputCache
              ? block_fp_cache[n * 5 + 3]
              : fp[atom + atom_stride *
                                (radial_dim + 3 * angular_descriptor_stride + n)];
          accumulate_l4_force(fp_l4, local_sum, gn, gnp, rinv, unit, f12);
        }
      } else if (angular_dim_per_n >= 4) {
        const float fp_l4 =
            fp[atom + atom_stride *
                          (radial_dim + 3 * angular_descriptor_stride + n)];
        accumulate_l4_force(fp_l4, local_sum, gn, gnp, rinv, unit, f12);
      }
      int high_body_channel = angular_dim_per_n;
      if constexpr (HasQ222 == 1) {
        const int q222_descriptor =
            radial_dim + high_body_channel * angular_descriptor_stride + n;
        const float fp_q222 = UseBlockInputCache
            ? block_fp_cache[n * 5 + 4]
            : fp[atom + atom_stride * q222_descriptor];
        accumulate_q222_force(fp_q222, local_sum, gn, gnp, r, rinv, unit, f12);
        ++high_body_channel;
      } else if constexpr (HasQ222 < 0) {
        if (has_q_222) {
          const int q222_descriptor =
              radial_dim + high_body_channel * angular_descriptor_stride + n;
          const float fp_q222 = fp[atom + atom_stride * q222_descriptor];
          accumulate_q222_force(fp_q222, local_sum, gn, gnp, r, rinv, unit, f12);
          ++high_body_channel;
        }
      }
      if constexpr (HasQ1111 == 1) {
        const int q1111_descriptor =
            radial_dim + high_body_channel * angular_descriptor_stride + n;
        const float fp_q1111 = fp[atom + atom_stride * q1111_descriptor];
        accumulate_q1111_force(fp_q1111, local_sum, gn, gnp, rinv, unit, f12);
        ++high_body_channel;
      } else if constexpr (HasQ1111 < 0) {
        if (has_q_1111) {
          const int q1111_descriptor =
              radial_dim + high_body_channel * angular_descriptor_stride + n;
          const float fp_q1111 = fp[atom + atom_stride * q1111_descriptor];
          accumulate_q1111_force(fp_q1111, local_sum, gn, gnp, rinv, unit, f12);
          ++high_body_channel;
        }
      }
      if constexpr (HasQ112 == 1) {
        const int q112_descriptor =
            radial_dim + high_body_channel * angular_descriptor_stride + n;
        const float fp_q112 = fp[atom + atom_stride * q112_descriptor];
        accumulate_q112_force(fp_q112, local_sum, gn, gnp, rinv, unit, f12);
        ++high_body_channel;
      } else if constexpr (HasQ112 < 0) {
        if (has_q_112) {
          const int q112_descriptor =
              radial_dim + high_body_channel * angular_descriptor_stride + n;
          const float fp_q112 = fp[atom + atom_stride * q112_descriptor];
          accumulate_q112_force(fp_q112, local_sum, gn, gnp, rinv, unit, f12);
          ++high_body_channel;
        }
      }
      if constexpr (HasQ123 == 1) {
        const int q123_descriptor =
            radial_dim + high_body_channel * angular_descriptor_stride + n;
        const float fp_q123 = fp[atom + atom_stride * q123_descriptor];
        accumulate_q123_force(fp_q123, local_sum, gn, gnp, rinv, unit, f12);
        ++high_body_channel;
      } else if constexpr (HasQ123 < 0) {
        if (has_q_123) {
          const int q123_descriptor =
              radial_dim + high_body_channel * angular_descriptor_stride + n;
          const float fp_q123 = fp[atom + atom_stride * q123_descriptor];
          accumulate_q123_force(fp_q123, local_sum, gn, gnp, rinv, unit, f12);
          ++high_body_channel;
        }
      }
      if constexpr (HasQ233 == 1) {
        const int q233_descriptor =
            radial_dim + high_body_channel * angular_descriptor_stride + n;
        const float fp_q233 = fp[atom + atom_stride * q233_descriptor];
        accumulate_q233_force(fp_q233, local_sum, gn, gnp, rinv, unit, f12);
        ++high_body_channel;
      } else if constexpr (HasQ233 < 0) {
        if (has_q_233) {
          const int q233_descriptor =
              radial_dim + high_body_channel * angular_descriptor_stride + n;
          const float fp_q233 = fp[atom + atom_stride * q233_descriptor];
          accumulate_q233_force(fp_q233, local_sum, gn, gnp, rinv, unit, f12);
          ++high_body_channel;
        }
      }
      if constexpr (HasQ134 == 1) {
        const int q134_descriptor =
            radial_dim + high_body_channel * angular_descriptor_stride + n;
        const float fp_q134 = fp[atom + atom_stride * q134_descriptor];
        accumulate_q134_force(fp_q134, local_sum, gn, gnp, rinv, unit, f12);
      } else if constexpr (HasQ134 < 0) {
        if (has_q_134) {
          const int q134_descriptor =
              radial_dim + high_body_channel * angular_descriptor_stride + n;
          const float fp_q134 = fp[atom + atom_stride * q134_descriptor];
          accumulate_q134_force(fp_q134, local_sum, gn, gnp, rinv, unit, f12);
        }
      }
    }
    }

    s_fx += f12[0];
    s_fy += f12[1];
    s_fz += f12[2];
    atomicAdd(&force_soa3[neighbor], -static_cast<double>(f12[0]));
    atomicAdd(&force_soa3[atom_stride + neighbor], -static_cast<double>(f12[1]));
    atomicAdd(&force_soa3[2 * atom_stride + neighbor], -static_cast<double>(f12[2]));

    if constexpr (AccumulateVirial) {
      if (virial_to_neighbor) {
        atomicAdd(&virial_soa9[neighbor], -static_cast<double>(x12 * f12[0]));
        atomicAdd(
            &virial_soa9[atom_stride + neighbor],
            -static_cast<double>(y12 * f12[1]));
        atomicAdd(
            &virial_soa9[2 * atom_stride + neighbor],
            -static_cast<double>(z12 * f12[2]));
        atomicAdd(
            &virial_soa9[3 * atom_stride + neighbor],
            -static_cast<double>(x12 * f12[1]));
        atomicAdd(
            &virial_soa9[4 * atom_stride + neighbor],
            -static_cast<double>(x12 * f12[2]));
        atomicAdd(
            &virial_soa9[5 * atom_stride + neighbor],
            -static_cast<double>(y12 * f12[2]));
        atomicAdd(
            &virial_soa9[6 * atom_stride + neighbor],
            -static_cast<double>(y12 * f12[0]));
        atomicAdd(
            &virial_soa9[7 * atom_stride + neighbor],
            -static_cast<double>(z12 * f12[0]));
        atomicAdd(
            &virial_soa9[8 * atom_stride + neighbor],
            -static_cast<double>(z12 * f12[1]));
      } else {
        s_sxx -= x12 * f12[0];
        s_syy -= y12 * f12[1];
        s_szz -= z12 * f12[2];
        s_sxy -= x12 * f12[1];
        s_sxz -= x12 * f12[2];
        s_syz -= y12 * f12[2];
        s_syx -= y12 * f12[0];
        s_szx -= z12 * f12[0];
        s_szy -= z12 * f12[1];
      }
    }
  }

  if constexpr (BlockParallel) {
    __shared__ float reduce_fx[kAngularForceThreads];
    __shared__ float reduce_fy[kAngularForceThreads];
    __shared__ float reduce_fz[kAngularForceThreads];
    reduce_fx[threadIdx.x] = s_fx;
    reduce_fy[threadIdx.x] = s_fy;
    reduce_fz[threadIdx.x] = s_fz;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce_fx[threadIdx.x] += reduce_fx[threadIdx.x + stride];
        reduce_fy[threadIdx.x] += reduce_fy[threadIdx.x + stride];
        reduce_fz[threadIdx.x] += reduce_fz[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      atomicAdd(&force_soa3[atom], static_cast<double>(reduce_fx[0]));
      atomicAdd(&force_soa3[atom_stride + atom], static_cast<double>(reduce_fy[0]));
      atomicAdd(&force_soa3[2 * atom_stride + atom], static_cast<double>(reduce_fz[0]));
    }
  } else {
    atomicAdd(&force_soa3[atom], static_cast<double>(s_fx));
    atomicAdd(&force_soa3[atom_stride + atom], static_cast<double>(s_fy));
    atomicAdd(&force_soa3[2 * atom_stride + atom], static_cast<double>(s_fz));
  }
  if constexpr (AccumulateVirial) {
    if (virial_to_neighbor) {
      return;
    }
    virial_soa9[atom] += static_cast<double>(s_sxx);
    virial_soa9[atom_stride + atom] += static_cast<double>(s_syy);
    virial_soa9[2 * atom_stride + atom] += static_cast<double>(s_szz);
    virial_soa9[3 * atom_stride + atom] += static_cast<double>(s_sxy);
    virial_soa9[4 * atom_stride + atom] += static_cast<double>(s_sxz);
    virial_soa9[5 * atom_stride + atom] += static_cast<double>(s_syz);
    virial_soa9[6 * atom_stride + atom] += static_cast<double>(s_syx);
    virial_soa9[7 * atom_stride + atom] += static_cast<double>(s_szx);
    virial_soa9[8 * atom_stride + atom] += static_cast<double>(s_szy);
  }
}

}  // namespace

void accumulate_l2_angular_forces_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool accumulate_virial,
    bool virial_to_neighbor) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.body_channels.l_max_3body >= 1 &&
              protocol.body_channels.l_max_3body <= 4,
          "angular force kernel supports l_max_3body 1..4");
  require(!protocol.body_channels.has_q_222 || protocol.body_channels.l_max_3body >= 2,
          "q222 angular force requires l_max_3body >= 2");
  require(!protocol.body_channels.has_q_112 || protocol.body_channels.l_max_3body >= 2,
          "q112 angular force requires l_max_3body >= 2");
  require(!protocol.body_channels.has_q_123 || protocol.body_channels.l_max_3body >= 3,
          "q123 angular force requires l_max_3body >= 3");
  require(!protocol.body_channels.has_q_233 || protocol.body_channels.l_max_3body >= 3,
          "q233 angular force requires l_max_3body >= 3");
  require(!protocol.body_channels.has_q_134 || protocol.body_channels.l_max_3body >= 4,
          "q134 angular force requires l_max_3body >= 4");
  require(protocol.cutoff_angular > 0.0, "angular cutoff must be positive");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.types != nullptr, "workspace missing atom types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_angular != nullptr, "workspace missing angular counts");
  require(view.nl_angular_slot_major != nullptr,
          "workspace missing angular neighbors");
  require(view.r12_angular != nullptr, "workspace missing angular distance cache");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.sum_fxyz != nullptr, "workspace missing angular sums");
  require(view.force_soa3 != nullptr, "workspace missing force output");
  require(view.virial_soa9 != nullptr, "workspace missing virial output");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_type_pair_major != nullptr,
          "model missing type-pair-major descriptor coefficients");

  const int threads = kAngularForceThreads;
  const int blocks = (atom_count + threads - 1) / threads;
  const bool block_parallel =
      !accumulate_virial && protocol.neighbor_capacity_angular >= threads / 2;
  const bool edge_parallel =
      !block_parallel &&
      !accumulate_virial && !protocol.has_zbl &&
      protocol.neighbor_capacity_angular > 0;
  if (blocks > 0) {
    const bool l4_q222_q1111 =
        protocol.body_channels.l_max_3body == 4 &&
        protocol.body_channels.has_q_222 &&
        protocol.body_channels.has_q_1111 &&
        !protocol.body_channels.has_q_112 &&
        !protocol.body_channels.has_q_123 &&
        !protocol.body_channels.has_q_233 &&
        !protocol.body_channels.has_q_134 &&
        protocol.n_max_angular == 4 &&
        protocol.basis_size_angular == 8;
    const bool l4_q222_only =
        protocol.body_channels.l_max_3body == 4 &&
        protocol.body_channels.has_q_222 &&
        !protocol.body_channels.has_q_1111 &&
        !protocol.body_channels.has_q_112 &&
        !protocol.body_channels.has_q_123 &&
        !protocol.body_channels.has_q_233 &&
        !protocol.body_channels.has_q_134 &&
        protocol.n_max_angular == 4 &&
        protocol.basis_size_angular == 8;
    const bool l4_compact_q222_only =
        protocol.body_channels.l_max_3body == 4 &&
        protocol.body_channels.has_q_222 &&
        !protocol.body_channels.has_q_1111 &&
        !protocol.body_channels.has_q_112 &&
        !protocol.body_channels.has_q_123 &&
        !protocol.body_channels.has_q_233 &&
        !protocol.body_channels.has_q_134 &&
        protocol.n_max_angular == 2 &&
        protocol.basis_size_angular == 4;
    if (l4_q222_q1111 && accumulate_virial) {
      accumulate_l2_angular_forces<false, false, false, true, 4, 1, 1, 0, 0, 0, 0, 4, 8><<<blocks, threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.num_types,
          protocol.num_types * protocol.num_types,
          protocol.n_max_radial,
          protocol.basis_size_radial,
          protocol.n_max_angular,
          protocol.basis_size_angular,
          protocol.body_channels.l_max_3body,
          protocol.body_channels.has_q_222 ? 1 : 0,
          protocol.body_channels.has_q_1111 ? 1 : 0,
          protocol.body_channels.has_q_112 ? 1 : 0,
          protocol.body_channels.has_q_123 ? 1 : 0,
          protocol.body_channels.has_q_233 ? 1 : 0,
          protocol.body_channels.has_q_134 ? 1 : 0,
          protocol.neighbor_capacity_angular,
          protocol.body_channels.abc_count(),
          static_cast<float>(protocol.cutoff_angular),
          box,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          view.types,
          view.positions_soa3,
          view.nn_angular,
          view.nl_angular_slot_major,
          view.r12_angular,
          view.f12x,
          view.f12y,
          view.f12z,
          view.fp,
          view.sum_fxyz,
          model_view.descriptor_coefficients_type_pair_major,
          view.force_soa3,
          view.virial_soa9,
          virial_to_neighbor ? 1 : 0);
    } else if (l4_q222_q1111) {
      accumulate_l2_angular_forces<false, false, false, false, 4, 1, 1, 0, 0, 0, 0, 4, 8><<<blocks, threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.num_types,
          protocol.num_types * protocol.num_types,
          protocol.n_max_radial,
          protocol.basis_size_radial,
          protocol.n_max_angular,
          protocol.basis_size_angular,
          protocol.body_channels.l_max_3body,
          protocol.body_channels.has_q_222 ? 1 : 0,
          protocol.body_channels.has_q_1111 ? 1 : 0,
          protocol.body_channels.has_q_112 ? 1 : 0,
          protocol.body_channels.has_q_123 ? 1 : 0,
          protocol.body_channels.has_q_233 ? 1 : 0,
          protocol.body_channels.has_q_134 ? 1 : 0,
          protocol.neighbor_capacity_angular,
          protocol.body_channels.abc_count(),
          static_cast<float>(protocol.cutoff_angular),
          box,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          view.types,
          view.positions_soa3,
          view.nn_angular,
          view.nl_angular_slot_major,
          view.r12_angular,
          view.f12x,
          view.f12y,
          view.f12z,
          view.fp,
          view.sum_fxyz,
          model_view.descriptor_coefficients_type_pair_major,
          view.force_soa3,
          view.virial_soa9,
          virial_to_neighbor ? 1 : 0);
    } else if (l4_q222_only && !accumulate_virial) {
      if (edge_parallel) {
        const dim3 edge_grid(
            (atom_count + threads - 1) / threads,
            protocol.neighbor_capacity_angular);
        accumulate_l2_angular_forces<true, false, false, false, 4, 1, 0, 0, 0, 0, 0, 4, 8><<<edge_grid, threads>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            protocol.num_types,
            protocol.num_types * protocol.num_types,
            protocol.n_max_radial,
            protocol.basis_size_radial,
            protocol.n_max_angular,
            protocol.basis_size_angular,
            protocol.body_channels.l_max_3body,
            protocol.body_channels.has_q_222 ? 1 : 0,
            protocol.body_channels.has_q_1111 ? 1 : 0,
            protocol.body_channels.has_q_112 ? 1 : 0,
            protocol.body_channels.has_q_123 ? 1 : 0,
            protocol.body_channels.has_q_233 ? 1 : 0,
            protocol.body_channels.has_q_134 ? 1 : 0,
            protocol.neighbor_capacity_angular,
            protocol.body_channels.abc_count(),
            static_cast<float>(protocol.cutoff_angular),
            box,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            view.types,
            view.positions_soa3,
            view.nn_angular,
            view.nl_angular_slot_major,
            view.r12_angular,
            view.f12x,
            view.f12y,
            view.f12z,
            view.fp,
            view.sum_fxyz,
            model_view.descriptor_coefficients_type_pair_major,
            view.force_soa3,
            view.virial_soa9,
          virial_to_neighbor ? 1 : 0);
      } else if (block_parallel) {
        accumulate_l2_angular_forces<false, true, false, false, 4, 1, 0, 0, 0, 0, 0, 4, 8><<<atom_count, threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.num_types,
          protocol.num_types * protocol.num_types,
          protocol.n_max_radial,
          protocol.basis_size_radial,
          protocol.n_max_angular,
          protocol.basis_size_angular,
          protocol.body_channels.l_max_3body,
          protocol.body_channels.has_q_222 ? 1 : 0,
          protocol.body_channels.has_q_1111 ? 1 : 0,
          protocol.body_channels.has_q_112 ? 1 : 0,
          protocol.body_channels.has_q_123 ? 1 : 0,
          protocol.body_channels.has_q_233 ? 1 : 0,
          protocol.body_channels.has_q_134 ? 1 : 0,
          protocol.neighbor_capacity_angular,
          protocol.body_channels.abc_count(),
          static_cast<float>(protocol.cutoff_angular),
          box,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          view.types,
          view.positions_soa3,
          view.nn_angular,
          view.nl_angular_slot_major,
          view.r12_angular,
          view.f12x,
          view.f12y,
          view.f12z,
          view.fp,
          view.sum_fxyz,
          model_view.descriptor_coefficients_type_pair_major,
          view.force_soa3,
          view.virial_soa9,
          virial_to_neighbor ? 1 : 0);
      } else {
        accumulate_l2_angular_forces<false, false, false, false, 4, 1, 0, 0, 0, 0, 0, 4, 8><<<blocks, threads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.num_types,
          protocol.num_types * protocol.num_types,
          protocol.n_max_radial,
          protocol.basis_size_radial,
          protocol.n_max_angular,
          protocol.basis_size_angular,
          protocol.body_channels.l_max_3body,
          protocol.body_channels.has_q_222 ? 1 : 0,
          protocol.body_channels.has_q_1111 ? 1 : 0,
          protocol.body_channels.has_q_112 ? 1 : 0,
          protocol.body_channels.has_q_123 ? 1 : 0,
          protocol.body_channels.has_q_233 ? 1 : 0,
          protocol.body_channels.has_q_134 ? 1 : 0,
          protocol.neighbor_capacity_angular,
          protocol.body_channels.abc_count(),
          static_cast<float>(protocol.cutoff_angular),
          box,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          view.types,
          view.positions_soa3,
          view.nn_angular,
          view.nl_angular_slot_major,
          view.r12_angular,
          view.f12x,
          view.f12y,
          view.f12z,
          view.fp,
          view.sum_fxyz,
          model_view.descriptor_coefficients_type_pair_major,
          view.force_soa3,
          view.virial_soa9,
          virial_to_neighbor ? 1 : 0);
      }
    } else if (l4_compact_q222_only && block_parallel) {
      accumulate_l2_angular_forces<
          false, true, false, false, 4, 1, 0, 0, 0, 0, 0, 2, 4>
          <<<atom_count, threads>>>(
              atom_count,
              static_cast<int>(view.atom_capacity),
              protocol.num_types,
              protocol.num_types * protocol.num_types,
              protocol.n_max_radial,
              protocol.basis_size_radial,
              protocol.n_max_angular,
              protocol.basis_size_angular,
              protocol.body_channels.l_max_3body,
              protocol.body_channels.has_q_222 ? 1 : 0,
              protocol.body_channels.has_q_1111 ? 1 : 0,
              protocol.body_channels.has_q_112 ? 1 : 0,
              protocol.body_channels.has_q_123 ? 1 : 0,
              protocol.body_channels.has_q_233 ? 1 : 0,
              protocol.body_channels.has_q_134 ? 1 : 0,
              protocol.neighbor_capacity_angular,
              protocol.body_channels.abc_count(),
              static_cast<float>(protocol.cutoff_angular),
              box,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              view.types,
              view.positions_soa3,
              view.nn_angular,
              view.nl_angular_slot_major,
              view.r12_angular,
              view.f12x,
              view.f12y,
              view.f12z,
              view.fp,
              view.sum_fxyz,
              model_view.descriptor_coefficients_type_pair_major,
              view.force_soa3,
              view.virial_soa9,
              virial_to_neighbor ? 1 : 0);
    } else if (accumulate_virial) {
      accumulate_l2_angular_forces<false, false, false, true, -1, -1, -1, -1, -1, -1, -1, -1, -1>
          <<<blocks, threads>>>(
              atom_count,
              static_cast<int>(view.atom_capacity),
              protocol.num_types,
              protocol.num_types * protocol.num_types,
              protocol.n_max_radial,
              protocol.basis_size_radial,
              protocol.n_max_angular,
              protocol.basis_size_angular,
              protocol.body_channels.l_max_3body,
              protocol.body_channels.has_q_222 ? 1 : 0,
              protocol.body_channels.has_q_1111 ? 1 : 0,
              protocol.body_channels.has_q_112 ? 1 : 0,
              protocol.body_channels.has_q_123 ? 1 : 0,
              protocol.body_channels.has_q_233 ? 1 : 0,
              protocol.body_channels.has_q_134 ? 1 : 0,
              protocol.neighbor_capacity_angular,
              protocol.body_channels.abc_count(),
              static_cast<float>(protocol.cutoff_angular),
              box,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              view.types,
              view.positions_soa3,
              view.nn_angular,
              view.nl_angular_slot_major,
              view.r12_angular,
              view.f12x,
              view.f12y,
              view.f12z,
              view.fp,
              view.sum_fxyz,
              model_view.descriptor_coefficients_type_pair_major,
              view.force_soa3,
              view.virial_soa9,
          virial_to_neighbor ? 1 : 0);
    } else if (block_parallel) {
      accumulate_l2_angular_forces<false, true, false, false, -1, -1, -1, -1, -1, -1, -1, -1, -1>
          <<<atom_count, threads>>>(
              atom_count,
              static_cast<int>(view.atom_capacity),
              protocol.num_types,
              protocol.num_types * protocol.num_types,
              protocol.n_max_radial,
              protocol.basis_size_radial,
              protocol.n_max_angular,
              protocol.basis_size_angular,
              protocol.body_channels.l_max_3body,
              protocol.body_channels.has_q_222 ? 1 : 0,
              protocol.body_channels.has_q_1111 ? 1 : 0,
              protocol.body_channels.has_q_112 ? 1 : 0,
              protocol.body_channels.has_q_123 ? 1 : 0,
              protocol.body_channels.has_q_233 ? 1 : 0,
              protocol.body_channels.has_q_134 ? 1 : 0,
              protocol.neighbor_capacity_angular,
              protocol.body_channels.abc_count(),
              static_cast<float>(protocol.cutoff_angular),
              box,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              view.types,
              view.positions_soa3,
              view.nn_angular,
              view.nl_angular_slot_major,
              view.r12_angular,
              view.f12x,
              view.f12y,
              view.f12z,
              view.fp,
              view.sum_fxyz,
              model_view.descriptor_coefficients_type_pair_major,
              view.force_soa3,
              view.virial_soa9,
          virial_to_neighbor ? 1 : 0);
    } else {
      accumulate_l2_angular_forces<false, false, false, false, -1, -1, -1, -1, -1, -1, -1, -1, -1>
          <<<blocks, threads>>>(
              atom_count,
              static_cast<int>(view.atom_capacity),
              protocol.num_types,
              protocol.num_types * protocol.num_types,
              protocol.n_max_radial,
              protocol.basis_size_radial,
              protocol.n_max_angular,
              protocol.basis_size_angular,
              protocol.body_channels.l_max_3body,
              protocol.body_channels.has_q_222 ? 1 : 0,
              protocol.body_channels.has_q_1111 ? 1 : 0,
              protocol.body_channels.has_q_112 ? 1 : 0,
              protocol.body_channels.has_q_123 ? 1 : 0,
              protocol.body_channels.has_q_233 ? 1 : 0,
              protocol.body_channels.has_q_134 ? 1 : 0,
              protocol.neighbor_capacity_angular,
              protocol.body_channels.abc_count(),
              static_cast<float>(protocol.cutoff_angular),
              box,
              nullptr,
              nullptr,
              nullptr,
              nullptr,
              view.types,
              view.positions_soa3,
              view.nn_angular,
              view.nl_angular_slot_major,
              view.r12_angular,
              view.f12x,
              view.f12y,
              view.f12z,
              view.fp,
              view.sum_fxyz,
              model_view.descriptor_coefficients_type_pair_major,
              view.force_soa3,
              view.virial_soa9,
          virial_to_neighbor ? 1 : 0);
    }
  }
  check_cuda(cudaGetLastError(), "accumulate angular forces kernel launch failed");
}

void accumulate_l2_angular_forces_batched(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.body_channels.l_max_3body >= 1 &&
              protocol.body_channels.l_max_3body <= 4,
          "angular force kernel supports l_max_3body 1..4");
  require(!protocol.body_channels.has_q_222 || protocol.body_channels.l_max_3body >= 2,
          "q222 angular force requires l_max_3body >= 2");
  require(!protocol.body_channels.has_q_112 || protocol.body_channels.l_max_3body >= 2,
          "q112 angular force requires l_max_3body >= 2");
  require(!protocol.body_channels.has_q_123 || protocol.body_channels.l_max_3body >= 3,
          "q123 angular force requires l_max_3body >= 3");
  require(!protocol.body_channels.has_q_233 || protocol.body_channels.l_max_3body >= 3,
          "q233 angular force requires l_max_3body >= 3");
  require(!protocol.body_channels.has_q_134 || protocol.body_channels.l_max_3body >= 4,
          "q134 angular force requires l_max_3body >= 4");
  require(protocol.cutoff_angular > 0.0, "angular cutoff must be positive");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.atom_to_structure != nullptr, "workspace missing atom_to_structure");
  require(view.boxes_row_major9 != nullptr, "workspace missing boxes");
  require(view.box_inverse_row_major9 != nullptr, "workspace missing box inverses");
  require(view.pbc_flags3 != nullptr, "workspace missing pbc flags");
  require(view.types != nullptr, "workspace missing atom types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_angular != nullptr, "workspace missing angular counts");
  require(view.nl_angular_slot_major != nullptr, "workspace missing angular neighbors");
  require(view.r12_angular != nullptr, "workspace missing angular distance cache");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.sum_fxyz != nullptr, "workspace missing angular sums");
  require(view.force_soa3 != nullptr, "workspace missing force output");
  require(view.virial_soa9 != nullptr, "workspace missing virial output");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_type_pair_major != nullptr,
          "model missing type-pair-major descriptor coefficients");

  const int threads = kAngularForceThreads;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    accumulate_l2_angular_forces<false, false, true, true, -1, -1, -1, -1, -1, -1, -1, -1, -1>
        <<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.num_types,
        protocol.num_types * protocol.num_types,
        protocol.n_max_radial,
        protocol.basis_size_radial,
        protocol.n_max_angular,
        protocol.basis_size_angular,
        protocol.body_channels.l_max_3body,
        protocol.body_channels.has_q_222 ? 1 : 0,
        protocol.body_channels.has_q_1111 ? 1 : 0,
        protocol.body_channels.has_q_112 ? 1 : 0,
        protocol.body_channels.has_q_123 ? 1 : 0,
        protocol.body_channels.has_q_233 ? 1 : 0,
        protocol.body_channels.has_q_134 ? 1 : 0,
        protocol.neighbor_capacity_angular,
        protocol.body_channels.abc_count(),
        static_cast<float>(protocol.cutoff_angular),
        SimulationBox{},
        view.atom_to_structure,
        view.boxes_row_major9,
        view.box_inverse_row_major9,
        view.pbc_flags3,
        view.types,
        view.positions_soa3,
        view.nn_angular,
        view.nl_angular_slot_major,
        view.r12_angular,
        view.f12x,
        view.f12y,
        view.f12z,
        view.fp,
        view.sum_fxyz,
        model_view.descriptor_coefficients_type_pair_major,
        view.force_soa3,
        view.virial_soa9,
        0);
  }
  check_cuda(cudaGetLastError(),
             "accumulate batched angular forces kernel launch failed");
  check_cuda(cudaDeviceSynchronize(), "accumulate batched angular forces kernel failed");
}

}  // namespace nep_adapters::cuda_backend
