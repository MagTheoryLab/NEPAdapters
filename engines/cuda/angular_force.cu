#include "device_operations.hpp"
#include "simulation_box_device.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace nep_adapters::cuda_backend {
namespace {

constexpr float kPi = 3.1415927f;
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

struct AngularPullScales {
  float l1[3];
  float l2[5];
  float l3[7];
  float l4[9];
};

__device__ __forceinline__ AngularPullScales find_q1111_pull(
    float fp,
    const float* sum) {
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
  AngularPullScales result = {};
#pragma unroll
  for (int alpha = 0; alpha < 3; ++alpha) {
    result.l1[alpha] = scales[alpha];
  }
  return result;
}

__device__ __forceinline__ AngularPullScales find_q222_pull(
    float fp,
    const float* sum) {
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
  AngularPullScales result = {};
  result.l2[0] = fp *
      (3.0f * c0 * s3 * s3 + c1 * (s4 * s4 + s5 * s5) +
       c2 * (s6 * s6 + s7 * s7));
  result.l2[1] = fp *
      (2.0f * c1 * s3 * s4 - 2.0f * c3 * s6 * s4 + c4 * s5 * s7);
  result.l2[2] = fp *
      (2.0f * c1 * s3 * s5 + 2.0f * c3 * s6 * s5 + c4 * s4 * s7);
  result.l2[3] = fp *
      (2.0f * c2 * s3 * s6 + c3 * (s5 * s5 - s4 * s4));
  result.l2[4] = fp * (2.0f * c2 * s3 * s7 + c4 * s4 * s5);
  return result;
}

__device__ __forceinline__ AngularPullScales find_q112_pull(
    float fp,
    const float* sum) {
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
  AngularPullScales result = {};
#pragma unroll
  for (int alpha = 0; alpha < 3; ++alpha) {
    result.l1[alpha] = l1_scales[alpha];
  }
#pragma unroll
  for (int alpha = 0; alpha < 5; ++alpha) {
    result.l2[alpha] = l2_scales[alpha];
  }
  return result;
}

__device__ __forceinline__ AngularPullScales find_q123_pull(
    float fp,
    const float* sum) {
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
  AngularPullScales result = {};
#pragma unroll
  for (int alpha = 0; alpha < 3; ++alpha) {
    result.l1[alpha] = l1_scales[alpha];
  }
#pragma unroll
  for (int alpha = 0; alpha < 5; ++alpha) {
    result.l2[alpha] = l2_scales[alpha];
  }
#pragma unroll
  for (int alpha = 0; alpha < 7; ++alpha) {
    result.l3[alpha] = l3_scales[alpha];
  }
  return result;
}

__device__ __forceinline__ AngularPullScales find_q233_pull(
    float fp,
    const float* sum) {
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
  AngularPullScales result = {};
#pragma unroll
  for (int alpha = 0; alpha < 5; ++alpha) {
    result.l2[alpha] = l2_scales[alpha];
  }
#pragma unroll
  for (int alpha = 0; alpha < 7; ++alpha) {
    result.l3[alpha] = l3_scales[alpha];
  }
  return result;
}

__device__ __forceinline__ AngularPullScales find_q134_pull(
    float fp,
    const float* sum) {
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
  AngularPullScales result = {};
#pragma unroll
  for (int alpha = 0; alpha < 3; ++alpha) {
    result.l1[alpha] = l1_scales[alpha];
  }
#pragma unroll
  for (int alpha = 0; alpha < 7; ++alpha) {
    result.l3[alpha] = l3_scales[alpha];
  }
#pragma unroll
  for (int alpha = 0; alpha < 9; ++alpha) {
    result.l4[alpha] = l4_scales[alpha];
  }
  return result;
}

__device__ __forceinline__ int regular_channel(int alpha) {
  if (alpha < 3) {
    return 0;
  }
  if (alpha < 8) {
    return 1;
  }
  if (alpha < 15) {
    return 2;
  }
  return 3;
}

__device__ __forceinline__ float regular_weight(int alpha) {
  switch (alpha) {
    case 0:
      return 2.0f * 0.238732414637843f;
    case 1:
    case 2:
      return 4.0f * 0.119366207318922f;
    case 3:
      return 2.0f * 0.099471839432435f;
    case 4:
    case 5:
      return 4.0f * 0.596831036594608f;
    case 6:
    case 7:
      return 4.0f * 0.149207759148652f;
    case 8:
      return 2.0f * 0.139260575205408f;
    case 9:
    case 10:
      return 4.0f * 0.104445431404056f;
    case 11:
    case 12:
      return 4.0f * 1.044454314040563f;
    case 13:
    case 14:
      return 4.0f * 0.174075719006761f;
    case 15:
      return 2.0f * 0.011190581936149f;
    case 16:
    case 17:
      return 4.0f * 0.223811638722978f;
    case 18:
    case 19:
      return 4.0f * 0.111905819361489f;
    case 20:
    case 21:
      return 4.0f * 1.566681471060845f;
    default:
      return 4.0f * 0.195835183882606f;
  }
}

__device__ __forceinline__ void add_pull_scales(
    const AngularPullScales& source,
    float* target) {
#pragma unroll
  for (int alpha = 0; alpha < 3; ++alpha) {
    target[alpha] += source.l1[alpha];
  }
#pragma unroll
  for (int alpha = 0; alpha < 5; ++alpha) {
    target[3 + alpha] += source.l2[alpha];
  }
#pragma unroll
  for (int alpha = 0; alpha < 7; ++alpha) {
    target[8 + alpha] += source.l3[alpha];
  }
#pragma unroll
  for (int alpha = 0; alpha < 9; ++alpha) {
    target[15 + alpha] += source.l4[alpha];
  }
}

__device__ __noinline__ void build_angular_pull(
    int l_max_3body,
    int has_q_222,
    int has_q_1111,
    int has_q_112,
    int has_q_123,
    int has_q_233,
    int has_q_134,
    const float* sum,
    const float* fp_channels,
    float* pull) {
  const int abc_count = (l_max_3body + 1) * (l_max_3body + 1) - 1;
#pragma unroll
  for (int alpha = 0; alpha < 24; ++alpha) {
    pull[alpha] = alpha < abc_count
        ? regular_weight(alpha) * sum[alpha] *
              fp_channels[regular_channel(alpha)]
        : 0.0f;
  }

  int channel = l_max_3body;
  if (has_q_222) {
    add_pull_scales(find_q222_pull(fp_channels[channel], sum), pull);
    ++channel;
  }
  if (has_q_1111) {
    add_pull_scales(find_q1111_pull(fp_channels[channel], sum), pull);
    ++channel;
  }
  if (has_q_112) {
    add_pull_scales(find_q112_pull(fp_channels[channel], sum), pull);
    ++channel;
  }
  if (has_q_123) {
    add_pull_scales(find_q123_pull(fp_channels[channel], sum), pull);
    ++channel;
  }
  if (has_q_233) {
    add_pull_scales(find_q233_pull(fp_channels[channel], sum), pull);
    ++channel;
  }
  if (has_q_134) {
    add_pull_scales(find_q134_pull(fp_channels[channel], sum), pull);
  }
}

__device__ __forceinline__ void accumulate_angular_component(
    int alpha,
    float gn_scale,
    float gnp_scale,
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

  switch (alpha) {
    case 0:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, z, 0.0f, 0.0f, 1.0f, rinv, unit, force);
      break;
    case 1:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, x, 1.0f, 0.0f, 0.0f, rinv, unit, force);
      break;
    case 2:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, y, 0.0f, 1.0f, 0.0f, rinv, unit, force);
      break;
    case 3:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, -1.0f + 3.0f * z2,
          0.0f, 0.0f, 6.0f * z, rinv, unit, force);
      break;
    case 4:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, z * x, z, 0.0f, x, rinv, unit, force);
      break;
    case 5:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, z * y, 0.0f, z, y, rinv, unit, force);
      break;
    case 6:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, x2_minus_y2,
          2.0f * x, -2.0f * y, 0.0f, rinv, unit, force);
      break;
    case 7:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, two_xy,
          2.0f * y, 2.0f * x, 0.0f, rinv, unit, force);
      break;
    case 8:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, -3.0f * z + 5.0f * z * z2,
          0.0f, 0.0f, -3.0f + 15.0f * z2, rinv, unit, force);
      break;
    case 9: {
      const float a = -1.0f + 5.0f * z2;
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, a * x,
          a, 0.0f, 10.0f * x * z, rinv, unit, force);
      break;
    }
    case 10: {
      const float a = -1.0f + 5.0f * z2;
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, a * y,
          0.0f, a, 10.0f * y * z, rinv, unit, force);
      break;
    }
    case 11:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, z * x2_minus_y2,
          2.0f * x * z, -2.0f * y * z, x2_minus_y2,
          rinv, unit, force);
      break;
    case 12:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, z * two_xy,
          2.0f * y * z, 2.0f * x * z, two_xy,
          rinv, unit, force);
      break;
    case 13:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, x3_minus_3xy2,
          3.0f * x2 - 3.0f * y2, -6.0f * x * y, 0.0f,
          rinv, unit, force);
      break;
    case 14:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, three_x2y_minus_y3,
          6.0f * x * y, 3.0f * x2 - 3.0f * y2, 0.0f,
          rinv, unit, force);
      break;
    case 15:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, 3.0f - 30.0f * z2 + 35.0f * z2 * z2,
          0.0f, 0.0f, -60.0f * z + 140.0f * z * z2,
          rinv, unit, force);
      break;
    case 16: {
      const float z_factor = -3.0f * z + 7.0f * z * z2;
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, z_factor * x,
          z_factor, 0.0f, (-3.0f + 21.0f * z2) * x,
          rinv, unit, force);
      break;
    }
    case 17: {
      const float z_factor = -3.0f * z + 7.0f * z * z2;
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, z_factor * y,
          0.0f, z_factor, (-3.0f + 21.0f * z2) * y,
          rinv, unit, force);
      break;
    }
    case 18: {
      const float a = -1.0f + 7.0f * z2;
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, a * x2_minus_y2,
          2.0f * x * a, -2.0f * y * a, 14.0f * z * x2_minus_y2,
          rinv, unit, force);
      break;
    }
    case 19: {
      const float a = -1.0f + 7.0f * z2;
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, a * two_xy,
          2.0f * y * a, 2.0f * x * a, 14.0f * z * two_xy,
          rinv, unit, force);
      break;
    }
    case 20:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, z * x3_minus_3xy2,
          z * (3.0f * x2 - 3.0f * y2), z * (-6.0f * x * y),
          x3_minus_3xy2, rinv, unit, force);
      break;
    case 21:
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, z * three_x2y_minus_y3,
          z * 6.0f * x * y, z * (3.0f * x2 - 3.0f * y2),
          three_x2y_minus_y3, rinv, unit, force);
      break;
    case 22: {
      const float value = x * x3_minus_3xy2 - y * three_x2y_minus_y3;
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, value,
          4.0f * x * x2 - 12.0f * x * y2,
          -12.0f * x2 * y + 4.0f * y * y2, 0.0f,
          rinv, unit, force);
      break;
    }
    default: {
      const float value = x * three_x2y_minus_y3 + y * x3_minus_3xy2;
      add_unit_spherical_derivative_accumulated(
          gn_scale, gnp_scale, value,
          12.0f * x2 * y - 4.0f * y * y2,
          4.0f * x * x2 - 12.0f * x * y2, 0.0f,
          rinv, unit, force);
      break;
    }
  }
}

// Keep the model-size specialization inside reusable fixed-extent primitives;
// the pull construction and edge traversal remain one algorithm.
template <int NCount>
__device__ __forceinline__ void evaluate_angular_radial_response(
    int basis_count,
    float cutoff_angular,
    float rcinv,
    float r,
    const float* coefficient_pair,
    float (&gn)[NCount],
    float (&gnp)[NCount]) {
  float fc = 0.0f;
  float fcp = 0.0f;
  find_fc_and_fcp(cutoff_angular, rcinv, r, fc, fcp);
  for (int k = 0; k < basis_count; ++k) {
    float fn = 0.0f;
    float fnp = 0.0f;
    find_fn_and_fnp(k, rcinv, r, fc, fcp, fn, fnp);
#pragma unroll
    for (int n = 0; n < NCount; ++n) {
      const float coefficient = coefficient_pair[n * basis_count + k];
      gn[n] += fn * coefficient;
      gnp[n] += fnp * coefficient;
    }
  }
}

template <int NCount, int WarpWidth>
__device__ __forceinline__ void contract_angular_pull_n(
    const float* owned_pull,
    int source_lane,
    const float (&gn)[NCount],
    const float (&gnp)[NCount],
    float& gn_scale,
    float& gnp_scale) {
  constexpr unsigned kFullWarpMask = 0xffffffffu;
  gn_scale = 0.0f;
  gnp_scale = 0.0f;
#pragma unroll
  for (int n = 0; n < NCount; ++n) {
    const float pull = __shfl_sync(
        kFullWarpMask, owned_pull[n], source_lane, WarpWidth);
    gn_scale += pull * gn[n];
    gnp_scale += pull * gnp[n];
  }
}

template <bool Enabled>
struct AngularVirialOutput;

template <>
struct AngularVirialOutput<false> {
  __device__ __forceinline__ void add_edge(
      float,
      float,
      float,
      const float*,
      int,
      int,
      double*,
      bool) {}

  template <int EdgesPerAtomBatch>
  __device__ __forceinline__ void reduce_and_store(
      int,
      bool,
      int,
      int,
      double*,
      bool) {}
};

template <>
struct AngularVirialOutput<true> {
  float sxx = 0.0f;
  float sxy = 0.0f;
  float sxz = 0.0f;
  float syx = 0.0f;
  float syy = 0.0f;
  float syz = 0.0f;
  float szx = 0.0f;
  float szy = 0.0f;
  float szz = 0.0f;

  __device__ __forceinline__ void add_edge(
      float x12,
      float y12,
      float z12,
      const float* force,
      int neighbor,
      int atom_stride,
      double* virial_soa9,
      bool virial_to_neighbor) {
    if (virial_to_neighbor) {
      atomicAdd(
          &virial_soa9[neighbor],
          -static_cast<double>(x12 * force[0]));
      atomicAdd(
          &virial_soa9[atom_stride + neighbor],
          -static_cast<double>(y12 * force[1]));
      atomicAdd(
          &virial_soa9[2 * atom_stride + neighbor],
          -static_cast<double>(z12 * force[2]));
      atomicAdd(
          &virial_soa9[3 * atom_stride + neighbor],
          -static_cast<double>(x12 * force[1]));
      atomicAdd(
          &virial_soa9[4 * atom_stride + neighbor],
          -static_cast<double>(x12 * force[2]));
      atomicAdd(
          &virial_soa9[5 * atom_stride + neighbor],
          -static_cast<double>(y12 * force[2]));
      atomicAdd(
          &virial_soa9[6 * atom_stride + neighbor],
          -static_cast<double>(y12 * force[0]));
      atomicAdd(
          &virial_soa9[7 * atom_stride + neighbor],
          -static_cast<double>(z12 * force[0]));
      atomicAdd(
          &virial_soa9[8 * atom_stride + neighbor],
          -static_cast<double>(z12 * force[1]));
      return;
    }
    sxx -= x12 * force[0];
    syy -= y12 * force[1];
    szz -= z12 * force[2];
    sxy -= x12 * force[1];
    sxz -= x12 * force[2];
    syz -= y12 * force[2];
    syx -= y12 * force[0];
    szx -= z12 * force[0];
    szy -= z12 * force[1];
  }

  template <int EdgesPerAtomBatch>
  __device__ __forceinline__ void reduce_and_store(
      int edge_lane,
      bool valid_atom,
      int atom,
      int atom_stride,
      double* virial_soa9,
      bool virial_to_neighbor) {
    if (virial_to_neighbor) {
      return;
    }
    constexpr unsigned kFullWarpMask = 0xffffffffu;
#pragma unroll
    for (int offset = EdgesPerAtomBatch / 2; offset > 0; offset >>= 1) {
      sxx += __shfl_down_sync(
          kFullWarpMask, sxx, offset, EdgesPerAtomBatch);
      syy += __shfl_down_sync(
          kFullWarpMask, syy, offset, EdgesPerAtomBatch);
      szz += __shfl_down_sync(
          kFullWarpMask, szz, offset, EdgesPerAtomBatch);
      sxy += __shfl_down_sync(
          kFullWarpMask, sxy, offset, EdgesPerAtomBatch);
      sxz += __shfl_down_sync(
          kFullWarpMask, sxz, offset, EdgesPerAtomBatch);
      syz += __shfl_down_sync(
          kFullWarpMask, syz, offset, EdgesPerAtomBatch);
      syx += __shfl_down_sync(
          kFullWarpMask, syx, offset, EdgesPerAtomBatch);
      szx += __shfl_down_sync(
          kFullWarpMask, szx, offset, EdgesPerAtomBatch);
      szy += __shfl_down_sync(
          kFullWarpMask, szy, offset, EdgesPerAtomBatch);
    }
    if (edge_lane != 0 || !valid_atom) {
      return;
    }
    virial_soa9[atom] += static_cast<double>(sxx);
    virial_soa9[atom_stride + atom] += static_cast<double>(syy);
    virial_soa9[2 * atom_stride + atom] += static_cast<double>(szz);
    virial_soa9[3 * atom_stride + atom] += static_cast<double>(sxy);
    virial_soa9[4 * atom_stride + atom] += static_cast<double>(sxz);
    virial_soa9[5 * atom_stride + atom] += static_cast<double>(syz);
    virial_soa9[6 * atom_stride + atom] += static_cast<double>(syx);
    virial_soa9[7 * atom_stride + atom] += static_cast<double>(szx);
    virial_soa9[8 * atom_stride + atom] += static_cast<double>(szy);
  }
};

// Cached geometry is a compiler seam: removing the reconstruction state from
// the common path keeps its register footprint at the lean tile level.
template <
    bool AccumulateVirial,
    bool UseCachedGeometry,
    int NCount,
    int AtomsPerWarp,
    int EdgesPerAtomBatch>
__global__ void accumulate_angular_forces_pull_tile(
    int atom_count,
    int atom_stride,
    int num_types,
    int num_type_pairs,
    int n_max_radial,
    int basis_size_radial,
    int basis_size_angular,
    int l_max_3body,
    int has_q_222,
    int has_q_1111,
    int has_q_112,
    int has_q_123,
    int has_q_233,
    int has_q_134,
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
  constexpr int kAngularComponents = 24;
  constexpr int kMaxFpChannels = 10;
  constexpr int kAlphaRounds =
      (kAngularComponents + EdgesPerAtomBatch - 1) / EdgesPerAtomBatch;
  constexpr int kSumPerAtom = NCount * kAngularComponents;
  constexpr int kFpPerAtom = NCount * kMaxFpChannels;
  constexpr unsigned kFullWarpMask = 0xffffffffu;
  static_assert(NCount > 0);
  static_assert(AtomsPerWarp * EdgesPerAtomBatch == 32);
  static_assert(
      EdgesPerAtomBatch == 4 || EdgesPerAtomBatch == 8 ||
      EdgesPerAtomBatch == 16);

  const int lane = threadIdx.x;
  const int atom_in_tile = lane / EdgesPerAtomBatch;
  const int edge_lane = lane - atom_in_tile * EdgesPerAtomBatch;
  const int atom_base = blockIdx.x * AtomsPerWarp;
  const int raw_atom = atom_base + atom_in_tile;
  const bool valid_atom = raw_atom < atom_count;
  const int atom = valid_atom ? raw_atom : atom_count - 1;
  const int radial_dim = n_max_radial + 1;
  const int channel_count = l_max_3body + has_q_222 + has_q_1111 +
      has_q_112 + has_q_123 + has_q_233 + has_q_134;
  SimulationBox atom_box = single_box;
  double xi = 0.0;
  double yi = 0.0;
  double zi = 0.0;
  if constexpr (!UseCachedGeometry) {
    if (atom_to_structure != nullptr) {
      atom_box = load_structure_box(
          atom_to_structure[atom],
          boxes_row_major9,
          box_inverse_row_major9,
          pbc_flags3);
    }
    xi = positions_soa3[atom];
    yi = positions_soa3[atom_stride + atom];
    zi = positions_soa3[2 * atom_stride + atom];
  }

  __shared__ float tile_sum[AtomsPerWarp * kSumPerAtom];
  __shared__ float tile_fp[AtomsPerWarp * kFpPerAtom];
  __shared__ float tile_pull[AtomsPerWarp * kSumPerAtom];

  for (int flat = lane; flat < AtomsPerWarp * kSumPerAtom;
       flat += blockDim.x) {
    const int local_atom = flat % AtomsPerWarp;
    const int component = flat / AtomsPerWarp;
    const int n = component / kAngularComponents;
    const int alpha = component - n * kAngularComponents;
    const int candidate_atom = atom_base + local_atom;
    const int source_atom =
        candidate_atom < atom_count ? candidate_atom : atom_count - 1;
    tile_sum[local_atom * kSumPerAtom + component] = alpha < abc_count
        ? sum_fxyz[source_atom + atom_stride * (n * abc_count + alpha)]
        : 0.0f;
  }
  for (int flat = lane; flat < AtomsPerWarp * kFpPerAtom;
       flat += blockDim.x) {
    const int local_atom = flat % AtomsPerWarp;
    const int component = flat / AtomsPerWarp;
    const int n = component / kMaxFpChannels;
    const int channel = component - n * kMaxFpChannels;
    const int candidate_atom = atom_base + local_atom;
    const int source_atom =
        candidate_atom < atom_count ? candidate_atom : atom_count - 1;
    const int descriptor = radial_dim + channel * NCount + n;
    tile_fp[local_atom * kFpPerAtom + component] = channel < channel_count
        ? fp[source_atom + atom_stride * descriptor]
        : 0.0f;
  }
  __syncthreads();

  for (int n = edge_lane; n < NCount; n += EdgesPerAtomBatch) {
    build_angular_pull(
        l_max_3body,
        has_q_222,
        has_q_1111,
        has_q_112,
        has_q_123,
        has_q_233,
        has_q_134,
        tile_sum + atom_in_tile * kSumPerAtom + n * kAngularComponents,
        tile_fp + atom_in_tile * kFpPerAtom + n * kMaxFpChannels,
        tile_pull + atom_in_tile * kSumPerAtom + n * kAngularComponents);
  }
  __syncthreads();

  float owned_pull[kAlphaRounds][NCount];
#pragma unroll
  for (int round = 0; round < kAlphaRounds; ++round) {
    const int alpha = round * EdgesPerAtomBatch + edge_lane;
#pragma unroll
    for (int n = 0; n < NCount; ++n) {
      owned_pull[round][n] = alpha < abc_count
          ? tile_pull[
                atom_in_tile * kSumPerAtom + n * kAngularComponents + alpha]
          : 0.0f;
    }
  }

  int edge_count = 0;
  int type1 = 0;
  if (edge_lane == 0 && valid_atom) {
    edge_count = nn_angular[atom];
    type1 = types[atom];
  }
  edge_count = __shfl_sync(
      kFullWarpMask, edge_count, 0, EdgesPerAtomBatch);
  type1 = __shfl_sync(kFullWarpMask, type1, 0, EdgesPerAtomBatch);

  int max_edge_count = edge_count;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const int other =
        __shfl_xor_sync(kFullWarpMask, max_edge_count, offset);
    max_edge_count = other > max_edge_count ? other : max_edge_count;
  }

  const int angular_coefficient_offset =
      num_type_pairs * radial_dim * (basis_size_radial + 1);
  const int basis_count = basis_size_angular + 1;
  const int angular_basis_count = NCount * basis_count;
  const float rcinv = 1.0f / cutoff_angular;
  float center_fx = 0.0f;
  float center_fy = 0.0f;
  float center_fz = 0.0f;
  AngularVirialOutput<AccumulateVirial> virial_output;
  const int batch_count =
      (max_edge_count + EdgesPerAtomBatch - 1) / EdgesPerAtomBatch;
  for (int batch = 0; batch < batch_count; ++batch) {
    const int slot = batch * EdgesPerAtomBatch + edge_lane;
    bool active_edge = valid_atom && slot < edge_count;
    int neighbor = atom;
    float r = 1.0f;
    float rinv = 1.0f;
    float x12 = 1.0f;
    float y12 = 0.0f;
    float z12 = 0.0f;
    float unit[3] = {1.0f, 0.0f, 0.0f};
    float gn[NCount] = {};
    float gnp[NCount] = {};

    if (active_edge) {
      const int pair_offset = atom + atom_stride * slot;
      neighbor = nl_angular[pair_offset];
      r = r12_angular[pair_offset];
      active_edge = r > 0.0f;
      if (active_edge) {
        if constexpr (UseCachedGeometry) {
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
        rinv = 1.0f / r;
        unit[0] = x12 * rinv;
        unit[1] = y12 * rinv;
        unit[2] = z12 * rinv;

        const int type2 = types[neighbor];
        const int type_pair = type1 * num_types + type2;
        const float* coefficient_pair = descriptor_coefficients +
            angular_coefficient_offset + type_pair * angular_basis_count;
        evaluate_angular_radial_response<NCount>(
            basis_count,
            cutoff_angular,
            rcinv,
            r,
            coefficient_pair,
            gn,
            gnp);
      }
    }

    float f12[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int round = 0; round < kAlphaRounds; ++round) {
#pragma unroll
      for (int source = 0; source < EdgesPerAtomBatch; ++source) {
        const int alpha = round * EdgesPerAtomBatch + source;
        float gn_scale = 0.0f;
        float gnp_scale = 0.0f;
        contract_angular_pull_n<NCount, EdgesPerAtomBatch>(
            owned_pull[round],
            source,
            gn,
            gnp,
            gn_scale,
            gnp_scale);
        if (alpha < abc_count) {
          accumulate_angular_component(
              alpha,
              gn_scale,
              gnp_scale,
              rinv,
              unit,
              f12);
        }
      }
    }

    if (active_edge) {
      center_fx += f12[0];
      center_fy += f12[1];
      center_fz += f12[2];
      atomicAdd(&force_soa3[neighbor], -static_cast<double>(f12[0]));
      atomicAdd(
          &force_soa3[atom_stride + neighbor],
          -static_cast<double>(f12[1]));
      atomicAdd(
          &force_soa3[2 * atom_stride + neighbor],
          -static_cast<double>(f12[2]));
      virial_output.add_edge(
          x12,
          y12,
          z12,
          f12,
          neighbor,
          atom_stride,
          virial_soa9,
          virial_to_neighbor != 0);
    }
  }

#pragma unroll
  for (int offset = EdgesPerAtomBatch / 2; offset > 0; offset >>= 1) {
    center_fx += __shfl_down_sync(
        kFullWarpMask, center_fx, offset, EdgesPerAtomBatch);
    center_fy += __shfl_down_sync(
        kFullWarpMask, center_fy, offset, EdgesPerAtomBatch);
    center_fz += __shfl_down_sync(
        kFullWarpMask, center_fz, offset, EdgesPerAtomBatch);
  }
  if (edge_lane == 0 && valid_atom) {
    atomicAdd(&force_soa3[atom], static_cast<double>(center_fx));
    atomicAdd(
        &force_soa3[atom_stride + atom],
        static_cast<double>(center_fy));
    atomicAdd(
        &force_soa3[2 * atom_stride + atom],
        static_cast<double>(center_fz));
  }
  virial_output.template reduce_and_store<EdgesPerAtomBatch>(
      edge_lane,
      valid_atom,
      atom,
      atom_stride,
      virial_soa9,
      virial_to_neighbor != 0);
}

template <
    bool AccumulateVirial,
    bool UseCachedGeometry,
    int NCount,
    int EdgesPerAtomBatch>
void launch_angular_pull_tile(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModelView& model_view,
    const DeviceWorkspaceView& view,
    bool batched,
    bool virial_to_neighbor) {
  constexpr int kAtomsPerWarp = 32 / EdgesPerAtomBatch;
  static_assert(
      EdgesPerAtomBatch == 4 || EdgesPerAtomBatch == 8 ||
      EdgesPerAtomBatch == 16);
  static_assert(32 % EdgesPerAtomBatch == 0);
  const int tile_blocks =
      (atom_count + kAtomsPerWarp - 1) / kAtomsPerWarp;
  accumulate_angular_forces_pull_tile<
      AccumulateVirial,
      UseCachedGeometry,
      NCount,
      kAtomsPerWarp,
      EdgesPerAtomBatch>
      <<<tile_blocks, kAngularForceThreads>>>(
          atom_count,
          static_cast<int>(view.atom_capacity),
          protocol.num_types,
          protocol.num_types * protocol.num_types,
          protocol.n_max_radial,
          protocol.basis_size_radial,
          protocol.basis_size_angular,
          protocol.body_channels.l_max_3body,
          protocol.body_channels.has_q_222 ? 1 : 0,
          protocol.body_channels.has_q_1111 ? 1 : 0,
          protocol.body_channels.has_q_112 ? 1 : 0,
          protocol.body_channels.has_q_123 ? 1 : 0,
          protocol.body_channels.has_q_233 ? 1 : 0,
          protocol.body_channels.has_q_134 ? 1 : 0,
          protocol.body_channels.abc_count(),
          static_cast<float>(protocol.cutoff_angular),
          box,
          batched ? view.atom_to_structure : nullptr,
          batched ? view.boxes_row_major9 : nullptr,
          batched ? view.box_inverse_row_major9 : nullptr,
          batched ? view.pbc_flags3 : nullptr,
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

template <bool AccumulateVirial, bool UseCachedGeometry>
void dispatch_angular_pull_tile_impl(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModelView& model_view,
    const DeviceWorkspaceView& view,
    bool batched,
    bool virial_to_neighbor) {
  switch (protocol.n_max_angular) {
    case 0:
      launch_angular_pull_tile<AccumulateVirial, UseCachedGeometry, 1, 4>(
          protocol, atom_count, box, model_view, view, batched,
          virial_to_neighbor);
      break;
    case 1:
      launch_angular_pull_tile<AccumulateVirial, UseCachedGeometry, 2, 4>(
          protocol, atom_count, box, model_view, view, batched,
          virial_to_neighbor);
      break;
    case 2:
      launch_angular_pull_tile<AccumulateVirial, UseCachedGeometry, 3, 8>(
          protocol, atom_count, box, model_view, view, batched,
          virial_to_neighbor);
      break;
    case 3:
      launch_angular_pull_tile<AccumulateVirial, UseCachedGeometry, 4, 8>(
          protocol, atom_count, box, model_view, view, batched,
          virial_to_neighbor);
      break;
    case 4:
      launch_angular_pull_tile<AccumulateVirial, UseCachedGeometry, 5, 8>(
          protocol, atom_count, box, model_view, view, batched,
          virial_to_neighbor);
      break;
    case 5:
      launch_angular_pull_tile<AccumulateVirial, UseCachedGeometry, 6, 8>(
          protocol, atom_count, box, model_view, view, batched,
          virial_to_neighbor);
      break;
    case 6:
      launch_angular_pull_tile<AccumulateVirial, UseCachedGeometry, 7, 16>(
          protocol, atom_count, box, model_view, view, batched,
          virial_to_neighbor);
      break;
    case 7:
      launch_angular_pull_tile<AccumulateVirial, UseCachedGeometry, 8, 16>(
          protocol, atom_count, box, model_view, view, batched,
          virial_to_neighbor);
      break;
    case 8:
      launch_angular_pull_tile<AccumulateVirial, UseCachedGeometry, 9, 16>(
          protocol, atom_count, box, model_view, view, batched,
          virial_to_neighbor);
      break;
  }
}

template <bool AccumulateVirial>
void dispatch_angular_pull_tile(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModelView& model_view,
    const DeviceWorkspaceView& view,
    bool batched,
    bool virial_to_neighbor) {
  if (view.f12x != nullptr) {
    dispatch_angular_pull_tile_impl<AccumulateVirial, true>(
        protocol,
        atom_count,
        box,
        model_view,
        view,
        batched,
        virial_to_neighbor);
  } else {
    dispatch_angular_pull_tile_impl<AccumulateVirial, false>(
        protocol,
        atom_count,
        box,
        model_view,
        view,
        batched,
        virial_to_neighbor);
  }
}

void validate_angular_force_inputs(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModelView& model_view,
    const DeviceWorkspaceView& view,
    bool batched) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.n_max_angular >= 0 && protocol.n_max_angular <= 8,
          "angular force kernel supports n_max_angular 0..8");
  require(protocol.body_channels.l_max_3body >= 1 &&
              protocol.body_channels.l_max_3body <= 4,
          "angular force kernel supports l_max_3body 1..4");
  require(!protocol.body_channels.has_q_222 ||
              protocol.body_channels.l_max_3body >= 2,
          "q222 angular force requires l_max_3body >= 2");
  require(!protocol.body_channels.has_q_112 ||
              protocol.body_channels.l_max_3body >= 2,
          "q112 angular force requires l_max_3body >= 2");
  require(!protocol.body_channels.has_q_123 ||
              protocol.body_channels.l_max_3body >= 3,
          "q123 angular force requires l_max_3body >= 3");
  require(!protocol.body_channels.has_q_233 ||
              protocol.body_channels.l_max_3body >= 3,
          "q233 angular force requires l_max_3body >= 3");
  require(!protocol.body_channels.has_q_134 ||
              protocol.body_channels.l_max_3body >= 4,
          "q134 angular force requires l_max_3body >= 4");
  require(protocol.cutoff_angular > 0.0, "angular cutoff must be positive");
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");

  if (batched) {
    require(view.atom_to_structure != nullptr,
            "workspace missing atom_to_structure");
    require(view.boxes_row_major9 != nullptr, "workspace missing boxes");
    require(view.box_inverse_row_major9 != nullptr,
            "workspace missing box inverses");
    require(view.pbc_flags3 != nullptr, "workspace missing pbc flags");
  }
  require(view.types != nullptr, "workspace missing atom types");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.nn_angular != nullptr, "workspace missing angular counts");
  require(view.nl_angular_slot_major != nullptr,
          "workspace missing angular neighbors");
  require(view.r12_angular != nullptr,
          "workspace missing angular distance cache");
  require(view.fp != nullptr, "workspace missing descriptor derivatives");
  require(view.sum_fxyz != nullptr, "workspace missing angular sums");
  require(view.force_soa3 != nullptr, "workspace missing force output");
  require(view.virial_soa9 != nullptr, "workspace missing virial output");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_type_pair_major != nullptr,
          "model missing type-pair-major descriptor coefficients");
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
  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  validate_angular_force_inputs(
      protocol, atom_count, model_view, view, false);

  if (atom_count > 0) {
    if (accumulate_virial) {
      dispatch_angular_pull_tile<true>(
          protocol,
          atom_count,
          box,
          model_view,
          view,
          false,
          virial_to_neighbor);
    } else {
      dispatch_angular_pull_tile<false>(
          protocol, atom_count, box, model_view, view, false, false);
    }
  }
  check_cuda(
      cudaGetLastError(), "accumulate angular forces kernel launch failed");
}

void accumulate_l2_angular_forces_batched(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView view = workspace.view();
  validate_angular_force_inputs(
      protocol, atom_count, model_view, view, true);

  if (atom_count > 0) {
    dispatch_angular_pull_tile<true>(
        protocol,
        atom_count,
        SimulationBox{},
        model_view,
        view,
        true,
        false);
  }
  check_cuda(cudaGetLastError(),
             "accumulate batched angular forces kernel launch failed");
  check_cuda(
      cudaDeviceSynchronize(),
      "accumulate batched angular forces kernel failed");
}

}  // namespace nep_adapters::cuda_backend
