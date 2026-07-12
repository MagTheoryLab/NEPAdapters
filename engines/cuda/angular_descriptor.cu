#include "device_operations.hpp"
#include "simulation_box_device.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace nep_adapters::cuda_backend {
namespace {

constexpr float kPi = 3.1415927f;
constexpr int kMaxFusedDescriptorDim = 64;
constexpr int kAngularOrderTile = 3;
constexpr std::size_t kDefaultDynamicSharedMemoryBytes = 48 * 1024;

void check_cuda(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(message) + ": " + cudaGetErrorString(status));
  }
}

__device__ __forceinline__ float angular_cutoff(float rc, float rcinv, float r) {
  if (r >= rc) {
    return 0.0f;
  }
  return 0.5f * cosf(kPi * r * rcinv) + 0.5f;
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

__device__ __forceinline__ void accumulate_s_l1(
    float x,
    float y,
    float z,
    float gn,
    float* s) {
  s[0] += z * gn;
  s[1] += x * gn;
  s[2] += y * gn;
}

__device__ __forceinline__ void accumulate_s_l2(
    float x,
    float y,
    float z,
    float gn,
    float* s) {
  s[3] += (-1.0f + 3.0f * z * z) * gn;
  s[4] += z * x * gn;
  s[5] += z * y * gn;
  s[6] += (x * x - y * y) * gn;
  s[7] += (2.0f * x * y) * gn;
}

__device__ __forceinline__ void accumulate_s_l3(
    float x,
    float y,
    float z,
    float gn,
    float* s) {
  const float x2_minus_y2 = x * x - y * y;
  const float two_xy = 2.0f * x * y;
  const float x3_minus_3xy2 = x * x2_minus_y2 - y * two_xy;
  const float three_x2y_minus_y3 = x * two_xy + y * x2_minus_y2;
  s[8] += (-3.0f * z + 5.0f * z * z * z) * gn;
  s[9] += (-1.0f + 5.0f * z * z) * x * gn;
  s[10] += (-1.0f + 5.0f * z * z) * y * gn;
  s[11] += z * x2_minus_y2 * gn;
  s[12] += z * two_xy * gn;
  s[13] += x3_minus_3xy2 * gn;
  s[14] += three_x2y_minus_y3 * gn;
}

__device__ __forceinline__ void accumulate_s_l4(
    float x,
    float y,
    float z,
    float gn,
    float* s) {
  const float z2 = z * z;
  const float x2_minus_y2 = x * x - y * y;
  const float two_xy = 2.0f * x * y;
  const float x3_minus_3xy2 = x * x2_minus_y2 - y * two_xy;
  const float three_x2y_minus_y3 = x * two_xy + y * x2_minus_y2;
  const float x4_minus_6x2y2_plus_y4 =
      x * x3_minus_3xy2 - y * three_x2y_minus_y3;
  const float four_x3y_minus_4xy3 =
      x * three_x2y_minus_y3 + y * x3_minus_3xy2;
  s[15] += (3.0f - 30.0f * z2 + 35.0f * z2 * z2) * gn;
  s[16] += (-3.0f * z + 7.0f * z * z2) * x * gn;
  s[17] += (-3.0f * z + 7.0f * z * z2) * y * gn;
  s[18] += (-1.0f + 7.0f * z2) * x2_minus_y2 * gn;
  s[19] += (-1.0f + 7.0f * z2) * two_xy * gn;
  s[20] += z * x3_minus_3xy2 * gn;
  s[21] += z * three_x2y_minus_y3 * gn;
  s[22] += x4_minus_6x2y2_plus_y4 * gn;
  s[23] += four_x3y_minus_4xy3 * gn;
}

__device__ __forceinline__ float find_q_l1(const float* s) {
  return 0.238732414637843f * s[0] * s[0] +
         2.0f *
             (0.119366207318922f * s[1] * s[1] +
              0.119366207318922f * s[2] * s[2]);
}

__device__ __forceinline__ float find_q_l2(const float* s) {
  return 0.099471839432435f * s[3] * s[3] +
         2.0f *
             (0.596831036594608f * s[4] * s[4] +
              0.596831036594608f * s[5] * s[5] +
              0.149207759148652f * s[6] * s[6] +
              0.149207759148652f * s[7] * s[7]);
}

__device__ __forceinline__ float find_q_l3(const float* s) {
  return 0.139260575205408f * s[8] * s[8] +
         2.0f *
             (0.104445431404056f * s[9] * s[9] +
              0.104445431404056f * s[10] * s[10] +
              1.044454314040563f * s[11] * s[11] +
              1.044454314040563f * s[12] * s[12] +
              0.174075719006761f * s[13] * s[13] +
              0.174075719006761f * s[14] * s[14]);
}

__device__ __forceinline__ float find_q_l4(const float* s) {
  return 0.011190581936149f * s[15] * s[15] +
         2.0f *
             (0.223811638722978f * s[16] * s[16] +
              0.223811638722978f * s[17] * s[17] +
              0.111905819361489f * s[18] * s[18] +
              0.111905819361489f * s[19] * s[19] +
              1.566681471060845f * s[20] * s[20] +
              1.566681471060845f * s[21] * s[21] +
              0.195835183882606f * s[22] * s[22] +
              0.195835183882606f * s[23] * s[23]);
}

__device__ __forceinline__ float find_q_222(const float* s) {
  return -0.007499480826664f * s[3] * s[3] * s[3] +
         -0.134990654879954f * s[3] * (s[4] * s[4] + s[5] * s[5]) +
         0.067495327439977f * s[3] * (s[6] * s[6] + s[7] * s[7]) +
         0.404971964639861f * s[6] * (s[5] * s[5] - s[4] * s[4]) +
         -0.809943929279723f * s[4] * s[5] * s[7];
}

__device__ __forceinline__ float find_q_1111(const float* s) {
  const float s0_sq = s[0] * s[0];
  const float s12_sq = s[1] * s[1] + s[2] * s[2];
  return 0.026596810706114f * s0_sq * s0_sq +
         0.053193621412227f * s0_sq * s12_sq +
         0.026596810706114f * s12_sq * s12_sq;
}

__device__ __forceinline__ float find_q_112(const float* s) {
  return 0.027493550848847f * s[0] * s[0] * s[3] +
         0.164961305093080f * s[0] * (s[1] * s[4] + s[2] * s[5]) +
         -0.013746775424423f * s[3] * (s[1] * s[1] + s[2] * s[2]) +
         0.041240326273270f * s[6] * (s[1] * s[1] - s[2] * s[2]) +
         0.082480652546540f * s[1] * s[2] * s[7];
}

__device__ __forceinline__ float find_q_123(const float* s) {
  float value = 0.0f;
  value += -0.168362926992344f *
           (s[12] * s[2] * s[4] - s[11] * s[2] * s[5] +
            s[1] * s[11] * s[4] + s[1] * s[12] * s[5]);
  value += -0.084181463496172f * (s[0] * s[11] * s[6] + s[0] * s[12] * s[7]);
  value += -0.042090731748086f *
           (s[14] * s[2] * s[6] - s[13] * s[2] * s[7] +
            s[1] * s[13] * s[6] + s[1] * s[14] * s[7]);
  value += -0.067345170796937f * (s[10] * s[0] * s[5] + s[0] * s[4] * s[9]);
  value += -0.016836292699234f *
           (s[10] * s[2] * s[3] + s[0] * s[3] * s[8] +
            s[1] * s[3] * s[9]);
  value += -0.008418146349617f *
           (s[10] * s[2] * s[6] - s[10] * s[1] * s[7] -
            s[2] * s[7] * s[9] - s[1] * s[6] * s[9]);
  value += -0.033672585398469f * (-s[2] * s[5] * s[8] - s[1] * s[4] * s[8]);
  return value;
}

__device__ __forceinline__ float find_q_233(const float* s) {
  float value = 0.0f;
  value += 0.008572620635186f * (s[3] * s[8] * s[8]);
  value += 0.009644198214584f * (s[10] * s[10] * s[3] + s[3] * s[9] * s[9]);
  value += 0.019288396429168f * (-s[10] * s[10] * s[6] + s[6] * s[9] * s[9]);
  value += 0.025717861905558f * (s[4] * s[8] * s[9] + s[10] * s[5] * s[8]);
  value += 0.026789439484956f * (-s[13] * s[13] * s[3] - s[14] * s[14] * s[3]);
  value += 0.032147327381947f *
           (-s[14] * s[7] * s[9] - s[13] * s[6] * s[9] -
            s[10] * s[14] * s[6] + s[10] * s[13] * s[7]);
  value += 0.038576792858337f * (s[10] * s[7] * s[9]);
  value += 0.128589309527790f * (-s[11] * s[6] * s[8] - s[12] * s[7] * s[8]);
  value += 0.192883964291685f *
           (s[11] * s[4] * s[9] + s[12] * s[5] * s[9] +
            s[10] * s[12] * s[4] - s[10] * s[11] * s[5]);
  value += 0.321473273819474f *
           (s[12] * s[14] * s[4] + s[11] * s[14] * s[5] +
            s[13] * s[11] * s[4] - s[13] * s[12] * s[5]);
  return value;
}

__device__ __forceinline__ float find_q_134(const float* s) {
  return 0.003645164295772f * (-s[10] * s[15] * s[2] - s[1] * s[15] * s[9]) +
         0.004860219061029f * (s[0] * s[15] * s[8]) +
         0.006075273826286f *
             (-s[1] * s[13] * s[18] - s[1] * s[14] * s[19] -
              s[2] * s[14] * s[18] + s[2] * s[13] * s[19]) +
         0.018225821478859f *
             (-s[10] * s[18] * s[2] + s[1] * s[10] * s[19] +
              s[1] * s[18] * s[9] + s[2] * s[19] * s[9]) +
         0.024301095305146f * (s[1] * s[16] * s[8] + s[2] * s[17] * s[8]) +
         0.036451642957719f *
             (s[0] * s[10] * s[17] + s[0] * s[16] * s[9] -
              s[1] * s[11] * s[16] - s[1] * s[12] * s[17] -
              s[2] * s[12] * s[16] + s[2] * s[11] * s[17]) +
         0.042526916784005f *
             (s[1] * s[13] * s[22] + s[1] * s[14] * s[23] -
              s[2] * s[14] * s[22] + s[2] * s[13] * s[23]) +
         0.072903285915437f * (s[0] * s[11] * s[18] + s[0] * s[12] * s[19]) +
         0.085053833568010f * (s[0] * s[13] * s[20] + s[0] * s[14] * s[21]) +
         0.255161500704030f *
             (s[1] * s[11] * s[20] + s[1] * s[12] * s[21] -
              s[2] * s[12] * s[20] + s[2] * s[11] * s[21]);
}

__global__ void build_angular_descriptors(
    int atom_count,
    int atom_stride,
    int descriptor_dim,
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
    const int* __restrict__ types,
    const int* __restrict__ nn_angular,
    const int* __restrict__ nl_angular,
    const float* __restrict__ f12x,
    const float* __restrict__ f12y,
    const float* __restrict__ f12z,
    const float* __restrict__ r12_angular,
    const float* __restrict__ fn_angular,
    const float* __restrict__ descriptor_coefficients,
    float* __restrict__ sum_fxyz,
    float* __restrict__ descriptors) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int radial_dim = n_max_radial + 1;
  const int angular_coefficient_offset =
      num_type_pairs * (n_max_radial + 1) * (basis_size_radial + 1);
  const int type1 = types[atom];
  const int angular_count = nn_angular[atom];

  for (int n = 0; n <= n_max_angular; ++n) {
    float s[24] = {0.0f};
    for (int slot = 0; slot < angular_count; ++slot) {
      const int offset = atom + atom_stride * slot;
      const int neighbor = nl_angular[offset];
      const int type2 = types[neighbor];
      const int type_pair = type1 * num_types + type2;
      float gn = 0.0f;
      for (int k = 0; k <= basis_size_angular; ++k) {
        const int coefficient_index =
            angular_coefficient_offset +
            (n * (basis_size_angular + 1) + k) * num_type_pairs + type_pair;
        const int fn_index = offset + atom_stride * angular_capacity * k;
        gn += fn_angular[fn_index] * descriptor_coefficients[coefficient_index];
      }

      const float r = r12_angular[offset];
      if (r <= 0.0f) {
        continue;
      }
      const float rinv = 1.0f / r;
      const float x = f12x[offset] * rinv;
      const float y = f12y[offset] * rinv;
      const float z = f12z[offset] * rinv;
      if (l_max_3body >= 1) {
        accumulate_s_l1(x, y, z, gn, s);
      }
      if (l_max_3body >= 2) {
        accumulate_s_l2(x, y, z, gn, s);
      }
      if (l_max_3body >= 3) {
        accumulate_s_l3(x, y, z, gn, s);
      }
      if (l_max_3body >= 4) {
        accumulate_s_l4(x, y, z, gn, s);
      }
    }

    for (int abc = 0; abc < abc_count; ++abc) {
      const int s_index = atom + atom_stride * (n * abc_count + abc);
      sum_fxyz[s_index] = abc < 24 ? s[abc] : 0.0f;
    }

    if (l_max_3body >= 1) {
      const int descriptor_index = radial_dim + n;
      descriptors[atom + atom_stride * descriptor_index] = find_q_l1(s);
    }
    if (l_max_3body >= 2) {
      const int descriptor_index = radial_dim + (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_l2(s);
      }
    }
    if (l_max_3body >= 3) {
      const int descriptor_index = radial_dim + 2 * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_l3(s);
      }
    }
    if (l_max_3body >= 4) {
      const int descriptor_index = radial_dim + 3 * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_l4(s);
      }
    }
    int channel = l_max_3body;
    if (has_q_222) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_222(s);
      }
      ++channel;
    }
    if (has_q_1111) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_1111(s);
      }
      ++channel;
    }
    if (has_q_112) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_112(s);
      }
      ++channel;
    }
    if (has_q_123) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_123(s);
      }
      ++channel;
    }
    if (has_q_233) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_233(s);
      }
      ++channel;
    }
    if (has_q_134) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_134(s);
      }
    }
  }
}

__global__ void build_angular_descriptors_from_geometry(
    int atom_count,
    int atom_stride,
    int descriptor_dim,
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
    const int* __restrict__ types,
    const int* __restrict__ nn_angular,
    const int* __restrict__ nl_angular,
    const float* __restrict__ f12x,
    const float* __restrict__ f12y,
    const float* __restrict__ f12z,
    float* __restrict__ r12_angular,
    const float* __restrict__ descriptor_coefficients,
    float* __restrict__ sum_fxyz,
    float* __restrict__ descriptors) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int radial_dim = n_max_radial + 1;
  const int angular_coefficient_offset =
      num_type_pairs * (n_max_radial + 1) * (basis_size_radial + 1);
  const int type1 = types[atom];
  const int angular_count = nn_angular[atom];
  const float rc = cutoff_angular;
  const float rcinv = 1.0f / rc;

  for (int n = 0; n <= n_max_angular; ++n) {
    float s[24] = {0.0f};
    for (int slot = 0; slot < angular_count; ++slot) {
      const int offset = atom + atom_stride * slot;
      const int neighbor = nl_angular[offset];
      const int type2 = types[neighbor];
      const int type_pair = type1 * num_types + type2;

      const float dx = f12x[offset];
      const float dy = f12y[offset];
      const float dz = f12z[offset];
      const float r = sqrtf(dx * dx + dy * dy + dz * dz);
      r12_angular[offset] = r;
      if (r <= 0.0f) {
        continue;
      }

      const float fc = angular_cutoff(rc, rcinv, r);
      const float x = 2.0f * (r * rcinv - 1.0f) * (r * rcinv - 1.0f) - 1.0f;
      const float half_fc = 0.5f * fc;
      float gn = 0.0f;
      float t_minus_2 = 1.0f;
      float t_minus_1 = x;
      for (int k = 0; k <= basis_size_angular; ++k) {
        float fn = fc;
        if (k == 1) {
          fn = (x + 1.0f) * half_fc;
        } else if (k >= 2) {
          const float t = 2.0f * x * t_minus_1 - t_minus_2;
          t_minus_2 = t_minus_1;
          t_minus_1 = t;
          fn = (t + 1.0f) * half_fc;
        }
        const int coefficient_index =
            angular_coefficient_offset +
            (n * (basis_size_angular + 1) + k) * num_type_pairs + type_pair;
        gn += fn * descriptor_coefficients[coefficient_index];
      }

      const float rinv = 1.0f / r;
      const float x12 = dx * rinv;
      const float y12 = dy * rinv;
      const float z12 = dz * rinv;
      if (l_max_3body >= 1) {
        accumulate_s_l1(x12, y12, z12, gn, s);
      }
      if (l_max_3body >= 2) {
        accumulate_s_l2(x12, y12, z12, gn, s);
      }
      if (l_max_3body >= 3) {
        accumulate_s_l3(x12, y12, z12, gn, s);
      }
      if (l_max_3body >= 4) {
        accumulate_s_l4(x12, y12, z12, gn, s);
      }
    }

    for (int abc = 0; abc < abc_count; ++abc) {
      const int s_index = atom + atom_stride * (n * abc_count + abc);
      sum_fxyz[s_index] = abc < 24 ? s[abc] : 0.0f;
    }

    if (l_max_3body >= 1) {
      const int descriptor_index = radial_dim + n;
      descriptors[atom + atom_stride * descriptor_index] = find_q_l1(s);
    }
    if (l_max_3body >= 2) {
      const int descriptor_index = radial_dim + (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_l2(s);
      }
    }
    if (l_max_3body >= 3) {
      const int descriptor_index = radial_dim + 2 * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_l3(s);
      }
    }
    if (l_max_3body >= 4) {
      const int descriptor_index = radial_dim + 3 * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_l4(s);
      }
    }
    int channel = l_max_3body;
    if (has_q_222) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_222(s);
      }
      ++channel;
    }
    if (has_q_1111) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_1111(s);
      }
      ++channel;
    }
    if (has_q_112) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_112(s);
      }
      ++channel;
    }
    if (has_q_123) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_123(s);
      }
      ++channel;
    }
    if (has_q_233) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_233(s);
      }
      ++channel;
    }
    if (has_q_134) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        descriptors[atom + atom_stride * descriptor_index] = find_q_134(s);
      }
    }
  }
}

__global__ void build_angular_descriptors_and_ann_from_geometry(
    int atom_count,
    int atom_stride,
    int version,
    int descriptor_dim,
    int hidden_neurons,
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
    const int* __restrict__ types,
    const int* __restrict__ nn_angular,
    const int* __restrict__ nl_angular,
    const float* __restrict__ f12x,
    const float* __restrict__ f12y,
    const float* __restrict__ f12z,
    float* __restrict__ r12_angular,
    const float* __restrict__ descriptor_coefficients,
    const float* __restrict__ q_scaler,
    const float* __restrict__ ann_type_major,
    float* __restrict__ sum_fxyz,
    float* __restrict__ descriptors,
    double* __restrict__ potential,
    float* __restrict__ fp) {
  (void)q_scaler;
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const int type1 = types[atom];
  if (type1 < 0 || type1 >= num_types) {
    return;
  }

  float q[kMaxFusedDescriptorDim];
  float fp_local[kMaxFusedDescriptorDim];
  for (int d = 0; d < descriptor_dim; ++d) {
    q[d] = descriptors[atom + atom_stride * d];
    fp_local[d] = 0.0f;
  }

  const int radial_dim = n_max_radial + 1;
  const int angular_coefficient_offset =
      num_type_pairs * (n_max_radial + 1) * (basis_size_radial + 1);
  const int angular_count = nn_angular[atom];
  const float rc = cutoff_angular;
  const float rcinv = 1.0f / rc;

  for (int n = 0; n <= n_max_angular; ++n) {
    float s[24] = {0.0f};
    for (int slot = 0; slot < angular_count; ++slot) {
      const int offset = atom + atom_stride * slot;
      const int neighbor = nl_angular[offset];
      const int type2 = types[neighbor];
      const int type_pair = type1 * num_types + type2;

      const float dx = f12x[offset];
      const float dy = f12y[offset];
      const float dz = f12z[offset];
      const float r = sqrtf(dx * dx + dy * dy + dz * dz);
      r12_angular[offset] = r;
      if (r <= 0.0f) {
        continue;
      }

      const float fc = angular_cutoff(rc, rcinv, r);
      const float x = 2.0f * (r * rcinv - 1.0f) * (r * rcinv - 1.0f) - 1.0f;
      const float half_fc = 0.5f * fc;
      float gn = 0.0f;
      float t_minus_2 = 1.0f;
      float t_minus_1 = x;
      for (int k = 0; k <= basis_size_angular; ++k) {
        float fn = fc;
        if (k == 1) {
          fn = (x + 1.0f) * half_fc;
        } else if (k >= 2) {
          const float t = 2.0f * x * t_minus_1 - t_minus_2;
          t_minus_2 = t_minus_1;
          t_minus_1 = t;
          fn = (t + 1.0f) * half_fc;
        }
        const int coefficient_index =
            angular_coefficient_offset +
            (n * (basis_size_angular + 1) + k) * num_type_pairs + type_pair;
        gn += fn * descriptor_coefficients[coefficient_index];
      }

      const float rinv = 1.0f / r;
      const float x12 = dx * rinv;
      const float y12 = dy * rinv;
      const float z12 = dz * rinv;
      if (l_max_3body >= 1) {
        accumulate_s_l1(x12, y12, z12, gn, s);
      }
      if (l_max_3body >= 2) {
        accumulate_s_l2(x12, y12, z12, gn, s);
      }
      if (l_max_3body >= 3) {
        accumulate_s_l3(x12, y12, z12, gn, s);
      }
      if (l_max_3body >= 4) {
        accumulate_s_l4(x12, y12, z12, gn, s);
      }
    }

    for (int abc = 0; abc < abc_count; ++abc) {
      const int s_index = atom + atom_stride * (n * abc_count + abc);
      sum_fxyz[s_index] = abc < 24 ? s[abc] : 0.0f;
    }

    if (l_max_3body >= 1) {
      const int descriptor_index = radial_dim + n;
      const float value = find_q_l1(s);
      descriptors[atom + atom_stride * descriptor_index] = value;
      q[descriptor_index] = value;
    }
    if (l_max_3body >= 2) {
      const int descriptor_index = radial_dim + (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        const float value = find_q_l2(s);
        descriptors[atom + atom_stride * descriptor_index] = value;
        q[descriptor_index] = value;
      }
    }
    if (l_max_3body >= 3) {
      const int descriptor_index = radial_dim + 2 * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        const float value = find_q_l3(s);
        descriptors[atom + atom_stride * descriptor_index] = value;
        q[descriptor_index] = value;
      }
    }
    if (l_max_3body >= 4) {
      const int descriptor_index = radial_dim + 3 * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        const float value = find_q_l4(s);
        descriptors[atom + atom_stride * descriptor_index] = value;
        q[descriptor_index] = value;
      }
    }
    int channel = l_max_3body;
    if (has_q_222) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        const float value = find_q_222(s);
        descriptors[atom + atom_stride * descriptor_index] = value;
        q[descriptor_index] = value;
      }
      ++channel;
    }
    if (has_q_1111) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        const float value = find_q_1111(s);
        descriptors[atom + atom_stride * descriptor_index] = value;
        q[descriptor_index] = value;
      }
      ++channel;
    }
    if (has_q_112) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        const float value = find_q_112(s);
        descriptors[atom + atom_stride * descriptor_index] = value;
        q[descriptor_index] = value;
      }
      ++channel;
    }
    if (has_q_123) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        const float value = find_q_123(s);
        descriptors[atom + atom_stride * descriptor_index] = value;
        q[descriptor_index] = value;
      }
      ++channel;
    }
    if (has_q_233) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        const float value = find_q_233(s);
        descriptors[atom + atom_stride * descriptor_index] = value;
        q[descriptor_index] = value;
      }
      ++channel;
    }
    if (has_q_134) {
      const int descriptor_index = radial_dim + channel * (n_max_angular + 1) + n;
      if (descriptor_index < descriptor_dim) {
        const float value = find_q_134(s);
        descriptors[atom + atom_stride * descriptor_index] = value;
        q[descriptor_index] = value;
      }
    }
  }

  const int w0_count = hidden_neurons * descriptor_dim;
  const int type_extra_bias_count = version == 5 ? 1 : 0;
  const int type_block_size =
      w0_count + hidden_neurons + hidden_neurons + type_extra_bias_count;
  const float* w0 = ann_type_major + type1 * type_block_size;
  const float* b0 = w0 + w0_count;
  const float* w1 = b0 + hidden_neurons;
  const float* b1 = ann_type_major + num_types * type_block_size;

  float energy = 0.0f;
  for (int neuron = 0; neuron < hidden_neurons; ++neuron) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < descriptor_dim; ++d) {
      w0_times_q += w0[neuron * descriptor_dim + d] * q[d];
    }
    const float x1 = tanhf(w0_times_q - b0[neuron]);
    const float tanh_derivative = 1.0f - x1 * x1;
    energy += w1[neuron] * x1;
    for (int d = 0; d < descriptor_dim; ++d) {
      fp_local[d] += w1[neuron] * tanh_derivative *
                     w0[neuron * descriptor_dim + d];
    }
  }

  const float type_bias = version == 5 ? w1[hidden_neurons] : 0.0f;
  potential[atom] = static_cast<double>(energy - type_bias - b1[0]);
  for (int d = 0; d < descriptor_dim; ++d) {
    fp[atom + atom_stride * d] = fp_local[d];
  }
}

template <bool StorePotential, int DescriptorDim>
__global__ void build_descriptors_and_ann_from_positions(
    int atom_count,
    int atom_stride,
    int version,
    int descriptor_dim,
    int hidden_neurons,
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
    float cutoff_radial,
    float cutoff_angular,
    SimulationBox box,
    const int* __restrict__ types,
    const double* __restrict__ positions_soa3,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const int* __restrict__ nn_angular,
    const int* __restrict__ nl_angular,
    float* __restrict__ r12_angular,
    float* __restrict__ f12x,
    float* __restrict__ f12y,
    float* __restrict__ f12z,
    const float* __restrict__ descriptor_coefficients,
    const float* __restrict__ q_scaler,
    const float* __restrict__ ann_type_major,
    float* __restrict__ sum_fxyz,
    float* __restrict__ descriptors,
    double* __restrict__ potential,
    float* __restrict__ fp) {
  (void)q_scaler;
  (void)descriptors;
  extern __shared__ float radial_basis_sums[];
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  const int radial_basis_size = basis_size_radial + 1;
  const int radial_basis_channels = num_types * radial_basis_size;
  for (int channel = 0; channel < radial_basis_channels; ++channel) {
    radial_basis_sums[channel * blockDim.x + threadIdx.x] = 0.0f;
  }
  if (atom >= atom_count) {
    return;
  }

  const int type1 = types[atom];
  if (type1 < 0 || type1 >= num_types) {
    return;
  }

  constexpr int kDescriptorCapacity =
      DescriptorDim > 0 ? DescriptorDim : kMaxFusedDescriptorDim;
  const int descriptor_count =
      DescriptorDim > 0 ? DescriptorDim : descriptor_dim;
  float q[kDescriptorCapacity];
  float fp_local[kDescriptorCapacity];
  for (int d = 0; d < descriptor_count; ++d) {
    q[d] = 0.0f;
    fp_local[d] = 0.0f;
  }

  const double xi = positions_soa3[atom];
  const double yi = positions_soa3[atom_stride + atom];
  const double zi = positions_soa3[2 * atom_stride + atom];
  const float radial_rcinv = 1.0f / cutoff_radial;
  const int radial_basis_count =
      (n_max_radial + 1) * radial_basis_size;
  const int angular_basis_count =
      (n_max_angular + 1) * (basis_size_angular + 1);
  const int radial_count = nn_radial[atom];
  for (int slot = 0; slot < radial_count; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    const int type2 = types[neighbor];
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    minimum_image_delta(
        box,
        positions_soa3[neighbor] - xi,
        positions_soa3[atom_stride + neighbor] - yi,
        positions_soa3[2 * atom_stride + neighbor] - zi,
        dx,
        dy,
        dz);
    const float r = sqrtf(dx * dx + dy * dy + dz * dz);
    const float fc = angular_cutoff(cutoff_radial, radial_rcinv, r);
    const float x = 2.0f * (r * radial_rcinv - 1.0f) *
                    (r * radial_rcinv - 1.0f) - 1.0f;
    const float half_fc = 0.5f * fc;
    float t_minus_2 = 1.0f;
    float t_minus_1 = x;
    for (int k = 0; k <= basis_size_radial; ++k) {
      float fn = fc;
      if (k == 1) {
        fn = (x + 1.0f) * half_fc;
      } else if (k >= 2) {
        const float t = 2.0f * x * t_minus_1 - t_minus_2;
        t_minus_2 = t_minus_1;
        t_minus_1 = t;
        fn = (t + 1.0f) * half_fc;
      }
      radial_basis_sums[
          (type2 * radial_basis_size + k) * blockDim.x + threadIdx.x] += fn;
    }
  }
  for (int n = 0; n <= n_max_radial; ++n) {
    for (int type2 = 0; type2 < num_types; ++type2) {
      const int type_pair = type1 * num_types + type2;
      const int coefficient_base =
          type_pair * radial_basis_count + n * radial_basis_size;
      for (int k = 0; k < radial_basis_size; ++k) {
        q[n] += radial_basis_sums[
                    (type2 * radial_basis_size + k) * blockDim.x +
                    threadIdx.x] *
                descriptor_coefficients[coefficient_base + k];
      }
    }
  }
  const int radial_dim = n_max_radial + 1;
  const int angular_coefficient_offset =
      num_type_pairs * radial_basis_count;
  const int angular_count = nn_angular[atom];
  const float angular_rcinv = 1.0f / cutoff_angular;

  const int angular_order_count = n_max_angular + 1;
  for (int n_base = 0; n_base < angular_order_count;
       n_base += kAngularOrderTile) {
    const int active_orders =
        n_base + kAngularOrderTile <= angular_order_count
        ? kAngularOrderTile
        : angular_order_count - n_base;
    float s[kAngularOrderTile][24];
#pragma unroll
    for (int tile = 0; tile < kAngularOrderTile; ++tile) {
#pragma unroll
      for (int abc = 0; abc < 24; ++abc) {
        s[tile][abc] = 0.0f;
      }
    }

    for (int slot = 0; slot < angular_count; ++slot) {
      const int offset = atom + atom_stride * slot;
      const int neighbor = nl_angular[offset];
      const int type2 = types[neighbor];
      const int type_pair = type1 * num_types + type2;
      float dx = 0.0f;
      float dy = 0.0f;
      float dz = 0.0f;
      float r = 0.0f;
      if (n_base == 0) {
        minimum_image_delta(
            box,
            positions_soa3[neighbor] - xi,
            positions_soa3[atom_stride + neighbor] - yi,
            positions_soa3[2 * atom_stride + neighbor] - zi,
            dx,
            dy,
            dz);
        if (f12x != nullptr) {
          f12x[offset] = dx;
          f12y[offset] = dy;
          f12z[offset] = dz;
        }
        r = sqrtf(dx * dx + dy * dy + dz * dz);
        r12_angular[offset] = r;
      } else if (f12x != nullptr) {
        dx = f12x[offset];
        dy = f12y[offset];
        dz = f12z[offset];
        r = r12_angular[offset];
      } else {
        minimum_image_delta(
            box,
            positions_soa3[neighbor] - xi,
            positions_soa3[atom_stride + neighbor] - yi,
            positions_soa3[2 * atom_stride + neighbor] - zi,
            dx,
            dy,
            dz);
        r = r12_angular[offset];
      }
      if (r <= 0.0f) {
        continue;
      }

      const float fc = angular_cutoff(cutoff_angular, angular_rcinv, r);
      const float x = 2.0f * (r * angular_rcinv - 1.0f) *
                      (r * angular_rcinv - 1.0f) - 1.0f;
      const float half_fc = 0.5f * fc;
      float gn[kAngularOrderTile] = {0.0f};
      float t_minus_2 = 1.0f;
      float t_minus_1 = x;
      const int coefficient_base =
          angular_coefficient_offset +
          type_pair * angular_basis_count +
          n_base * (basis_size_angular + 1);
      for (int k = 0; k <= basis_size_angular; ++k) {
        float fn = fc;
        if (k == 1) {
          fn = (x + 1.0f) * half_fc;
        } else if (k >= 2) {
          const float t = 2.0f * x * t_minus_1 - t_minus_2;
          t_minus_2 = t_minus_1;
          t_minus_1 = t;
          fn = (t + 1.0f) * half_fc;
        }
#pragma unroll
        for (int tile = 0; tile < kAngularOrderTile; ++tile) {
          if (tile < active_orders) {
            const int coefficient_index =
                coefficient_base + tile * (basis_size_angular + 1) + k;
            gn[tile] +=
                fn * descriptor_coefficients[coefficient_index];
          }
        }
      }

      const float rinv = 1.0f / r;
      const float x12 = dx * rinv;
      const float y12 = dy * rinv;
      const float z12 = dz * rinv;
#pragma unroll
      for (int tile = 0; tile < kAngularOrderTile; ++tile) {
        if (tile < active_orders) {
          if (l_max_3body >= 1) {
            accumulate_s_l1(x12, y12, z12, gn[tile], s[tile]);
          }
          if (l_max_3body >= 2) {
            accumulate_s_l2(x12, y12, z12, gn[tile], s[tile]);
          }
          if (l_max_3body >= 3) {
            accumulate_s_l3(x12, y12, z12, gn[tile], s[tile]);
          }
          if (l_max_3body >= 4) {
            accumulate_s_l4(x12, y12, z12, gn[tile], s[tile]);
          }
        }
      }
    }

#pragma unroll
    for (int tile = 0; tile < kAngularOrderTile; ++tile) {
      if (tile < active_orders) {
        const int n = n_base + tile;
        const float* s_order = s[tile];
        for (int abc = 0; abc < abc_count; ++abc) {
          const int s_index = atom + atom_stride * (n * abc_count + abc);
          sum_fxyz[s_index] = abc < 24 ? s_order[abc] : 0.0f;
        }

        if (l_max_3body >= 1) {
          const int descriptor_index = radial_dim + n;
          const float value = find_q_l1(s_order);
          q[descriptor_index] = value;
        }
        if (l_max_3body >= 2) {
          const int descriptor_index = radial_dim + angular_order_count + n;
          if (descriptor_index < descriptor_dim) {
            const float value = find_q_l2(s_order);
            q[descriptor_index] = value;
          }
        }
        if (l_max_3body >= 3) {
          const int descriptor_index =
              radial_dim + 2 * angular_order_count + n;
          if (descriptor_index < descriptor_dim) {
            const float value = find_q_l3(s_order);
            q[descriptor_index] = value;
          }
        }
        if (l_max_3body >= 4) {
          const int descriptor_index =
              radial_dim + 3 * angular_order_count + n;
          if (descriptor_index < descriptor_dim) {
            const float value = find_q_l4(s_order);
            q[descriptor_index] = value;
          }
        }
        int channel = l_max_3body;
        if (has_q_222) {
          const int descriptor_index =
              radial_dim + channel * angular_order_count + n;
          if (descriptor_index < descriptor_dim) {
            const float value = find_q_222(s_order);
            q[descriptor_index] = value;
          }
          ++channel;
        }
        if (has_q_1111) {
          const int descriptor_index =
              radial_dim + channel * angular_order_count + n;
          if (descriptor_index < descriptor_dim) {
            const float value = find_q_1111(s_order);
            q[descriptor_index] = value;
          }
          ++channel;
        }
        if (has_q_112) {
          const int descriptor_index =
              radial_dim + channel * angular_order_count + n;
          if (descriptor_index < descriptor_dim) {
            const float value = find_q_112(s_order);
            q[descriptor_index] = value;
          }
          ++channel;
        }
        if (has_q_123) {
          const int descriptor_index =
              radial_dim + channel * angular_order_count + n;
          if (descriptor_index < descriptor_dim) {
            const float value = find_q_123(s_order);
            q[descriptor_index] = value;
          }
          ++channel;
        }
        if (has_q_233) {
          const int descriptor_index =
              radial_dim + channel * angular_order_count + n;
          if (descriptor_index < descriptor_dim) {
            const float value = find_q_233(s_order);
            q[descriptor_index] = value;
          }
          ++channel;
        }
        if (has_q_134) {
          const int descriptor_index =
              radial_dim + channel * angular_order_count + n;
          if (descriptor_index < descriptor_dim) {
            const float value = find_q_134(s_order);
            q[descriptor_index] = value;
          }
        }
      }
    }
  }

  const int w0_count = hidden_neurons * descriptor_dim;
  const int type_extra_bias_count = version == 5 ? 1 : 0;
  const int type_block_size =
      w0_count + hidden_neurons + hidden_neurons + type_extra_bias_count;
  const float* w0 = ann_type_major + type1 * type_block_size;
  const float* b0 = w0 + w0_count;
  const float* w1 = b0 + hidden_neurons;
  const float* b1 = ann_type_major + num_types * type_block_size;

  float energy = 0.0f;
  for (int neuron = 0; neuron < hidden_neurons; ++neuron) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < descriptor_count; ++d) {
      w0_times_q += w0[neuron * descriptor_dim + d] * q[d];
    }
    const float x1 = tanhf(w0_times_q - b0[neuron]);
    const float tanh_derivative = 1.0f - x1 * x1;
    if constexpr (StorePotential) {
      energy += w1[neuron] * x1;
    }
    for (int d = 0; d < descriptor_count; ++d) {
      fp_local[d] += w1[neuron] * tanh_derivative *
                     w0[neuron * descriptor_dim + d];
    }
  }

  if constexpr (StorePotential) {
    const float type_bias = version == 5 ? w1[hidden_neurons] : 0.0f;
    potential[atom] = static_cast<double>(energy - type_bias - b1[0]);
  }
  for (int d = 0; d < descriptor_count; ++d) {
    fp[atom + atom_stride * d] = fp_local[d];
  }
}

}  // namespace

void build_angular_descriptors_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.descriptor_dim > 0, "descriptor_dim must be positive");
  require(protocol.n_max_radial >= 0, "n_max_radial must be non-negative");
  require(protocol.n_max_angular >= 0, "n_max_angular must be non-negative");
  require(protocol.basis_size_radial >= 0, "radial basis size must be non-negative");
  require(protocol.basis_size_angular >= 0, "angular basis size must be non-negative");
  require(protocol.neighbor_capacity_angular > 0,
          "angular neighbor capacity must be positive");
  require(protocol.body_channels.l_max_3body <= 4,
          "angular descriptor kernel supports l_max_3body <= 4");
  require(!protocol.body_channels.has_q_222 || protocol.body_channels.l_max_3body >= 2,
          "q_222 requires l_max_3body >= 2");
  require(!protocol.body_channels.has_q_112 || protocol.body_channels.l_max_3body >= 2,
          "q_112 requires l_max_3body >= 2");
  require(!protocol.body_channels.has_q_123 || protocol.body_channels.l_max_3body >= 3,
          "q_123 requires l_max_3body >= 3");
  require(!protocol.body_channels.has_q_233 || protocol.body_channels.l_max_3body >= 3,
          "q_233 requires l_max_3body >= 3");
  require(!protocol.body_channels.has_q_134 || protocol.body_channels.l_max_3body >= 4,
          "q_134 requires l_max_3body >= 4");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView workspace_view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= workspace_view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  require(workspace_view.types != nullptr, "workspace missing atom types");
  require(workspace_view.nn_angular != nullptr, "workspace missing angular counts");
  require(workspace_view.nl_angular_slot_major != nullptr,
          "workspace missing angular neighbor list");
  require(workspace_view.f12x != nullptr && workspace_view.f12y != nullptr &&
              workspace_view.f12z != nullptr,
          "workspace missing angular pair geometry");
  require(workspace_view.r12_angular != nullptr,
          "workspace missing angular distance cache");
  require(workspace_view.sum_fxyz != nullptr, "workspace missing sum_fxyz cache");
  require(workspace_view.descriptors != nullptr, "workspace missing descriptor cache");

  const int threads = 128;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    build_angular_descriptors<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(workspace_view.atom_capacity),
        protocol.descriptor_dim,
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
        workspace_view.types,
        workspace_view.nn_angular,
        workspace_view.nl_angular_slot_major,
        workspace_view.f12x,
        workspace_view.f12y,
        workspace_view.f12z,
        workspace_view.r12_angular,
        workspace_view.fn_angular,
        model_view.descriptor_coefficients,
        workspace_view.sum_fxyz,
        workspace_view.descriptors);
  }
  check_cuda(cudaGetLastError(), "build angular descriptors kernel launch failed");
  check_cuda(cudaDeviceSynchronize(), "build angular descriptors kernel failed");
}

void build_angular_descriptors_from_geometry_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.descriptor_dim > 0, "descriptor_dim must be positive");
  require(protocol.n_max_radial >= 0, "n_max_radial must be non-negative");
  require(protocol.n_max_angular >= 0, "n_max_angular must be non-negative");
  require(protocol.basis_size_radial >= 0, "radial basis size must be non-negative");
  require(protocol.basis_size_angular >= 0, "angular basis size must be non-negative");
  require(protocol.cutoff_angular > 0.0, "angular cutoff must be positive");
  require(protocol.neighbor_capacity_angular > 0,
          "angular neighbor capacity must be positive");
  require(protocol.body_channels.l_max_3body <= 4,
          "angular descriptor kernel supports l_max_3body <= 4");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView workspace_view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= workspace_view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  require(workspace_view.types != nullptr, "workspace missing atom types");
  require(workspace_view.nn_angular != nullptr, "workspace missing angular counts");
  require(workspace_view.nl_angular_slot_major != nullptr,
          "workspace missing angular neighbor list");
  require(workspace_view.f12x != nullptr && workspace_view.f12y != nullptr &&
              workspace_view.f12z != nullptr,
          "workspace missing angular pair geometry");
  require(workspace_view.r12_angular != nullptr,
          "workspace missing angular distance cache");
  require(workspace_view.sum_fxyz != nullptr, "workspace missing sum_fxyz cache");
  require(workspace_view.descriptors != nullptr, "workspace missing descriptor cache");

  const int threads = 64;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    build_angular_descriptors_from_geometry<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(workspace_view.atom_capacity),
        protocol.descriptor_dim,
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
        workspace_view.types,
        workspace_view.nn_angular,
        workspace_view.nl_angular_slot_major,
        workspace_view.f12x,
        workspace_view.f12y,
        workspace_view.f12z,
        workspace_view.r12_angular,
        model_view.descriptor_coefficients,
        workspace_view.sum_fxyz,
        workspace_view.descriptors);
  }
  check_cuda(cudaGetLastError(),
             "build fused angular descriptors kernel launch failed");
  check_cuda(cudaDeviceSynchronize(), "build fused angular descriptors kernel failed");
}

bool try_build_angular_descriptors_and_ann_from_geometry_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const DeviceModel& model,
    DeviceWorkspace& workspace) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.version == 4 || protocol.version == 5,
          "fused descriptor/ANN kernel supports NEP4/NEP5");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.descriptor_dim > 0, "descriptor_dim must be positive");
  require(protocol.hidden_neurons > 0, "hidden_neurons must be positive");
  if (protocol.descriptor_dim > kMaxFusedDescriptorDim) {
    return false;
  }
  require(protocol.n_max_radial >= 0, "n_max_radial must be non-negative");
  require(protocol.n_max_angular >= 0, "n_max_angular must be non-negative");
  require(protocol.basis_size_radial >= 0, "radial basis size must be non-negative");
  require(protocol.basis_size_angular >= 0, "angular basis size must be non-negative");
  require(protocol.cutoff_angular > 0.0, "angular cutoff must be positive");
  require(protocol.neighbor_capacity_angular > 0,
          "angular neighbor capacity must be positive");
  require(protocol.body_channels.l_max_3body <= 4,
          "angular descriptor kernel supports l_max_3body <= 4");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView workspace_view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= workspace_view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  require(model_view.descriptor_coefficients_type_pair_major != nullptr,
          "model missing type-pair-major descriptor coefficients");
  require(model_view.descriptor_coefficients_type_pair_major_count >=
              protocol.descriptor_parameter_count,
          "model type-pair-major descriptor coefficient buffer is too small");
  require(model_view.ann_type_major != nullptr, "model missing ANN parameters");
  require(model_view.ann_type_major_qscaled != nullptr,
          "model missing q-scaled ANN parameters");
  require(model_view.q_scaler != nullptr, "model missing q_scaler");
  require(model_view.ann_type_major_count >= protocol.ann_parameter_count,
          "ANN parameter buffer is too small");
  require(model_view.ann_type_major_qscaled_count >= protocol.ann_parameter_count,
          "q-scaled ANN parameter buffer is too small");
  require(model_view.q_scaler_count >= static_cast<std::size_t>(protocol.descriptor_dim),
          "q_scaler buffer is too small");
  require(workspace_view.types != nullptr, "workspace missing atom types");
  require(workspace_view.nn_angular != nullptr, "workspace missing angular counts");
  require(workspace_view.nl_angular_slot_major != nullptr,
          "workspace missing angular neighbor list");
  require(workspace_view.f12x != nullptr && workspace_view.f12y != nullptr &&
              workspace_view.f12z != nullptr,
          "workspace missing angular pair geometry");
  require(workspace_view.r12_angular != nullptr,
          "workspace missing angular distance cache");
  require(workspace_view.sum_fxyz != nullptr, "workspace missing sum_fxyz cache");
  require(workspace_view.descriptors != nullptr, "workspace missing descriptor cache");
  require(workspace_view.potential != nullptr, "workspace missing potential output");
  require(workspace_view.fp != nullptr, "workspace missing descriptor derivative cache");

  const int threads = 64;
  const int blocks = (atom_count + threads - 1) / threads;
  if (blocks > 0) {
    build_angular_descriptors_and_ann_from_geometry<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(workspace_view.atom_capacity),
        protocol.version,
        protocol.descriptor_dim,
        protocol.hidden_neurons,
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
        workspace_view.types,
        workspace_view.nn_angular,
        workspace_view.nl_angular_slot_major,
        workspace_view.f12x,
        workspace_view.f12y,
        workspace_view.f12z,
        workspace_view.r12_angular,
        model_view.descriptor_coefficients,
        model_view.q_scaler,
        model_view.ann_type_major_qscaled,
        workspace_view.sum_fxyz,
        workspace_view.descriptors,
        workspace_view.potential,
        workspace_view.fp);
  }
  check_cuda(cudaGetLastError(),
             "build fused angular descriptors/ANN kernel launch failed");
  return true;
}

bool try_build_descriptors_and_ann_from_positions_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    const DeviceModel& model,
    DeviceWorkspace& workspace,
    bool store_potential) {
  require(atom_count >= 0, "atom_count must be non-negative");
  require(protocol.version == 4 || protocol.version == 5,
          "fused descriptor/ANN kernel supports NEP4/NEP5");
  require(protocol.num_types > 0, "num_types must be positive");
  require(protocol.descriptor_dim > 0, "descriptor_dim must be positive");
  require(protocol.hidden_neurons > 0, "hidden_neurons must be positive");
  if (protocol.descriptor_dim > kMaxFusedDescriptorDim) {
    return false;
  }
  require(protocol.n_max_radial >= 0, "n_max_radial must be non-negative");
  require(protocol.n_max_angular >= 0, "n_max_angular must be non-negative");
  require(protocol.basis_size_radial >= 0, "radial basis size must be non-negative");
  require(protocol.basis_size_angular >= 0, "angular basis size must be non-negative");
  require(protocol.cutoff_radial > 0.0, "radial cutoff must be positive");
  require(protocol.cutoff_angular > 0.0, "angular cutoff must be positive");
  require(protocol.neighbor_capacity_radial > 0,
          "radial neighbor capacity must be positive");
  require(protocol.neighbor_capacity_angular > 0,
          "angular neighbor capacity must be positive");
  require(protocol.body_channels.l_max_3body <= 4,
          "angular descriptor kernel supports l_max_3body <= 4");

  const DeviceModelView model_view = model.view();
  const DeviceWorkspaceView workspace_view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= workspace_view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(model_view.descriptor_coefficients != nullptr,
          "model missing descriptor coefficients");
  require(model_view.descriptor_coefficients_count >= protocol.descriptor_parameter_count,
          "model descriptor coefficient buffer is too small");
  require(model_view.ann_type_major != nullptr, "model missing ANN parameters");
  require(model_view.ann_type_major_qscaled != nullptr,
          "model missing q-scaled ANN parameters");
  require(model_view.q_scaler != nullptr, "model missing q_scaler");
  require(model_view.ann_type_major_count >= protocol.ann_parameter_count,
          "ANN parameter buffer is too small");
  require(model_view.ann_type_major_qscaled_count >= protocol.ann_parameter_count,
          "q-scaled ANN parameter buffer is too small");
  require(model_view.q_scaler_count >= static_cast<std::size_t>(protocol.descriptor_dim),
          "q_scaler buffer is too small");
  require(workspace_view.types != nullptr, "workspace missing atom types");
  require(workspace_view.positions_soa3 != nullptr, "workspace missing positions");
  require(workspace_view.nn_radial != nullptr, "workspace missing radial counts");
  require(workspace_view.nl_radial_slot_major != nullptr,
          "workspace missing radial neighbor list");
  require(workspace_view.nn_angular != nullptr, "workspace missing angular counts");
  require(workspace_view.nl_angular_slot_major != nullptr,
          "workspace missing angular neighbor list");
  require(workspace_view.r12_angular != nullptr,
          "workspace missing angular distance cache");
  require(workspace_view.sum_fxyz != nullptr, "workspace missing sum_fxyz cache");
  require(workspace_view.descriptors != nullptr, "workspace missing descriptor cache");
  require(workspace_view.potential != nullptr, "workspace missing potential output");
  require(workspace_view.fp != nullptr, "workspace missing descriptor derivative cache");

  const int threads = 128;
  const int blocks = (atom_count + threads - 1) / threads;
  const std::size_t radial_basis_sum_shared_bytes =
      static_cast<std::size_t>(threads) *
      static_cast<std::size_t>(protocol.num_types) *
      static_cast<std::size_t>(protocol.basis_size_radial + 1) * sizeof(float);
  const bool needs_shared_memory_optin =
      radial_basis_sum_shared_bytes > kDefaultDynamicSharedMemoryBytes;
  if (needs_shared_memory_optin) {
    int device = 0;
    check_cuda(cudaGetDevice(&device), "get CUDA device for radial basis sum");
    cudaDeviceProp device_properties{};
    check_cuda(
        cudaGetDeviceProperties(&device_properties, device),
        "get CUDA device properties for radial basis sum");
    const std::size_t optin_shared_bytes = std::max(
        static_cast<std::size_t>(device_properties.sharedMemPerBlock),
        static_cast<std::size_t>(device_properties.sharedMemPerBlockOptin));
    if (radial_basis_sum_shared_bytes > optin_shared_bytes) {
      return false;
    }
  }
  if (blocks > 0) {
    const auto launch = [&](auto store_potential_tag, auto descriptor_dim_tag) {
      constexpr bool kStorePotential = decltype(store_potential_tag)::value;
      constexpr int kDescriptorDim = decltype(descriptor_dim_tag)::value;
      if (needs_shared_memory_optin) {
        check_cuda(
            cudaFuncSetAttribute(
                build_descriptors_and_ann_from_positions<
                    kStorePotential,
                    kDescriptorDim>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(radial_basis_sum_shared_bytes)),
            "configure fused radial basis-sum shared memory");
      }
      build_descriptors_and_ann_from_positions<kStorePotential, kDescriptorDim>
          <<<blocks, threads, radial_basis_sum_shared_bytes>>>(
          atom_count,
          static_cast<int>(workspace_view.atom_capacity),
          protocol.version,
          protocol.descriptor_dim,
          protocol.hidden_neurons,
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
          static_cast<float>(protocol.cutoff_radial),
          static_cast<float>(protocol.cutoff_angular),
          box,
          workspace_view.types,
          workspace_view.positions_soa3,
          workspace_view.nn_radial,
          workspace_view.nl_radial_slot_major,
          workspace_view.nn_angular,
          workspace_view.nl_angular_slot_major,
          workspace_view.r12_angular,
          workspace_view.f12x,
          workspace_view.f12y,
          workspace_view.f12z,
          model_view.descriptor_coefficients_type_pair_major,
          model_view.q_scaler,
          model_view.ann_type_major_qscaled,
          workspace_view.sum_fxyz,
          workspace_view.descriptors,
          workspace_view.potential,
          workspace_view.fp);
    };
    if (store_potential) {
      if (protocol.descriptor_dim == 35) {
        launch(
            std::true_type{},
            std::integral_constant<int, 35>{});
      } else {
        launch(
            std::true_type{},
            std::integral_constant<int, -1>{});
      }
    } else {
      if (protocol.descriptor_dim == 35) {
        launch(
            std::false_type{},
            std::integral_constant<int, 35>{});
      } else {
        launch(
            std::false_type{},
            std::integral_constant<int, -1>{});
      }
    }
  }
  check_cuda(cudaGetLastError(),
             "build fused descriptors/ANN kernel launch failed");
  return true;
}

}  // namespace nep_adapters::cuda_backend
