#pragma once

#include <cuda_runtime.h>

namespace nep_adapters::cuda_backend::angular_harmonics {

constexpr int kMaxAngularComponents = 80;

// These coefficients follow the GPUMD NEP angular basis convention.  Keep
// internal linkage because this header is used by independent CUDA translation
// units and the project does not enable relocatable device code.
static __device__ __constant__ float kC3B[kMaxAngularComponents] = {
    0.238732414637843f, 0.119366207318922f, 0.119366207318922f,
    0.099471839432435f, 0.596831036594608f, 0.596831036594608f,
    0.149207759148652f, 0.149207759148652f, 0.139260575205408f,
    0.104445431404056f, 0.104445431404056f, 1.044454314040563f,
    1.044454314040563f, 0.174075719006761f, 0.174075719006761f,
    0.011190581936149f, 0.223811638722978f, 0.223811638722978f,
    0.111905819361489f, 0.111905819361489f, 1.566681471060845f,
    1.566681471060845f, 0.195835183882606f, 0.195835183882606f,
    0.013677377921960f, 0.102580334414698f, 0.102580334414698f,
    2.872249363611549f, 2.872249363611549f, 0.119677056817148f,
    0.119677056817148f, 2.154187022708661f, 2.154187022708661f,
    0.215418702270866f, 0.215418702270866f, 0.004041043476943f,
    0.169723826031592f, 0.169723826031592f, 0.106077391269745f,
    0.106077391269745f, 0.424309565078979f, 0.424309565078979f,
    0.127292869523694f, 0.127292869523694f, 2.800443129521260f,
    2.800443129521260f, 0.233370260793438f, 0.233370260793438f,
    0.004662742473395f, 0.004079899664221f, 0.004079899664221f,
    0.024479397985326f, 0.024479397985326f, 0.012239698992663f,
    0.012239698992663f, 0.538546755677165f, 0.538546755677165f,
    0.134636688919291f, 0.134636688919291f, 3.500553911901575f,
    3.500553911901575f, 0.250039565135827f, 0.250039565135827f,
    0.000082569397966f, 0.005944996653579f, 0.005944996653579f,
    0.104037441437634f, 0.104037441437634f, 0.762941237209318f,
    0.762941237209318f, 0.114441185581398f, 0.114441185581398f,
    5.950941650232678f, 5.950941650232678f, 0.141689086910302f,
    0.141689086910302f, 4.250672607309055f, 4.250672607309055f,
    0.265667037956816f, 0.265667037956816f};

static __device__ __constant__ float kZ5[6][6] = {
    {0.0f, 15.0f, 0.0f, -70.0f, 0.0f, 63.0f},
    {1.0f, 0.0f, -14.0f, 0.0f, 21.0f, 0.0f},
    {0.0f, -1.0f, 0.0f, 3.0f, 0.0f, 0.0f},
    {-1.0f, 0.0f, 9.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

static __device__ __constant__ float kZ6[7][7] = {
    {-5.0f, 0.0f, 105.0f, 0.0f, -315.0f, 0.0f, 231.0f},
    {0.0f, 5.0f, 0.0f, -30.0f, 0.0f, 33.0f, 0.0f},
    {1.0f, 0.0f, -18.0f, 0.0f, 33.0f, 0.0f, 0.0f},
    {0.0f, -3.0f, 0.0f, 11.0f, 0.0f, 0.0f, 0.0f},
    {-1.0f, 0.0f, 11.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

static __device__ __constant__ float kZ7[8][8] = {
    {0.0f, -35.0f, 0.0f, 315.0f, 0.0f, -693.0f, 0.0f, 429.0f},
    {-5.0f, 0.0f, 135.0f, 0.0f, -495.0f, 0.0f, 429.0f, 0.0f},
    {0.0f, 15.0f, 0.0f, -110.0f, 0.0f, 143.0f, 0.0f, 0.0f},
    {3.0f, 0.0f, -66.0f, 0.0f, 143.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, -3.0f, 0.0f, 13.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {-1.0f, 0.0f, 13.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

static __device__ __constant__ float kZ8[9][9] = {
    {35.0f, 0.0f, -1260.0f, 0.0f, 6930.0f, 0.0f, -12012.0f, 0.0f,
     6435.0f},
    {0.0f, -35.0f, 0.0f, 385.0f, 0.0f, -1001.0f, 0.0f, 715.0f, 0.0f},
    {-1.0f, 0.0f, 33.0f, 0.0f, -143.0f, 0.0f, 143.0f, 0.0f, 0.0f},
    {0.0f, 3.0f, 0.0f, -26.0f, 0.0f, 39.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, -26.0f, 0.0f, 65.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, -1.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {-1.0f, 0.0f, 15.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};

template <int L>
__device__ __forceinline__ float z_coefficient(int n1, int n2) {
  static_assert(L >= 5 && L <= 8);
  if constexpr (L == 5) {
    return kZ5[n1][n2];
  } else if constexpr (L == 6) {
    return kZ6[n1][n2];
  } else if constexpr (L == 7) {
    return kZ7[n1][n2];
  } else {
    return kZ8[n1][n2];
  }
}

__device__ __forceinline__ void complex_product(
    float a,
    float b,
    float& real_part,
    float& imag_part) {
  const float real_temp = real_part;
  real_part = a * real_temp - b * imag_part;
  imag_part = a * imag_part + b * real_temp;
}

template <int L>
__device__ __forceinline__ void accumulate_order(
    float x,
    float y,
    float z,
    float radial,
    float* values) {
  static_assert(L >= 5 && L <= 8);
  int index = L * L - 1;
  float z_pow[L + 1] = {1.0f};
#pragma unroll
  for (int power = 1; power <= L; ++power) {
    z_pow[power] = z * z_pow[power - 1];
  }
  float real_part = x;
  float imag_part = y;
#pragma unroll
  for (int n1 = 0; n1 <= L; ++n1) {
    const int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    float z_factor = 0.0f;
#pragma unroll
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      z_factor += z_coefficient<L>(n1, n2) * z_pow[n2];
    }
    z_factor *= radial;
    if (n1 == 0) {
      values[index++] += z_factor;
    } else {
      values[index++] += z_factor * real_part;
      values[index++] += z_factor * imag_part;
      complex_product(x, y, real_part, imag_part);
    }
  }
}

template <int L>
__device__ __forceinline__ float invariant(const float* values) {
  static_assert(L >= 5 && L <= 8);
  constexpr int kStart = L * L - 1;
  constexpr int kCount = 2 * L + 1;
  float result = 0.0f;
#pragma unroll
  for (int component = 1; component < kCount; ++component) {
    const float value = values[kStart + component];
    result += kC3B[kStart + component] * value * value;
  }
  result *= 2.0f;
  const float first = values[kStart];
  return result + kC3B[kStart] * first * first;
}

}  // namespace nep_adapters::cuda_backend::angular_harmonics
