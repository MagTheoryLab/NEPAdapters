#include "device_operations.hpp"

#include "simulation_box_device.cuh"

#if defined(NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM)
#include <cufft.h>
#endif
#include <cuda_runtime.h>

#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace nep_adapters::cuda_backend {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kCoulomb = 14.399645;
#if defined(NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM)
constexpr float kTwoPiF = 6.28318530717958647692f;

__constant__ float kPppmSincCoeff[6] = {
    1.0f,
    -1.6666667e-1f,
    8.3333333e-3f,
    -1.9841270e-4f,
    2.7557319e-6f,
    -2.5052108e-8f};
__constant__ float kPppmGCoeff[5] = {
    1.0000000e+00f,
    -1.6666667e+00f,
    7.7777778e-01f,
    -8.9947090e-02f,
    7.0546737e-04f};
__constant__ float kPppmWCoeff[5][5] = {
    {2.6041667e-03f, -2.0833333e-02f, 6.2500000e-02f, -8.3333333e-02f, 4.1666667e-02f},
    {1.9791667e-01f, -4.5833333e-01f, 2.5000000e-01f, 1.6666667e-01f, -1.6666667e-01f},
    {5.9895833e-01f, 0.0000000e+00f, -6.2500000e-01f, 0.0000000e+00f, 2.5000000e-01f},
    {1.9791667e-01f, 4.5833333e-01f, 2.5000000e-01f, -1.6666667e-01f, -1.6666667e-01f},
    {2.6041667e-03f, 2.0833333e-02f, 6.2500000e-02f, 8.3333333e-02f, 4.1666667e-02f}};
#endif

void check_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(action) + ": " + cudaGetErrorString(status));
  }
}

#if defined(NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM)
void check_cufft(cufftResult status, const char* action) {
  if (status != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string(action) + ": cuFFT error " + std::to_string(status));
  }
}
#endif

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void cross_product(const double a[3], const double b[3], double c[3]) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

double area(const double* a, const double* b) {
  const double x = a[1] * b[2] - a[2] * b[1];
  const double y = a[2] * b[0] - a[0] * b[2];
  const double z = a[0] * b[1] - a[1] * b[0];
  return std::sqrt(x * x + y * y + z * z);
}

#if defined(NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM)
double volume(const SimulationBox& box) {
  const double* h = box.frac_to_cart;
  return std::abs(
      h[0] * (h[4] * h[8] - h[5] * h[7]) +
      h[1] * (h[5] * h[6] - h[3] * h[8]) +
      h[2] * (h[3] * h[7] - h[4] * h[6]));
}

int best_pppm_mesh_dim(int minimum) {
  int dim = 16;
  while (dim < minimum) {
    dim *= 2;
  }
  return dim;
}
#endif

struct QnepKSpace {
  std::vector<double> kx;
  std::vector<double> ky;
  std::vector<double> kz;
  std::vector<double> g;
};

QnepKSpace make_qnep_kspace(const SimulationBox& box, double alpha) {
  double a1[3] = {box.frac_to_cart[0], box.frac_to_cart[3], box.frac_to_cart[6]};
  double a2[3] = {box.frac_to_cart[1], box.frac_to_cart[4], box.frac_to_cart[7]};
  double a3[3] = {box.frac_to_cart[2], box.frac_to_cart[5], box.frac_to_cart[8]};
  double b1[3] = {};
  double b2[3] = {};
  double b3[3] = {};
  cross_product(a2, a3, b1);
  cross_product(a3, a1, b2);
  cross_product(a1, a2, b3);
  const double* h = box.frac_to_cart;
  const double det =
      h[0] * (h[4] * h[8] - h[5] * h[7]) +
      h[1] * (h[5] * h[6] - h[3] * h[8]) +
      h[2] * (h[3] * h[7] - h[4] * h[6]);
  const double two_pi = 2.0 * kPi;
  const double two_pi_over_det = two_pi / det;
  for (int d = 0; d < 3; ++d) {
    b1[d] *= two_pi_over_det;
    b2[d] *= two_pi_over_det;
    b3[d] *= two_pi_over_det;
  }
  const double volume_k = two_pi * two_pi * two_pi / std::abs(det);
  const int n1_max = static_cast<int>(alpha * two_pi * area(b2, b3) / volume_k);
  const int n2_max = static_cast<int>(alpha * two_pi * area(b3, b1) / volume_k);
  const int n3_max = static_cast<int>(alpha * two_pi * area(b1, b2) / volume_k);
  const double ksq_max = two_pi * two_pi * alpha * alpha;
  const double alpha_factor = 0.25 / (alpha * alpha);

  QnepKSpace out;
  for (int n1 = 0; n1 <= n1_max; ++n1) {
    for (int n2 = -n2_max; n2 <= n2_max; ++n2) {
      for (int n3 = -n3_max; n3 <= n3_max; ++n3) {
        const int nsq = n1 * n1 + n2 * n2 + n3 * n3;
        if (nsq == 0 || (n1 == 0 && n2 < 0) ||
            (n1 == 0 && n2 == 0 && n3 < 0)) {
          continue;
        }
        const double kx = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
        const double ky = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
        const double kz = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
        const double ksq = kx * kx + ky * ky + kz * kz;
        if (ksq < ksq_max) {
          out.kx.push_back(kx);
          out.ky.push_back(ky);
          out.kz.push_back(kz);
          out.g.push_back(
              2.0 * std::abs(two_pi_over_det) / ksq *
              std::exp(-ksq * alpha_factor));
        }
      }
    }
  }
  return out;
}

template <typename T>
T* copy_to_device(const std::vector<T>& host, const char* action) {
  if (host.empty()) {
    return nullptr;
  }
  T* device = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(T)), action);
  check_cuda(
      cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice),
      action);
  return device;
}

struct DeviceKSpace {
  int count = 0;
  double* kx = nullptr;
  double* ky = nullptr;
  double* kz = nullptr;
  double* g = nullptr;
  double* s_real = nullptr;
  double* s_imag = nullptr;
};

void free_kspace(DeviceKSpace& kspace) {
  cudaFree(kspace.kx);
  cudaFree(kspace.ky);
  cudaFree(kspace.kz);
  cudaFree(kspace.g);
  cudaFree(kspace.s_real);
  cudaFree(kspace.s_imag);
}

DeviceKSpace copy_kspace_to_device(const QnepKSpace& host) {
  DeviceKSpace device;
  device.count = static_cast<int>(host.g.size());
  if (device.count == 0) {
    return device;
  }
  device.kx = copy_to_device(host.kx, "copy qNEP kx");
  device.ky = copy_to_device(host.ky, "copy qNEP ky");
  device.kz = copy_to_device(host.kz, "copy qNEP kz");
  device.g = copy_to_device(host.g, "copy qNEP G");
  check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&device.s_real), host.g.size() * sizeof(double)),
      "allocate qNEP S_real");
  check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&device.s_imag), host.g.size() * sizeof(double)),
      "allocate qNEP S_imag");
  return device;
}

__global__ void reciprocal_sums(
    int atom_count,
    int atom_stride,
    const double* __restrict__ charge,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ kx,
    const double* __restrict__ ky,
    const double* __restrict__ kz,
    double* __restrict__ s_real,
    double* __restrict__ s_imag) {
  extern __shared__ double reciprocal_partial[];
  double* real_partial = reciprocal_partial;
  double* imag_partial = reciprocal_partial + blockDim.x;
  const int k = blockIdx.x;
  double real = 0.0;
  double imag = 0.0;
  for (int atom = threadIdx.x; atom < atom_count; atom += blockDim.x) {
    const double kr = kx[k] * positions_soa3[atom] +
                      ky[k] * positions_soa3[atom_stride + atom] +
                      kz[k] * positions_soa3[2 * atom_stride + atom];
    const double q = charge[atom];
    real += q * cos(kr);
    imag -= q * sin(kr);
  }
  real_partial[threadIdx.x] = real;
  imag_partial[threadIdx.x] = imag;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      real_partial[threadIdx.x] += real_partial[threadIdx.x + stride];
      imag_partial[threadIdx.x] += imag_partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    s_real[k] = real_partial[0];
    s_imag[k] = imag_partial[0];
  }
}

__global__ void reciprocal_apply(
    int atom_count,
    int atom_stride,
    int k_count,
    double alpha_factor,
    const double* __restrict__ charge,
    const double* __restrict__ positions_soa3,
    const double* __restrict__ kx,
    const double* __restrict__ ky,
    const double* __restrict__ kz,
    const double* __restrict__ g,
    const double* __restrict__ s_real,
    const double* __restrict__ s_imag,
    double* __restrict__ potential,
    double* __restrict__ force_soa3,
    double* __restrict__ virial_soa9,
    float* __restrict__ d_real) {
  const int atom = blockIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const double q = charge[atom];
  double energy_sum = 0.0;
  double virial_sum[6] = {};
  double force_sum[3] = {};
  double d_real_sum = 0.0;
  for (int k = threadIdx.x; k < k_count; k += blockDim.x) {
    const double kxv = kx[k];
    const double kyv = ky[k];
    const double kzv = kz[k];
    const double kr = kxv * positions_soa3[atom] +
                      kyv * positions_soa3[atom_stride + atom] +
                      kzv * positions_soa3[2 * atom_stride + atom];
    double sin_kr = 0.0;
    double cos_kr = 0.0;
    sincos(kr, &sin_kr, &cos_kr);
    const double gv = g[k];
    const double imag_term = gv * (s_real[k] * sin_kr + s_imag[k] * cos_kr);
    const double gse = gv * (s_real[k] * cos_kr - s_imag[k] * sin_kr);
    const double qgse = q * gse;
    const double ksq = kxv * kxv + kyv * kyv + kzv * kzv;
    const double alpha_k_factor = 2.0 * alpha_factor + 2.0 / ksq;
    energy_sum += qgse;
    virial_sum[0] += qgse * (1.0 - alpha_k_factor * kxv * kxv);
    virial_sum[1] += qgse * (1.0 - alpha_k_factor * kyv * kyv);
    virial_sum[2] += qgse * (1.0 - alpha_k_factor * kzv * kzv);
    virial_sum[3] -= qgse * (alpha_k_factor * kxv * kyv);
    virial_sum[4] -= qgse * (alpha_k_factor * kyv * kzv);
    virial_sum[5] -= qgse * (alpha_k_factor * kzv * kxv);
    d_real_sum += gse;
    force_sum[0] += kxv * imag_term;
    force_sum[1] += kyv * imag_term;
    force_sum[2] += kzv * imag_term;
  }
  extern __shared__ double reciprocal_apply_partial[];
  double* partial = reciprocal_apply_partial;
  partial[threadIdx.x] = energy_sum;
  partial[blockDim.x + threadIdx.x] = virial_sum[0];
  partial[2 * blockDim.x + threadIdx.x] = virial_sum[1];
  partial[3 * blockDim.x + threadIdx.x] = virial_sum[2];
  partial[4 * blockDim.x + threadIdx.x] = virial_sum[3];
  partial[5 * blockDim.x + threadIdx.x] = virial_sum[4];
  partial[6 * blockDim.x + threadIdx.x] = virial_sum[5];
  partial[7 * blockDim.x + threadIdx.x] = d_real_sum;
  partial[8 * blockDim.x + threadIdx.x] = force_sum[0];
  partial[9 * blockDim.x + threadIdx.x] = force_sum[1];
  partial[10 * blockDim.x + threadIdx.x] = force_sum[2];
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      for (int item = 0; item < 11; ++item) {
        partial[item * blockDim.x + threadIdx.x] +=
            partial[item * blockDim.x + threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x != 0) {
    return;
  }
  energy_sum = partial[0];
  virial_sum[0] = partial[blockDim.x];
  virial_sum[1] = partial[2 * blockDim.x];
  virial_sum[2] = partial[3 * blockDim.x];
  virial_sum[3] = partial[4 * blockDim.x];
  virial_sum[4] = partial[5 * blockDim.x];
  virial_sum[5] = partial[6 * blockDim.x];
  d_real_sum = partial[7 * blockDim.x];
  force_sum[0] = partial[8 * blockDim.x];
  force_sum[1] = partial[9 * blockDim.x];
  force_sum[2] = partial[10 * blockDim.x];
  potential[atom] += kCoulomb * energy_sum;
  virial_soa9[atom] += kCoulomb * virial_sum[0];
  virial_soa9[atom_stride + atom] += kCoulomb * virial_sum[1];
  virial_soa9[2 * atom_stride + atom] += kCoulomb * virial_sum[2];
  virial_soa9[3 * atom_stride + atom] += kCoulomb * virial_sum[3];
  virial_soa9[4 * atom_stride + atom] += kCoulomb * virial_sum[5];
  virial_soa9[5 * atom_stride + atom] += kCoulomb * virial_sum[4];
  virial_soa9[6 * atom_stride + atom] += kCoulomb * virial_sum[3];
  virial_soa9[7 * atom_stride + atom] += kCoulomb * virial_sum[5];
  virial_soa9[8 * atom_stride + atom] += kCoulomb * virial_sum[4];
  d_real[atom] = static_cast<float>(2.0 * kCoulomb * d_real_sum);
  const double charge_factor = kCoulomb * 2.0 * q;
  force_soa3[atom] += charge_factor * force_sum[0];
  force_soa3[atom_stride + atom] += charge_factor * force_sum[1];
  force_soa3[2 * atom_stride + atom] += charge_factor * force_sum[2];
}

#if defined(NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM)
struct PppmPara {
  int k[3] = {};
  int k_half[3] = {};
  int k0k1 = 0;
  int k0k1k2 = 0;
  float alpha_factor = 0.0f;
  float two_pi_over_volume = 0.0f;
  float potential_factor = 0.0f;
  float reciprocal[9] = {};
  float two_pi_over_k[3] = {};
};

struct DevicePppm {
  PppmPara para;
  int mesh_count = 0;
  cufftHandle plan = 0;
  cufftHandle plan_virial = 0;
  float* kx = nullptr;
  float* ky = nullptr;
  float* kz = nullptr;
  float* g = nullptr;
  cufftComplex* mesh = nullptr;
  cufftComplex* mesh_g = nullptr;
  cufftComplex* mesh_x = nullptr;
  cufftComplex* mesh_y = nullptr;
  cufftComplex* mesh_z = nullptr;
  cufftComplex* mesh_virial = nullptr;
};

void free_pppm(DevicePppm& pppm) {
  cudaFree(pppm.kx);
  cudaFree(pppm.ky);
  cudaFree(pppm.kz);
  cudaFree(pppm.g);
  cudaFree(pppm.mesh);
  cudaFree(pppm.mesh_g);
  cudaFree(pppm.mesh_x);
  cudaFree(pppm.mesh_y);
  cudaFree(pppm.mesh_z);
  cudaFree(pppm.mesh_virial);
  if (pppm.plan != 0) {
    cufftDestroy(pppm.plan);
  }
  if (pppm.plan_virial != 0) {
    cufftDestroy(pppm.plan_virial);
  }
  pppm = DevicePppm{};
}

struct PppmCache {
  DevicePppm scratch;
  ~PppmCache() {
    free_pppm(scratch);
  }
};

PppmPara make_pppm_para(const SimulationBox& box, float alpha, int atom_count) {
  double a1[3] = {box.frac_to_cart[0], box.frac_to_cart[3], box.frac_to_cart[6]};
  double a2[3] = {box.frac_to_cart[1], box.frac_to_cart[4], box.frac_to_cart[7]};
  double a3[3] = {box.frac_to_cart[2], box.frac_to_cart[5], box.frac_to_cart[8]};
  const double vol = volume(box);
  const double thickness[3] = {
      vol / area(a2, a3),
      vol / area(a3, a1),
      vol / area(a1, a2)};
  PppmPara para;
  for (int d = 0; d < 3; ++d) {
    para.k[d] = best_pppm_mesh_dim(static_cast<int>(thickness[d]));
    para.k_half[d] = para.k[d] / 2;
    para.two_pi_over_k[d] = kTwoPiF / static_cast<float>(para.k[d]);
  }
  para.k0k1 = para.k[0] * para.k[1];
  para.k0k1k2 = para.k0k1 * para.k[2];
  para.alpha_factor = 0.25f / (alpha * alpha);
  para.two_pi_over_volume = kTwoPiF / static_cast<float>(vol);
  para.potential_factor = static_cast<float>(kCoulomb / atom_count);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      para.reciprocal[3 * row + col] =
          kTwoPiF * static_cast<float>(box.cart_to_frac[3 * row + col]);
    }
  }
  return para;
}

void ensure_pppm_storage(DevicePppm& pppm, const PppmPara& para) {
  if (pppm.mesh_count == para.k0k1k2 &&
      pppm.para.k[0] == para.k[0] &&
      pppm.para.k[1] == para.k[1] &&
      pppm.para.k[2] == para.k[2]) {
    pppm.para = para;
    return;
  }
  free_pppm(pppm);
  pppm.para = para;
  pppm.mesh_count = para.k0k1k2;
  const std::size_t count = static_cast<std::size_t>(para.k0k1k2);
  check_cuda(cudaMalloc(&pppm.kx, count * sizeof(float)), "allocate PPPM kx");
  check_cuda(cudaMalloc(&pppm.ky, count * sizeof(float)), "allocate PPPM ky");
  check_cuda(cudaMalloc(&pppm.kz, count * sizeof(float)), "allocate PPPM kz");
  check_cuda(cudaMalloc(&pppm.g, count * sizeof(float)), "allocate PPPM G");
  check_cuda(cudaMalloc(&pppm.mesh, count * sizeof(cufftComplex)), "allocate PPPM mesh");
  check_cuda(cudaMalloc(&pppm.mesh_g, count * sizeof(cufftComplex)), "allocate PPPM mesh_G");
  check_cuda(cudaMalloc(&pppm.mesh_x, count * sizeof(cufftComplex)), "allocate PPPM mesh_x");
  check_cuda(cudaMalloc(&pppm.mesh_y, count * sizeof(cufftComplex)), "allocate PPPM mesh_y");
  check_cuda(cudaMalloc(&pppm.mesh_z, count * sizeof(cufftComplex)), "allocate PPPM mesh_z");
  check_cuda(
      cudaMalloc(&pppm.mesh_virial, count * 6 * sizeof(cufftComplex)),
      "allocate PPPM virial mesh");
  check_cufft(
      cufftPlan3d(&pppm.plan, para.k[2], para.k[1], para.k[0], CUFFT_C2C),
      "create PPPM FFT plan");
  int dims[3] = {para.k[2], para.k[1], para.k[0]};
  check_cufft(
      cufftPlanMany(
          &pppm.plan_virial,
          3,
          dims,
          nullptr,
          1,
          para.k0k1k2,
          nullptr,
          1,
          para.k0k1k2,
          CUFFT_C2C,
          6),
      "create PPPM virial FFT plan");
}

__device__ __forceinline__ float pppm_sinc(float x) {
  float y = 0.0f;
  if (x * x <= 1.0f) {
    float term = 1.0f;
    for (int i = 0; i < 6; ++i) {
      y += kPppmSincCoeff[i] * term;
      term *= x * x;
    }
  } else {
    y = sinf(x) / x;
  }
  return y;
}

__device__ __forceinline__ int wrap_pppm_mesh_index(int dim, int index) {
  if (index >= dim) {
    return index - dim;
  }
  if (index < 0) {
    return index + dim;
  }
  return index;
}

__device__ __forceinline__ void pppm_weights(float d, float* weight) {
  for (int i = 0; i < 5; ++i) {
    weight[i] =
        (((kPppmWCoeff[i][4] * d + kPppmWCoeff[i][3]) * d +
          kPppmWCoeff[i][2]) * d +
         kPppmWCoeff[i][1]) * d +
        kPppmWCoeff[i][0];
  }
}

__global__ void pppm_find_k_and_g(
    PppmPara para,
    float* __restrict__ kx,
    float* __restrict__ ky,
    float* __restrict__ kz,
    float* __restrict__ g) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= para.k0k1k2) {
    return;
  }
  int nk2 = index / para.k0k1;
  int nk1 = (index - nk2 * para.k0k1) / para.k[0];
  int nk0 = index % para.k[0];
  int nk[3] = {nk0, nk1, nk2};
  float denominator[3] = {};
  for (int d = 0; d < 3; ++d) {
    if (nk[d] >= para.k_half[d]) {
      nk[d] -= para.k[d];
    }
    float term = sinf(0.5f * para.two_pi_over_k[d] * nk[d]);
    term *= term;
    term = (((kPppmGCoeff[4] * term + kPppmGCoeff[3]) * term +
             kPppmGCoeff[2]) * term +
            kPppmGCoeff[1]) * term +
           kPppmGCoeff[0];
    denominator[d] = term * term;
  }
  const float kxv =
      nk[0] * para.reciprocal[0] +
      nk[1] * para.reciprocal[3] +
      nk[2] * para.reciprocal[6];
  const float kyv =
      nk[0] * para.reciprocal[1] +
      nk[1] * para.reciprocal[4] +
      nk[2] * para.reciprocal[7];
  const float kzv =
      nk[0] * para.reciprocal[2] +
      nk[1] * para.reciprocal[5] +
      nk[2] * para.reciprocal[8];
  kx[index] = kxv;
  ky[index] = kyv;
  kz[index] = kzv;
  const float ksq = kxv * kxv + kyv * kyv + kzv * kzv;
  if (ksq == 0.0f) {
    g[index] = 0.0f;
    return;
  }
  float numerator = pppm_sinc(0.5f * para.two_pi_over_k[0] * nk[0]);
  numerator *= pppm_sinc(0.5f * para.two_pi_over_k[1] * nk[1]);
  numerator *= pppm_sinc(0.5f * para.two_pi_over_k[2] * nk[2]);
  numerator = numerator * numerator * numerator * numerator * numerator;
  numerator *= numerator;
  g[index] =
      numerator * para.two_pi_over_volume / ksq * expf(-ksq * para.alpha_factor) /
      (denominator[0] * denominator[1] * denominator[2]);
}

__global__ void pppm_zero_mesh(int count, cufftComplex* __restrict__ mesh) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    mesh[index].x = 0.0f;
    mesh[index].y = 0.0f;
  }
}

__global__ void pppm_find_mesh(
    int atom_count,
    int atom_stride,
    PppmPara para,
    SimulationBox box,
    const double* __restrict__ charge,
    const double* __restrict__ positions_soa3,
    cufftComplex* __restrict__ mesh) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const double x = positions_soa3[atom];
  const double y = positions_soa3[atom_stride + atom];
  const double z = positions_soa3[2 * atom_stride + atom];
  const float q = static_cast<float>(charge[atom]);
  const float sx = static_cast<float>(
      (box.cart_to_frac[0] * x + box.cart_to_frac[1] * y + box.cart_to_frac[2] * z) *
      para.k[0]);
  const float sy = static_cast<float>(
      (box.cart_to_frac[3] * x + box.cart_to_frac[4] * y + box.cart_to_frac[5] * z) *
      para.k[1]);
  const float sz = static_cast<float>(
      (box.cart_to_frac[6] * x + box.cart_to_frac[7] * y + box.cart_to_frac[8] * z) *
      para.k[2]);
  const int ix = static_cast<int>(sx + 0.5f);
  const int iy = static_cast<int>(sy + 0.5f);
  const int iz = static_cast<int>(sz + 0.5f);
  float wx[5] = {};
  float wy[5] = {};
  float wz[5] = {};
  pppm_weights(sx - ix, wx);
  pppm_weights(sy - iy, wy);
  pppm_weights(sz - iz, wz);
  for (int dx = -2; dx <= 2; ++dx) {
    const int nx = wrap_pppm_mesh_index(para.k[0], ix + dx);
    for (int dy = -2; dy <= 2; ++dy) {
      const int ny = wrap_pppm_mesh_index(para.k[1], iy + dy);
      for (int dz = -2; dz <= 2; ++dz) {
        const int nz = wrap_pppm_mesh_index(para.k[2], iz + dz);
        const int mesh_index = nx + para.k[0] * (ny + para.k[1] * nz);
        const float weight = wx[dx + 2] * wy[dy + 2] * wz[dz + 2];
        atomicAdd(&mesh[mesh_index].x, q * weight);
      }
    }
  }
}

__global__ void pppm_ik_times_mesh_g(
    int count,
    const float* __restrict__ kx,
    const float* __restrict__ ky,
    const float* __restrict__ kz,
    const float* __restrict__ g,
    const cufftComplex* __restrict__ mesh,
    cufftComplex* __restrict__ mesh_x,
    cufftComplex* __restrict__ mesh_y,
    cufftComplex* __restrict__ mesh_z) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const cufftComplex value = mesh[index];
  const float gv = g[index];
  mesh_x[index] = {value.y * kx[index] * gv, -value.x * kx[index] * gv};
  mesh_y[index] = {value.y * ky[index] * gv, -value.x * ky[index] * gv};
  mesh_z[index] = {value.y * kz[index] * gv, -value.x * kz[index] * gv};
}

__global__ void pppm_find_mesh_g(
    int count,
    const float* __restrict__ g,
    const cufftComplex* __restrict__ mesh,
    cufftComplex* __restrict__ mesh_g) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    mesh_g[index] = {mesh[index].x * g[index], mesh[index].y * g[index]};
  }
}

__global__ void pppm_find_mesh_virial(
    int count,
    PppmPara para,
    const float* __restrict__ kx,
    const float* __restrict__ ky,
    const float* __restrict__ kz,
    const float* __restrict__ g,
    const cufftComplex* __restrict__ mesh,
    cufftComplex* __restrict__ virial) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float kxv = kx[index];
  const float kyv = ky[index];
  const float kzv = kz[index];
  const float ksq = kxv * kxv + kyv * kyv + kzv * kzv;
  float factor[6] = {};
  if (ksq != 0.0f) {
    const float alpha_k_factor = 2.0f * para.alpha_factor + 2.0f / ksq;
    factor[0] = 1.0f - alpha_k_factor * kxv * kxv;
    factor[1] = 1.0f - alpha_k_factor * kyv * kyv;
    factor[2] = 1.0f - alpha_k_factor * kzv * kzv;
    factor[3] = -alpha_k_factor * kxv * kyv;
    factor[4] = -alpha_k_factor * kyv * kzv;
    factor[5] = -alpha_k_factor * kzv * kxv;
  }
  const float gx = g[index] * mesh[index].x;
  const float gy = g[index] * mesh[index].y;
  for (int component = 0; component < 6; ++component) {
    virial[component * count + index] = {factor[component] * gx, factor[component] * gy};
  }
}

__global__ void pppm_apply_field(
    int atom_count,
    int atom_stride,
    PppmPara para,
    SimulationBox box,
    const double* __restrict__ charge,
    const double* __restrict__ positions_soa3,
    const cufftComplex* __restrict__ mesh_g,
    const cufftComplex* __restrict__ mesh_x,
    const cufftComplex* __restrict__ mesh_y,
    const cufftComplex* __restrict__ mesh_z,
    const cufftComplex* __restrict__ mesh_virial,
    float* __restrict__ d_real,
    double* __restrict__ force_soa3,
    double* __restrict__ virial_soa9,
    double* __restrict__ potential,
    bool need_per_atom_potential) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const double x = positions_soa3[atom];
  const double y = positions_soa3[atom_stride + atom];
  const double z = positions_soa3[2 * atom_stride + atom];
  const float q = static_cast<float>(charge[atom]);
  const float sx = static_cast<float>(
      (box.cart_to_frac[0] * x + box.cart_to_frac[1] * y + box.cart_to_frac[2] * z) *
      para.k[0]);
  const float sy = static_cast<float>(
      (box.cart_to_frac[3] * x + box.cart_to_frac[4] * y + box.cart_to_frac[5] * z) *
      para.k[1]);
  const float sz = static_cast<float>(
      (box.cart_to_frac[6] * x + box.cart_to_frac[7] * y + box.cart_to_frac[8] * z) *
      para.k[2]);
  const int ix = static_cast<int>(sx + 0.5f);
  const int iy = static_cast<int>(sy + 0.5f);
  const int iz = static_cast<int>(sz + 0.5f);
  float wx[5] = {};
  float wy[5] = {};
  float wz[5] = {};
  pppm_weights(sx - ix, wx);
  pppm_weights(sy - iy, wy);
  pppm_weights(sz - iz, wz);
  float d_sum = 0.0f;
  float field[3] = {};
  float virial[6] = {};
  for (int dx = -2; dx <= 2; ++dx) {
    const int nx = wrap_pppm_mesh_index(para.k[0], ix + dx);
    for (int dy = -2; dy <= 2; ++dy) {
      const int ny = wrap_pppm_mesh_index(para.k[1], iy + dy);
      for (int dz = -2; dz <= 2; ++dz) {
        const int nz = wrap_pppm_mesh_index(para.k[2], iz + dz);
        const int mesh_index = nx + para.k[0] * (ny + para.k[1] * nz);
        const float weight = wx[dx + 2] * wy[dy + 2] * wz[dz + 2];
        d_sum += weight * mesh_g[mesh_index].x;
        field[0] += weight * mesh_x[mesh_index].x;
        field[1] += weight * mesh_y[mesh_index].x;
        field[2] += weight * mesh_z[mesh_index].x;
        if (mesh_virial != nullptr) {
          for (int component = 0; component < 6; ++component) {
            virial[component] +=
                weight * mesh_virial[component * para.k0k1k2 + mesh_index].x;
          }
        }
      }
    }
  }
  d_real[atom] = 2.0f * static_cast<float>(kCoulomb) * d_sum;
  const double force_factor = 2.0 * kCoulomb * static_cast<double>(q);
  force_soa3[atom] += force_factor * field[0];
  force_soa3[atom_stride + atom] += force_factor * field[1];
  force_soa3[2 * atom_stride + atom] += force_factor * field[2];
  if (mesh_virial != nullptr) {
    const double virial_factor = kCoulomb * static_cast<double>(q);
    virial_soa9[atom] += virial_factor * virial[0];
    virial_soa9[atom_stride + atom] += virial_factor * virial[1];
    virial_soa9[2 * atom_stride + atom] += virial_factor * virial[2];
    virial_soa9[3 * atom_stride + atom] += virial_factor * virial[3];
    virial_soa9[4 * atom_stride + atom] += virial_factor * virial[5];
    virial_soa9[5 * atom_stride + atom] += virial_factor * virial[4];
    virial_soa9[6 * atom_stride + atom] += virial_factor * virial[3];
    virial_soa9[7 * atom_stride + atom] += virial_factor * virial[5];
    virial_soa9[8 * atom_stride + atom] += virial_factor * virial[4];
  }
  if (need_per_atom_potential) {
    potential[atom] += kCoulomb * static_cast<double>(q) * d_sum;
  }
}

__global__ void pppm_add_total_virial_and_potential(
    int atom_count,
    int atom_stride,
    PppmPara para,
    const cufftComplex* __restrict__ mesh,
    const float* __restrict__ kx,
    const float* __restrict__ ky,
    const float* __restrict__ kz,
    const float* __restrict__ g,
    double* __restrict__ virial_soa9,
    double* __restrict__ potential,
    bool add_total_potential) {
  const int tid = threadIdx.x;
  float value = 0.0f;
  for (int index = tid; index < para.k0k1k2; index += blockDim.x) {
    const float kxv = kx[index];
    const float kyv = ky[index];
    const float kzv = kz[index];
    const float ksq = kxv * kxv + kyv * kyv + kzv * kzv;
    if (ksq == 0.0f) {
      continue;
    }
    const cufftComplex s = mesh[index];
    const float gss = g[index] * (s.x * s.x + s.y * s.y);
    const float alpha_k_factor = 2.0f * para.alpha_factor + 2.0f / ksq;
    switch (blockIdx.x) {
      case 0:
        value += gss * (1.0f - alpha_k_factor * kxv * kxv);
        break;
      case 1:
        value += gss * (1.0f - alpha_k_factor * kyv * kyv);
        break;
      case 2:
        value += gss * (1.0f - alpha_k_factor * kzv * kzv);
        break;
      case 3:
        value -= gss * (alpha_k_factor * kxv * kyv);
        break;
      case 4:
        value -= gss * (alpha_k_factor * kyv * kzv);
        break;
      case 5:
        value -= gss * (alpha_k_factor * kzv * kxv);
        break;
      case 6:
        if (!add_total_potential) {
          return;
        }
        value += gss;
        break;
    }
  }
  extern __shared__ float partial[];
  partial[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }
  const double per_atom_value =
      static_cast<double>(partial[0]) * static_cast<double>(para.potential_factor);
  for (int atom = tid; atom < atom_count; atom += blockDim.x) {
    switch (blockIdx.x) {
      case 0:
        virial_soa9[atom] += per_atom_value;
        break;
      case 1:
        virial_soa9[atom_stride + atom] += per_atom_value;
        break;
      case 2:
        virial_soa9[2 * atom_stride + atom] += per_atom_value;
        break;
      case 3:
        virial_soa9[3 * atom_stride + atom] += per_atom_value;
        virial_soa9[6 * atom_stride + atom] += per_atom_value;
        break;
      case 4:
        virial_soa9[5 * atom_stride + atom] += per_atom_value;
        virial_soa9[8 * atom_stride + atom] += per_atom_value;
        break;
      case 5:
        virial_soa9[4 * atom_stride + atom] += per_atom_value;
        virial_soa9[7 * atom_stride + atom] += per_atom_value;
        break;
      case 6:
        if (!add_total_potential) {
          return;
        }
        potential[atom] += per_atom_value;
        break;
    }
  }
}

void apply_qnep_pppm(
    int atom_count,
    int atom_stride,
    const SimulationBox& box,
    double alpha,
    const DeviceWorkspaceView& view,
    bool need_per_atom_kspace_potential,
    bool need_per_atom_kspace_virial) {
  thread_local PppmCache cache;
  DevicePppm& pppm = cache.scratch;
  ensure_pppm_storage(
      pppm, make_pppm_para(box, static_cast<float>(alpha), atom_count));
  const int mesh_threads = 64;
  const int mesh_blocks = (pppm.para.k0k1k2 + mesh_threads - 1) / mesh_threads;
  pppm_find_k_and_g<<<mesh_blocks, mesh_threads>>>(
      pppm.para, pppm.kx, pppm.ky, pppm.kz, pppm.g);
  check_cuda(cudaGetLastError(), "qNEP PPPM k/G launch failed");
  pppm_zero_mesh<<<mesh_blocks, mesh_threads>>>(pppm.para.k0k1k2, pppm.mesh);
  check_cuda(cudaGetLastError(), "qNEP PPPM zero mesh launch failed");
  const int atom_threads = 64;
  const int atom_blocks = (atom_count + atom_threads - 1) / atom_threads;
  pppm_find_mesh<<<atom_blocks, atom_threads>>>(
      atom_count, atom_stride, pppm.para, box, view.charge, view.positions_soa3, pppm.mesh);
  check_cuda(cudaGetLastError(), "qNEP PPPM mesh launch failed");
  check_cufft(cufftExecC2C(pppm.plan, pppm.mesh, pppm.mesh, CUFFT_FORWARD),
              "qNEP PPPM forward FFT");
  pppm_ik_times_mesh_g<<<mesh_blocks, mesh_threads>>>(
      pppm.para.k0k1k2,
      pppm.kx,
      pppm.ky,
      pppm.kz,
      pppm.g,
      pppm.mesh,
      pppm.mesh_x,
      pppm.mesh_y,
      pppm.mesh_z);
  check_cuda(cudaGetLastError(), "qNEP PPPM field mesh launch failed");
  pppm_find_mesh_g<<<mesh_blocks, mesh_threads>>>(
      pppm.para.k0k1k2, pppm.g, pppm.mesh, pppm.mesh_g);
  check_cuda(cudaGetLastError(), "qNEP PPPM potential mesh launch failed");
  check_cufft(cufftExecC2C(pppm.plan, pppm.mesh_g, pppm.mesh_g, CUFFT_INVERSE),
              "qNEP PPPM potential inverse FFT");
  check_cufft(cufftExecC2C(pppm.plan, pppm.mesh_x, pppm.mesh_x, CUFFT_INVERSE),
              "qNEP PPPM x inverse FFT");
  check_cufft(cufftExecC2C(pppm.plan, pppm.mesh_y, pppm.mesh_y, CUFFT_INVERSE),
              "qNEP PPPM y inverse FFT");
  check_cufft(cufftExecC2C(pppm.plan, pppm.mesh_z, pppm.mesh_z, CUFFT_INVERSE),
              "qNEP PPPM z inverse FFT");
  if (need_per_atom_kspace_virial) {
    pppm_find_mesh_virial<<<mesh_blocks, mesh_threads>>>(
        pppm.para.k0k1k2,
        pppm.para,
        pppm.kx,
        pppm.ky,
        pppm.kz,
        pppm.g,
        pppm.mesh,
        pppm.mesh_virial);
    check_cuda(cudaGetLastError(), "qNEP PPPM virial mesh launch failed");
    check_cufft(
        cufftExecC2C(pppm.plan_virial, pppm.mesh_virial, pppm.mesh_virial, CUFFT_INVERSE),
        "qNEP PPPM virial inverse FFT");
    pppm_apply_field<<<atom_blocks, atom_threads>>>(
        atom_count,
        atom_stride,
        pppm.para,
        box,
        view.charge,
        view.positions_soa3,
        pppm.mesh_g,
        pppm.mesh_x,
        pppm.mesh_y,
        pppm.mesh_z,
        pppm.mesh_virial,
        view.d_real,
        view.force_soa3,
        view.virial_soa9,
        view.potential,
        true);
    check_cuda(cudaGetLastError(), "qNEP PPPM field apply launch failed");
  } else {
    pppm_apply_field<<<atom_blocks, atom_threads>>>(
        atom_count,
        atom_stride,
        pppm.para,
        box,
        view.charge,
        view.positions_soa3,
        pppm.mesh_g,
        pppm.mesh_x,
        pppm.mesh_y,
        pppm.mesh_z,
        nullptr,
        view.d_real,
        view.force_soa3,
        view.virial_soa9,
        view.potential,
        need_per_atom_kspace_potential);
    check_cuda(cudaGetLastError(), "qNEP PPPM field apply launch failed");
    const bool add_total_potential = !need_per_atom_kspace_potential;
    const int total_blocks = add_total_potential ? 7 : 6;
    pppm_add_total_virial_and_potential<<<total_blocks, 1024, 1024 * sizeof(float)>>>(
        atom_count,
        atom_stride,
        pppm.para,
        pppm.mesh,
        pppm.kx,
        pppm.ky,
        pppm.kz,
        pppm.g,
        view.virial_soa9,
        view.potential,
        add_total_potential);
    check_cuda(cudaGetLastError(), "qNEP PPPM total virial launch failed");
  }
}
#endif

__global__ void real_space_charge(
    int atom_count,
    int atom_stride,
    int charge_mode,
    double cutoff_radial,
    SimulationBox box,
    const int* __restrict__ nn_radial,
    const int* __restrict__ nl_radial,
    const double* __restrict__ charge,
    const double* __restrict__ positions_soa3,
    double* __restrict__ potential,
    double* __restrict__ force_soa3,
    double* __restrict__ virial_soa9,
    float* __restrict__ d_real) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  const double alpha = kPi / cutoff_radial;
  const double two_alpha_over_sqrt_pi = 2.0 * alpha / sqrt(kPi);
  const double a = erfc(kPi) / (cutoff_radial * cutoff_radial) +
                   two_alpha_over_sqrt_pi * exp(-kPi * kPi) / cutoff_radial;
  const double b = -erfc(kPi) / cutoff_radial - a * cutoff_radial;
  const bool real_space_only = charge_mode == 3;
  const double q1 = charge[atom];
  const double x1 = positions_soa3[atom];
  const double y1 = positions_soa3[atom_stride + atom];
  const double z1 = positions_soa3[2 * atom_stride + atom];

  double s_fx = 0.0;
  double s_fy = 0.0;
  double s_fz = 0.0;
  double s_sxx = 0.0;
  double s_sxy = 0.0;
  double s_sxz = 0.0;
  double s_syx = 0.0;
  double s_syy = 0.0;
  double s_syz = 0.0;
  double s_szx = 0.0;
  double s_szy = 0.0;
  double s_szz = 0.0;
  double s_pe = real_space_only ? 0.0 : -two_alpha_over_sqrt_pi * 0.5 * q1 * q1;
  double d_sum = real_space_only ? 0.0 : -q1 * two_alpha_over_sqrt_pi;
  for (int slot = 0; slot < nn_radial[atom]; ++slot) {
    const int neighbor = nl_radial[atom + atom_stride * slot];
    const double q2 = charge[neighbor];
    const double qq = q1 * q2;
    float x12 = 0.0f;
    float y12 = 0.0f;
    float z12 = 0.0f;
    minimum_image_delta(
        box,
        positions_soa3[neighbor] - x1,
        positions_soa3[atom_stride + neighbor] - y1,
        positions_soa3[2 * atom_stride + neighbor] - z1,
        x12,
        y12,
        z12);
    const double r = sqrt(
        static_cast<double>(x12) * x12 +
        static_cast<double>(y12) * y12 +
        static_cast<double>(z12) * z12);
    const double rinv = 1.0 / r;
    const double erfc_r = erfc(alpha * r) * rinv;
    const double exp_term = exp(-alpha * alpha * r * r);
    if (real_space_only) {
      d_sum += q2 * (erfc_r + a * r + b);
      s_pe += 0.5 * qq * (erfc_r + a * r + b);
    } else {
      d_sum += q2 * erfc_r;
      s_pe += 0.5 * qq * erfc_r;
    }
    double f2 = erfc_r + two_alpha_over_sqrt_pi * exp_term;
    if (real_space_only) {
      f2 = -0.5 * kCoulomb * qq * (f2 * rinv * rinv - a * rinv);
    } else {
      f2 *= -0.5 * kCoulomb * qq * rinv * rinv;
    }
    const double f12x = x12 * f2;
    const double f12y = y12 * f2;
    const double f12z = z12 * f2;
    s_fx += 2.0 * f12x;
    s_fy += 2.0 * f12y;
    s_fz += 2.0 * f12z;
    s_sxx -= x12 * f12x;
    s_sxy -= x12 * f12y;
    s_sxz -= x12 * f12z;
    s_syx -= y12 * f12x;
    s_syy -= y12 * f12y;
    s_syz -= y12 * f12z;
    s_szx -= z12 * f12x;
    s_szy -= z12 * f12y;
    s_szz -= z12 * f12z;
  }
  force_soa3[atom] += s_fx;
  force_soa3[atom_stride + atom] += s_fy;
  force_soa3[2 * atom_stride + atom] += s_fz;
  virial_soa9[atom] += s_sxx;
  virial_soa9[atom_stride + atom] += s_syy;
  virial_soa9[2 * atom_stride + atom] += s_szz;
  virial_soa9[3 * atom_stride + atom] += s_sxy;
  virial_soa9[4 * atom_stride + atom] += s_sxz;
  virial_soa9[5 * atom_stride + atom] += s_syz;
  virial_soa9[6 * atom_stride + atom] += s_syx;
  virial_soa9[7 * atom_stride + atom] += s_szx;
  virial_soa9[8 * atom_stride + atom] += s_szy;
  d_real[atom] += static_cast<float>(kCoulomb * d_sum);
  potential[atom] += kCoulomb * s_pe;
}

__global__ void zero_mean_d_real_kernel(
    int atom_count,
    float* __restrict__ d_real) {
  extern __shared__ float d_real_partial[];
  float sum = 0.0f;
  for (int atom = threadIdx.x; atom < atom_count; atom += blockDim.x) {
    sum += d_real[atom];
  }
  d_real_partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      d_real_partial[threadIdx.x] += d_real_partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float mean = d_real_partial[0] / static_cast<float>(atom_count);
  for (int atom = threadIdx.x; atom < atom_count; atom += blockDim.x) {
    d_real[atom] -= mean;
  }
}

}  // namespace

void apply_qnep_charge_terms_on_device(
    const ModelProtocol& protocol,
    int atom_count,
    const SimulationBox& box,
    DeviceWorkspace& workspace,
    bool need_per_atom_kspace_potential,
    bool need_per_atom_kspace_virial) {
  require(atom_count > 0, "atom_count must be positive");
  require(protocol.charge_mode > 0, "qNEP charge terms require charge mode");
  const DeviceWorkspaceView view = workspace.view();
  require(static_cast<std::size_t>(atom_count) <= view.atom_capacity,
          "atom_count exceeds workspace atom capacity");
  require(view.charge != nullptr, "workspace missing charge output");
  require(view.positions_soa3 != nullptr, "workspace missing positions");
  require(view.potential != nullptr, "workspace missing potential output");
  require(view.force_soa3 != nullptr, "workspace missing force output");
  require(view.virial_soa9 != nullptr, "workspace missing virial output");
  require(view.d_real != nullptr, "workspace missing D_real cache");

  check_cuda(
      cudaMemset(view.force_soa3, 0, view.atom_capacity * 3 * sizeof(double)),
      "clear qNEP charge force");
  check_cuda(
      cudaMemset(view.virial_soa9, 0, view.atom_capacity * 9 * sizeof(double)),
      "clear qNEP charge virial");
  check_cuda(
      cudaMemset(view.d_real, 0, view.atom_capacity * sizeof(float)),
      "clear qNEP D_real");

  const double alpha = kPi / protocol.cutoff_radial;
  const int threads = 32;
  const int blocks = (atom_count + threads - 1) / threads;
  if (protocol.charge_mode == 1 || protocol.charge_mode == 2) {
    const char* kspace = std::getenv("NEP_ADAPTERS_QNEP_KSPACE");
    const std::string kspace_mode = kspace == nullptr ? "direct" : kspace;
    require(
        kspace_mode == "direct" || kspace_mode == "pppm",
        "NEP_ADAPTERS_QNEP_KSPACE must be 'direct' or 'pppm'");
    if (kspace_mode == "pppm") {
#if defined(NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM)
      apply_qnep_pppm(
          atom_count,
          static_cast<int>(view.atom_capacity),
          box,
          alpha,
          view,
          need_per_atom_kspace_potential,
          need_per_atom_kspace_virial);
#else
      throw std::runtime_error(
          "qNEP PPPM support is disabled; rebuild with "
          "NEP_ADAPTERS_CUDA_ENABLE_QNEP_PPPM=ON");
#endif
    } else {
      const QnepKSpace host_kspace = make_qnep_kspace(box, alpha);
      DeviceKSpace device_kspace = copy_kspace_to_device(host_kspace);
      if (device_kspace.count > 0) {
        reciprocal_sums<<<device_kspace.count, 256, 2 * 256 * sizeof(double)>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            view.charge,
            view.positions_soa3,
            device_kspace.kx,
            device_kspace.ky,
            device_kspace.kz,
            device_kspace.s_real,
            device_kspace.s_imag);
        check_cuda(cudaGetLastError(), "qNEP reciprocal sums launch failed");
        reciprocal_apply<<<atom_count, threads, 11 * threads * sizeof(double)>>>(
            atom_count,
            static_cast<int>(view.atom_capacity),
            device_kspace.count,
            0.25 / (alpha * alpha),
            view.charge,
            view.positions_soa3,
            device_kspace.kx,
            device_kspace.ky,
            device_kspace.kz,
            device_kspace.g,
            device_kspace.s_real,
            device_kspace.s_imag,
            view.potential,
            view.force_soa3,
            view.virial_soa9,
            view.d_real);
        check_cuda(cudaGetLastError(), "qNEP reciprocal apply launch failed");
      }
      free_kspace(device_kspace);
    }
  }
  if (protocol.charge_mode == 1 || protocol.charge_mode == 3) {
    require(view.nn_radial != nullptr, "workspace missing radial counts");
    require(view.nl_radial_slot_major != nullptr, "workspace missing radial neighbors");
    real_space_charge<<<blocks, threads>>>(
        atom_count,
        static_cast<int>(view.atom_capacity),
        protocol.charge_mode,
        protocol.cutoff_radial,
        box,
        view.nn_radial,
        view.nl_radial_slot_major,
        view.charge,
        view.positions_soa3,
        view.potential,
        view.force_soa3,
        view.virial_soa9,
        view.d_real);
    check_cuda(cudaGetLastError(), "qNEP real-space charge launch failed");
  }
  zero_mean_d_real_kernel<<<1, 256, 256 * sizeof(float)>>>(atom_count, view.d_real);
  check_cuda(cudaGetLastError(), "qNEP zero-mean D_real launch failed");
  check_cuda(cudaDeviceSynchronize(), "qNEP charge terms failed");
}

}  // namespace nep_adapters::cuda_backend
