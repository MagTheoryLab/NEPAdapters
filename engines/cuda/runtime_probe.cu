#include "nep_adapters/engines/cuda.hpp"

#include <cuda_runtime.h>

#include <string>

namespace {

constexpr int kProbeExpected = 0x4e455041;

__global__ void runtime_probe_kernel(int* result) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *result = kProbeExpected;
  }
}

std::string cuda_failure_detail(const char* action, cudaError_t status) {
  return std::string(action) + ": " + cudaGetErrorString(status);
}

}  // namespace

namespace nep_adapters {

bool probe_cuda_device(int device_index, std::string& detail) {
  detail.clear();
  cudaGetLastError();

  cudaError_t status = cudaSetDevice(device_index);
  if (status != cudaSuccess) {
    detail = cuda_failure_detail("cudaSetDevice failed", status);
    cudaGetLastError();
    return false;
  }

  int* device_result = nullptr;
  status = cudaMalloc(
      reinterpret_cast<void**>(&device_result),
      sizeof(*device_result));
  if (status != cudaSuccess) {
    detail = cuda_failure_detail("CUDA probe allocation failed", status);
    cudaGetLastError();
    return false;
  }

  int host_result = 0;
  runtime_probe_kernel<<<1, 1>>>(device_result);
  status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaDeviceSynchronize();
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(
        &host_result,
        device_result,
        sizeof(host_result),
        cudaMemcpyDeviceToHost);
  }

  const cudaError_t free_status = cudaFree(device_result);
  if (status != cudaSuccess) {
    detail = cuda_failure_detail("CUDA probe kernel failed", status);
    cudaGetLastError();
    return false;
  }
  if (free_status != cudaSuccess) {
    detail = cuda_failure_detail("CUDA probe cleanup failed", free_status);
    cudaGetLastError();
    return false;
  }
  if (host_result != kProbeExpected) {
    detail = "CUDA probe kernel returned an invalid result";
    return false;
  }

  detail = "CUDA allocation, kernel launch, synchronization, and copyback passed.";
  return true;
}

}  // namespace nep_adapters
