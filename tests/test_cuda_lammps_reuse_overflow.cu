#include "nep_adapters/api.h"
#include "nep_adapters/engines/cuda.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    std::cerr << action << ": " << cudaGetErrorString(status) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

template <typename T>
T* copy_to_device(const std::vector<T>& host, const char* action) {
  T* device = nullptr;
  check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(T)),
      action);
  check_cuda(
      cudaMemcpy(
          device,
          host.data(),
          host.size() * sizeof(T),
          cudaMemcpyHostToDevice),
      action);
  return device;
}

std::string write_small_neighbor_model() {
  const std::string model_path =
      (std::filesystem::temp_directory_path() /
       "cuda_lammps_reuse_overflow.nep")
          .string();
  std::ofstream out(model_path);
  out << "nep4 1 C\n"
      << "cutoff 5 1 1 1\n"
      << "n_max 1 0\n"
      << "basis_size 2 0\n"
      << "l_max 0 0 0\n"
      << "ANN 2 0\n";
  const double values[] = {
      0.20, -0.10, 0.05, 0.15,
      0.01, -0.02,
      0.30, -0.25,
      0.04,
      0.70, -0.15, 0.05, -0.30, 0.20, -0.10, 0.0,
      0.80, 1.10,
  };
  for (double value : values) {
    out << value << "\n";
  }
  return model_path;
}

NepaStatus run_case(
    NepaModel* model,
    NepaLammpsDeviceNeighborInput& input,
    NepaLammpsDeviceNeighborResult& result,
    double* device_positions,
    double* device_forces,
    const std::vector<double>& positions) {
  check_cuda(
      cudaMemcpy(
          device_positions,
          positions.data(),
          positions.size() * sizeof(double),
          cudaMemcpyHostToDevice),
      "update positions");
  check_cuda(
      cudaMemset(device_forces, 0, positions.size() * sizeof(double)),
      "clear forces");
  return nepa_find_force_lammps_device_neighbors(model, &input, &result);
}

}  // namespace

int main() {
  if (!nep_adapters::register_cuda_engine()) {
    std::cerr << "failed to register CUDA engine\n";
    return EXIT_FAILURE;
  }

  constexpr int atom_count = 4;
  constexpr int max_neighbors = 3;
  const std::string model_path = write_small_neighbor_model();
  NepaModel* model = nullptr;
  if (nepa_load_model("cuda", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    std::cerr << "failed to load CUDA model\n";
    return EXIT_FAILURE;
  }

  const std::vector<int> ilist = {0, 1, 2, 3};
  const std::vector<int> numneigh(atom_count, max_neighbors);
  const std::vector<int> neighbors = {
      1, 2, 3,
      0, 2, 3,
      0, 1, 3,
      0, 1, 2,
  };
  const std::vector<int> types(atom_count, 0);
  const std::vector<double> sparse_positions = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      2.0, 0.0, 0.0,
      8.0, 0.0, 0.0,
  };
  const std::vector<double> dense_positions = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      2.0, 0.0, 0.0,
      3.0, 0.0, 0.0,
  };

  int* d_ilist = copy_to_device(ilist, "copy ilist");
  int* d_numneigh = copy_to_device(numneigh, "copy numneigh");
  int* d_neighbors = copy_to_device(neighbors, "copy neighbors");
  int* d_types = copy_to_device(types, "copy types");
  double* d_positions = copy_to_device(sparse_positions, "copy positions");
  double* d_forces = nullptr;
  check_cuda(
      cudaMalloc(
          reinterpret_cast<void**>(&d_forces),
          sparse_positions.size() * sizeof(double)),
      "allocate forces");

  NepaLammpsDeviceNeighborInput input{};
  input.nlocal = atom_count;
  input.nall = atom_count;
  input.inum = atom_count;
  input.max_neighbors = max_neighbors;
  input.neighbor_rows = atom_count;
  input.numneigh_length = atom_count;
  input.ilist = d_ilist;
  input.numneigh = d_numneigh;
  input.neighbors = d_neighbors;
  input.neighbor_atom_stride = max_neighbors;
  input.neighbor_slot_stride = 1;
  input.types = d_types;
  input.positions = d_positions;
  input.position_atom_stride = 3;
  input.position_component_stride = 1;

  NepaLammpsDeviceNeighborResult result{};
  result.forces = d_forces;
  result.force_atom_stride = 3;
  result.force_component_stride = 1;

  const NepaStatus sparse_status =
      run_case(
          model,
          input,
          result,
          d_positions,
          d_forces,
          sparse_positions);
  const NepaStatus dense_status =
      run_case(
          model,
          input,
          result,
          d_positions,
          d_forces,
          dense_positions);
  const std::string dense_error =
      nepa_last_error_message() != nullptr ? nepa_last_error_message() : "";
  const NepaStatus recovered_status =
      run_case(
          model,
          input,
          result,
          d_positions,
          d_forces,
          sparse_positions);

  cudaFree(d_ilist);
  cudaFree(d_numneigh);
  cudaFree(d_neighbors);
  cudaFree(d_types);
  cudaFree(d_positions);
  cudaFree(d_forces);
  nepa_free_model(model);
  std::filesystem::remove(model_path);

  if (sparse_status != NEPA_STATUS_OK ||
      dense_status != NEPA_STATUS_INVALID_ARGUMENT ||
      dense_error.find("neighbor count exceeds workspace capacity") ==
          std::string::npos ||
      recovered_status != NEPA_STATUS_OK) {
    std::cerr << "unexpected statuses sparse=" << sparse_status
              << " dense=" << dense_status
              << " recovered=" << recovered_status
              << " dense_error=" << dense_error << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
