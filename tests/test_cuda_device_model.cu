#include "device_operations.hpp"
#include "device_model.hpp"
#include "device_workspace.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool copy_matches(const float* device, const std::vector<float>& expected) {
  std::vector<float> actual(expected.size(), 0.0f);
  const cudaError_t status = cudaMemcpy(
      actual.data(),
      device,
      actual.size() * sizeof(float),
      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    return false;
  }
  return actual == expected;
}

std::vector<int> copy_ints(const int* device, std::size_t count) {
  std::vector<int> actual(count, 0);
  const cudaError_t status = cudaMemcpy(
      actual.data(),
      device,
      actual.size() * sizeof(int),
      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    return {};
  }
  return actual;
}

bool invert_row_major3(const double* matrix, double* inverse) {
  const double det =
      matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
      matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
      matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
  if (std::abs(det) <= 1.0e-12) {
    return false;
  }
  const double inv_det = 1.0 / det;
  inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inv_det;
  inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inv_det;
  inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inv_det;
  inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inv_det;
  inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inv_det;
  inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inv_det;
  inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inv_det;
  inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inv_det;
  inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inv_det;
  return true;
}

nep_adapters::cuda_backend::SimulationBox make_simulation_box(
    const double* cell_row_major9,
    const int* pbc_flags3) {
  nep_adapters::cuda_backend::SimulationBox box;
  for (int component = 0; component < 9; ++component) {
    box.frac_to_cart[component] = cell_row_major9[component];
  }
  if (!invert_row_major3(box.frac_to_cart, box.cart_to_frac)) {
    throw std::runtime_error("singular test box");
  }
  box.pbc[0] = pbc_flags3[0];
  box.pbc[1] = pbc_flags3[1];
  box.pbc[2] = pbc_flags3[2];
  return box;
}

std::vector<double> minimum_image_delta_aos3(
    const std::vector<double>& positions_aos3,
    int center,
    int neighbor,
    const nep_adapters::cuda_backend::SimulationBox& box) {
  const double dx = positions_aos3[3 * static_cast<std::size_t>(neighbor)] -
                    positions_aos3[3 * static_cast<std::size_t>(center)];
  const double dy = positions_aos3[3 * static_cast<std::size_t>(neighbor) + 1] -
                    positions_aos3[3 * static_cast<std::size_t>(center) + 1];
  const double dz = positions_aos3[3 * static_cast<std::size_t>(neighbor) + 2] -
                    positions_aos3[3 * static_cast<std::size_t>(center) + 2];
  double sx = box.cart_to_frac[0] * dx + box.cart_to_frac[1] * dy +
              box.cart_to_frac[2] * dz;
  double sy = box.cart_to_frac[3] * dx + box.cart_to_frac[4] * dy +
              box.cart_to_frac[5] * dz;
  double sz = box.cart_to_frac[6] * dx + box.cart_to_frac[7] * dy +
              box.cart_to_frac[8] * dz;
  if (box.pbc[0]) {
    sx -= std::nearbyint(sx);
  }
  if (box.pbc[1]) {
    sy -= std::nearbyint(sy);
  }
  if (box.pbc[2]) {
    sz -= std::nearbyint(sz);
  }
  return {
      box.frac_to_cart[0] * sx + box.frac_to_cart[1] * sy +
          box.frac_to_cart[2] * sz,
      box.frac_to_cart[3] * sx + box.frac_to_cart[4] * sy +
          box.frac_to_cart[5] * sz,
      box.frac_to_cart[6] * sx + box.frac_to_cart[7] * sy +
          box.frac_to_cart[8] * sz};
}

std::vector<std::vector<int>> brute_force_neighbors(
    const std::vector<double>& positions_aos3,
    double cutoff,
    const nep_adapters::cuda_backend::SimulationBox& box) {
  const int atom_count = static_cast<int>(positions_aos3.size() / 3);
  std::vector<std::vector<int>> neighbors(static_cast<std::size_t>(atom_count));
  const double cutoff_sq = cutoff * cutoff;
  for (int i = 0; i < atom_count; ++i) {
    for (int j = 0; j < atom_count; ++j) {
      if (i == j) {
        continue;
      }
      const std::vector<double> delta =
          minimum_image_delta_aos3(positions_aos3, i, j, box);
      if (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2] <
          cutoff_sq) {
        neighbors[static_cast<std::size_t>(i)].push_back(j);
      }
    }
    std::sort(
        neighbors[static_cast<std::size_t>(i)].begin(),
        neighbors[static_cast<std::size_t>(i)].end());
  }
  return neighbors;
}

bool neighbor_sets_match(
    const std::vector<int>& counts,
    const std::vector<int>& slot_major_neighbors,
    int atom_stride,
    const std::vector<std::vector<int>>& expected) {
  if (counts.size() < expected.size()) {
    return false;
  }
  for (std::size_t atom = 0; atom < expected.size(); ++atom) {
    if (counts[atom] != static_cast<int>(expected[atom].size())) {
      return false;
    }
    std::vector<int> actual;
    actual.reserve(expected[atom].size());
    for (int slot = 0; slot < counts[atom]; ++slot) {
      actual.push_back(
          slot_major_neighbors[atom + static_cast<std::size_t>(atom_stride * slot)]);
    }
    std::sort(actual.begin(), actual.end());
    if (actual != expected[atom]) {
      return false;
    }
  }
  return true;
}

__global__ void parameter_smoke_kernel(
    nep_adapters::cuda_backend::DeviceModelView model,
    float* output) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float value = 0.0f;
    if (model.ann_type_major_count > 0) {
      value += model.ann_type_major[0];
    }
    if (model.descriptor_coefficients_count > 1) {
      value += 2.0f * model.descriptor_coefficients[1];
    }
    if (model.q_scaler_count > 0) {
      value += 3.0f * model.q_scaler[0];
    }
    output[0] = value;
  }
}

__global__ void internal_workspace_smoke_kernel(
    nep_adapters::cuda_backend::DeviceModelView model,
    nep_adapters::cuda_backend::DeviceWorkspaceView workspace,
    float* output) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const int slot_one_neighbor =
        workspace.nl_radial_slot_major[workspace.atom_capacity];
    const float value =
        model.ann_type_major[0] +
        static_cast<float>(workspace.types[1]) +
        static_cast<float>(10 * workspace.atom_to_structure[2]) +
        static_cast<float>(workspace.positions_soa3[1]) +
        static_cast<float>(workspace.boxes_row_major9[0]) +
        static_cast<float>(workspace.pbc_flags3[2]) +
        static_cast<float>(slot_one_neighbor);
    workspace.fp[0] = value;
    output[0] = value;
  }
}

__global__ void external_workspace_smoke_kernel(
    nep_adapters::cuda_backend::DeviceModelView model,
    nep_adapters::cuda_backend::DeviceWorkspaceView workspace,
    float* output) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    const int active_atom = workspace.active_atom_indices[1];
    const int slot_one_neighbor =
        workspace.nl_angular_slot_major[workspace.atom_capacity + active_atom];
    const float value =
        model.q_scaler[0] +
        static_cast<float>(workspace.types[active_atom]) +
        static_cast<float>(workspace.nn_angular[active_atom]) +
        static_cast<float>(slot_one_neighbor) +
        static_cast<float>(workspace.neighbor_source);
    workspace.fp[1] = value;
    output[0] = value;
  }
}

}  // namespace

int main() {
  const std::string model_path =
      (std::filesystem::temp_directory_path() / "cuda_device_model.nep").string();
  {
    std::ofstream out(model_path);
    out << "nep4 1 C\n"
        << "cutoff 5 4 8 6\n"
        << "n_max 0 0\n"
        << "basis_size 0 0\n"
        << "l_max 1 0 0\n"
        << "ANN 2 0\n";
    for (int value = 1; value <= 13; ++value) {
      out << value << "\n";
    }
  }

  const nep_adapters::cuda_backend::HostModelParameters host =
      nep_adapters::cuda_backend::load_host_model_parameters(model_path);
  nep_adapters::cuda_backend::DeviceModel device(host);
  const nep_adapters::cuda_backend::DeviceModelUploadSummary summary =
      device.upload_summary();

  if (summary.ann_type_major_bytes != host.ann_type_major.size() * sizeof(float) ||
      summary.ann_type_major_qscaled_bytes !=
          host.ann_type_major_qscaled.size() * sizeof(float) ||
      summary.descriptor_coefficients_bytes !=
          host.descriptor_coefficients.size() * sizeof(float) ||
      summary.descriptor_coefficients_type_pair_major_bytes !=
          host.descriptor_coefficients_type_pair_major.size() * sizeof(float) ||
      summary.angular_coefficients_center_type_major_bytes !=
          host.angular_coefficients_center_type_major.size() * sizeof(float) ||
      summary.q_scaler_bytes != host.q_scaler.size() * sizeof(float) ||
      summary.cutoff_radial_pair_bytes !=
          host.cutoff_radial_pair.size() * sizeof(float) ||
      summary.cutoff_angular_pair_bytes !=
          host.cutoff_angular_pair.size() * sizeof(float) ||
      summary.zbl_parameters_pair_bytes !=
          host.zbl_parameters_pair.size() * sizeof(float) ||
      summary.spin_baseline_bytes !=
          host.spin_baseline.size() * sizeof(double) ||
      summary.atomic_numbers_bytes != host.atomic_numbers.size() * sizeof(int) ||
      summary.total_bytes !=
          (host.ann_type_major.size() + host.ann_type_major_qscaled.size() +
           host.descriptor_coefficients.size() +
           host.descriptor_coefficients_type_pair_major.size() +
           host.angular_coefficients_center_type_major.size() +
           host.q_scaler.size() + host.cutoff_radial_pair.size() +
           host.cutoff_angular_pair.size() +
           host.zbl_parameters_pair.size()) *
                  sizeof(float) +
              host.spin_baseline.size() * sizeof(double) +
              host.atomic_numbers.size() * sizeof(int)) {
    std::fprintf(stderr, "device model upload summary contract failed\n");
    return EXIT_FAILURE;
  }

  if (!copy_matches(device.view().ann_type_major, host.ann_type_major) ||
      !copy_matches(
          device.view().ann_type_major_qscaled,
          host.ann_type_major_qscaled) ||
      !copy_matches(
          device.view().descriptor_coefficients,
          host.descriptor_coefficients) ||
      !copy_matches(
          device.view().descriptor_coefficients_type_pair_major,
          host.descriptor_coefficients_type_pair_major) ||
      !copy_matches(
          device.view().angular_coefficients_center_type_major,
          host.angular_coefficients_center_type_major) ||
      !copy_matches(device.view().q_scaler, host.q_scaler) ||
      !copy_matches(
          device.view().cutoff_radial_pair, host.cutoff_radial_pair) ||
      !copy_matches(
          device.view().cutoff_angular_pair, host.cutoff_angular_pair) ||
      !copy_matches(
          device.view().zbl_parameters_pair, host.zbl_parameters_pair) ||
      copy_ints(device.view().atomic_numbers, host.atomic_numbers.size()) !=
          host.atomic_numbers) {
    std::fprintf(stderr, "device model upload copy contract failed\n");
    return EXIT_FAILURE;
  }

  float* device_output = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(float)) !=
      cudaSuccess) {
    std::fprintf(stderr, "device output allocation failed\n");
    return EXIT_FAILURE;
  }
  parameter_smoke_kernel<<<1, 32>>>(device.view(), device_output);
  if (cudaGetLastError() != cudaSuccess) {
    std::fprintf(stderr, "parameter smoke kernel launch failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  float smoke_value = 0.0f;
  if (cudaMemcpy(
          &smoke_value,
          device_output,
          sizeof(float),
          cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "parameter smoke copyback failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  if (smoke_value != host.ann_type_major[0] +
                         2.0f * host.descriptor_coefficients[1] +
                         3.0f * host.q_scaler[0]) {
    std::fprintf(stderr, "parameter smoke value mismatch\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }

  const nep_adapters::cuda_backend::WorkspacePlan internal_plan =
      nep_adapters::cuda_backend::make_internal_neighbor_workspace_plan(
          host.protocol, 4, 2);
  nep_adapters::cuda_backend::DeviceWorkspace internal_workspace(internal_plan);
  if (internal_workspace.summary().array_count != internal_plan.arrays.size() ||
      internal_workspace.summary().total_bytes != internal_plan.total_bytes() ||
      internal_workspace.view().boxes_row_major9 == nullptr ||
      internal_workspace.view().active_atom_indices != nullptr) {
    std::fprintf(stderr, "internal workspace contract failed\n");
    return EXIT_FAILURE;
  }
  int batch_counts[] = {2, 2};
  int batch_offsets[] = {0, 2};
  int batch_types[] = {0, 0, 0, 0};
  double batch_positions[] = {
      0.0, 10.0, 20.0,
      1.0, 11.0, 21.0,
      2.0, 12.0, 22.0,
      3.0, 13.0, 23.0,
  };
  double batch_boxes[] = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
      2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0};
  int batch_pbc[] = {1, 1, 1, 0, 0, 0};
  NepaStructureBatch batch{};
  batch.num_structures = 2;
  batch.total_atoms = 4;
  batch.atom_counts = batch_counts;
  batch.atom_offsets = batch_offsets;
  batch.types = batch_types;
  batch.positions_aos3 = batch_positions;
  batch.boxes_row_major9 = batch_boxes;
  batch.pbc_flags3 = batch_pbc;
  nep_adapters::cuda_backend::stage_batch_on_device(
      batch, host.protocol.num_types, internal_workspace);
  const nep_adapters::cuda_backend::DeviceWorkspaceView internal_view =
      internal_workspace.view();
  const int internal_radial_counts[] = {2, 1, 0, 0};
  const int internal_radial_neighbors[] = {
      1, 0, 0, 0, 2, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0};
  if (cudaMemcpy(
          internal_view.nn_radial,
          internal_radial_counts,
          sizeof(internal_radial_counts),
          cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(
          internal_view.nl_radial_slot_major,
          internal_radial_neighbors,
          sizeof(internal_radial_neighbors),
          cudaMemcpyHostToDevice) != cudaSuccess) {
    std::fprintf(stderr, "internal workspace smoke setup failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }

  internal_workspace_smoke_kernel<<<1, 32>>>(
      device.view(),
      internal_workspace.view(),
      device_output);
  if (cudaGetLastError() != cudaSuccess) {
    std::fprintf(stderr, "internal workspace smoke kernel launch failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  if (cudaMemcpy(
          &smoke_value,
          device_output,
          sizeof(float),
          cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "internal workspace smoke copyback failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  const float expected_internal_value =
      host.ann_type_major[0] + 0.0f + 10.0f + 1.0f + 1.0f + 1.0f + 2.0f;
  if (smoke_value != expected_internal_value) {
    std::fprintf(stderr, "internal workspace smoke value mismatch\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }

  const nep_adapters::cuda_backend::WorkspacePlan external_plan =
      nep_adapters::cuda_backend::make_external_neighbor_workspace_plan(
          host.protocol, 4, 2);
  nep_adapters::cuda_backend::DeviceWorkspace external_workspace(external_plan);
  if (external_workspace.summary().array_count != external_plan.arrays.size() ||
      external_workspace.summary().total_bytes != external_plan.total_bytes() ||
      external_workspace.view().boxes_row_major9 != nullptr ||
      external_workspace.view().active_atom_indices == nullptr ||
      external_workspace.view().neighbor_source != 1) {
    std::fprintf(stderr, "external workspace contract failed\n");
    return EXIT_FAILURE;
  }
  int lmp_ilist[] = {0, 2};
  int lmp_numneigh[] = {1, 0, 2, 0};
  int lmp_neigh0[] = {1};
  int lmp_neigh2[] = {3, 1};
  int* lmp_firstneigh[] = {lmp_neigh0, nullptr, lmp_neigh2, nullptr};
  int lmp_types[] = {1, 2, 1, 2};
  int lmp_type_map[] = {-1, 0, 1};
  double lmp_x0[] = {0.0, 10.0, 20.0};
  double lmp_x1[] = {1.0, 11.0, 21.0};
  double lmp_x2[] = {2.0, 12.0, 22.0};
  double lmp_x3[] = {3.0, 13.0, 23.0};
  double* lmp_positions[] = {lmp_x0, lmp_x1, lmp_x2, lmp_x3};
  NepaLammpsNeighborInput lmp_input{};
  lmp_input.nlocal = 4;
  lmp_input.inum = 2;
  lmp_input.ilist = lmp_ilist;
  lmp_input.numneigh = lmp_numneigh;
  lmp_input.firstneigh = lmp_firstneigh;
  lmp_input.types = lmp_types;
  lmp_input.type_map = lmp_type_map;
  lmp_input.positions = lmp_positions;
  nep_adapters::cuda_backend::stage_lammps_external_neighbors_on_device(
      lmp_input,
      host.protocol,
      external_workspace);
  external_workspace_smoke_kernel<<<1, 32>>>(
      device.view(),
      external_workspace.view(),
      device_output);
  if (cudaGetLastError() != cudaSuccess) {
    std::fprintf(stderr, "external workspace smoke kernel launch failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  if (cudaMemcpy(
          &smoke_value,
          device_output,
          sizeof(float),
          cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::fprintf(stderr, "external workspace smoke copyback failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  const float expected_external_value = host.q_scaler[0] + 0.0f + 2.0f + 1.0f + 1.0f;
  if (smoke_value != expected_external_value) {
    std::fprintf(stderr, "external workspace smoke value mismatch\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }

  const std::string neighbor_model_path =
      (std::filesystem::temp_directory_path() / "cuda_neighbor_builder.nep").string();
  {
    std::ofstream out(neighbor_model_path);
    out << "nep4 1 C\n"
        << "cutoff 5 4 8 6\n"
        << "n_max 1 1\n"
        << "basis_size 2 2\n"
        << "l_max 4 1 1 1 1 1 1\n"
        << "ANN 1 0\n";
    for (int value = 1; value <= 59; ++value) {
      out << value << "\n";
    }
  }
  const nep_adapters::cuda_backend::ModelProtocol neighbor_protocol =
      nep_adapters::cuda_backend::parse_model_protocol(neighbor_model_path);
  const nep_adapters::cuda_backend::WorkspacePlan neighbor_plan =
      nep_adapters::cuda_backend::make_internal_neighbor_workspace_plan(
          neighbor_protocol, 8, 1);
  nep_adapters::cuda_backend::DeviceWorkspace neighbor_workspace(neighbor_plan);
  int neighbor_batch_count[] = {4};
  int neighbor_batch_offset[] = {0};
  int neighbor_types[] = {0, 0, 0, 0};
  double neighbor_positions[] = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      6.0, 0.0, 0.0,
      11.0, 0.0, 0.0,
  };
  double neighbor_box[] = {
      12.0, 0.0, 0.0,
      0.0, 12.0, 0.0,
      0.0, 0.0, 12.0,
  };
  int neighbor_pbc[] = {1, 1, 1};
  NepaStructureBatch neighbor_batch{};
  neighbor_batch.num_structures = 1;
  neighbor_batch.total_atoms = 4;
  neighbor_batch.atom_counts = neighbor_batch_count;
  neighbor_batch.atom_offsets = neighbor_batch_offset;
  neighbor_batch.types = neighbor_types;
  neighbor_batch.positions_aos3 = neighbor_positions;
  neighbor_batch.boxes_row_major9 = neighbor_box;
  neighbor_batch.pbc_flags3 = neighbor_pbc;
  nep_adapters::cuda_backend::stage_batch_on_device(
      neighbor_batch,
      neighbor_protocol.num_types,
      neighbor_workspace);
  const nep_adapters::cuda_backend::SimulationBox neighbor_simulation_box =
      make_simulation_box(neighbor_box, neighbor_pbc);
  nep_adapters::cuda_backend::build_internal_neighbors_on_device(
      neighbor_protocol,
      4,
      neighbor_simulation_box,
      neighbor_workspace);

  const nep_adapters::cuda_backend::DeviceWorkspaceView neighbor_view =
      neighbor_workspace.view();
  const std::vector<int> cell_dims = copy_ints(neighbor_view.cell_dims, 4);
  const std::vector<int> radial_counts = copy_ints(neighbor_view.nn_radial, 4);
  const std::vector<int> angular_counts = copy_ints(neighbor_view.nn_angular, 4);
  const std::vector<int> radial_neighbors = copy_ints(
      neighbor_view.nl_radial_slot_major,
      8 * static_cast<std::size_t>(neighbor_protocol.neighbor_capacity_radial));
  const std::vector<int> angular_neighbors = copy_ints(
      neighbor_view.nl_angular_slot_major,
      8 * static_cast<std::size_t>(neighbor_protocol.neighbor_capacity_angular));
  const std::vector<double> neighbor_positions_vec(
      neighbor_positions,
      neighbor_positions + 12);
  const std::vector<std::vector<int>> expected_radial = brute_force_neighbors(
      neighbor_positions_vec,
      neighbor_protocol.cutoff_radial,
      neighbor_simulation_box);
  const std::vector<std::vector<int>> expected_angular = brute_force_neighbors(
      neighbor_positions_vec,
      neighbor_protocol.cutoff_angular,
      neighbor_simulation_box);
  if (cell_dims.size() != 4 || cell_dims[0] != 2 || cell_dims[1] != 2 ||
      cell_dims[2] != 2 || cell_dims[3] != 8 ||
      !neighbor_sets_match(radial_counts, radial_neighbors, 8, expected_radial) ||
      !neighbor_sets_match(angular_counts, angular_neighbors, 8, expected_angular)) {
    std::fprintf(stderr, "orthorhombic neighbor contract failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  const double triclinic_box[] = {
      12.0, 2.0, 1.0,
      0.0, 11.0, 1.5,
      0.0, 0.0, 10.0,
  };
  const double triclinic_positions[] = {
      0.2, 0.2, 0.2,
      11.4, 1.9, 1.0,
      2.4, 0.5, 0.3,
      7.0, 7.0, 6.0,
  };
  const nep_adapters::cuda_backend::SimulationBox triclinic_simulation_box =
      make_simulation_box(triclinic_box, neighbor_pbc);
  neighbor_batch.positions_aos3 = triclinic_positions;
  neighbor_batch.boxes_row_major9 = triclinic_box;
  nep_adapters::cuda_backend::stage_batch_on_device(
      neighbor_batch,
      neighbor_protocol.num_types,
      neighbor_workspace);
  nep_adapters::cuda_backend::build_internal_neighbors_on_device(
      neighbor_protocol,
      4,
      triclinic_simulation_box,
      neighbor_workspace);
  const std::vector<int> triclinic_radial_counts =
      copy_ints(neighbor_view.nn_radial, 4);
  const std::vector<int> triclinic_angular_counts =
      copy_ints(neighbor_view.nn_angular, 4);
  const std::vector<int> triclinic_radial_neighbors = copy_ints(
      neighbor_view.nl_radial_slot_major,
      8 * static_cast<std::size_t>(neighbor_protocol.neighbor_capacity_radial));
  const std::vector<int> triclinic_angular_neighbors = copy_ints(
      neighbor_view.nl_angular_slot_major,
      8 * static_cast<std::size_t>(neighbor_protocol.neighbor_capacity_angular));
  const std::vector<double> triclinic_positions_vec(
      triclinic_positions,
      triclinic_positions + 12);
  const std::vector<std::vector<int>> expected_triclinic_radial =
      brute_force_neighbors(
          triclinic_positions_vec,
          neighbor_protocol.cutoff_radial,
          triclinic_simulation_box);
  const std::vector<std::vector<int>> expected_triclinic_angular =
      brute_force_neighbors(
          triclinic_positions_vec,
          neighbor_protocol.cutoff_angular,
          triclinic_simulation_box);
  if (!neighbor_sets_match(
          triclinic_radial_counts,
          triclinic_radial_neighbors,
          8,
          expected_triclinic_radial) ||
      !neighbor_sets_match(
          triclinic_angular_counts,
          triclinic_angular_neighbors,
          8,
          expected_triclinic_angular)) {
    std::fprintf(stderr, "triclinic neighbor contract failed\n");
    cudaFree(device_output);
    return EXIT_FAILURE;
  }
  cudaFree(device_output);

  nep_adapters::cuda_backend::DeviceModel moved(std::move(device));
  if (moved.view().ann_type_major == nullptr ||
      moved.view().descriptor_coefficients == nullptr ||
      moved.view().q_scaler == nullptr) {
    std::fprintf(stderr, "moved device model lost pointers\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
