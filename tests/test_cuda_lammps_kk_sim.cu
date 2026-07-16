#include "nep_adapters/api.h"
#include "nep_adapters/engines/cuda.hpp"
#include "device_workspace.hpp"

#include <cuda_runtime.h>

#include <cmath>
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

std::string write_radial_model() {
  const std::string model_path =
      (std::filesystem::temp_directory_path() /
       "cuda_lammps_kk_radial.nep")
          .string();
  std::ofstream out(model_path);
  out << "nep4 1 C\n"
      << "cutoff 5 1 8 1\n"
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

std::string write_small_mn_radial_model() {
  const std::string model_path =
      (std::filesystem::temp_directory_path() /
       "cuda_lammps_kk_small_mn.nep")
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

std::string write_zbl_model() {
  const std::string model_path =
      (std::filesystem::temp_directory_path() /
       "cuda_lammps_kk_zbl.nep")
          .string();
  std::ofstream out(model_path);
  out << "nep4_zbl 1 C\n"
      << "zbl 0.5 2.0\n"
      << "cutoff 5 1 8 1\n"
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

bool check_external_workspace_plan() {
  const nep_adapters::cuda_backend::ModelProtocol protocol =
      nep_adapters::cuda_backend::parse_model_protocol(write_radial_model());
  const nep_adapters::cuda_backend::WorkspacePlan workspace =
      nep_adapters::cuda_backend::make_external_neighbor_workspace_plan(
          protocol, 8, 5);

  const nep_adapters::cuda_backend::DeviceArrayPlan* active =
      workspace.find_array("active_atom_indices");
  const nep_adapters::cuda_backend::DeviceArrayPlan* positions =
      workspace.find_array("positions_soa3");
  const nep_adapters::cuda_backend::DeviceArrayPlan* nl_radial =
      workspace.find_array("nl_radial_slot_major");
  const nep_adapters::cuda_backend::DeviceArrayPlan* nl_angular =
      workspace.find_array("nl_angular_slot_major");
  const nep_adapters::cuda_backend::DeviceArrayPlan* forces =
      workspace.find_array("force_soa3");
  const nep_adapters::cuda_backend::DeviceArrayPlan* virials =
      workspace.find_array("virial_soa9");

  return workspace.neighbor_source ==
             nep_adapters::cuda_backend::NeighborSource::external &&
         workspace.active_atom_capacity == 5 &&
         workspace.find_array("boxes_row_major9") == nullptr &&
         active != nullptr && active->element_count == 5 &&
         positions != nullptr && positions->element_count == 24 &&
         forces != nullptr && forces->element_count == 24 &&
         virials != nullptr && virials->element_count == 72 &&
         nl_radial != nullptr &&
         nl_radial->element_count ==
             8 * static_cast<std::size_t>(protocol.neighbor_capacity_radial) &&
         nl_angular != nullptr &&
         nl_angular->element_count ==
             8 * static_cast<std::size_t>(protocol.neighbor_capacity_angular) &&
         workspace.total_bytes() > 0;
}

int lammps_raw9_to_nep_row_major9(int component) {
  const int map[9] = {0, 4, 8, 1, 2, 5, 3, 6, 7};
  return map[component];
}

bool run_device_layout_case(
    NepaModel* model,
    const char* case_name,
    bool soa_layout,
    bool use_type_map,
    double expected_energy,
    const double expected_forces[6],
    const double expected_potential_per_atom[2],
    const double expected_virials_per_atom_row_major9[18],
    double force_tolerance = 1.0e-8,
    double virial_tolerance = 1.0e-8) {
  constexpr int nlocal = 2;
  constexpr int nall = 2;
  constexpr int inum = 2;
  constexpr int max_neighbors = 1;
  constexpr int pitch = 4;
  constexpr int force_pitch = 5;
  constexpr int virial_pitch = 11;

  std::vector<int> ilist = {0, 1};
  std::vector<int> numneigh = {1, 1};
  std::vector<int> neighbors(
      soa_layout ? pitch * max_neighbors : nall * max_neighbors,
      -1);
  const int neighbor_atom_stride = soa_layout ? 1 : max_neighbors;
  const int neighbor_slot_stride = soa_layout ? pitch : 1;
  neighbors[0 * neighbor_atom_stride] = 1;
  neighbors[1 * neighbor_atom_stride] = 0;

  std::vector<int> types = use_type_map ? std::vector<int>{1, 1}
                                        : std::vector<int>{0, 0};
  std::vector<int> type_map = {-1, 0};
  const int position_atom_stride = soa_layout ? 1 : 3;
  const int position_component_stride = soa_layout ? pitch : 1;
  std::vector<double> positions(soa_layout ? 3 * pitch : nall * 3, 999.0);
  positions[0 * position_atom_stride + 0 * position_component_stride] = 0.0;
  positions[0 * position_atom_stride + 1 * position_component_stride] = 0.0;
  positions[0 * position_atom_stride + 2 * position_component_stride] = 0.0;
  positions[1 * position_atom_stride + 0 * position_component_stride] = 1.5;
  positions[1 * position_atom_stride + 1 * position_component_stride] = 0.0;
  positions[1 * position_atom_stride + 2 * position_component_stride] = 0.0;

  int* d_ilist = copy_to_device(ilist, "copy ilist");
  int* d_numneigh = copy_to_device(numneigh, "copy numneigh");
  int* d_neighbors = copy_to_device(neighbors, "copy neighbors");
  int* d_types = copy_to_device(types, "copy types");
  int* d_type_map = use_type_map ?
      copy_to_device(type_map, "copy type map") :
      nullptr;
  double* d_positions = copy_to_device(positions, "copy positions");

  const int force_atom_stride = soa_layout ? 1 : 3;
  const int force_component_stride = soa_layout ? force_pitch : 1;
  const int virial_atom_stride = soa_layout ? 1 : 9;
  const int virial_component_stride = soa_layout ? virial_pitch : 1;
  const int force_count = soa_layout ? 3 * force_pitch : nlocal * 3;
  const int virial_count = soa_layout ? 9 * virial_pitch : nlocal * 9;

  double* d_total_potential = nullptr;
  double* d_total_virial6 = nullptr;
  double* d_potential_per_atom = nullptr;
  double* d_forces = nullptr;
  double* d_virials = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_total_potential), sizeof(double)),
             "allocate total potential");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_total_virial6), 6 * sizeof(double)),
             "allocate total virial");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_potential_per_atom),
                        nlocal * sizeof(double)),
             "allocate per-atom potential");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_forces),
                        force_count * sizeof(double)),
             "allocate forces");
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_virials),
                        virial_count * sizeof(double)),
             "allocate virials");
  check_cuda(cudaMemset(d_forces, 0, force_count * sizeof(double)),
             "clear forces");
  check_cuda(cudaMemset(d_virials, 0, virial_count * sizeof(double)),
             "clear virials");

  NepaLammpsDeviceNeighborInput device_input{};
  device_input.nlocal = nlocal;
  device_input.nall = nall;
  device_input.inum = inum;
  device_input.max_neighbors = max_neighbors;
  device_input.neighbor_rows = nall;
  device_input.numneigh_length = nall;
  device_input.ilist = d_ilist;
  device_input.numneigh = d_numneigh;
  device_input.neighbors = d_neighbors;
  device_input.neighbor_owner = nullptr;
  device_input.neighbor_atom_stride = neighbor_atom_stride;
  device_input.neighbor_slot_stride = neighbor_slot_stride;
  device_input.types = d_types;
  device_input.type_map = d_type_map;
  device_input.type_map_length = use_type_map ?
      static_cast<int>(type_map.size()) :
      0;
  device_input.positions = d_positions;
  device_input.position_atom_stride = position_atom_stride;
  device_input.position_component_stride = position_component_stride;

  NepaLammpsDeviceNeighborResult device_result{};
  device_result.total_potential = d_total_potential;
  device_result.total_virial6 = d_total_virial6;
  device_result.potential_per_atom = d_potential_per_atom;
  device_result.forces = d_forces;
  device_result.force_atom_stride = force_atom_stride;
  device_result.force_component_stride = force_component_stride;
  device_result.virials_per_atom9 = d_virials;
  device_result.virial_atom_stride = virial_atom_stride;
  device_result.virial_component_stride = virial_component_stride;

  const NepaStatus device_status =
      nepa_find_force_lammps_device_neighbors(
          model,
          &device_input,
          &device_result);
  if (device_status != NEPA_STATUS_OK) {
    std::cerr << case_name << " device LAMMPS status=" << device_status << "\n";
    if (nepa_last_error_message() != nullptr) {
      std::cerr << case_name << " error=" << nepa_last_error_message() << "\n";
    }
    return false;
  }

  double total_potential = 0.0;
  std::vector<double> potential_per_atom(nlocal, 0.0);
  std::vector<double> forces(force_count, 0.0);
  std::vector<double> virials(virial_count, 0.0);
  check_cuda(
      cudaMemcpy(
          &total_potential,
          d_total_potential,
          sizeof(double),
          cudaMemcpyDeviceToHost),
      "copy total potential");
  check_cuda(
      cudaMemcpy(
          potential_per_atom.data(),
          d_potential_per_atom,
          potential_per_atom.size() * sizeof(double),
          cudaMemcpyDeviceToHost),
      "copy per-atom potential");
  check_cuda(
      cudaMemcpy(
          forces.data(),
          d_forces,
          forces.size() * sizeof(double),
          cudaMemcpyDeviceToHost),
      "copy forces");
  check_cuda(
      cudaMemcpy(
          virials.data(),
          d_virials,
          virials.size() * sizeof(double),
          cudaMemcpyDeviceToHost),
      "copy per-atom virials");

  if (std::abs(total_potential - expected_energy) > 1.0e-10) {
    std::cerr << case_name << " energy mismatch device=" << total_potential
              << " expected=" << expected_energy << "\n";
    return false;
  }
  for (int atom = 0; atom < nlocal; ++atom) {
    const double potential_diff =
        std::abs(potential_per_atom[atom] - expected_potential_per_atom[atom]);
    if (potential_diff > 1.0e-10) {
      std::cerr << case_name << " per-atom potential mismatch atom=" << atom
                << " device=" << potential_per_atom[atom]
                << " expected=" << expected_potential_per_atom[atom]
                << " diff=" << potential_diff << "\n";
      return false;
    }
    for (int component = 0; component < 3; ++component) {
      const double device_force =
          forces[atom * force_atom_stride + component * force_component_stride];
      const double expected_force = expected_forces[3 * atom + component];
      const double diff = std::abs(device_force - expected_force);
      if (diff > force_tolerance) {
        std::cerr << case_name << " force mismatch atom=" << atom
                  << " component=" << component
                  << " device=" << device_force
                  << " expected=" << expected_force
                  << " diff=" << diff << "\n";
        return false;
      }
    }
    for (int component = 0; component < 9; ++component) {
      const double device_virial =
          virials[
              atom * virial_atom_stride +
              component * virial_component_stride];
      const int expected_component =
          lammps_raw9_to_nep_row_major9(component);
      const double expected_virial =
          expected_virials_per_atom_row_major9[
              9 * atom + expected_component];
      const double diff = std::abs(device_virial - expected_virial);
      if (diff > virial_tolerance) {
        std::cerr << case_name << " per-atom virial mismatch atom=" << atom
                  << " component=" << component
                  << " device=" << device_virial
                  << " expected=" << expected_virial
                  << " diff=" << diff << "\n";
        return false;
      }
    }
  }

  const NepaStatus repeat_status =
      nepa_find_force_lammps_device_neighbors(
          model,
          &device_input,
          &device_result);
  if (repeat_status != NEPA_STATUS_OK) {
    std::cerr << case_name << " repeat status=" << repeat_status << "\n";
    return false;
  }
  check_cuda(
      cudaMemcpy(
          forces.data(),
          d_forces,
          forces.size() * sizeof(double),
          cudaMemcpyDeviceToHost),
      "copy repeat forces");
  for (int atom = 0; atom < nlocal; ++atom) {
    for (int component = 0; component < 3; ++component) {
      const double device_force =
          forces[atom * force_atom_stride + component * force_component_stride];
      const double expected_force = expected_forces[3 * atom + component];
      const double diff = std::abs(device_force - expected_force);
      if (diff > force_tolerance) {
        std::cerr << case_name << " repeat force mismatch atom=" << atom
                  << " component=" << component
                  << " device=" << device_force
                  << " expected=" << expected_force
                  << " diff=" << diff << "\n";
        return false;
      }
    }
  }

  device_result.total_potential = nullptr;
  device_result.total_virial6 = nullptr;
  device_result.potential_per_atom = nullptr;
  device_result.virials_per_atom9 = nullptr;
  check_cuda(cudaMemset(d_forces, 0, force_count * sizeof(double)),
             "clear force-only forces");
  const NepaStatus force_only_status =
      nepa_find_force_lammps_device_neighbors(
          model,
          &device_input,
          &device_result);
  if (force_only_status != NEPA_STATUS_OK) {
    std::cerr << case_name << " force-only status="
              << force_only_status << "\n";
    return false;
  }
  check_cuda(
      cudaMemcpy(
          forces.data(),
          d_forces,
          forces.size() * sizeof(double),
          cudaMemcpyDeviceToHost),
      "copy force-only forces");
  for (int atom = 0; atom < nlocal; ++atom) {
    for (int component = 0; component < 3; ++component) {
      const double device_force =
          forces[atom * force_atom_stride + component * force_component_stride];
      const double expected_force = expected_forces[3 * atom + component];
      const double diff = std::abs(device_force - expected_force);
      if (diff > force_tolerance) {
        std::cerr << case_name << " force-only mismatch atom=" << atom
                  << " component=" << component
                  << " device=" << device_force
                  << " expected=" << expected_force
                  << " diff=" << diff << "\n";
        return false;
      }
    }
  }

  device_result.total_potential = d_total_potential;
  device_result.total_virial6 = d_total_virial6;
  device_result.potential_per_atom = d_potential_per_atom;
  device_result.virials_per_atom9 = d_virials;
  device_result.forces = nullptr;
  const NepaStatus invalid_status =
      nepa_find_force_lammps_device_neighbors(
          model,
          &device_input,
          &device_result);
  if (invalid_status != NEPA_STATUS_INVALID_ARGUMENT) {
    std::cerr << case_name << " invalid argument status="
              << invalid_status << "\n";
    return false;
  }

  cudaFree(d_ilist);
  cudaFree(d_numneigh);
  cudaFree(d_neighbors);
  cudaFree(d_types);
  cudaFree(d_type_map);
  cudaFree(d_positions);
  cudaFree(d_total_potential);
  cudaFree(d_total_virial6);
  cudaFree(d_potential_per_atom);
  cudaFree(d_forces);
  cudaFree(d_virials);
  return true;
}

bool run_zbl_force_only_case() {
  const std::string model_path = write_zbl_model();
  NepaModel* model = nullptr;
  const NepaStatus load_status =
      nepa_load_model("cuda", model_path.c_str(), &model);
  if (load_status != NEPA_STATUS_OK || model == nullptr) {
    std::cerr << "failed to load ZBL model status=" << load_status
              << " path=" << model_path << "\n";
    return false;
  }

  int atom_counts[] = {2};
  int atom_offsets[] = {0};
  int batch_types[] = {0, 0};
  double positions_aos[] = {
      0.0, 0.0, 0.0,
      1.5, 0.0, 0.0,
  };
  double box[] = {
      8.0, 0.0, 0.0,
      0.0, 8.0, 0.0,
      0.0, 0.0, 8.0,
  };
  int pbc[] = {1, 1, 1};
  double batch_energy[] = {0.0};
  double batch_potential[2] = {};
  double batch_forces[6] = {};
  double batch_virials_per_atom[18] = {};
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = 2;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = batch_types;
  batch.positions_aos3 = positions_aos;
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;
  NepaFindForceResult batch_result{};
  batch_result.energy_per_structure = batch_energy;
  batch_result.potential_per_atom = batch_potential;
  batch_result.forces_aos3 = batch_forces;
  batch_result.virials_per_atom_row_major9 = batch_virials_per_atom;
  const NepaStatus batch_status =
      nepa_find_force_batch(model, &batch, &batch_result);
  if (batch_status != NEPA_STATUS_OK) {
    std::cerr << "ZBL batch status=" << batch_status << "\n";
    nepa_free_model(model);
    return false;
  }

  const bool ok = run_device_layout_case(
      model,
      "zbl_force_only",
      true,
      false,
      batch_energy[0],
      batch_forces,
      batch_potential,
      batch_virials_per_atom,
      1.0e-6,
      1.0e-6);
  nepa_free_model(model);
  return ok;
}

bool run_compact_device_neighbor_capacity_case() {
  const std::string model_path = write_small_mn_radial_model();
  NepaModel* model = nullptr;
  const NepaStatus load_status =
      nepa_load_model("cuda", model_path.c_str(), &model);
  if (load_status != NEPA_STATUS_OK || model == nullptr) {
    std::cerr << "failed to load small-MN model status=" << load_status
              << " path=" << model_path << "\n";
    return false;
  }

  constexpr int nlocal = 4;
  constexpr int nall = 4;
  constexpr int inum = 4;
  constexpr int max_neighbors = 3;
  std::vector<int> ilist = {0, 1, 2, 3};
  std::vector<int> numneigh = {3, 3, 3, 3};
  std::vector<int> neighbors = {
      1, 2, 3,
      0, 2, 3,
      0, 1, 3,
      0, 1, 2,
  };
  std::vector<int> types = {0, 0, 0, 0};
  std::vector<double> positions = {
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      2.0, 0.0, 0.0,
      8.0, 0.0, 0.0,
  };

  int* d_ilist = copy_to_device(ilist, "copy compact ilist");
  int* d_numneigh = copy_to_device(numneigh, "copy compact numneigh");
  int* d_neighbors = copy_to_device(neighbors, "copy compact neighbors");
  int* d_types = copy_to_device(types, "copy compact types");
  double* d_positions = copy_to_device(positions, "copy compact positions");
  double* d_forces = nullptr;
  check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&d_forces), nlocal * 3 * sizeof(double)),
      "allocate compact forces");
  check_cuda(cudaMemset(d_forces, 0, nlocal * 3 * sizeof(double)),
             "clear compact forces");

  NepaLammpsDeviceNeighborInput device_input{};
  device_input.nlocal = nlocal;
  device_input.nall = nall;
  device_input.inum = inum;
  device_input.max_neighbors = max_neighbors;
  device_input.neighbor_rows = nall;
  device_input.numneigh_length = nall;
  device_input.ilist = d_ilist;
  device_input.numneigh = d_numneigh;
  device_input.neighbors = d_neighbors;
  device_input.neighbor_atom_stride = max_neighbors;
  device_input.neighbor_slot_stride = 1;
  device_input.types = d_types;
  device_input.positions = d_positions;
  device_input.position_atom_stride = 3;
  device_input.position_component_stride = 1;

  NepaLammpsDeviceNeighborResult device_result{};
  device_result.forces = d_forces;
  device_result.force_atom_stride = 3;
  device_result.force_component_stride = 1;

  const NepaStatus status =
      nepa_find_force_lammps_device_neighbors(
          model,
          &device_input,
          &device_result);
  if (status != NEPA_STATUS_OK) {
    std::cerr << "compact device neighbor status=" << status
              << " error=" << nepa_last_error_message() << "\n";
    return false;
  }

  std::vector<double> forces(nlocal * 3, 0.0);
  check_cuda(
      cudaMemcpy(
          forces.data(),
          d_forces,
          forces.size() * sizeof(double),
          cudaMemcpyDeviceToHost),
      "copy compact forces");
  for (double value : forces) {
    if (!std::isfinite(value)) {
      std::cerr << "compact device neighbor force is not finite\n";
      return false;
    }
  }

  cudaFree(d_ilist);
  cudaFree(d_numneigh);
  cudaFree(d_neighbors);
  cudaFree(d_types);
  cudaFree(d_positions);
  cudaFree(d_forces);
  nepa_free_model(model);
  return true;
}

bool run_ghost_per_atom_virial_case(NepaModel* model) {
  constexpr int nlocal = 1;
  constexpr int nall = 2;
  std::vector<int> ilist = {0};
  std::vector<int> numneigh = {1, 0};
  std::vector<int> neighbors = {1, -1};
  std::vector<int> types = {0, 0};
  std::vector<double> positions = {
      0.0, 0.0, 0.0,
      1.5, 0.0, 0.0,
  };
  std::vector<double> virial_sentinel(nall * 9, 123.0);

  int* d_ilist = copy_to_device(ilist, "copy ghost ilist");
  int* d_numneigh = copy_to_device(numneigh, "copy ghost numneigh");
  int* d_neighbors = copy_to_device(neighbors, "copy ghost neighbors");
  int* d_types = copy_to_device(types, "copy ghost types");
  double* d_positions = copy_to_device(positions, "copy ghost positions");
  double* d_virials =
      copy_to_device(virial_sentinel, "copy ghost virial sentinel");
  double* d_forces = nullptr;
  double* d_total_potential = nullptr;
  double* d_total_virial6 = nullptr;
  check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&d_forces), nall * 3 * sizeof(double)),
      "allocate ghost forces");
  check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&d_total_potential), sizeof(double)),
      "allocate ghost total potential");
  check_cuda(
      cudaMalloc(reinterpret_cast<void**>(&d_total_virial6), 6 * sizeof(double)),
      "allocate ghost total virial");

  NepaLammpsDeviceNeighborInput input{};
  input.nlocal = nlocal;
  input.nall = nall;
  input.inum = nlocal;
  input.max_neighbors = 1;
  input.neighbor_rows = nall;
  input.numneigh_length = nall;
  input.ilist = d_ilist;
  input.numneigh = d_numneigh;
  input.neighbors = d_neighbors;
  input.neighbor_atom_stride = 1;
  input.neighbor_slot_stride = 1;
  input.types = d_types;
  input.positions = d_positions;
  input.position_atom_stride = 3;
  input.position_component_stride = 1;

  NepaLammpsDeviceNeighborResult result{};
  result.total_potential = d_total_potential;
  result.total_virial6 = d_total_virial6;
  result.forces = d_forces;
  result.force_atom_stride = 3;
  result.force_component_stride = 1;
  result.virials_per_atom9 = d_virials;
  result.virial_atom_stride = 9;
  result.virial_component_stride = 1;

  const NepaStatus status =
      nepa_find_force_lammps_device_neighbors(model, &input, &result);
  bool ok = status == NEPA_STATUS_OK;
  if (!ok) {
    std::cerr << "ghost per-atom virial status=" << status
              << " error=" << nepa_last_error_message() << "\n";
  }

  std::vector<double> virials(nall * 9, 0.0);
  std::vector<double> total_virial6(6, 0.0);
  if (ok) {
    check_cuda(
        cudaMemcpy(
            virials.data(),
            d_virials,
            virials.size() * sizeof(double),
            cudaMemcpyDeviceToHost),
        "copy ghost per-atom virial");
    check_cuda(
        cudaMemcpy(
            total_virial6.data(),
            d_total_virial6,
            total_virial6.size() * sizeof(double),
            cudaMemcpyDeviceToHost),
        "copy ghost total virial");
    bool ghost_nonzero = false;
    for (int component = 0; component < 9; ++component) {
      ghost_nonzero =
          ghost_nonzero || std::abs(virials[9 + component]) > 1.0e-12;
    }
    if (!ghost_nonzero) {
      std::cerr << "ghost per-atom virial was not written\n";
      ok = false;
    }
    const double sum6[6] = {
        virials[0] + virials[9],
        virials[1] + virials[10],
        virials[2] + virials[11],
        0.5 * (virials[3] + virials[6] + virials[12] + virials[15]),
        0.5 * (virials[4] + virials[7] + virials[13] + virials[16]),
        0.5 * (virials[5] + virials[8] + virials[14] + virials[17]),
    };
    for (int component = 0; component < 6; ++component) {
      if (std::abs(sum6[component] - total_virial6[component]) > 1.0e-6) {
        std::cerr << "ghost virial sum mismatch component=" << component
                  << " atom_sum=" << sum6[component]
                  << " total=" << total_virial6[component] << "\n";
        ok = false;
      }
    }
  }

  cudaFree(d_ilist);
  cudaFree(d_numneigh);
  cudaFree(d_neighbors);
  cudaFree(d_types);
  cudaFree(d_positions);
  cudaFree(d_virials);
  cudaFree(d_forces);
  cudaFree(d_total_potential);
  cudaFree(d_total_virial6);
  return ok;
}

}  // namespace

int main() {
  if (!check_external_workspace_plan()) {
    std::cerr << "external workspace contract failed\n";
    return EXIT_FAILURE;
  }
  if (!nep_adapters::register_cuda_engine()) {
    std::cerr << "failed to register CUDA engine\n";
    return EXIT_FAILURE;
  }

  const std::string model_path = write_radial_model();
  NepaModel* model = nullptr;
  const NepaStatus load_status =
      nepa_load_model("cuda", model_path.c_str(), &model);
  if (load_status != NEPA_STATUS_OK || model == nullptr) {
    std::cerr << "failed to load model status=" << load_status
              << " path=" << model_path << "\n";
    return EXIT_FAILURE;
  }

  int atom_counts[] = {2};
  int atom_offsets[] = {0};
  int batch_types[] = {0, 0};
  double positions_aos[] = {
      0.0, 0.0, 0.0,
      1.5, 0.0, 0.0,
  };
  double box[] = {
      8.0, 0.0, 0.0,
      0.0, 8.0, 0.0,
      0.0, 0.0, 8.0,
  };
  int pbc[] = {1, 1, 1};
  double batch_energy[] = {0.0};
  double batch_potential[2] = {};
  double batch_forces[6] = {};
  double batch_virials_per_atom[18] = {};
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = 2;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = batch_types;
  batch.positions_aos3 = positions_aos;
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;
  NepaFindForceResult batch_result{};
  batch_result.energy_per_structure = batch_energy;
  batch_result.potential_per_atom = batch_potential;
  batch_result.forces_aos3 = batch_forces;
  batch_result.virials_per_atom_row_major9 = batch_virials_per_atom;
  const NepaStatus batch_status =
      nepa_find_force_batch(model, &batch, &batch_result);
  if (batch_status != NEPA_STATUS_OK) {
    std::cerr << "batch status=" << batch_status << "\n";
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  const bool legacy_ok = run_device_layout_case(
      model,
      "legacy_aos",
      false,
      false,
      batch_energy[0],
      batch_forces,
      batch_potential,
      batch_virials_per_atom);
  const bool nolegacy_ok = run_device_layout_case(
      model,
      "nolegacy_soa",
      true,
      true,
      batch_energy[0],
      batch_forces,
      batch_potential,
      batch_virials_per_atom);
  const bool legacy_after_nolegacy_ok = run_device_layout_case(
      model,
      "legacy_aos_after_nolegacy_soa",
      false,
      true,
      batch_energy[0],
      batch_forces,
      batch_potential,
      batch_virials_per_atom);
  const bool ghost_virial_ok = run_ghost_per_atom_virial_case(model);

  nepa_free_model(model);
  const bool zbl_ok = run_zbl_force_only_case();
  const bool compact_ok = run_compact_device_neighbor_capacity_case();
  return legacy_ok && nolegacy_ok && legacy_after_nolegacy_ok &&
          ghost_virial_ok && zbl_ok && compact_ok ?
      EXIT_SUCCESS :
      EXIT_FAILURE;
}
