#include "nep_adapters/api.h"
#include "nep_adapters/engines/cuda.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

std::string write_radial_model() {
  const std::string model_path =
      (std::filesystem::temp_directory_path() /
       "cuda_lammps_host_radial.nep")
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

bool run_ghost_output_case(NepaModel* model) {
  int ilist[] = {0};
  int numneigh[] = {1, 0};
  int neigh0[] = {1};
  int* firstneigh[] = {neigh0, nullptr};
  int types[] = {1, 1};
  int type_map[] = {-1, 0};
  double x0[] = {0.0, 0.0, 0.0};
  double x1[] = {1.5, 0.0, 0.0};
  double* positions[] = {x0, x1};

  double total_potential = 0.0;
  double total_virial6[6] = {};
  double potential_per_atom[1] = {};
  double f0[] = {123.0, 123.0, 123.0};
  double f1[] = {123.0, 123.0, 123.0};
  double* forces[] = {f0, f1};
  double v0[9] = {};
  double v1[9] = {};
  double* virials[] = {v0, v1};

  NepaLammpsNeighborInput input{};
  input.nlocal = 1;
  input.inum = 1;
  input.ilist = ilist;
  input.numneigh = numneigh;
  input.firstneigh = firstneigh;
  input.types = types;
  input.type_map = type_map;
  input.positions = positions;

  NepaLammpsNeighborResult result{};
  result.total_potential = &total_potential;
  result.total_virial6 = total_virial6;
  result.potential_per_atom = potential_per_atom;
  result.forces = forces;
  result.virials_per_atom9 = virials;

  const NepaStatus status =
      nepa_find_force_lammps_neighbors(model, &input, &result);
  if (status != NEPA_STATUS_OK) {
    std::cerr << "ghost output status=" << status
              << " error=" << nepa_last_error_message() << "\n";
    return false;
  }

  for (int component = 0; component < 3; ++component) {
    if (!std::isfinite(f0[component]) || f1[component] != 123.0) {
      std::cerr << "ghost force ownership mismatch component=" << component
                << " local=" << f0[component]
                << " ghost=" << f1[component] << "\n";
      return false;
    }
  }

  double ghost_virial_norm = 0.0;
  for (double value : v1) {
    if (!std::isfinite(value)) {
      std::cerr << "ghost virial is not finite\n";
      return false;
    }
    ghost_virial_norm += std::abs(value);
  }
  if (ghost_virial_norm <= 1.0e-12 ||
      std::abs(v0[0] + v1[0] - total_virial6[0]) > 1.0e-10) {
    std::cerr << "ghost virial copyback mismatch local_xx=" << v0[0]
              << " ghost_xx=" << v1[0]
              << " total_xx=" << total_virial6[0] << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
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

  int ilist[] = {0, 1};
  int numneigh[] = {1, 1};
  int neigh0[] = {1};
  int neigh1[] = {0};
  int* firstneigh[] = {neigh0, neigh1};
  int types[] = {1, 1};
  int type_map[] = {-1, 0};
  double x0[] = {0.0, 0.0, 0.0};
  double x1[] = {1.5, 0.0, 0.0};
  double* positions[] = {x0, x1};
  double total_potential = 0.0;
  double total_virial6[6] = {};
  double potential_per_atom[2] = {};
  double f0[] = {0.0, 0.0, 0.0};
  double f1[] = {0.0, 0.0, 0.0};
  double* forces[] = {f0, f1};
  double v0[9] = {};
  double v1[9] = {};
  double* virials[] = {v0, v1};

  NepaLammpsNeighborInput input{};
  input.nlocal = 2;
  input.inum = 2;
  input.ilist = ilist;
  input.numneigh = numneigh;
  input.firstneigh = firstneigh;
  input.types = types;
  input.type_map = type_map;
  input.positions = positions;

  NepaLammpsNeighborResult result{};
  result.total_potential = &total_potential;
  result.total_virial6 = total_virial6;
  result.potential_per_atom = potential_per_atom;
  result.forces = forces;
  result.virials_per_atom9 = virials;

  const NepaStatus lammps_status =
      nepa_find_force_lammps_neighbors(model, &input, &result);
  if (lammps_status != NEPA_STATUS_OK || !std::isfinite(total_potential)) {
    std::cerr << "LAMMPS CUDA status=" << lammps_status
              << " energy=" << total_potential << "\n";
    nepa_free_model(model);
    return EXIT_FAILURE;
  }
  for (double* force : forces) {
    for (int component = 0; component < 3; ++component) {
      if (!std::isfinite(force[component])) {
        nepa_free_model(model);
        return EXIT_FAILURE;
      }
    }
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
  double batch_forces[6] = {};
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
  batch_result.forces_aos3 = batch_forces;
  const NepaStatus batch_status = nepa_find_force_batch(model, &batch, &batch_result);
  if (batch_status != NEPA_STATUS_OK ||
      std::abs(total_potential - batch_energy[0]) > 1.0e-10) {
    std::cerr << "batch status=" << batch_status
              << " lammps_energy=" << total_potential
              << " batch_energy=" << batch_energy[0] << "\n";
    nepa_free_model(model);
    return EXIT_FAILURE;
  }
  for (int atom = 0; atom < 2; ++atom) {
    for (int component = 0; component < 3; ++component) {
      const double diff =
          std::abs(forces[atom][component] - batch_forces[3 * atom + component]);
      if (diff > 1.0e-10) {
        std::cerr << "force mismatch atom=" << atom
                  << " component=" << component
                  << " lammps=" << forces[atom][component]
                  << " batch=" << batch_forces[3 * atom + component]
                  << " diff=" << diff << "\n";
        nepa_free_model(model);
        return EXIT_FAILURE;
      }
    }
  }

  result.forces = nullptr;
  const NepaStatus invalid_status =
      nepa_find_force_lammps_neighbors(model, &input, &result);
  if (invalid_status != NEPA_STATUS_INVALID_ARGUMENT) {
    std::cerr << "invalid argument status=" << invalid_status << "\n";
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  if (!run_ghost_output_case(model)) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  nepa_free_model(model);
  return EXIT_SUCCESS;
}
