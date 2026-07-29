#include "nep_adapters/api.h"
#include "nep_adapters/engines/cuda.hpp"
#include "../engines/cuda/device_model.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string write_zbl_model(bool flexible) {
  const std::string model_path =
      (std::filesystem::temp_directory_path() /
       (flexible ? "cuda_flexible_zbl_batch_force.nep"
                 : "cuda_typewise_zbl_batch_force.nep"))
          .string();
  std::ofstream out(model_path);
  out << "nep4_zbl 1 C\n"
      << (flexible ? "zbl 0 0\n" : "zbl 0.8 2.5 0.7\n")
      << "cutoff 0.9 0.5 8 1\n"
      << "n_max 0 0\n"
      << "basis_size 0 0\n"
      << "l_max 0 0 0\n"
      << "ANN 3 0\n";
  const double values[] = {
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0,
      0.0, 0.0,
      1.0,
  };
  for (double value : values) {
    out << value << "\n";
  }
  if (flexible) {
    const double zbl_parameters[] = {
        0.8, 1.6,
        0.18175, 3.1998, 0.50986, 0.94229,
        0.28022, 0.4029, 0.02817, 0.20162,
    };
    for (double value : zbl_parameters) {
      out << value << "\n";
    }
  }
  return model_path;
}

struct Prediction {
  double energy = 0.0;
  std::vector<double> forces;
};

Prediction evaluate(NepaModel* model, const std::vector<double>& positions) {
  int atom_counts[] = {3};
  int atom_offsets[] = {0};
  int types[] = {0, 0, 0};
  double box[] = {
      12.0, 0.0, 0.0,
      0.0, 12.0, 0.0,
      0.0, 0.0, 12.0,
  };
  int pbc[] = {1, 1, 1};
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = 3;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions.data();
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;

  Prediction prediction;
  prediction.forces.assign(positions.size(), 0.0);
  NepaFindForceResult result{};
  result.energy_per_structure = &prediction.energy;
  result.forces_aos3 = prediction.forces.data();
  const NepaStatus status = nepa_find_force_batch(model, &batch, &result);
  if (status != NEPA_STATUS_OK) {
    std::cerr << "CUDA ZBL force status=" << status << "\n";
    std::exit(EXIT_FAILURE);
  }
  return prediction;
}

Prediction evaluate_lammps(
    NepaModel* model,
    const std::vector<double>& positions) {
  int ilist[] = {0, 1, 2};
  int numneigh[] = {2, 2, 2};
  int neigh0[] = {1, 2};
  int neigh1[] = {0, 2};
  int neigh2[] = {0, 1};
  int* firstneigh[] = {neigh0, neigh1, neigh2};
  int types[] = {1, 1, 1};
  int type_map[] = {-1, 0};
  double x0[] = {positions[0], positions[1], positions[2]};
  double x1[] = {positions[3], positions[4], positions[5]};
  double x2[] = {positions[6], positions[7], positions[8]};
  double* coordinates[] = {x0, x1, x2};

  NepaLammpsNeighborInput input{};
  input.nlocal = 3;
  input.inum = 3;
  input.ilist = ilist;
  input.numneigh = numneigh;
  input.firstneigh = firstneigh;
  input.types = types;
  input.type_map = type_map;
  input.positions = coordinates;

  Prediction prediction;
  prediction.forces.assign(positions.size(), 0.0);
  double* forces[] = {
      prediction.forces.data(),
      prediction.forces.data() + 3,
      prediction.forces.data() + 6,
  };
  double total_virial6[6] = {};
  NepaLammpsNeighborResult result{};
  result.total_potential = &prediction.energy;
  result.total_virial6 = total_virial6;
  result.forces = forces;
  const NepaStatus status =
      nepa_find_force_lammps_neighbors(model, &input, &result);
  if (status != NEPA_STATUS_OK) {
    std::cerr << "CUDA LAMMPS ZBL force status=" << status
              << " error=" << nepa_last_error_message() << "\n";
    std::exit(EXIT_FAILURE);
  }
  return prediction;
}

}  // namespace

int main() {
  if (!nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }

  for (const bool flexible : {false, true}) {
    const std::string model_path = write_zbl_model(flexible);
    const auto host =
        nep_adapters::cuda_backend::load_host_model_parameters(model_path);
    if (host.zbl_parameters_pair.size() != 10 ||
        host.zbl_parameters_pair[1] <= 1.0f) {
      std::cerr << "CUDA ZBL pair packing mismatch flexible=" << flexible
                << " pair_count=" << host.zbl_parameters_pair.size()
                << " outer="
                << (host.zbl_parameters_pair.size() >= 2
                        ? host.zbl_parameters_pair[1]
                        : -1.0f)
                << "\n";
      return EXIT_FAILURE;
    }
    NepaModel* model = nullptr;
    const NepaStatus load_status =
        nepa_load_model("cuda", model_path.c_str(), &model);
    if (load_status != NEPA_STATUS_OK ||
        model == nullptr) {
      std::cerr << "CUDA ZBL load failed flexible=" << flexible
                << " status=" << load_status
                << " error=" << nepa_last_error_message() << "\n";
      return EXIT_FAILURE;
    }

    NepaModelInfo info{};
    const NepaStatus info_status = nepa_model_info(model, &info);
    if (info_status != NEPA_STATUS_OK ||
        (info.capabilities & NEPA_CAPABILITY_BATCH_FIND_FORCE) == 0 ||
        info.descriptor_dim != 1 || info.cutoff_max < 1.6) {
      std::cerr << "CUDA ZBL model info mismatch flexible=" << flexible
                << " status=" << info_status
                << " capabilities=" << info.capabilities
                << " descriptor_dim=" << info.descriptor_dim
                << " cutoff_max=" << info.cutoff_max << "\n";
      nepa_free_model(model);
      return EXIT_FAILURE;
    }

    const std::vector<double> positions = {
        0.10, 0.10, 0.20,
        1.10, 0.10, 0.20,
        2.45, 0.42, 0.30,
    };
    const Prediction base = evaluate(model, positions);
    if (!std::isfinite(base.energy) || std::abs(base.energy) < 1.0e-8) {
      std::cerr << "CUDA ZBL invalid energy flexible=" << flexible
                << " energy=" << base.energy;
      for (double force : base.forces) {
        std::cerr << " force=" << force;
      }
      std::cerr << "\n";
      nepa_free_model(model);
      return EXIT_FAILURE;
    }
    for (double force : base.forces) {
      if (!std::isfinite(force)) {
        nepa_free_model(model);
        return EXIT_FAILURE;
      }
    }
    const Prediction lammps = evaluate_lammps(model, positions);
    if (std::abs(lammps.energy - base.energy) > 1.0e-4) {
      std::cerr << "CUDA ZBL batch/LAMMPS energy mismatch flexible="
                << flexible << " batch=" << base.energy
                << " lammps=" << lammps.energy << "\n";
      nepa_free_model(model);
      return EXIT_FAILURE;
    }
    for (std::size_t coordinate = 0;
         coordinate < base.forces.size();
         ++coordinate) {
      if (std::abs(lammps.forces[coordinate] - base.forces[coordinate]) >
          1.0e-4) {
        std::cerr << "CUDA ZBL batch/LAMMPS force mismatch flexible="
                  << flexible << " coordinate=" << coordinate
                  << " batch=" << base.forces[coordinate]
                  << " lammps=" << lammps.forces[coordinate] << "\n";
        nepa_free_model(model);
        return EXIT_FAILURE;
      }
    }

    constexpr double eps = 1.0e-4;
    constexpr double tolerance = 5.0e-2;
    for (std::size_t coordinate = 0; coordinate < positions.size(); ++coordinate) {
      std::vector<double> plus = positions;
      std::vector<double> minus = positions;
      plus[coordinate] += eps;
      minus[coordinate] -= eps;
      const double e_plus = evaluate(model, plus).energy;
      const double e_minus = evaluate(model, minus).energy;
      const double finite_difference_force =
          -(e_plus - e_minus) / (2.0 * eps);
      const double diff =
          std::abs(base.forces[coordinate] - finite_difference_force);
      if (diff > tolerance) {
        std::cerr << "ZBL finite-difference mismatch flexible=" << flexible
                  << " coordinate=" << coordinate
                  << " force=" << base.forces[coordinate]
                  << " fd=" << finite_difference_force
                  << " diff=" << diff << "\n";
        nepa_free_model(model);
        return EXIT_FAILURE;
      }
    }
    nepa_free_model(model);
  }
  return EXIT_SUCCESS;
}
