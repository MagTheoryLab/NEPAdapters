#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Prediction {
  double energy = 0.0;
  std::vector<double> forces;
  std::vector<double> virial;
  std::vector<double> descriptors;
};

std::string write_model(
    const std::string& name,
    const std::vector<std::string>& lines) {
  const std::string path =
      (std::filesystem::temp_directory_path() / name).string();
  std::ofstream out(path);
  for (const std::string& line : lines) {
    out << line << '\n';
  }
  return path;
}

Prediction evaluate(
    NepaModel* model,
    const std::vector<int>& types,
    const std::vector<double>& positions,
    double box_length = 10.0) {
  const int atom_count = static_cast<int>(types.size());
  int atom_counts[] = {atom_count};
  int atom_offsets[] = {0};
  int pbc[] = {1, 1, 1};
  double box[] = {
      box_length, 0.0, 0.0,
      0.0, box_length, 0.0,
      0.0, 0.0, box_length,
  };
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = atom_count;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types.data();
  batch.positions_aos3 = positions.data();
  batch.boxes_row_major9 = box;
  batch.pbc_flags3 = pbc;

  Prediction prediction;
  prediction.forces.resize(positions.size());
  prediction.virial.resize(9);
  NepaFindForceResult result{};
  result.energy_per_structure = &prediction.energy;
  result.forces_aos3 = prediction.forces.data();
  result.virials_row_major9 = prediction.virial.data();
  if (nepa_find_force_batch(model, &batch, &result) != NEPA_STATUS_OK) {
    std::cerr << "protocol extension force evaluation failed: "
              << nepa_last_error_message() << '\n';
    std::exit(EXIT_FAILURE);
  }
  NepaModelInfo info{};
  if (nepa_model_info(model, &info) != NEPA_STATUS_OK ||
      info.descriptor_dim <= 0) {
    std::cerr << "protocol extension model info failed\n";
    std::exit(EXIT_FAILURE);
  }
  prediction.descriptors.resize(
      static_cast<std::size_t>(atom_count) *
      static_cast<std::size_t>(info.descriptor_dim));
  NepaFindDescriptorResult descriptor_result{};
  descriptor_result.descriptors = prediction.descriptors.data();
  if (nepa_find_descriptors(model, &batch, &descriptor_result) !=
      NEPA_STATUS_OK) {
    std::cerr << "protocol extension descriptor evaluation failed: "
              << nepa_last_error_message() << '\n';
    std::exit(EXIT_FAILURE);
  }
  return prediction;
}

double max_difference(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs) {
  double result = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    result = std::max(result, std::abs(lhs[i] - rhs[i]));
  }
  return result;
}

bool check_cpu_cuda_parity(
    const std::string& model_path,
    const std::vector<int>& types,
    const std::vector<double>& positions,
    double box_length = 10.0) {
  NepaModel* cpu = nullptr;
  NepaModel* cuda = nullptr;
  if (nepa_load_model("cpu", model_path.c_str(), &cpu) != NEPA_STATUS_OK ||
      nepa_load_model("cuda", model_path.c_str(), &cuda) != NEPA_STATUS_OK ||
      cpu == nullptr || cuda == nullptr) {
    std::cerr << "protocol extension model load failed: "
              << nepa_last_error_message() << '\n';
    nepa_free_model(cpu);
    nepa_free_model(cuda);
    return false;
  }
  const Prediction cpu_result = evaluate(cpu, types, positions, box_length);
  const Prediction cuda_result = evaluate(cuda, types, positions, box_length);
  nepa_free_model(cpu);
  nepa_free_model(cuda);
  const double energy_error =
      std::abs(cpu_result.energy - cuda_result.energy);
  const double force_error =
      max_difference(cpu_result.forces, cuda_result.forces);
  const double virial_error =
      max_difference(cpu_result.virial, cuda_result.virial);
  const double descriptor_error =
      max_difference(cpu_result.descriptors, cuda_result.descriptors);
  if (energy_error > 2.0e-5 || force_error > 2.0e-5 ||
      virial_error > 2.0e-4 || descriptor_error > 2.0e-5) {
    std::cerr << "protocol extension parity mismatch energy=" << energy_error
              << " force=" << force_error
              << " virial=" << virial_error
              << " descriptor=" << descriptor_error << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (!nep_adapters::register_cpu_engine() ||
      !nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }
  if (argc == 2) {
    NepaModel* external = nullptr;
    const NepaStatus status = nepa_load_model("cuda", argv[1], &external);
    if (status != NEPA_STATUS_OK || external == nullptr) {
      std::cerr << "external CUDA model load failed: "
                << nepa_last_error_message() << '\n';
      return EXIT_FAILURE;
    }
    NepaModelInfo info{};
    const bool valid =
        nepa_model_info(external, &info) == NEPA_STATUS_OK &&
        (info.capabilities & NEPA_CAPABILITY_BATCH_FIND_FORCE) != 0;
    if (valid && info.num_types == 36) {
      const Prediction prediction = evaluate(
          external,
          {8, 8},
          {0.0, 0.0, 0.0, 1.0, 0.0, 0.0});
      if (!std::isfinite(prediction.energy) ||
          !std::all_of(
              prediction.forces.begin(),
              prediction.forces.end(),
              [](double value) { return std::isfinite(value); })) {
        nepa_free_model(external);
        return EXIT_FAILURE;
      }
    }
    nepa_free_model(external);
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  const std::string two_hidden = write_model(
      "cuda_two_hidden_protocol.nep",
      {
          "nep4 1 C",
          "cutoff 5 1 8 1",
          "n_max 0 0",
          "basis_size 0 0",
          "l_max 0 0 0",
          "ANN 2 2",
          "0.7", "-0.3", "0.1", "-0.2",
          "0.4", "-0.5", "0.6", "0.2",
          "0.05", "-0.1", "0.8", "-0.7", "0.03",
          "0.9", "0.0", "1.2",
      });
  if (!check_cpu_cuda_parity(
          two_hidden,
          {0, 0},
          {0.0, 0.0, 0.0, 1.3, 0.2, 0.1})) {
    return EXIT_FAILURE;
  }
  if (!check_cpu_cuda_parity(
          two_hidden,
          {0},
          {0.0, 0.0, 0.0},
          4.0)) {
    std::cerr << "small periodic box parity failed\n";
    return EXIT_FAILURE;
  }

  const std::string per_type_cutoff = write_model(
      "cuda_per_type_cutoff_protocol.nep",
      {
          "nep4 2 C H",
          "cutoff 1 0.5 3 0.5 8 1",
          "n_max 0 0",
          "basis_size 0 0",
          "l_max 0 0 0",
          "ANN 1 0",
          "0.8", "0.1", "0.7",
          "-0.4", "-0.2", "0.5",
          "0.05",
          "0.9", "0.6", "0.4", "0.8",
          "0.0", "0.0", "0.0", "0.0",
          "1.1",
      });
  if (!check_cpu_cuda_parity(
          per_type_cutoff,
          {0, 1},
          {0.0, 0.0, 0.0, 1.5, 0.0, 0.0})) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
