#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(NepaStatus status) {
  if (status != NEPA_STATUS_OK) {
    throw std::runtime_error(nepa_last_error_message());
  }
}

struct Model {
  NepaModel* handle = nullptr;
  Model(const char* backend, const std::string& path) {
    check(nepa_load_model(backend, path.c_str(), &handle));
  }
  ~Model() { nepa_free_model(handle); }
};

std::string write_model(int radial, int angular) {
  const auto path = std::filesystem::temp_directory_path() /
      ("cuda_internal_growth_" + std::to_string(radial) + "_" +
       std::to_string(angular) + ".nep");
  std::ofstream out(path);
  out << "nep4 1 C\ncutoff 6 5 " << radial << " " << angular << "\n"
      << "n_max 0 0\nbasis_size 0 1\nl_max 4 0 0\nANN 3 0\n";
  const double values[] = {
      0.05, -0.04, 0.03, 0.02, -0.01,
      -0.02, 0.06, 0.01, 0.04, -0.03,
      0.04, 0.02, -0.05, 0.03, 0.01,
      0.01, -0.02, 0.03, 0.20, -0.15, 0.10, 0.0,
      0.45, 0.80, -0.30, 1.0, 0.5, 0.4, 0.3, 0.2,
  };
  for (double value : values) {
    out << value << "\n";
  }
  return path.string();
}

struct Batch {
  std::vector<int> counts, offsets, types, pbc;
  std::vector<double> positions, boxes;
  NepaStructureBatch input{};

  Batch(int atoms, int structures, bool skew, bool small) {
    for (int s = 0; s < structures; ++s) {
      counts.push_back(atoms);
      offsets.push_back(s * atoms);
      const double length = small ? 7.0 : 30.0;
      const double box[] = {
          length, skew ? 1.1 : 0.0, skew ? -0.4 : 0.0,
          0.0, length, skew ? 0.7 : 0.0,
          0.0, 0.0, length};
      boxes.insert(boxes.end(), box, box + 9);
      pbc.insert(pbc.end(), {1, 1, 1});
      for (int a = 0; a < atoms; ++a) {
        types.push_back(0);
        // All atoms are distinct; increasing this cluster forces repeated growth.
        positions.insert(positions.end(), {
            0.3 + 1.2 * (a % 3),
            0.4 + 1.1 * ((a / 3) % 3),
            0.5 + 1.3 * (a / 9)});
      }
    }
    input.num_structures = structures;
    input.total_atoms = atoms * structures;
    input.atom_counts = counts.data();
    input.atom_offsets = offsets.data();
    input.types = types.data();
    input.positions_aos3 = positions.data();
    input.boxes_row_major9 = boxes.data();
    input.pbc_flags3 = pbc.data();
  }
};

std::vector<double> forces(NepaModel* model, const Batch& batch) {
  const int n = batch.input.total_atoms;
  std::vector<double> energy(batch.counts.size()), potential(n), force(n * 3), virial(n * 9);
  NepaFindForceResult result{};
  result.energy_per_structure = energy.data();
  result.potential_per_atom = potential.data();
  result.forces_aos3 = force.data();
  result.virials_per_atom_row_major9 = virial.data();
  check(nepa_find_force_batch(model, &batch.input, &result));
  energy.insert(energy.end(), potential.begin(), potential.end());
  energy.insert(energy.end(), force.begin(), force.end());
  energy.insert(energy.end(), virial.begin(), virial.end());
  return energy;
}

std::vector<double> descriptors(NepaModel* model, const Batch& batch) {
  std::vector<double> values(batch.input.total_atoms * 5);
  NepaFindDescriptorResult result{};
  result.descriptors = values.data();
  check(nepa_find_descriptors(model, &batch.input, &result));
  return values;
}

void compare(const std::vector<double>& actual, const std::vector<double>& expected,
             const char* label, double tolerance) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("output size mismatch");
  }
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
        std::abs(actual[i] - expected[i]) > tolerance * (1.0 + std::abs(expected[i]))) {
      std::cerr << label << " mismatch at " << i << ": " << actual[i]
                << " vs " << expected[i] << "\n";
      throw std::runtime_error("neighbor growth changed prediction");
    }
  }
}

void run(int radial, int angular, bool skew, bool small) {
  const auto small_path = write_model(radial, angular);
  const auto reference_path = write_model(128, 128);
  {
    Model force_model("cuda", small_path);
    Model descriptor_model("cuda", small_path);
    Model reference("cuda", reference_path);
    Model cpu("cpu", small_path);
    // First fit the initial capacity, grow twice, then reuse it with fewer atoms
    // and a different batch shape. Descriptor growth starts from a fresh model.
    const int atoms[] = {2, 8, 27, 8, 2};
    for (int step = 0; step < 5; ++step) {
      Batch batch(atoms[step], step % 2 + 1, skew, small);
      const auto force = forces(force_model.handle, batch);
      compare(force, forces(reference.handle, batch), "force / wide CUDA", 1.0e-6);
      compare(force, forces(cpu.handle, batch), "force / CPU", 2.0e-4);
      const auto descriptor = descriptors(descriptor_model.handle, batch);
      compare(descriptor, descriptors(reference.handle, batch), "descriptor / wide CUDA", 1.0e-6);
      compare(descriptor, descriptors(cpu.handle, batch), "descriptor / CPU", 2.0e-4);
    }
  }
  std::filesystem::remove(small_path);
  std::filesystem::remove(reference_path);
}

}  // namespace

int main() {
  try {
    if (!nep_adapters::register_cuda_engine() || !nep_adapters::register_cpu_engine()) {
      return EXIT_FAILURE;
    }
    for (bool skew : {false, true}) {
      run(1, 128, skew, false);  // radial-only overflow
      run(128, 1, skew, false);  // angular-only overflow
      run(1, 1, skew, false);   // both lists overflow
      run(1, 1, skew, true);    // periodic expansion and output reduction
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
