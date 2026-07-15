#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_nep3.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Prediction {
  std::vector<double> energy;
  std::vector<double> potential;
  std::vector<double> forces;
  std::vector<double> virial;
  std::vector<double> per_atom_virial;
};

struct Case {
  std::string name;
  std::string model_text;
  std::vector<double> positions;
  double tolerance = 1.0e-3;
  int type_cycle = 1;
};

std::string write_model(const Case& test_case) {
  const std::string model_path =
      (std::filesystem::temp_directory_path() /
       ("cuda_cpu_triclinic_" + test_case.name + ".nep"))
          .string();
  std::ofstream out(model_path);
  out << test_case.model_text;
  return model_path;
}

std::vector<double> triclinic_box() {
  return {
      12.0, 2.0, 1.0,
      0.0, 11.0, 1.5,
      0.0, 0.0, 10.0,
  };
}

std::vector<double> orthorhombic_box() {
  return {
      10.0, 0.0, 0.0,
      0.0, 10.0, 0.0,
      0.0, 0.0, 10.0,
  };
}

Prediction evaluate_batch(
    const char* engine_name,
    const std::string& model_path,
    const std::vector<int>& atom_counts,
    const std::vector<int>& atom_offsets,
    const std::vector<double>& positions,
    const std::vector<double>& boxes,
    const std::vector<int>& pbc,
    int type_cycle = 1) {
  NepaModel* model = nullptr;
  if (nepa_load_model(engine_name, model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    std::cerr << "failed to load " << engine_name << " model " << model_path << "\n";
    std::exit(EXIT_FAILURE);
  }

  const int total_atoms = static_cast<int>(positions.size() / 3);
  std::vector<int> types(static_cast<std::size_t>(total_atoms), 0);
  for (std::size_t structure = 0; structure < atom_counts.size(); ++structure) {
    for (int atom = 0; atom < atom_counts[structure]; ++atom) {
      types[static_cast<std::size_t>(atom_offsets[structure] + atom)] =
          atom % type_cycle;
    }
  }

  NepaStructureBatch batch{};
  batch.num_structures = static_cast<int>(atom_counts.size());
  batch.total_atoms = total_atoms;
  batch.atom_counts = atom_counts.data();
  batch.atom_offsets = atom_offsets.data();
  batch.types = types.data();
  batch.positions_aos3 = positions.data();
  batch.boxes_row_major9 = boxes.data();
  batch.pbc_flags3 = pbc.data();

  Prediction prediction;
  prediction.energy.assign(atom_counts.size(), 0.0);
  prediction.potential.assign(static_cast<std::size_t>(total_atoms), 0.0);
  prediction.forces.assign(positions.size(), 0.0);
  prediction.virial.assign(atom_counts.size() * 9, 0.0);
  prediction.per_atom_virial.assign(static_cast<std::size_t>(total_atoms) * 9, 0.0);
  NepaFindForceResult result{};
  result.energy_per_structure = prediction.energy.data();
  result.potential_per_atom = prediction.potential.data();
  result.forces_aos3 = prediction.forces.data();
  result.virials_row_major9 = prediction.virial.data();
  result.virials_per_atom_row_major9 = prediction.per_atom_virial.data();

  const NepaStatus status = nepa_find_force_batch(model, &batch, &result);
  nepa_free_model(model);
  if (status != NEPA_STATUS_OK) {
    std::cerr << engine_name << " find_force_batch status=" << status
              << " model=" << model_path
              << " error=" << nepa_last_error_message() << "\n";
    std::exit(EXIT_FAILURE);
  }
  return prediction;
}

Prediction evaluate_single(
    const char* engine_name,
    const std::string& model_path,
    const std::vector<double>& positions,
    int type_cycle) {
  return evaluate_batch(
      engine_name,
      model_path,
      {3},
      {0},
      positions,
      triclinic_box(),
      {1, 1, 1},
      type_cycle);
}

Prediction evaluate_structures_independently(
    const char* engine_name,
    const std::string& model_path,
    const std::vector<int>& atom_counts,
    const std::vector<int>& atom_offsets,
    const std::vector<double>& positions,
    const std::vector<double>& boxes,
    const std::vector<int>& pbc,
    int type_cycle) {
  const int total_atoms = static_cast<int>(positions.size() / 3);
  Prediction prediction;
  prediction.energy.assign(atom_counts.size(), 0.0);
  prediction.potential.assign(static_cast<std::size_t>(total_atoms), 0.0);
  prediction.forces.assign(positions.size(), 0.0);
  prediction.virial.assign(atom_counts.size() * 9, 0.0);
  prediction.per_atom_virial.assign(static_cast<std::size_t>(total_atoms) * 9, 0.0);

  for (std::size_t structure = 0; structure < atom_counts.size(); ++structure) {
    const int atom_count = atom_counts[structure];
    const int atom_offset = atom_offsets[structure];
    std::vector<double> local_positions(static_cast<std::size_t>(atom_count) * 3);
    for (int atom = 0; atom < atom_count; ++atom) {
      const std::size_t global_atom = static_cast<std::size_t>(atom_offset + atom);
      local_positions[3 * static_cast<std::size_t>(atom) + 0] =
          positions[3 * global_atom + 0];
      local_positions[3 * static_cast<std::size_t>(atom) + 1] =
          positions[3 * global_atom + 1];
      local_positions[3 * static_cast<std::size_t>(atom) + 2] =
          positions[3 * global_atom + 2];
    }

    const std::vector<double> local_box(
        boxes.begin() + static_cast<std::ptrdiff_t>(9 * structure),
        boxes.begin() + static_cast<std::ptrdiff_t>(9 * structure + 9));
    const std::vector<int> local_pbc(
        pbc.begin() + static_cast<std::ptrdiff_t>(3 * structure),
        pbc.begin() + static_cast<std::ptrdiff_t>(3 * structure + 3));
    const Prediction local = evaluate_batch(
        engine_name,
        model_path,
        {atom_count},
        {0},
        local_positions,
        local_box,
        local_pbc,
        type_cycle);

    prediction.energy[structure] = local.energy[0];
    for (int atom = 0; atom < atom_count; ++atom) {
      const std::size_t global_atom = static_cast<std::size_t>(atom_offset + atom);
      const std::size_t local_atom = static_cast<std::size_t>(atom);
      prediction.potential[global_atom] = local.potential[local_atom];
      prediction.forces[3 * global_atom + 0] = local.forces[3 * local_atom + 0];
      prediction.forces[3 * global_atom + 1] = local.forces[3 * local_atom + 1];
      prediction.forces[3 * global_atom + 2] = local.forces[3 * local_atom + 2];
      for (int component = 0; component < 9; ++component) {
        prediction.per_atom_virial[9 * global_atom + component] =
            local.per_atom_virial[9 * local_atom + component];
      }
    }
    for (int component = 0; component < 9; ++component) {
      prediction.virial[9 * structure + component] =
          local.virial[component];
    }
  }

  return prediction;
}

bool close_value(double lhs, double rhs, double tolerance) {
  const double scale = std::max(1.0, std::max(std::abs(lhs), std::abs(rhs)));
  return std::abs(lhs - rhs) <= tolerance * scale;
}

bool compare_prediction(
    const Case& test_case,
    const Prediction& cpu,
    const Prediction& cuda) {
  for (std::size_t structure = 0; structure < cpu.energy.size(); ++structure) {
    if (!close_value(
            cpu.energy[structure],
            cuda.energy[structure],
            test_case.tolerance)) {
      std::cerr << test_case.name << " energy mismatch structure=" << structure
                << " cpu=" << cpu.energy[structure]
                << " cuda=" << cuda.energy[structure] << "\n";
      return false;
    }
  }
  for (std::size_t atom = 0; atom < cpu.potential.size(); ++atom) {
    if (!close_value(cpu.potential[atom], cuda.potential[atom], test_case.tolerance)) {
      std::cerr << test_case.name << " potential mismatch atom=" << atom
                << " cpu=" << cpu.potential[atom]
                << " cuda=" << cuda.potential[atom] << "\n";
      return false;
    }
  }
  for (std::size_t component = 0; component < cpu.forces.size(); ++component) {
    if (!close_value(cpu.forces[component], cuda.forces[component], test_case.tolerance)) {
      std::cerr << test_case.name << " force mismatch component=" << component
                << " cpu=" << cpu.forces[component]
                << " cuda=" << cuda.forces[component] << "\n";
      return false;
    }
  }
  for (std::size_t component = 0; component < cpu.virial.size(); ++component) {
    if (!close_value(cpu.virial[component], cuda.virial[component], test_case.tolerance)) {
      std::cerr << test_case.name << " virial mismatch component=" << component
                << " cpu=" << cpu.virial[component]
                << " cuda=" << cuda.virial[component] << "\n";
      return false;
    }
  }
  for (std::size_t component = 0;
       component < cpu.per_atom_virial.size();
       ++component) {
    if (!close_value(
            cpu.per_atom_virial[component],
            cuda.per_atom_virial[component],
            test_case.tolerance)) {
      std::cerr << test_case.name << " per-atom virial mismatch component="
                << component << " cpu=" << cpu.per_atom_virial[component]
                << " cuda=" << cuda.per_atom_virial[component] << "\n";
      return false;
    }
  }
  return true;
}

std::string values_text(const std::vector<double>& values) {
  std::string text;
  for (double value : values) {
    text += std::to_string(value) + "\n";
  }
  return text;
}

std::vector<double> generated_model_values(
    std::size_t ann_parameter_count,
    std::size_t descriptor_parameter_count,
    int descriptor_dim) {
  std::vector<double> values;
  values.reserve(
      ann_parameter_count + descriptor_parameter_count +
      static_cast<std::size_t>(descriptor_dim));
  for (std::size_t index = 0; index < ann_parameter_count; ++index) {
    const int centered = static_cast<int>(index % 17) - 8;
    values.push_back(5.0e-4 * static_cast<double>(centered));
  }
  for (std::size_t index = 0; index < descriptor_parameter_count; ++index) {
    const int centered = static_cast<int>(index % 11) - 5;
    values.push_back(2.0e-3 * static_cast<double>(centered));
  }
  for (int descriptor = 0; descriptor < descriptor_dim; ++descriptor) {
    values.push_back(1.0);
  }
  return values;
}

std::vector<Case> make_cases() {
  const std::vector<double> radial_values = {
      0.20, -0.10, 0.05, 0.15,
      0.01, -0.02,
      0.30, -0.25,
      0.04,
      0.70, -0.15, 0.05, -0.30, 0.20, -0.10, 0.0,
      0.80, 1.10,
  };
  const std::vector<double> angular_values = {
      0.05, -0.04, 0.03, 0.02, -0.01,
      -0.02, 0.06, 0.01, 0.04, -0.03,
      0.04, 0.02, -0.05, 0.03, 0.01,
      0.01, -0.02, 0.03,
      0.20, -0.15, 0.10,
      0.0,
      0.0, 0.80, -0.30,
      1.0, 0.5, 0.4, 0.3, 0.2,
  };
  const std::vector<double> q222_values = {
      0.04, -0.03, 0.05, 0.02,
      -0.02, 0.06, 0.02, -0.04,
      0.03, -0.04, 0.05, 0.01,
      0.01, -0.015, 0.02,
      0.18, -0.12, 0.09,
      0.0,
      0.0, 0.75, -0.25,
      1.0, 0.45, 0.35, 0.25,
  };
  const std::vector<double> nep5_values = {
      0.12, -0.08, 0.05,
      0.01, -0.02, 0.03,
      0.20, -0.15, 0.10,
      0.07,
      0.02,
      0.80, 0.0,
      1.0,
  };
  const std::vector<double> zbl_values = {
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0,
      0.0, 0.0,
      1.0,
  };
  constexpr int kWideDescriptorDim = 65;
  constexpr int kWideHiddenNeurons = 3;
  const std::vector<double> wide_descriptor_values = generated_model_values(
      static_cast<std::size_t>(kWideDescriptorDim + 2) *
              kWideHiddenNeurons +
          1,
      66,
      kWideDescriptorDim);
  constexpr int kSharedPressureTypes = 8;
  constexpr int kSharedPressureHiddenNeurons = 120;
  constexpr int kSharedPressureDescriptorDim = 1;
  const std::vector<double> shared_pressure_values = generated_model_values(
      static_cast<std::size_t>(kSharedPressureDescriptorDim + 2) *
              kSharedPressureHiddenNeurons * kSharedPressureTypes +
          1,
      static_cast<std::size_t>(kSharedPressureTypes) *
          kSharedPressureTypes * 18,
      kSharedPressureDescriptorDim);

  return {
      {
          "radial",
          "nep4 1 C\n"
          "cutoff 5 1 8 1\n"
          "n_max 1 0\n"
          "basis_size 2 0\n"
          "l_max 0 0 0\n"
          "ANN 2 0\n" +
              values_text(radial_values),
          {0.20, 0.20, 0.20, 11.40, 1.90, 1.00, 2.40, 0.50, 0.30},
          2.0e-3,
      },
      {
          "angular_l4",
          "nep4 1 C\n"
          "cutoff 0.5 4 1 8\n"
          "n_max 0 0\n"
          "basis_size 0 1\n"
          "l_max 4 0 0\n"
          "ANN 3 0\n" +
              values_text(angular_values),
          {0.20, 0.20, 0.20, 11.40, 1.90, 1.00, 2.40, 0.50, 0.30},
          5.0e-3,
      },
      {
          "q222",
          "nep4 1 C\n"
          "cutoff 0.5 4 1 8\n"
          "n_max 0 0\n"
          "basis_size 0 1\n"
          "l_max 2 2 0\n"
          "ANN 3 0\n" +
              values_text(q222_values),
          {0.20, 0.20, 0.20, 11.40, 1.90, 1.00, 2.40, 0.50, 0.30},
          5.0e-3,
      },
      {
          "nep5",
          "nep5 1 C\n"
          "cutoff 4 0.5 8 1\n"
          "n_max 0 0\n"
          "basis_size 0 0\n"
          "l_max 0 0 0\n"
          "ANN 3 0\n" +
              values_text(nep5_values),
          {0.20, 0.20, 0.20, 11.40, 1.90, 1.00, 2.40, 0.50, 0.30},
          2.0e-3,
      },
      {
          "zbl",
          "nep4_zbl 1 C\n"
          "zbl 0.8 1.6\n"
          "cutoff 4 0.5 8 1\n"
          "n_max 0 0\n"
          "basis_size 0 0\n"
          "l_max 0 0 0\n"
          "ANN 3 0\n" +
              values_text(zbl_values),
          {0.20, 0.20, 0.20, 11.20, 1.85, 0.95, 2.40, 0.50, 0.30},
          5.0e-2,
      },
      {
          "wide_descriptor",
          "nep4 1 C\n"
          "cutoff 5 1 8 1\n"
          "n_max 64 0\n"
          "basis_size 0 0\n"
          "l_max 0 0 0\n"
          "ANN 3 0\n" +
              values_text(wide_descriptor_values),
          {0.20, 0.20, 0.20, 11.40, 1.90, 1.00, 2.40, 0.50, 0.30},
          8.0e-3,
      },
      {
          "shared_pressure",
          "nep4 8 C H N O F Si P S\n"
          "cutoff 5 1 8 1\n"
          "n_max 0 0\n"
          "basis_size 16 0\n"
          "l_max 0 0 0\n"
          "ANN 120 0\n" +
              values_text(shared_pressure_values),
          {0.20, 0.20, 0.20, 11.40, 1.90, 1.00, 2.40, 0.50, 0.30},
          1.0e-4,
          kSharedPressureTypes,
      },
  };
}

}  // namespace

int main() {
  if (!nep_adapters::register_cpu_nep3_engine() ||
      !nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }

  for (const Case& test_case : make_cases()) {
    const std::string model_path = write_model(test_case);
    const Prediction cpu = evaluate_single(
        "cpu_nep3",
        model_path,
        test_case.positions,
        test_case.type_cycle);
    const Prediction cuda = evaluate_single(
        "cuda",
        model_path,
        test_case.positions,
        test_case.type_cycle);
    if (!compare_prediction(test_case, cpu, cuda)) {
      return EXIT_FAILURE;
    }

    std::vector<double> batch_positions = test_case.positions;
    const std::vector<double> second_positions = {
        0.30, 0.20, 0.10,
        2.20, 0.20, 0.20,
        4.10, 0.60, 0.30,
    };
    batch_positions.insert(
        batch_positions.end(),
        second_positions.begin(),
        second_positions.end());
    std::vector<double> batch_boxes = triclinic_box();
    const std::vector<double> second_box = orthorhombic_box();
    batch_boxes.insert(batch_boxes.end(), second_box.begin(), second_box.end());
    const Prediction cpu_batch = evaluate_structures_independently(
        "cpu_nep3",
        model_path,
        {3, 3},
        {0, 3},
        batch_positions,
        batch_boxes,
        {1, 1, 1, 1, 1, 1},
        test_case.type_cycle);
    const Prediction cuda_batch = evaluate_batch(
        "cuda",
        model_path,
        {3, 3},
        {0, 3},
        batch_positions,
        batch_boxes,
        {1, 1, 1, 1, 1, 1},
        test_case.type_cycle);
    if (!compare_prediction(test_case, cpu_batch, cuda_batch)) {
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
