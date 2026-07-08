#include "nep_adapters/api.h"
#include "nep_adapters/engines/cuda.hpp"

#if defined(NEP_ADAPTERS_BENCH_DEVICE_PIPELINE)
#include "ann_energy.hpp"
#include "angular_basis_cache.hpp"
#include "angular_descriptor.hpp"
#include "angular_force.hpp"
#include "batch_output.hpp"
#include "device_model.hpp"
#include "device_staging.hpp"
#include "device_workspace.hpp"
#include "internal_neighbor_builder.hpp"
#include "model_parameters.hpp"
#include "pair_geometry_cache.hpp"
#include "radial_basis_cache.hpp"
#include "radial_descriptor.hpp"
#include "radial_force.hpp"
#include "workspace_plan.hpp"
#include "zbl_force.hpp"

#include <cuda_runtime.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class ModelKind {
  radial,
  angular,
  q134,
};

struct Options {
  ModelKind model = ModelKind::radial;
  int structures = 64;
  int atoms_per_structure = 64;
  int warmup = 3;
  int iterations = 20;
  bool compare_single_loop = true;
  bool device_pipeline = false;
};

int parse_int_arg(const char* text, const char* name) {
  const int value = std::atoi(text);
  if (value <= 0) {
    std::cerr << "Invalid " << name << ": " << text << '\n';
    std::exit(EXIT_FAILURE);
  }
  return value;
}

ModelKind parse_model_kind(const std::string& text) {
  if (text == "radial") {
    return ModelKind::radial;
  }
  if (text == "angular") {
    return ModelKind::angular;
  }
  if (text == "q134") {
    return ModelKind::q134;
  }
  std::cerr << "Invalid --model: " << text << '\n';
  std::exit(EXIT_FAILURE);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      options.model = parse_model_kind(argv[++i]);
    } else if (arg == "--structures" && i + 1 < argc) {
      options.structures = parse_int_arg(argv[++i], "--structures");
    } else if (arg == "--atoms" && i + 1 < argc) {
      options.atoms_per_structure = parse_int_arg(argv[++i], "--atoms");
    } else if (arg == "--warmup" && i + 1 < argc) {
      options.warmup = parse_int_arg(argv[++i], "--warmup");
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = parse_int_arg(argv[++i], "--iterations");
    } else if (arg == "--batch-only") {
      options.compare_single_loop = false;
    } else if (arg == "--device-pipeline") {
      options.device_pipeline = true;
      options.compare_single_loop = false;
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--model radial|angular|q134]"
                << " [--structures N] [--atoms N] [--warmup N] [--iterations N]"
                << " [--batch-only] [--device-pipeline]\n";
      std::exit(EXIT_FAILURE);
    }
  }
  return options;
}

const char* model_kind_name(ModelKind kind) {
  switch (kind) {
    case ModelKind::radial:
      return "radial";
    case ModelKind::angular:
      return "angular";
    case ModelKind::q134:
      return "q134";
  }
  return "unknown";
}

std::string write_radial_model() {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "nep_adapters_cuda_batch_bench.nep";
  std::ofstream out(path);
  out
      << "nep4 1 C\n"
      << "cutoff 5 1 8 1\n"
      << "n_max 1 0\n"
      << "basis_size 2 0\n"
      << "l_max 0 0 0\n"
      << "ANN 2 0\n"
      << "0.20\n-0.10\n0.05\n0.15\n"
      << "0.01\n-0.02\n"
      << "0.30\n-0.25\n"
      << "0.04\n"
      << "0.70\n-0.15\n0.05\n-0.30\n0.20\n-0.10\n0.0\n"
      << "0.80\n1.10\n";
  return path.string();
}

std::string write_angular_model() {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "nep_adapters_cuda_batch_angular.nep";
  std::ofstream out(path);
  out << "nep4 1 C\n"
      << "cutoff 0.5 4 1 8\n"
      << "n_max 0 0\n"
      << "basis_size 0 1\n"
      << "l_max 4 0 0\n"
      << "ANN 3 0\n";
  const double values[] = {
      0.05, -0.04, 0.03, 0.02, -0.01,
      -0.02, 0.06, 0.01, 0.04, -0.03,
      0.04, 0.02, -0.05, 0.03, 0.01,
      0.01, -0.02, 0.03,
      0.20, -0.15, 0.10,
      0.0,
      0.0, 0.80, -0.30,
      1.0, 0.5, 0.4, 0.3, 0.2,
  };
  for (double value : values) {
    out << value << "\n";
  }
  return path.string();
}

std::string write_q134_model() {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "nep_adapters_cuda_batch_q134.nep";
  std::ofstream out(path);
  out << "nep4 1 C\n"
      << "cutoff 0.5 4 1 8\n"
      << "n_max 0 0\n"
      << "basis_size 0 1\n"
      << "l_max 4 0 0 0 0 0 1\n"
      << "ANN 3 0\n";
  const double values[] = {
      0.03, -0.02, 0.04, -0.01, 0.05, -0.03,
      -0.01, 0.05, -0.03, 0.02, -0.04, 0.01,
      0.04, 0.01, 0.03, -0.05, 0.02, -0.02,
      0.01, -0.012, 0.018,
      0.14, -0.11, 0.09,
      0.0,
      0.0, 0.70, -0.20,
      1.0, 0.44, 0.33, 0.28, 0.21, 0.17,
  };
  for (double value : values) {
    out << value << "\n";
  }
  return path.string();
}

std::string write_model(ModelKind kind) {
  switch (kind) {
    case ModelKind::radial:
      return write_radial_model();
    case ModelKind::angular:
      return write_angular_model();
    case ModelKind::q134:
      return write_q134_model();
  }
  return write_radial_model();
}

struct BatchStorage {
  std::vector<int> atom_counts;
  std::vector<int> atom_offsets;
  std::vector<int> types;
  std::vector<double> positions;
  std::vector<double> boxes;
  std::vector<int> pbc;
  std::vector<double> energy;
  std::vector<double> potential;
  std::vector<double> forces;
  std::vector<double> virial;
  std::vector<double> per_atom_virial;
};

int grid_width(int atoms_per_structure) {
  int width = 1;
  while (width * width < atoms_per_structure) {
    ++width;
  }
  return width;
}

BatchStorage make_batch(const Options& options) {
  BatchStorage storage;
  const int total_atoms = options.structures * options.atoms_per_structure;
  storage.atom_counts.assign(static_cast<std::size_t>(options.structures),
                             options.atoms_per_structure);
  storage.atom_offsets.resize(static_cast<std::size_t>(options.structures));
  storage.types.assign(static_cast<std::size_t>(total_atoms), 0);
  storage.positions.resize(static_cast<std::size_t>(total_atoms) * 3);
  storage.boxes.resize(static_cast<std::size_t>(options.structures) * 9, 0.0);
  storage.pbc.assign(static_cast<std::size_t>(options.structures) * 3, 0);
  storage.energy.assign(static_cast<std::size_t>(options.structures), 0.0);
  storage.potential.assign(static_cast<std::size_t>(total_atoms), 0.0);
  storage.forces.assign(static_cast<std::size_t>(total_atoms) * 3, 0.0);
  storage.virial.assign(static_cast<std::size_t>(options.structures) * 9, 0.0);
  storage.per_atom_virial.assign(static_cast<std::size_t>(total_atoms) * 9, 0.0);

  for (int structure = 0; structure < options.structures; ++structure) {
    const int atom_offset = structure * options.atoms_per_structure;
    storage.atom_offsets[static_cast<std::size_t>(structure)] = atom_offset;
    const int width = grid_width(options.atoms_per_structure);
    const double length =
        options.model == ModelKind::radial
            ? 4.0 * options.atoms_per_structure + 8.0
            : 2.4 * width + 8.0;
    double* box = storage.boxes.data() + 9 * static_cast<std::size_t>(structure);
    box[0] = length;
    box[4] = options.model == ModelKind::radial ? 12.0 : length;
    box[8] = 12.0;
    for (int atom = 0; atom < options.atoms_per_structure; ++atom) {
      const std::size_t global = static_cast<std::size_t>(atom_offset + atom);
      if (options.model == ModelKind::radial) {
        storage.positions[3 * global + 0] = 4.0 * atom + 2.0;
        storage.positions[3 * global + 1] = 1.5 + 0.1 * (atom % 3);
        storage.positions[3 * global + 2] = 1.5 + 0.1 * (atom % 5);
      } else {
        const int ix = atom % width;
        const int iy = atom / width;
        storage.positions[3 * global + 0] = 2.2 * ix + 2.0;
        storage.positions[3 * global + 1] = 2.4 * iy + 2.0;
        storage.positions[3 * global + 2] = 1.5 + 0.12 * ((atom + structure) % 3);
      }
    }
  }
  return storage;
}

NepaStructureBatch make_view(
    const BatchStorage& storage,
    int structures,
    int total_atoms) {
  NepaStructureBatch batch{};
  batch.num_structures = structures;
  batch.total_atoms = total_atoms;
  batch.atom_counts = storage.atom_counts.data();
  batch.atom_offsets = storage.atom_offsets.data();
  batch.types = storage.types.data();
  batch.positions_aos3 = storage.positions.data();
  batch.boxes_row_major9 = storage.boxes.data();
  batch.pbc_flags3 = storage.pbc.data();
  return batch;
}

NepaFindForceResult make_result(BatchStorage& storage) {
  NepaFindForceResult result{};
  result.energy_per_structure = storage.energy.data();
  result.potential_per_atom = storage.potential.data();
  result.forces_aos3 = storage.forces.data();
  result.virials_row_major9 = storage.virial.data();
  result.virials_per_atom_row_major9 = storage.per_atom_virial.data();
  return result;
}

void run_or_die(NepaModel* model, const NepaStructureBatch& batch, NepaFindForceResult& result) {
  const NepaStatus status = nepa_find_force_batch(model, &batch, &result);
  if (status != NEPA_STATUS_OK) {
    std::cerr << "nepa_find_force_batch failed: " << status << '\n';
    std::exit(EXIT_FAILURE);
  }
}

double elapsed_ms(std::chrono::steady_clock::time_point begin,
                  std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

#if defined(NEP_ADAPTERS_BENCH_DEVICE_PIPELINE)
void check_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(action) + ": " + cudaGetErrorString(status));
  }
}

bool batch_boxes_are_orthorhombic(const NepaStructureBatch& batch) {
  constexpr double kTolerance = 1.0e-14;
  for (int structure = 0; structure < batch.num_structures; ++structure) {
    const double* box = batch.boxes_row_major9 + 9 * static_cast<std::size_t>(structure);
    if (!std::isfinite(box[0]) || !std::isfinite(box[4]) ||
        !std::isfinite(box[8]) || box[0] <= 0.0 || box[4] <= 0.0 ||
        box[8] <= 0.0) {
      return false;
    }
    for (int component = 0; component < 9; ++component) {
      if (component == 0 || component == 4 || component == 8) {
        continue;
      }
      if (std::abs(box[component]) > kTolerance) {
        return false;
      }
    }
  }
  return true;
}

void clear_device_outputs(nep_adapters::cuda_backend::DeviceWorkspace& workspace) {
  const nep_adapters::cuda_backend::DeviceWorkspaceView view = workspace.view();
  check_cuda(
      cudaMemset(view.potential, 0, view.atom_capacity * sizeof(double)),
      "clear potential");
  check_cuda(
      cudaMemset(view.force_soa3, 0, 3 * view.atom_capacity * sizeof(double)),
      "clear force");
  check_cuda(
      cudaMemset(view.virial_soa9, 0, 9 * view.atom_capacity * sizeof(double)),
      "clear virial");
}

void run_device_pipeline(
    const nep_adapters::cuda_backend::ModelProtocol& protocol,
    int structure_count,
    int atom_count,
    bool orthorhombic_fast_path,
    const nep_adapters::cuda_backend::DeviceModel& model,
    nep_adapters::cuda_backend::DeviceWorkspace& workspace) {
  clear_device_outputs(workspace);
  nep_adapters::cuda_backend::build_internal_neighbors_batched(
      protocol,
      structure_count,
      atom_count,
      workspace,
      orthorhombic_fast_path);
  if (!orthorhombic_fast_path) {
    nep_adapters::cuda_backend::build_pair_geometry_cache_batched(
        protocol,
        atom_count,
        workspace);
  }
  nep_adapters::cuda_backend::build_radial_basis_cache_on_device(
      protocol,
      atom_count,
      workspace);
  nep_adapters::cuda_backend::build_radial_descriptors_on_device(
      protocol,
      atom_count,
      model,
      workspace);
  if (protocol.body_channels.l_max_3body > 0) {
    nep_adapters::cuda_backend::build_angular_basis_cache_on_device(
        protocol,
        atom_count,
        workspace);
    nep_adapters::cuda_backend::build_angular_descriptors_on_device(
        protocol,
        atom_count,
        model,
        workspace);
  }
  nep_adapters::cuda_backend::evaluate_ann_energy_on_device(
      protocol,
      atom_count,
      model,
      workspace);
  nep_adapters::cuda_backend::accumulate_radial_forces_batched(
      protocol,
      atom_count,
      model,
      workspace);
  if (protocol.body_channels.l_max_3body > 0) {
    nep_adapters::cuda_backend::accumulate_l2_angular_forces_batched(
        protocol,
        atom_count,
        model,
        workspace);
  }
  if (protocol.has_zbl) {
    nep_adapters::cuda_backend::accumulate_zbl_forces_batched(
        protocol,
        atom_count,
        model,
        workspace);
  }
  nep_adapters::cuda_backend::prepare_batched_outputs(
      structure_count,
      atom_count,
      workspace);
}

double time_device_pipeline(
    const Options& options,
    const std::string& model_path,
    const NepaStructureBatch& batch,
    std::size_t& workspace_bytes) {
  const nep_adapters::cuda_backend::HostModelParameters host =
      nep_adapters::cuda_backend::load_host_model_parameters(model_path);
  nep_adapters::cuda_backend::DeviceModel model(host);
  nep_adapters::cuda_backend::DeviceWorkspace workspace(
      nep_adapters::cuda_backend::make_internal_neighbor_workspace_plan(
          host.protocol,
          static_cast<std::size_t>(batch.total_atoms),
          static_cast<std::size_t>(batch.num_structures)));
  workspace_bytes = workspace.summary().total_bytes;
  nep_adapters::cuda_backend::stage_batch_on_device(batch, workspace);
  const bool orthorhombic_fast_path = batch_boxes_are_orthorhombic(batch);

  for (int i = 0; i < options.warmup; ++i) {
    run_device_pipeline(
        host.protocol,
        batch.num_structures,
        batch.total_atoms,
        orthorhombic_fast_path,
        model,
        workspace);
  }

  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check_cuda(cudaEventCreate(&begin), "create begin event");
  check_cuda(cudaEventCreate(&end), "create end event");
  check_cuda(cudaEventRecord(begin), "record begin event");
  for (int i = 0; i < options.iterations; ++i) {
    run_device_pipeline(
        host.protocol,
        batch.num_structures,
        batch.total_atoms,
        orthorhombic_fast_path,
        model,
        workspace);
  }
  check_cuda(cudaEventRecord(end), "record end event");
  check_cuda(cudaEventSynchronize(end), "synchronize end event");
  float milliseconds = 0.0f;
  check_cuda(cudaEventElapsedTime(&milliseconds, begin, end), "read elapsed time");
  check_cuda(cudaEventDestroy(begin), "destroy begin event");
  check_cuda(cudaEventDestroy(end), "destroy end event");
  return static_cast<double>(milliseconds) / options.iterations;
}
#endif

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if (!nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }

  const std::string model_path = write_model(options.model);
  BatchStorage storage = make_batch(options);
  const int total_atoms = options.structures * options.atoms_per_structure;
  NepaStructureBatch batch = make_view(storage, options.structures, total_atoms);
  NepaFindForceResult result = make_result(storage);

#if defined(NEP_ADAPTERS_BENCH_DEVICE_PIPELINE)
  if (options.device_pipeline) {
    try {
      std::size_t workspace_bytes = 0;
      const double device_ms =
          time_device_pipeline(options, model_path, batch, workspace_bytes);
      std::cout << "model=" << model_kind_name(options.model) << '\n'
                << "structures=" << options.structures
                << " atoms_per_structure=" << options.atoms_per_structure
                << " total_atoms=" << total_atoms << '\n'
                << "device_pipeline_ms=" << device_ms << '\n'
                << "workspace_mb="
                << static_cast<double>(workspace_bytes) / (1024.0 * 1024.0)
                << '\n';
      return EXIT_SUCCESS;
    } catch (const std::exception& error) {
      std::cerr << "device pipeline benchmark failed: " << error.what() << '\n';
      return EXIT_FAILURE;
    }
  }
#else
  if (options.device_pipeline) {
    std::cerr << "--device-pipeline requires NEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME\n";
    return EXIT_FAILURE;
  }
#endif

  NepaModel* model = nullptr;
  if (nepa_load_model("cuda", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    std::cerr << "failed to load CUDA model\n";
    return EXIT_FAILURE;
  }

  for (int i = 0; i < options.warmup; ++i) {
    run_or_die(model, batch, result);
  }
  const auto batch_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < options.iterations; ++i) {
    run_or_die(model, batch, result);
  }
  const auto batch_end = std::chrono::steady_clock::now();

  int single_count = options.atoms_per_structure;
  int single_offset = 0;
  std::vector<double> single_energy(1, 0.0);
  std::vector<double> single_potential(static_cast<std::size_t>(single_count), 0.0);
  std::vector<double> single_forces(static_cast<std::size_t>(single_count) * 3, 0.0);
  std::vector<double> single_virial(9, 0.0);
  std::vector<double> single_per_atom_virial(
      static_cast<std::size_t>(single_count) * 9,
      0.0);
  NepaFindForceResult single_result{};
  single_result.energy_per_structure = single_energy.data();
  single_result.potential_per_atom = single_potential.data();
  single_result.forces_aos3 = single_forces.data();
  single_result.virials_row_major9 = single_virial.data();
  single_result.virials_per_atom_row_major9 = single_per_atom_virial.data();

  double loop_ms = 0.0;
  if (options.compare_single_loop) {
    const auto loop_begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      for (int structure = 0; structure < options.structures; ++structure) {
        NepaStructureBatch single = batch;
        single.num_structures = 1;
        single.total_atoms = options.atoms_per_structure;
        single.atom_counts = &single_count;
        single.atom_offsets = &single_offset;
        single.types = storage.types.data() + storage.atom_offsets[structure];
        single.positions_aos3 =
            storage.positions.data() + 3 * storage.atom_offsets[structure];
        single.boxes_row_major9 = storage.boxes.data() + 9 * structure;
        single.pbc_flags3 = storage.pbc.data() + 3 * structure;
        run_or_die(model, single, single_result);
      }
    }
    const auto loop_end = std::chrono::steady_clock::now();
    loop_ms = elapsed_ms(loop_begin, loop_end) / options.iterations;
  }

  nepa_free_model(model);

  const double batch_ms = elapsed_ms(batch_begin, batch_end) / options.iterations;
  std::cout << "model=" << model_kind_name(options.model) << '\n'
            << "structures=" << options.structures
            << " atoms_per_structure=" << options.atoms_per_structure
            << " total_atoms=" << total_atoms << '\n'
            << "batched_ms=" << batch_ms << '\n';
  if (options.compare_single_loop) {
    std::cout << "single_loop_ms=" << loop_ms << '\n'
              << "batched_vs_single_loop=" << loop_ms / batch_ms << '\n';
  }
  return EXIT_SUCCESS;
}
