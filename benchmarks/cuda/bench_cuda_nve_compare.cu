#include "nep_adapters/api.h"
#include "nep_adapters/engines/cuda.hpp"

#include "ann_energy.hpp"
#include "device_model.hpp"
#include "device_staging.hpp"
#include "device_workspace.hpp"
#include "internal_neighbor_builder.hpp"
#include "model_parameters.hpp"
#include "radial_basis_cache.hpp"
#include "radial_descriptor.hpp"
#include "radial_force.hpp"
#include "simulation_box.hpp"
#include "workspace_plan.hpp"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  int nx = 125;
  int ny = 125;
  int nz = 128;
  int warmup = 5;
  int iterations = 100;
  double spacing = 4.0;
  double dt = 0.001;
  bool integrate = false;
};

int parse_positive_int(const char* text, const char* name) {
  const int value = std::atoi(text);
  if (value <= 0) {
    std::cerr << "Invalid " << name << ": " << text << '\n';
    std::exit(EXIT_FAILURE);
  }
  return value;
}

double parse_positive_double(const char* text, const char* name) {
  const double value = std::atof(text);
  if (value <= 0.0) {
    std::cerr << "Invalid " << name << ": " << text << '\n';
    std::exit(EXIT_FAILURE);
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--nx" && i + 1 < argc) {
      options.nx = parse_positive_int(argv[++i], "--nx");
    } else if (arg == "--ny" && i + 1 < argc) {
      options.ny = parse_positive_int(argv[++i], "--ny");
    } else if (arg == "--nz" && i + 1 < argc) {
      options.nz = parse_positive_int(argv[++i], "--nz");
    } else if (arg == "--warmup" && i + 1 < argc) {
      options.warmup = parse_positive_int(argv[++i], "--warmup");
    } else if (arg == "--iterations" && i + 1 < argc) {
      options.iterations = parse_positive_int(argv[++i], "--iterations");
    } else if (arg == "--spacing" && i + 1 < argc) {
      options.spacing = parse_positive_double(argv[++i], "--spacing");
    } else if (arg == "--dt" && i + 1 < argc) {
      options.dt = parse_positive_double(argv[++i], "--dt");
    } else if (arg == "--mode" && i + 1 < argc) {
      const std::string mode = argv[++i];
      if (mode == "force") {
        options.integrate = false;
      } else if (mode == "nve") {
        options.integrate = true;
      } else {
        std::cerr << "Invalid --mode: " << mode << '\n';
        std::exit(EXIT_FAILURE);
      }
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--nx N] [--ny N] [--nz N]"
                << " [--warmup N] [--iterations N]"
                << " [--spacing A] [--dt T] [--mode force|nve]\n";
      std::exit(EXIT_FAILURE);
    }
  }
  return options;
}

void check_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(action) + ": " + cudaGetErrorString(status));
  }
}

std::string write_radial_model() {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "nep_adapters_cuda_nve_compare.nep";
  std::ofstream out(path);
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
    out << value << '\n';
  }
  return path.string();
}

int atom_count(const Options& options) {
  return options.nx * options.ny * options.nz;
}

std::array<double, 9> make_box(const Options& options) {
  return {
      options.spacing * options.nx, 0.0, 0.0,
      0.0, options.spacing * options.ny, 0.0,
      0.0, 0.0, options.spacing * options.nz,
  };
}

std::vector<double> make_positions_aos3(const Options& options) {
  const int total_atoms = atom_count(options);
  std::vector<double> positions(static_cast<std::size_t>(total_atoms) * 3);
  int atom = 0;
  for (int iz = 0; iz < options.nz; ++iz) {
    for (int iy = 0; iy < options.ny; ++iy) {
      for (int ix = 0; ix < options.nx; ++ix) {
        const std::size_t base = 3 * static_cast<std::size_t>(atom++);
        positions[base + 0] = options.spacing * (ix + 0.5);
        positions[base + 1] = options.spacing * (iy + 0.5);
        positions[base + 2] = options.spacing * (iz + 0.5);
      }
    }
  }
  return positions;
}

std::vector<double> make_velocities_soa3(const Options& options) {
  const int total_atoms = atom_count(options);
  std::vector<double> velocities(static_cast<std::size_t>(total_atoms) * 3);
  for (int atom = 0; atom < total_atoms; ++atom) {
    velocities[static_cast<std::size_t>(atom)] =
        1.0e-4 * ((atom % 7) - 3);
    velocities[static_cast<std::size_t>(total_atoms + atom)] =
        1.0e-4 * (((atom / 7) % 7) - 3);
    velocities[static_cast<std::size_t>(2 * total_atoms + atom)] =
        1.0e-4 * (((atom / 49) % 7) - 3);
  }
  return velocities;
}

nep_adapters::cuda_backend::SimulationBox make_simulation_box(
    const Options& options) {
  nep_adapters::cuda_backend::SimulationBox box{};
  const std::array<double, 9> matrix = make_box(options);
  for (int i = 0; i < 9; ++i) {
    box.frac_to_cart[i] = matrix[static_cast<std::size_t>(i)];
  }
  box.cart_to_frac[0] = 1.0 / matrix[0];
  box.cart_to_frac[4] = 1.0 / matrix[4];
  box.cart_to_frac[8] = 1.0 / matrix[8];
  box.pbc[0] = 1;
  box.pbc[1] = 1;
  box.pbc[2] = 1;
  return box;
}

NepaStructureBatch make_batch(
    const Options& options,
    const std::vector<int>& atom_counts,
    const std::vector<int>& atom_offsets,
    const std::vector<int>& types,
    const std::vector<double>& positions,
    const std::vector<double>& boxes,
    const std::vector<int>& pbc) {
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = atom_count(options);
  batch.atom_counts = atom_counts.data();
  batch.atom_offsets = atom_offsets.data();
  batch.types = types.data();
  batch.positions_aos3 = positions.data();
  batch.boxes_row_major9 = boxes.data();
  batch.pbc_flags3 = pbc.data();
  return batch;
}

__global__ void integrate_positions(
    int atom_count,
    int atom_stride,
    double dt,
    double lx,
    double ly,
    double lz,
    const double* __restrict__ forces,
    double* __restrict__ velocities,
    double* __restrict__ positions) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }

  const double inv_mass = 1.0 / 12.011;
  const double vx = velocities[atom] + dt * forces[atom] * inv_mass;
  const double vy =
      velocities[atom_stride + atom] + dt * forces[atom_stride + atom] * inv_mass;
  const double vz =
      velocities[2 * atom_stride + atom] +
      dt * forces[2 * atom_stride + atom] * inv_mass;
  double x = positions[atom] + dt * vx;
  double y = positions[atom_stride + atom] + dt * vy;
  double z = positions[2 * atom_stride + atom] + dt * vz;

  x -= floor(x / lx) * lx;
  y -= floor(y / ly) * ly;
  z -= floor(z / lz) * lz;

  velocities[atom] = vx;
  velocities[atom_stride + atom] = vy;
  velocities[2 * atom_stride + atom] = vz;
  positions[atom] = x;
  positions[atom_stride + atom] = y;
  positions[2 * atom_stride + atom] = z;
}

void clear_potential(nep_adapters::cuda_backend::DeviceWorkspace& workspace) {
  const nep_adapters::cuda_backend::DeviceWorkspaceView view = workspace.view();
  check_cuda(
      cudaMemset(view.potential, 0, view.atom_capacity * sizeof(double)),
      "clear potential");
}

void run_step(
    const nep_adapters::cuda_backend::ModelProtocol& protocol,
    const nep_adapters::cuda_backend::DeviceModel& model,
    const nep_adapters::cuda_backend::SimulationBox& box,
    const Options& options,
    double* velocities,
    nep_adapters::cuda_backend::DeviceWorkspace& workspace) {
  const int total_atoms = atom_count(options);
  clear_potential(workspace);
  nep_adapters::cuda_backend::build_internal_neighbors_on_device(
      protocol,
      total_atoms,
      box,
      workspace);
  nep_adapters::cuda_backend::build_radial_geometry_basis_cache_on_device(
      protocol,
      total_atoms,
      box,
      workspace);
  nep_adapters::cuda_backend::build_radial_descriptors_on_device(
      protocol,
      total_atoms,
      model,
      workspace);
  nep_adapters::cuda_backend::evaluate_ann_energy_on_device(
      protocol,
      total_atoms,
      model,
      workspace);
  nep_adapters::cuda_backend::accumulate_radial_forces_on_device(
      protocol,
      total_atoms,
      box,
      model,
      workspace,
      false);

  if (!options.integrate) {
    return;
  }

  const nep_adapters::cuda_backend::DeviceWorkspaceView view = workspace.view();
  constexpr int threads = 256;
  const int blocks = (total_atoms + threads - 1) / threads;
  integrate_positions<<<blocks, threads>>>(
      total_atoms,
      static_cast<int>(view.atom_capacity),
      options.dt,
      box.frac_to_cart[0],
      box.frac_to_cart[4],
      box.frac_to_cart[8],
      view.force_soa3,
      velocities,
      view.positions_soa3);
  check_cuda(cudaGetLastError(), "integrate positions");
}

double time_nve(
    const Options& options,
    const std::string& model_path,
    std::size_t& workspace_bytes) {
  const int total_atoms = atom_count(options);
  const std::vector<int> atom_counts = {total_atoms};
  const std::vector<int> atom_offsets = {0};
  const std::vector<int> types(static_cast<std::size_t>(total_atoms), 0);
  const std::vector<double> positions = make_positions_aos3(options);
  const std::array<double, 9> box_matrix = make_box(options);
  const std::vector<double> boxes(box_matrix.begin(), box_matrix.end());
  const std::vector<int> pbc = {1, 1, 1};
  const NepaStructureBatch batch =
      make_batch(options, atom_counts, atom_offsets, types, positions, boxes, pbc);

  const nep_adapters::cuda_backend::HostModelParameters host =
      nep_adapters::cuda_backend::load_host_model_parameters(model_path);
  nep_adapters::cuda_backend::DeviceModel model(host);
  nep_adapters::cuda_backend::DeviceWorkspace workspace(
      nep_adapters::cuda_backend::make_internal_neighbor_workspace_plan(
          host.protocol,
          static_cast<std::size_t>(total_atoms),
          1));
  workspace_bytes = workspace.summary().total_bytes;
  nep_adapters::cuda_backend::stage_batch_on_device(batch, workspace);

  std::vector<double> host_velocities = make_velocities_soa3(options);
  double* velocities = nullptr;
  check_cuda(
      cudaMalloc(
          reinterpret_cast<void**>(&velocities),
          host_velocities.size() * sizeof(double)),
      "allocate velocities");
  check_cuda(
      cudaMemcpy(
          velocities,
          host_velocities.data(),
          host_velocities.size() * sizeof(double),
          cudaMemcpyHostToDevice),
      "copy velocities");

  const nep_adapters::cuda_backend::SimulationBox box = make_simulation_box(options);
  for (int i = 0; i < options.warmup; ++i) {
    run_step(host.protocol, model, box, options, velocities, workspace);
  }
  check_cuda(cudaDeviceSynchronize(), "sync warmup");

  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  check_cuda(cudaEventCreate(&begin), "create begin event");
  check_cuda(cudaEventCreate(&end), "create end event");
  check_cuda(cudaEventRecord(begin), "record begin event");
  for (int i = 0; i < options.iterations; ++i) {
    run_step(host.protocol, model, box, options, velocities, workspace);
  }
  check_cuda(cudaEventRecord(end), "record end event");
  check_cuda(cudaEventSynchronize(end), "sync end event");

  float milliseconds = 0.0f;
  check_cuda(cudaEventElapsedTime(&milliseconds, begin, end), "elapsed time");
  check_cuda(cudaEventDestroy(begin), "destroy begin event");
  check_cuda(cudaEventDestroy(end), "destroy end event");
  cudaFree(velocities);
  return static_cast<double>(milliseconds) / options.iterations;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  if (!nep_adapters::register_cuda_engine()) {
    std::cerr << "failed to register CUDA engine\n";
    return EXIT_FAILURE;
  }

  try {
    const std::string model_path = write_radial_model();
    std::size_t workspace_bytes = 0;
    const double ms_per_step = time_nve(options, model_path, workspace_bytes);
    const int atoms = atom_count(options);
    const double matom_per_second =
        static_cast<double>(atoms) / ms_per_step / 1000.0;
    std::cout << "atoms=" << atoms << '\n'
              << "grid=" << options.nx << "x" << options.ny << "x"
              << options.nz << '\n'
              << "warmup=" << options.warmup
              << " iterations=" << options.iterations << '\n'
              << "mode=" << (options.integrate ? "nve" : "force") << '\n'
              << "spacing=" << options.spacing << " dt=" << options.dt << '\n'
              << "ms_per_step=" << ms_per_step << '\n'
              << "matom_per_s=" << matom_per_second << '\n'
              << "workspace_mb="
              << static_cast<double>(workspace_bytes) / (1024.0 * 1024.0)
              << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
