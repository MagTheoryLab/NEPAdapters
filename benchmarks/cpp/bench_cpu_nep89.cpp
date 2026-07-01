#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu_nep3.hpp"

#include "cpu_nep3_test_utils.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#if NEP_ADAPTERS_BENCH_OPENMP_ENABLED
#include <omp.h>
#endif

namespace {

bool run_find_force(
    NepaModel* model,
    const NepaStructureBatch& batch,
    NepaFindForceResult& result) {
  return nepa_find_force_batch(model, &batch, &result) == NEPA_STATUS_OK;
}

struct Replicate {
  int nx = 1;
  int ny = 1;
  int nz = 1;
};

Replicate parse_replicate(const std::string& text) {
  Replicate replicate;
  char x1 = '\0';
  char x2 = '\0';
  std::istringstream stream(text);
  stream >> replicate.nx >> x1 >> replicate.ny >> x2 >> replicate.nz;
  if (!stream || x1 != 'x' || x2 != 'x' || replicate.nx <= 0 ||
      replicate.ny <= 0 || replicate.nz <= 0) {
    std::cerr << "Invalid --replicate value: " << text << '\n';
    std::exit(EXIT_FAILURE);
  }
  return replicate;
}

cpu_nep3_test::Frame make_supercell(
    const cpu_nep3_test::Frame& frame,
    const Replicate& replicate) {
  const std::size_t atom_count = frame.types.size();
  const std::size_t image_count =
      static_cast<std::size_t>(replicate.nx) * replicate.ny * replicate.nz;
  cpu_nep3_test::Frame supercell;
  supercell.types.reserve(atom_count * image_count);
  supercell.positions_aos3.reserve(atom_count * image_count * 3);
  supercell.reference_energy = frame.reference_energy * image_count;

  const double a[] = {frame.box[0], frame.box[3], frame.box[6]};
  const double b[] = {frame.box[1], frame.box[4], frame.box[7]};
  const double c[] = {frame.box[2], frame.box[5], frame.box[8]};

  for (int ix = 0; ix < replicate.nx; ++ix) {
    for (int iy = 0; iy < replicate.ny; ++iy) {
      for (int iz = 0; iz < replicate.nz; ++iz) {
        const double shift[] = {
            ix * a[0] + iy * b[0] + iz * c[0],
            ix * a[1] + iy * b[1] + iz * c[1],
            ix * a[2] + iy * b[2] + iz * c[2],
        };
        for (std::size_t atom = 0; atom < atom_count; ++atom) {
          supercell.types.push_back(frame.types[atom]);
          supercell.positions_aos3.push_back(
              frame.positions_aos3[3 * atom + 0] + shift[0]);
          supercell.positions_aos3.push_back(
              frame.positions_aos3[3 * atom + 1] + shift[1]);
          supercell.positions_aos3.push_back(
              frame.positions_aos3[3 * atom + 2] + shift[2]);
        }
      }
    }
  }

  supercell.box[0] = frame.box[0] * replicate.nx;
  supercell.box[3] = frame.box[3] * replicate.nx;
  supercell.box[6] = frame.box[6] * replicate.nx;
  supercell.box[1] = frame.box[1] * replicate.ny;
  supercell.box[4] = frame.box[4] * replicate.ny;
  supercell.box[7] = frame.box[7] * replicate.ny;
  supercell.box[2] = frame.box[2] * replicate.nz;
  supercell.box[5] = frame.box[5] * replicate.nz;
  supercell.box[8] = frame.box[8] * replicate.nz;
  return supercell;
}

}  // namespace

int main(int argc, char** argv) {
  int iterations = 10;
  int warmup = 1;
  Replicate replicate;

  for (int arg = 1; arg < argc; ++arg) {
    if (std::strcmp(argv[arg], "--iterations") == 0 && arg + 1 < argc) {
      iterations = std::atoi(argv[++arg]);
    } else if (std::strcmp(argv[arg], "--warmup") == 0 && arg + 1 < argc) {
      warmup = std::atoi(argv[++arg]);
    } else if (std::strcmp(argv[arg], "--replicate") == 0 && arg + 1 < argc) {
      replicate = parse_replicate(argv[++arg]);
    } else if (arg == 1 && argv[arg][0] != '-') {
      // Backward-compatible positional iteration count.
      iterations = std::atoi(argv[arg]);
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--iterations N] [--warmup N] [--replicate NxMxK]\n";
      return EXIT_FAILURE;
    }
  }

  if (iterations <= 0 || warmup < 0) {
    return EXIT_FAILURE;
  }

  const std::string model_path = NEP_ADAPTERS_NEP89_MODEL_PATH;
  const std::string xyz_path = NEP_ADAPTERS_NEP89_XYZ_PATH;

  if (!nep_adapters::register_cpu_nep3_engine()) {
    return EXIT_FAILURE;
  }

  const auto type_map = cpu_nep3_test::read_type_map(model_path);
  cpu_nep3_test::Frame frame =
      cpu_nep3_test::read_first_frame(xyz_path, type_map);
  frame = make_supercell(frame, replicate);

  const std::int32_t atom_count =
      static_cast<std::int32_t>(frame.types.size());

  NepaModel* model = nullptr;
  if (nepa_load_model("cpu_nep3", model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }

  std::int32_t atom_counts[] = {atom_count};
  std::int32_t atom_offsets[] = {0};
  std::int32_t pbc[] = {1, 1, 1};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = atom_count;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = frame.types.data();
  batch.positions_aos3 = frame.positions_aos3.data();
  batch.boxes_row_major9 = frame.box;
  batch.pbc_flags3 = pbc;

  double energy[] = {0.0};
  std::vector<double> forces(static_cast<std::size_t>(atom_count) * 3, 0.0);
  double virial[9] = {};

  NepaFindForceResult result{};
  result.energy_per_structure = energy;
  result.forces_aos3 = forces.data();
  result.virials_row_major9 = virial;

  for (int i = 0; i < warmup; ++i) {
    if (!run_find_force(model, batch, result)) {
      nepa_free_model(model);
      return EXIT_FAILURE;
    }
  }

  const auto started = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    if (!run_find_force(model, batch, result)) {
      nepa_free_model(model);
      return EXIT_FAILURE;
    }
  }
  const auto finished = std::chrono::steady_clock::now();

  const double force_l1 = std::accumulate(
      forces.begin(),
      forces.end(),
      0.0,
      [](double sum, double value) { return sum + std::abs(value); });
  if (!std::isfinite(energy[0]) || !cpu_nep3_test::all_finite(forces) ||
      force_l1 <= 0.0) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  nepa_free_model(model);

  const int total_iterations = iterations;
  const double seconds =
      std::chrono::duration<double>(finished - started).count();
  const double evals_per_second = static_cast<double>(total_iterations) / seconds;
  const double atom_steps_per_second =
      static_cast<double>(total_iterations) * atom_count / seconds;
#if NEP_ADAPTERS_BENCH_OPENMP_ENABLED
  const int omp_threads = omp_get_max_threads();
#else
  const int omp_threads = 1;
#endif

  std::cout << "{\"benchmark\":\"cpu_nep3_nep89_find_force_batch\","
            << "\"engine\":\"cpu_nep3\","
            << "\"model\":\"nep89\","
            << "\"openmp_enabled\":" << NEP_ADAPTERS_BENCH_OPENMP_ENABLED << ','
            << "\"omp_threads\":" << omp_threads << ','
            << "\"replicate\":\"" << replicate.nx << 'x' << replicate.ny
            << 'x' << replicate.nz << "\","
            << "\"atoms\":" << atom_count << ','
            << "\"warmup\":" << warmup << ','
            << "\"iterations\":" << iterations << ','
            << "\"total_iterations\":" << total_iterations << ','
            << "\"seconds\":" << seconds << ','
            << "\"evals_per_second\":" << evals_per_second << ','
            << "\"atom_steps_per_second\":" << atom_steps_per_second
            << "}\n";

  return EXIT_SUCCESS;
}
