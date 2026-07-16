#include "nep_adapters/api.h"
#include "nep_adapters/engines/cpu.hpp"
#if defined(NEP_ADAPTERS_FP64_COMPARE_CUDA)
#include "nep_adapters/engines/cuda.hpp"
#include <cuda_runtime_api.h>
#endif

#include "cpu_engine_adapter.hpp"
#include "cpu_test_utils.hpp"
#include "nep.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP) || defined(USE_TABLE_FOR_RADIAL_FUNCTIONS) || \
    defined(NEP_ADAPTERS_CPU_USE_CBLAS)
#error "The FP64 oracle must use the scalar, untabulated CPU path"
#endif

namespace {

struct CaseData {
  std::string name;
  std::string model_path;
  std::vector<std::int32_t> types;
  std::vector<double> positions;
  std::vector<double> spins;
  std::array<double, 9> box{};
  std::array<std::int32_t, 3> pbc{};

  bool is_spin() const { return !spins.empty(); }
  int atom_count() const { return static_cast<int>(types.size()); }
};

struct Prediction {
  double energy = 0.0;
  std::vector<double> potential;
  std::vector<double> force;
  std::vector<double> virial;
  std::vector<double> atom_virial;
  std::vector<double> mforce;
  std::vector<double> tau;
  std::vector<double> descriptor;
};

struct LammpsPrediction {
  double energy = 0.0;
  std::vector<double> virial6;
  std::vector<double> potential;
  std::vector<double> force;
  std::vector<double> mforce;
  std::vector<double> atom_virial9;
};

struct Budget {
  double atol = 0.0;
  double rtol = 0.0;
};

struct Budgets {
  Budget energy_per_atom;
  Budget potential;
  Budget force;
  Budget virial;
  Budget atom_virial;
  Budget mforce;
  Budget tau;
  Budget descriptor;
  Budget radial_basis_sum;
};

struct ErrorStats {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double rms = 0.0;
  std::uint64_t max_float_ulp = 0;
  std::size_t max_index = 0;
  bool finite = true;
};

void require_status(NepaStatus status, const std::string& operation) {
  if (status != NEPA_STATUS_OK) {
    throw std::runtime_error(
        operation + " failed with status " + std::to_string(status));
  }
}

std::uint32_t ordered_float_bits(float value) {
  if (value == 0.0f) {
    return 0x80000000u;
  }
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

std::uint64_t float_ulp_distance(double lhs, double rhs) {
  const float a = static_cast<float>(lhs);
  const float b = static_cast<float>(rhs);
  if (!std::isfinite(a) || !std::isfinite(b)) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  const std::uint32_t oa = ordered_float_bits(a);
  const std::uint32_t ob = ordered_float_bits(b);
  return oa >= ob ? static_cast<std::uint64_t>(oa - ob)
                  : static_cast<std::uint64_t>(ob - oa);
}

ErrorStats error_stats(
    const std::vector<double>& candidate,
    const std::vector<double>& reference) {
  ErrorStats out;
  if (candidate.size() != reference.size() || candidate.empty()) {
    out.finite = false;
    return out;
  }
  long double sum_sq = 0.0L;
  for (std::size_t i = 0; i < candidate.size(); ++i) {
    if (!std::isfinite(candidate[i]) || !std::isfinite(reference[i])) {
      out.finite = false;
      continue;
    }
    const double abs_error = std::abs(candidate[i] - reference[i]);
    const double denominator = std::max(std::abs(reference[i]), 1.0e-12);
    const double rel_error = abs_error / denominator;
    const std::uint64_t ulp = float_ulp_distance(candidate[i], reference[i]);
    if (abs_error > out.max_abs) {
      out.max_abs = abs_error;
      out.max_index = i;
    }
    out.max_rel = std::max(out.max_rel, rel_error);
    out.max_float_ulp = std::max(out.max_float_ulp, ulp);
    sum_sq += static_cast<long double>(abs_error) * abs_error;
  }
  out.rms = std::sqrt(static_cast<double>(sum_sq / candidate.size()));
  return out;
}

bool within_budget(
    const std::vector<double>& candidate,
    const std::vector<double>& reference,
    const Budget& budget) {
  if (candidate.size() != reference.size()) {
    return false;
  }
  for (std::size_t i = 0; i < candidate.size(); ++i) {
    if (!std::isfinite(candidate[i]) || !std::isfinite(reference[i])) {
      return false;
    }
    const double limit = budget.atol + budget.rtol * std::abs(reference[i]);
    if (std::abs(candidate[i] - reference[i]) > limit) {
      return false;
    }
  }
  return true;
}

bool report_field(
    const std::string& backend,
    const std::string& case_name,
    const std::string& field,
    const std::vector<double>& candidate,
    const std::vector<double>& reference,
    const Budget& budget) {
  const ErrorStats stats = error_stats(candidate, reference);
  const bool ok = stats.finite && within_budget(candidate, reference, budget);
  std::cout << std::scientific << std::setprecision(9)
            << "FP64_ACCURACY backend=" << backend
            << " case=" << case_name
            << " field=" << field
            << " count=" << candidate.size()
            << " max_abs=" << stats.max_abs
            << " max_rel=" << stats.max_rel
            << " rms=" << stats.rms
            << " max_float_ulp=" << stats.max_float_ulp
            << " max_index=" << stats.max_index
            << " candidate_at_max="
            << (stats.max_index < candidate.size()
                    ? candidate[stats.max_index]
                    : std::numeric_limits<double>::quiet_NaN())
            << " reference_at_max="
            << (stats.max_index < reference.size()
                    ? reference[stats.max_index]
                    : std::numeric_limits<double>::quiet_NaN())
            << " status=" << (ok ? "pass" : "fail") << '\n';
  return ok;
}

double max_abs_diff(
    const std::vector<double>& lhs,
    const std::vector<double>& rhs) {
  if (lhs.size() != rhs.size()) {
    return INFINITY;
  }
  double out = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    out = std::max(out, std::abs(lhs[i] - rhs[i]));
  }
  return out;
}

std::vector<double> read_reference_vector(
    const std::string& path,
    const std::string& wanted) {
  std::ifstream input(path);
  std::string label;
  std::size_t count = 0;
  while (input >> label >> count) {
    std::vector<double> values(count, 0.0);
    for (double& value : values) {
      input >> value;
    }
    if (label == wanted) {
      return values;
    }
  }
  throw std::runtime_error("missing FP64 reference block: " + wanted);
}

int read_radial_descriptor_dim(const std::string& model_path) {
  std::ifstream input(model_path);
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream tokens(line);
    std::string label;
    int n_max_radial = -1;
    if (tokens >> label && label == "n_max" &&
        tokens >> n_max_radial && n_max_radial >= 0) {
      return n_max_radial + 1;
    }
  }
  throw std::runtime_error("missing radial n_max in model: " + model_path);
}

std::vector<double> radial_descriptor_channels(
    const std::vector<double>& descriptors,
    int atom_count,
    int radial_dim) {
  if (atom_count <= 0 || radial_dim <= 0 ||
      descriptors.size() % static_cast<std::size_t>(atom_count) != 0) {
    throw std::runtime_error("invalid descriptor shape for radial split");
  }
  const std::size_t descriptor_dim =
      descriptors.size() / static_cast<std::size_t>(atom_count);
  if (static_cast<std::size_t>(radial_dim) > descriptor_dim) {
    throw std::runtime_error("radial descriptor dimension exceeds total dimension");
  }
  std::vector<double> radial;
  radial.reserve(static_cast<std::size_t>(atom_count) * radial_dim);
  for (int atom = 0; atom < atom_count; ++atom) {
    const std::size_t offset = static_cast<std::size_t>(atom) * descriptor_dim;
    radial.insert(
        radial.end(),
        descriptors.begin() + offset,
        descriptors.begin() + offset + radial_dim);
  }
  return radial;
}

class OracleRunner {
 public:
  explicit OracleRunner(const std::string& model_path) : model_(model_path) {}

  NepaStatus model_info(NepaModelInfo& info) { return model_.model_info(info); }
  NepaStatus find_force_batch(
      const NepaStructureBatch& batch,
      NepaFindForceResult& result) {
    return model_.find_force_batch(batch, result);
  }
  NepaStatus find_descriptors(
      const NepaStructureBatch& batch,
      NepaFindDescriptorResult& result) {
    return model_.find_descriptors(batch, result);
  }
  NepaStatus find_force_lammps_neighbors(
      const NepaLammpsNeighborInput& input,
      NepaLammpsNeighborResult& result) {
    return model_.find_force_lammps_neighbors(input, result);
  }

 private:
  nep_adapters::CpuModel<NEP> model_;
};

class ApiRunner {
 public:
  ApiRunner(const std::string& backend, const std::string& model_path)
      : backend_(backend) {
    const NepaStatus status = nepa_load_model(
        backend.c_str(), model_path.c_str(), &model_);
    if (status != NEPA_STATUS_OK || model_ == nullptr) {
      throw std::runtime_error(
          "failed to load backend " + backend + " status=" +
          std::to_string(status));
    }
  }

  ApiRunner(const ApiRunner&) = delete;
  ApiRunner& operator=(const ApiRunner&) = delete;

  ~ApiRunner() {
    if (model_ != nullptr) {
      nepa_free_model(model_);
    }
  }

  NepaStatus model_info(NepaModelInfo& info) {
    return nepa_model_info(model_, &info);
  }
  NepaStatus find_force_batch(
      const NepaStructureBatch& batch,
      NepaFindForceResult& result) {
    return nepa_find_force_batch(model_, &batch, &result);
  }
  NepaStatus find_descriptors(
      const NepaStructureBatch& batch,
      NepaFindDescriptorResult& result) {
    return nepa_find_descriptors(model_, &batch, &result);
  }
  NepaStatus find_force_lammps_neighbors(
      const NepaLammpsNeighborInput& input,
      NepaLammpsNeighborResult& result) {
    return nepa_find_force_lammps_neighbors(model_, &input, &result);
  }
#if defined(NEP_ADAPTERS_FP64_COMPARE_CUDA)
  NepaStatus find_force_lammps_device_neighbors(
      const NepaLammpsDeviceNeighborInput& input,
      NepaLammpsDeviceNeighborResult& result) {
    return nepa_find_force_lammps_device_neighbors(model_, &input, &result);
  }
#endif

 private:
  std::string backend_;
  NepaModel* model_ = nullptr;
};

template <typename Runner>
Prediction evaluate_batch(
    Runner& runner,
    const CaseData& test_case,
    bool include_descriptors = true) {
  NepaModelInfo info{};
  require_status(runner.model_info(info), "model_info");
  const int atom_count = test_case.atom_count();
  if (atom_count <= 0 || test_case.positions.size() !=
                             static_cast<std::size_t>(atom_count) * 3 ||
      (test_case.is_spin() && test_case.spins.size() !=
                                  static_cast<std::size_t>(atom_count) * 3)) {
    throw std::runtime_error("invalid FP64 oracle case shape");
  }

  const std::int32_t atom_counts[] = {atom_count};
  const std::int32_t atom_offsets[] = {0};
  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = atom_count;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = test_case.types.data();
  batch.positions_aos3 = test_case.positions.data();
  batch.spins_aos3 = test_case.is_spin() ? test_case.spins.data() : nullptr;
  batch.boxes_row_major9 = test_case.box.data();
  batch.pbc_flags3 = test_case.pbc.data();

  Prediction out;
  out.potential.assign(atom_count, 0.0);
  out.force.assign(static_cast<std::size_t>(atom_count) * 3, 0.0);
  out.virial.assign(9, 0.0);
  out.atom_virial.assign(static_cast<std::size_t>(atom_count) * 9, 0.0);
  if (test_case.is_spin()) {
    out.mforce.assign(static_cast<std::size_t>(atom_count) * 3, 0.0);
    out.tau.assign(static_cast<std::size_t>(atom_count) * 3, 0.0);
  }

  NepaFindForceResult result{};
  result.energy_per_structure = &out.energy;
  result.potential_per_atom = out.potential.data();
  result.forces_aos3 = out.force.data();
  result.virials_row_major9 = out.virial.data();
  result.virials_per_atom_row_major9 = out.atom_virial.data();
  result.mforces_aos3 = out.mforce.empty() ? nullptr : out.mforce.data();
  result.tau_aos3 = out.tau.empty() ? nullptr : out.tau.data();
  require_status(runner.find_force_batch(batch, result), "find_force_batch");

  if (include_descriptors) {
    out.descriptor.assign(
        static_cast<std::size_t>(atom_count) * info.descriptor_dim, 0.0);
    NepaFindDescriptorResult descriptor_result{};
    descriptor_result.descriptors = out.descriptor.data();
    require_status(
        runner.find_descriptors(batch, descriptor_result),
        "find_descriptors");
  }
  return out;
}

CaseData make_nonmag_fixture() {
  const std::string model_path = NEP_ADAPTERS_FP64_MODEL_PATH;
  const auto type_map = cpu_test::read_type_map(model_path);
  const cpu_test::Frame frame =
      cpu_test::read_first_frame(NEP_ADAPTERS_FP64_XYZ_PATH, type_map);
  CaseData out;
  out.name = "nonmag_fixture";
  out.model_path = model_path;
  out.types = frame.types;
  out.positions = frame.positions_aos3;
  std::copy(frame.box, frame.box + 9, out.box.begin());
  out.pbc = {1, 1, 1};
  return out;
}

CaseData make_dense_nonmag_case() {
  CaseData out;
  out.name = "nonmag_dense_neighbors";
  out.model_path = NEP_ADAPTERS_FP64_MODEL_PATH;
  constexpr int width = 6;
  constexpr double spacing = 3.28;
  constexpr int atom_count = width * width * width;
  out.types.reserve(atom_count);
  out.positions.reserve(static_cast<std::size_t>(atom_count) * 3);
  int atom = 0;
  for (int iz = 0; iz < width; ++iz) {
    for (int iy = 0; iy < width; ++iy) {
      for (int ix = 0; ix < width; ++ix, ++atom) {
        out.types.push_back((ix + iy + iz) & 1);
        out.positions.push_back(
            (ix + 0.5) * spacing + 0.035 * std::sin(0.73 * atom));
        out.positions.push_back(
            (iy + 0.5) * spacing + 0.031 * std::cos(0.51 * atom));
        out.positions.push_back(
            (iz + 0.5) * spacing + 0.029 * std::sin(0.37 * atom + 0.2));
      }
    }
  }
  const double length = width * spacing;
  out.box = {length, 0.0, 0.0, 0.0, length, 0.0, 0.0, 0.0, length};
  out.pbc = {1, 1, 1};
  return out;
}

CaseData make_spin_reference_case() {
  CaseData out;
  out.name = "spin_chiral_reference";
  out.model_path = NEP_ADAPTERS_FP64_SPIN_MODEL_PATH;
  out.types = {0, 0, 0, 0};
  out.positions = {
      0.2, 0.2, 0.2,
      3.7, 0.3, 0.2,
      0.4, 3.6, 0.5,
      1.8, 1.7, 3.5};
  out.spins = {
      1.0, 0.2, 0.0,
      0.4, -0.3, 0.7,
      -0.2, 0.8, 0.5,
      0.6, 0.1, -0.4};
  out.box = {4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0};
  out.pbc = {1, 1, 1};
  return out;
}

CaseData make_spin_finite_difference_case() {
  CaseData out = make_spin_reference_case();
  out.name = "spin_chiral_isolated_periodic";
  out.box = {16.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0, 16.0};
  out.pbc = {1, 1, 1};
  return out;
}

CaseData make_dense_spin_case() {
  CaseData out;
  out.name = "spin_chiral_dense_neighbors";
  out.model_path = NEP_ADAPTERS_FP64_SPIN_MODEL_PATH;
  // This shape has more than 64 radial spin neighbors while keeping the
  // shorter angular list below its capacity, so the compact primitive core
  // must carry moments across multiple 64-slot chunks.
  constexpr int width = 6;
  constexpr double spacing = 2.4;
  constexpr int atom_count = width * width * width;
  out.types.assign(atom_count, 0);
  out.positions.reserve(static_cast<std::size_t>(atom_count) * 3);
  out.spins.reserve(static_cast<std::size_t>(atom_count) * 3);
  int atom = 0;
  for (int iz = 0; iz < width; ++iz) {
    for (int iy = 0; iy < width; ++iy) {
      for (int ix = 0; ix < width; ++ix, ++atom) {
        out.positions.push_back(
            (ix + 0.5) * spacing + 0.031 * std::sin(0.43 * atom));
        out.positions.push_back(
            (iy + 0.5) * spacing + 0.027 * std::cos(0.37 * atom));
        out.positions.push_back(
            (iz + 0.5) * spacing + 0.029 * std::sin(0.29 * atom + 0.4));
        out.spins.push_back(0.7 * std::cos(0.19 * atom));
        out.spins.push_back(0.6 * std::sin(0.23 * atom + 0.2));
        out.spins.push_back(0.5 * std::cos(0.31 * atom - 0.1));
      }
    }
  }
  const double length = width * spacing;
  out.box = {length, 0.0, 0.0, 0.0, length, 0.0, 0.0, 0.0, length};
  out.pbc = {1, 1, 1};
  return out;
}

bool validate_nonmag_frozen_reference(
    const Prediction& oracle,
    const CaseData& test_case) {
  const auto type_map = cpu_test::read_type_map(test_case.model_path);
  const cpu_test::Frame frame =
      cpu_test::read_first_frame(NEP_ADAPTERS_FP64_XYZ_PATH, type_map);
  const cpu_test::Matrix descriptors =
      cpu_test::read_matrix(NEP_ADAPTERS_FP64_DESCRIPTOR_PATH);
  const std::vector<double> energy = {oracle.energy};
  const std::vector<double> reference_energy = {frame.reference_energy};
  const std::vector<double> reference_virial(
      frame.reference_virial_row_major9,
      frame.reference_virial_row_major9 + 9);
  const double energy_diff = max_abs_diff(energy, reference_energy);
  const double force_diff = max_abs_diff(oracle.force, frame.reference_forces_aos3);
  const double virial_diff = max_abs_diff(oracle.virial, reference_virial);
  const double descriptor_diff = max_abs_diff(oracle.descriptor, descriptors.values);
  const bool ok = frame.has_reference_forces && frame.has_reference_virial &&
                  descriptors.rows == static_cast<std::size_t>(test_case.atom_count()) &&
                  energy_diff <= 1.0e-10 && force_diff <= 1.0e-10 &&
                  virial_diff <= 1.0e-10 && descriptor_diff <= 1.0e-10;
  std::cout << std::scientific << std::setprecision(9)
            << "FP64_ORACLE_SELF_CHECK case=" << test_case.name
            << " energy=" << energy_diff
            << " force=" << force_diff
            << " virial=" << virial_diff
            << " descriptor=" << descriptor_diff
            << " status=" << (ok ? "pass" : "fail") << '\n';
  return ok;
}

bool validate_spin_frozen_reference(
    const Prediction& oracle,
    const CaseData& test_case) {
  const std::string path = NEP_ADAPTERS_FP64_SPIN_REFERENCE_PATH;
  const double energy_diff = max_abs_diff(
      std::vector<double>{oracle.energy},
      read_reference_vector(path, "energy_total"));
  const double potential_diff = max_abs_diff(
      oracle.potential, read_reference_vector(path, "energy_atom"));
  const double force_diff =
      max_abs_diff(oracle.force, read_reference_vector(path, "force"));
  const double mforce_diff =
      max_abs_diff(oracle.mforce, read_reference_vector(path, "mforce"));
  const double virial_diff =
      max_abs_diff(oracle.virial, read_reference_vector(path, "virial9"));
  const double descriptor_diff = max_abs_diff(
      oracle.descriptor, read_reference_vector(path, "descriptor"));
  const bool ok = energy_diff <= 1.0e-10 && potential_diff <= 1.0e-10 &&
                  force_diff <= 1.0e-10 && mforce_diff <= 1.0e-10 &&
                  virial_diff <= 1.0e-9 && descriptor_diff <= 1.0e-10;
  std::cout << std::scientific << std::setprecision(9)
            << "FP64_ORACLE_SELF_CHECK case=" << test_case.name
            << " energy=" << energy_diff
            << " potential=" << potential_diff
            << " force=" << force_diff
            << " mforce=" << mforce_diff
            << " virial=" << virial_diff
            << " descriptor=" << descriptor_diff
            << " status=" << (ok ? "pass" : "fail") << '\n';
  return ok;
}

CaseData displaced_case(
    const CaseData& base,
    bool spin_coordinate,
    std::size_t coordinate,
    double delta) {
  CaseData out = base;
  std::vector<double>& values = spin_coordinate ? out.spins : out.positions;
  values[coordinate] += delta;
  return out;
}

CaseData strained_case(const CaseData& base, int axis, double strain) {
  CaseData out = base;
  for (int atom = 0; atom < out.atom_count(); ++atom) {
    out.positions[3 * static_cast<std::size_t>(atom) + axis] *= 1.0 + strain;
  }
  for (int column = 0; column < 3; ++column) {
    out.box[3 * axis + column] *= 1.0 + strain;
  }
  return out;
}

template <typename Runner>
double energy_only(Runner& runner, const CaseData& test_case) {
  return evaluate_batch(runner, test_case, false).energy;
}

template <typename Runner>
bool validate_oracle_derivatives(
    Runner& oracle,
    const CaseData& test_case) {
  const Prediction base = evaluate_batch(oracle, test_case, false);
  constexpr double h = 2.0e-4;
  double force_diff = 0.0;
  for (std::size_t coordinate = 0; coordinate < test_case.positions.size(); ++coordinate) {
    const double em2 = energy_only(
        oracle, displaced_case(test_case, false, coordinate, -2.0 * h));
    const double em1 = energy_only(
        oracle, displaced_case(test_case, false, coordinate, -h));
    const double ep1 = energy_only(
        oracle, displaced_case(test_case, false, coordinate, h));
    const double ep2 = energy_only(
        oracle, displaced_case(test_case, false, coordinate, 2.0 * h));
    const double finite_difference_force =
        (-em2 + 8.0 * em1 - 8.0 * ep1 + ep2) / (12.0 * h);
    force_diff = std::max(
        force_diff,
        std::abs(base.force[coordinate] - finite_difference_force));
  }

  double mforce_diff = 0.0;
  if (test_case.is_spin()) {
    for (std::size_t coordinate = 0; coordinate < test_case.spins.size(); ++coordinate) {
      const double em2 = energy_only(
          oracle, displaced_case(test_case, true, coordinate, -2.0 * h));
      const double em1 = energy_only(
          oracle, displaced_case(test_case, true, coordinate, -h));
      const double ep1 = energy_only(
          oracle, displaced_case(test_case, true, coordinate, h));
      const double ep2 = energy_only(
          oracle, displaced_case(test_case, true, coordinate, 2.0 * h));
      const double finite_difference_mforce =
          (-em2 + 8.0 * em1 - 8.0 * ep1 + ep2) / (12.0 * h);
      mforce_diff = std::max(
          mforce_diff,
          std::abs(base.mforce[coordinate] - finite_difference_mforce));
    }
  }

  double virial_diff = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double em2 = energy_only(oracle, strained_case(test_case, axis, -2.0 * h));
    const double em1 = energy_only(oracle, strained_case(test_case, axis, -h));
    const double ep1 = energy_only(oracle, strained_case(test_case, axis, h));
    const double ep2 = energy_only(oracle, strained_case(test_case, axis, 2.0 * h));
    // Public NEP virial follows the negative strain derivative convention.
    const double finite_difference_virial =
        (-em2 + 8.0 * em1 - 8.0 * ep1 + ep2) / (12.0 * h);
    virial_diff = std::max(
        virial_diff,
        std::abs(base.virial[4 * axis] - finite_difference_virial));
  }

  std::array<double, 3> force_sum{};
  for (int atom = 0; atom < test_case.atom_count(); ++atom) {
    for (int component = 0; component < 3; ++component) {
      force_sum[component] +=
          base.force[3 * static_cast<std::size_t>(atom) + component];
    }
  }
  const double force_sum_max = std::max(
      std::abs(force_sum[0]),
      std::max(std::abs(force_sum[1]), std::abs(force_sum[2])));
  const bool ok = force_diff <= 2.0e-7 && mforce_diff <= 2.0e-7 &&
                  virial_diff <= 2.0e-7 && force_sum_max <= 2.0e-10;
  std::cout << std::scientific << std::setprecision(9)
            << "FP64_ORACLE_DERIVATIVE case=" << test_case.name
            << " force=" << force_diff
            << " mforce=" << mforce_diff
            << " virial_diag=" << virial_diff
            << " force_sum=" << force_sum_max
            << " status=" << (ok ? "pass" : "fail") << '\n';
  return ok;
}

Budgets cpu_budgets() {
  return {
      {2.0e-8, 2.0e-12},
      {2.0e-8, 2.0e-12},
      {5.0e-8, 2.0e-11},
      {5.0e-7, 2.0e-11},
      {5.0e-7, 2.0e-11},
      {5.0e-8, 2.0e-11},
      {5.0e-8, 2.0e-11},
      {2.0e-8, 2.0e-11},
      {2.0e-8, 2.0e-11},
  };
}

Budgets cuda_budgets() {
  return {
      {5.0e-4, 0.0},
      {5.0e-4, 0.0},
      {1.0e-3, 0.0},
      {5.0e-3, 0.0},
      {5.0e-3, 0.0},
      {1.0e-3, 0.0},
      {1.0e-3, 0.0},
      {5.0e-4, 0.0},
      {2.0e-6, 2.0e-6},
  };
}

bool compare_prediction(
    const std::string& backend,
    const CaseData& test_case,
    const Prediction& candidate,
    const Prediction& oracle,
    const Budgets& budgets,
    bool compare_atom_virial = true) {
  const double atoms = static_cast<double>(test_case.atom_count());
  bool ok = true;
  ok = report_field(
           backend,
           test_case.name,
           "energy_per_atom",
           {candidate.energy / atoms},
           {oracle.energy / atoms},
           budgets.energy_per_atom) && ok;
  ok = report_field(
           backend, test_case.name, "potential", candidate.potential,
           oracle.potential, budgets.potential) && ok;
  ok = report_field(
           backend, test_case.name, "force", candidate.force,
           oracle.force, budgets.force) && ok;
  ok = report_field(
           backend, test_case.name, "virial", candidate.virial,
           oracle.virial, budgets.virial) && ok;
  if (compare_atom_virial) {
    ok = report_field(
             backend, test_case.name, "atom_virial", candidate.atom_virial,
             oracle.atom_virial, budgets.atom_virial) && ok;
  }
  ok = report_field(
           backend, test_case.name, "descriptor", candidate.descriptor,
           oracle.descriptor, budgets.descriptor) && ok;
  const int radial_dim = read_radial_descriptor_dim(test_case.model_path);
  ok = report_field(
           backend,
           test_case.name,
           "radial_basis_sum_scaled",
           radial_descriptor_channels(
               candidate.descriptor, test_case.atom_count(), radial_dim),
           radial_descriptor_channels(
               oracle.descriptor, test_case.atom_count(), radial_dim),
           budgets.radial_basis_sum) && ok;
  if (test_case.is_spin()) {
    ok = report_field(
             backend, test_case.name, "mforce", candidate.mforce,
             oracle.mforce, budgets.mforce) && ok;
    ok = report_field(
             backend, test_case.name, "tau", candidate.tau,
             oracle.tau, budgets.tau) && ok;
  }
  return ok;
}

template <typename Runner>
double model_cutoff_max(Runner& runner) {
  NepaModelInfo info{};
  require_status(runner.model_info(info), "model_info");
  if (!(info.cutoff_max > 0.0)) {
    throw std::runtime_error("model cutoff must be positive");
  }
  return info.cutoff_max;
}

std::vector<std::vector<int>> make_isolated_neighbor_rows(
    const CaseData& test_case,
    double cutoff) {
  const int atom_count = test_case.atom_count();
  const double cutoff_squared = cutoff * cutoff;
  std::vector<std::vector<int>> rows(atom_count);
  for (int atom = 0; atom < atom_count; ++atom) {
    const std::size_t atom_offset = 3 * static_cast<std::size_t>(atom);
    for (int neighbor = 0; neighbor < atom_count; ++neighbor) {
      if (neighbor == atom) {
        continue;
      }
      const std::size_t neighbor_offset =
          3 * static_cast<std::size_t>(neighbor);
      double distance_squared = 0.0;
      for (int component = 0; component < 3; ++component) {
        const double delta = test_case.positions[neighbor_offset + component] -
                             test_case.positions[atom_offset + component];
        distance_squared += delta * delta;
      }
      if (distance_squared < cutoff_squared) {
        rows[atom].push_back(neighbor);
      }
    }
  }
  return rows;
}

struct LammpsStorage {
  LammpsStorage(const CaseData& test_case, double cutoff) {
    const int atom_count = test_case.atom_count();
    ilist.resize(atom_count);
    std::iota(ilist.begin(), ilist.end(), 0);
    neighbor_rows = make_isolated_neighbor_rows(test_case, cutoff);
    numneigh.resize(atom_count);
    firstneigh.resize(atom_count, nullptr);
    for (int atom = 0; atom < atom_count; ++atom) {
      numneigh[atom] = static_cast<int>(neighbor_rows[atom].size());
      firstneigh[atom] = neighbor_rows[atom].data();
    }

    int max_type = 0;
    lammps_types.resize(atom_count);
    for (int atom = 0; atom < atom_count; ++atom) {
      max_type = std::max(max_type, static_cast<int>(test_case.types[atom]));
      lammps_types[atom] = test_case.types[atom] + 1;
    }
    type_map.assign(max_type + 2, -1);
    for (int type = 0; type <= max_type; ++type) {
      type_map[type + 1] = type;
    }

    position_rows.resize(atom_count);
    position_ptrs.resize(atom_count, nullptr);
    for (int atom = 0; atom < atom_count; ++atom) {
      for (int component = 0; component < 3; ++component) {
        position_rows[atom][component] =
            test_case.positions[3 * static_cast<std::size_t>(atom) + component];
      }
      position_ptrs[atom] = position_rows[atom].data();
    }

    if (test_case.is_spin()) {
      spin_rows.resize(atom_count);
      spin_ptrs.resize(atom_count, nullptr);
      for (int atom = 0; atom < atom_count; ++atom) {
        for (int component = 0; component < 3; ++component) {
          spin_rows[atom][component] =
              test_case.spins[3 * static_cast<std::size_t>(atom) + component];
        }
        spin_rows[atom][3] = 1.0;
        spin_ptrs[atom] = spin_rows[atom].data();
      }
    }

    input.nlocal = atom_count;
    input.inum = atom_count;
    input.ilist = ilist.data();
    input.numneigh = numneigh.data();
    input.firstneigh = firstneigh.data();
    input.types = lammps_types.data();
    input.type_map = type_map.data();
    input.positions = position_ptrs.data();
    input.spins = spin_ptrs.empty() ? nullptr : spin_ptrs.data();
  }

  NepaLammpsNeighborInput input{};
  std::vector<int> ilist;
  std::vector<int> numneigh;
  std::vector<std::vector<int>> neighbor_rows;
  std::vector<int*> firstneigh;
  std::vector<int> lammps_types;
  std::vector<int> type_map;
  std::vector<std::array<double, 3>> position_rows;
  std::vector<double*> position_ptrs;
  std::vector<std::array<double, 4>> spin_rows;
  std::vector<double*> spin_ptrs;
};

#if defined(NEP_ADAPTERS_FP64_COMPARE_CUDA)
void require_cuda(cudaError_t status, const std::string& operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        operation + " failed: " + cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      require_cuda(
          cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
          "cudaMalloc");
    }
  }

  explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
    if (count_ > 0) {
      require_cuda(
          cudaMemcpy(
              data_,
              host.data(),
              count_ * sizeof(T),
              cudaMemcpyHostToDevice),
          "cudaMemcpy host to device");
    }
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() { return data_; }
  const T* data() const { return data_; }

  void copy_to(std::vector<T>& host) const {
    if (host.size() != count_) {
      throw std::runtime_error("device copyback size mismatch");
    }
    if (count_ > 0) {
      require_cuda(
          cudaMemcpy(
              host.data(),
              data_,
              count_ * sizeof(T),
              cudaMemcpyDeviceToHost),
          "cudaMemcpy device to host");
    }
  }

 private:
  std::size_t count_ = 0;
  T* data_ = nullptr;
};

LammpsPrediction evaluate_lammps_device(
    ApiRunner& runner,
    const CaseData& test_case) {
  const int atom_count = test_case.atom_count();
  const std::vector<std::vector<int>> neighbor_rows =
      make_isolated_neighbor_rows(test_case, model_cutoff_max(runner));
  int max_neighbors = 0;
  for (const auto& row : neighbor_rows) {
    max_neighbors = std::max(max_neighbors, static_cast<int>(row.size()));
  }
  std::vector<int> ilist(atom_count);
  std::iota(ilist.begin(), ilist.end(), 0);
  std::vector<int> counts(atom_count, 0);
  std::vector<int> neighbors(
      static_cast<std::size_t>(atom_count) * max_neighbors);
  for (int atom = 0; atom < atom_count; ++atom) {
    counts[atom] = static_cast<int>(neighbor_rows[atom].size());
    for (int slot = 0; slot < counts[atom]; ++slot) {
      neighbors[static_cast<std::size_t>(atom) * max_neighbors + slot] =
          neighbor_rows[atom][slot];
    }
  }
  std::vector<int> types(test_case.types.begin(), test_case.types.end());
  std::vector<double> lammps_spins;
  if (test_case.is_spin()) {
    lammps_spins.resize(static_cast<std::size_t>(atom_count) * 4);
    for (int atom = 0; atom < atom_count; ++atom) {
      for (int component = 0; component < 3; ++component) {
        lammps_spins[4 * static_cast<std::size_t>(atom) + component] =
            test_case.spins[3 * static_cast<std::size_t>(atom) + component];
      }
      lammps_spins[4 * static_cast<std::size_t>(atom) + 3] = 1.0;
    }
  }

  DeviceBuffer<int> d_ilist(ilist);
  DeviceBuffer<int> d_counts(counts);
  DeviceBuffer<int> d_neighbors(neighbors);
  DeviceBuffer<int> d_types(types);
  DeviceBuffer<double> d_positions(test_case.positions);
  DeviceBuffer<double> d_spins(lammps_spins);
  DeviceBuffer<double> d_energy(1);
  DeviceBuffer<double> d_virial6(6);
  DeviceBuffer<double> d_potential(atom_count);
  DeviceBuffer<double> d_force(static_cast<std::size_t>(atom_count) * 3);
  DeviceBuffer<double> d_mforce(
      test_case.is_spin() ? static_cast<std::size_t>(atom_count) * 3 : 0);
  DeviceBuffer<double> d_atom_virial(
      static_cast<std::size_t>(atom_count) * 9);

  NepaLammpsDeviceNeighborInput input{};
  input.nlocal = atom_count;
  input.nall = atom_count;
  input.inum = atom_count;
  input.max_neighbors = max_neighbors;
  input.neighbor_rows = atom_count;
  input.numneigh_length = atom_count;
  input.ilist = d_ilist.data();
  input.numneigh = d_counts.data();
  input.neighbors = d_neighbors.data();
  input.neighbor_atom_stride = max_neighbors;
  input.neighbor_slot_stride = 1;
  input.types = d_types.data();
  input.positions = d_positions.data();
  input.position_atom_stride = 3;
  input.position_component_stride = 1;
  input.spins = test_case.is_spin() ? d_spins.data() : nullptr;
  input.spin_atom_stride = test_case.is_spin() ? 4 : 0;
  input.spin_component_stride = test_case.is_spin() ? 1 : 0;

  NepaLammpsDeviceNeighborResult result{};
  result.total_potential = d_energy.data();
  result.total_virial6 = d_virial6.data();
  result.potential_per_atom = d_potential.data();
  result.forces = d_force.data();
  result.force_atom_stride = 3;
  result.force_component_stride = 1;
  result.mforces = test_case.is_spin() ? d_mforce.data() : nullptr;
  result.mforce_atom_stride = test_case.is_spin() ? 3 : 0;
  result.mforce_component_stride = test_case.is_spin() ? 1 : 0;
  result.virials_per_atom9 = d_atom_virial.data();
  result.virial_atom_stride = 9;
  result.virial_component_stride = 1;
  require_status(
      runner.find_force_lammps_device_neighbors(input, result),
      "find_force_lammps_device_neighbors");

  LammpsPrediction out;
  std::vector<double> energy(1, 0.0);
  out.virial6.assign(6, 0.0);
  out.potential.assign(atom_count, 0.0);
  out.force.assign(static_cast<std::size_t>(atom_count) * 3, 0.0);
  if (test_case.is_spin()) {
    out.mforce.assign(static_cast<std::size_t>(atom_count) * 3, 0.0);
  }
  out.atom_virial9.assign(static_cast<std::size_t>(atom_count) * 9, 0.0);
  d_energy.copy_to(energy);
  d_virial6.copy_to(out.virial6);
  d_potential.copy_to(out.potential);
  d_force.copy_to(out.force);
  if (test_case.is_spin()) {
    d_mforce.copy_to(out.mforce);
  }
  d_atom_virial.copy_to(out.atom_virial9);
  out.energy = energy[0];
  return out;
}
#endif

template <typename Runner>
LammpsPrediction evaluate_lammps(
    Runner& runner,
    const CaseData& test_case,
    LammpsStorage& storage) {
  const int atom_count = test_case.atom_count();
  std::vector<std::array<double, 3>> force_rows(atom_count);
  std::vector<double*> force_ptrs(atom_count, nullptr);
  std::vector<std::array<double, 3>> mforce_rows(
      test_case.is_spin() ? atom_count : 0);
  std::vector<double*> mforce_ptrs(
      test_case.is_spin() ? atom_count : 0, nullptr);
  std::vector<std::array<double, 9>> virial_rows(atom_count);
  std::vector<double*> virial_ptrs(atom_count, nullptr);
  for (int atom = 0; atom < atom_count; ++atom) {
    force_ptrs[atom] = force_rows[atom].data();
    virial_ptrs[atom] = virial_rows[atom].data();
    if (test_case.is_spin()) {
      mforce_ptrs[atom] = mforce_rows[atom].data();
    }
  }

  LammpsPrediction out;
  out.virial6.assign(6, 0.0);
  out.potential.assign(atom_count, 0.0);
  out.force.assign(static_cast<std::size_t>(atom_count) * 3, 0.0);
  out.atom_virial9.assign(static_cast<std::size_t>(atom_count) * 9, 0.0);
  if (test_case.is_spin()) {
    out.mforce.assign(static_cast<std::size_t>(atom_count) * 3, 0.0);
  }

  NepaLammpsNeighborResult result{};
  result.total_potential = &out.energy;
  result.total_virial6 = out.virial6.data();
  result.potential_per_atom = out.potential.data();
  result.forces = force_ptrs.data();
  result.mforces = mforce_ptrs.empty() ? nullptr : mforce_ptrs.data();
  result.virials_per_atom9 = virial_ptrs.data();
  require_status(
      runner.find_force_lammps_neighbors(storage.input, result),
      "find_force_lammps_neighbors");

  for (int atom = 0; atom < atom_count; ++atom) {
    for (int component = 0; component < 3; ++component) {
      out.force[3 * static_cast<std::size_t>(atom) + component] =
          force_rows[atom][component];
      if (test_case.is_spin()) {
        out.mforce[3 * static_cast<std::size_t>(atom) + component] =
            mforce_rows[atom][component];
      }
    }
    for (int component = 0; component < 9; ++component) {
      out.atom_virial9[9 * static_cast<std::size_t>(atom) + component] =
          virial_rows[atom][component];
    }
  }
  return out;
}

std::vector<double> batch_virial6(const Prediction& batch) {
  return {
      batch.virial[0], batch.virial[4], batch.virial[8],
      0.5 * (batch.virial[1] + batch.virial[3]),
      0.5 * (batch.virial[2] + batch.virial[6]),
      0.5 * (batch.virial[5] + batch.virial[7])};
}

std::vector<double> lammps_atom_virial_to_batch_order(
    const std::vector<double>& values) {
  static constexpr int map[9] = {0, 4, 8, 1, 2, 5, 3, 6, 7};
  std::vector<double> out(values.size(), 0.0);
  for (std::size_t atom = 0; atom < values.size() / 9; ++atom) {
    for (int lammps_component = 0; lammps_component < 9; ++lammps_component) {
      out[9 * atom + map[lammps_component]] =
          values[9 * atom + lammps_component];
    }
  }
  return out;
}

#if defined(NEP_ADAPTERS_FP64_COMPARE_CUDA)
std::array<double, 3> heat_current_from_n2_virial(
    const std::vector<double>& atom_virial,
    const std::vector<double>& velocities) {
  if (atom_virial.size() % 9 != 0 ||
      velocities.size() != atom_virial.size() / 3) {
    throw std::runtime_error("invalid n2 heat-current input shape");
  }
  std::array<double, 3> heat_current{};
  for (std::size_t atom = 0; atom < velocities.size() / 3; ++atom) {
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        heat_current[row] +=
            atom_virial[9 * atom + 3 * row + column] *
            velocities[3 * atom + column];
      }
    }
  }
  return heat_current;
}

struct DirectHeatCurrentReference {
  std::array<double, 3> neighbor_owned{};
  std::array<double, 3> center_owned{};
};

template <typename Runner>
DirectHeatCurrentReference finite_difference_heat_current_reference(
    Runner& runner,
    const CaseData& test_case,
    const std::vector<double>& velocities) {
  if (velocities.size() != test_case.positions.size()) {
    throw std::runtime_error("invalid random-velocity shape");
  }

  constexpr double h = 1.0e-3;
  DirectHeatCurrentReference out;
  for (int center = 0; center < test_case.atom_count(); ++center) {
    for (int neighbor = 0; neighbor < test_case.atom_count(); ++neighbor) {
      if (neighbor == center) {
        continue;
      }
      for (int derivative_component = 0; derivative_component < 3;
           ++derivative_component) {
        CaseData em2_case = test_case;
        CaseData em1_case = test_case;
        CaseData ep1_case = test_case;
        CaseData ep2_case = test_case;
        const std::size_t coordinate =
            3 * static_cast<std::size_t>(neighbor) + derivative_component;
        em2_case.positions[coordinate] -= 2.0 * h;
        em1_case.positions[coordinate] -= h;
        ep1_case.positions[coordinate] += h;
        ep2_case.positions[coordinate] += 2.0 * h;
        const double em2 =
            evaluate_batch(runner, em2_case, false).potential[center];
        const double em1 =
            evaluate_batch(runner, em1_case, false).potential[center];
        const double ep1 =
            evaluate_batch(runner, ep1_case, false).potential[center];
        const double ep2 =
            evaluate_batch(runner, ep2_case, false).potential[center];
        const double edge_gradient =
            (em2 - 8.0 * em1 + 8.0 * ep1 - ep2) / (12.0 * h);
        for (int current_component = 0; current_component < 3;
             ++current_component) {
          const double rij =
              test_case.positions[3 * static_cast<std::size_t>(neighbor) +
                                  current_component] -
              test_case.positions[3 * static_cast<std::size_t>(center) +
                                  current_component];
          out.neighbor_owned[current_component] -=
              rij * edge_gradient *
              velocities[3 * static_cast<std::size_t>(neighbor) +
                         derivative_component];
          out.center_owned[current_component] -=
              rij * edge_gradient *
              velocities[3 * static_cast<std::size_t>(center) +
                         derivative_component];
        }
      }
    }
  }
  return out;
}

double max_abs_diff(
    const std::array<double, 3>& lhs,
    const std::array<double, 3>& rhs) {
  double out = 0.0;
  for (int component = 0; component < 3; ++component) {
    out = std::max(out, std::abs(lhs[component] - rhs[component]));
  }
  return out;
}

bool report_n2_heat_current(
    const std::string& path,
    const std::array<double, 3>& candidate,
    const DirectHeatCurrentReference& reference) {
  constexpr double tolerance = 1.0e-4;
  const double error = max_abs_diff(candidate, reference.neighbor_owned);
  const double center_gap =
      max_abs_diff(reference.center_owned, reference.neighbor_owned);
  const bool ok = error <= tolerance && center_gap > tolerance;
  std::cout << std::scientific << std::setprecision(9)
            << "CUDA_N2_HEAT_CURRENT path=" << path
            << " random_seed=0x6e325f68656174"
            << " direct_x=" << reference.neighbor_owned[0]
            << " direct_y=" << reference.neighbor_owned[1]
            << " direct_z=" << reference.neighbor_owned[2]
            << " virial_x=" << candidate[0]
            << " virial_y=" << candidate[1]
            << " virial_z=" << candidate[2]
            << " max_abs=" << error
            << " wrong_center_gap=" << center_gap
            << " tolerance=" << tolerance
            << " status=" << (ok ? "pass" : "fail") << '\n';
  return ok;
}

bool validate_cuda_n2_heat_current(
    ApiRunner& cuda_runner,
    OracleRunner& direct_runner,
    const CaseData& test_case) {
  std::mt19937_64 generator(0x6e325f68656174ULL);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  std::vector<double> velocities(test_case.positions.size(), 0.0);
  for (double& velocity : velocities) {
    velocity = distribution(generator);
  }
  for (int component = 0; component < 3; ++component) {
    double mean = 0.0;
    for (int atom = 0; atom < test_case.atom_count(); ++atom) {
      mean += velocities[3 * static_cast<std::size_t>(atom) + component];
    }
    mean /= static_cast<double>(test_case.atom_count());
    for (int atom = 0; atom < test_case.atom_count(); ++atom) {
      velocities[3 * static_cast<std::size_t>(atom) + component] -= mean;
    }
  }

  const DirectHeatCurrentReference reference =
      finite_difference_heat_current_reference(
          direct_runner, test_case, velocities);
  const Prediction batch = evaluate_batch(cuda_runner, test_case, false);
  LammpsStorage host_storage(test_case, model_cutoff_max(cuda_runner));
  const LammpsPrediction host =
      evaluate_lammps(cuda_runner, test_case, host_storage);
  const LammpsPrediction device =
      evaluate_lammps_device(cuda_runner, test_case);

  bool ok = report_n2_heat_current(
      "batch",
      heat_current_from_n2_virial(batch.atom_virial, velocities),
      reference);
  ok = report_n2_heat_current(
           "host_neighbors",
           heat_current_from_n2_virial(
               lammps_atom_virial_to_batch_order(host.atom_virial9),
               velocities),
           reference) && ok;
  ok = report_n2_heat_current(
           "device_neighbors",
           heat_current_from_n2_virial(
               lammps_atom_virial_to_batch_order(device.atom_virial9),
               velocities),
           reference) && ok;
  return ok;
}
#endif

std::vector<double> sum_lammps_atom_virial6(
    const std::vector<double>& values) {
  std::vector<double> out(6, 0.0);
  for (std::size_t atom = 0; atom < values.size() / 9; ++atom) {
    const double* raw9 = values.data() + 9 * atom;
    out[0] += raw9[0];
    out[1] += raw9[1];
    out[2] += raw9[2];
    out[3] += 0.5 * (raw9[3] + raw9[6]);
    out[4] += 0.5 * (raw9[4] + raw9[7]);
    out[5] += 0.5 * (raw9[5] + raw9[8]);
  }
  return out;
}

CaseData symmetrically_strained_case(
    const CaseData& base,
    int lammps_component,
    double strain) {
  static constexpr int first_axis[6] = {0, 1, 2, 0, 0, 1};
  static constexpr int second_axis[6] = {0, 1, 2, 1, 2, 2};
  if (lammps_component < 0 || lammps_component >= 6) {
    throw std::runtime_error("invalid LAMMPS virial component");
  }

  CaseData out = base;
  const int a = first_axis[lammps_component];
  const int b = second_axis[lammps_component];
  for (int atom = 0; atom < out.atom_count(); ++atom) {
    const std::size_t offset = 3 * static_cast<std::size_t>(atom);
    if (a == b) {
      out.positions[offset + a] =
          base.positions[offset + a] * (1.0 + strain);
    } else {
      // Engineering shear gamma: x_a += gamma*x_b/2 and
      // x_b += gamma*x_a/2.  Its negative energy derivative is the
      // symmetric LAMMPS off-diagonal virial.
      out.positions[offset + a] =
          base.positions[offset + a] +
          0.5 * strain * base.positions[offset + b];
      out.positions[offset + b] =
          base.positions[offset + b] +
          0.5 * strain * base.positions[offset + a];
    }
  }
  return out;
}

std::vector<double> force_moment_virial6(
    const CaseData& test_case,
    const std::vector<double>& force) {
  if (force.size() != test_case.positions.size()) {
    throw std::runtime_error("force-moment input shape mismatch");
  }
  std::array<long double, 3> origin{};
  for (int atom = 0; atom < test_case.atom_count(); ++atom) {
    for (int component = 0; component < 3; ++component) {
      origin[component] += test_case.positions[
          3 * static_cast<std::size_t>(atom) + component];
    }
  }
  for (long double& value : origin) {
    value /= static_cast<long double>(test_case.atom_count());
  }

  std::array<long double, 9> raw{};
  for (int atom = 0; atom < test_case.atom_count(); ++atom) {
    const std::size_t offset = 3 * static_cast<std::size_t>(atom);
    for (int position_component = 0; position_component < 3;
         ++position_component) {
      const long double centered_position =
          static_cast<long double>(test_case.positions[
              offset + position_component]) - origin[position_component];
      for (int force_component = 0; force_component < 3; ++force_component) {
        raw[3 * position_component + force_component] +=
            centered_position *
            static_cast<long double>(force[offset + force_component]);
      }
    }
  }
  return {
      static_cast<double>(raw[0]),
      static_cast<double>(raw[4]),
      static_cast<double>(raw[8]),
      static_cast<double>(0.5L * (raw[1] + raw[3])),
      static_cast<double>(0.5L * (raw[2] + raw[6])),
      static_cast<double>(0.5L * (raw[5] + raw[7])),
  };
}

template <typename EnergyEvaluator>
double five_point_negative_strain_derivative(
    EnergyEvaluator&& energy,
    const CaseData& test_case,
    int component,
    double h) {
  const double em2 =
      energy(symmetrically_strained_case(test_case, component, -2.0 * h));
  const double em1 =
      energy(symmetrically_strained_case(test_case, component, -h));
  const double ep1 =
      energy(symmetrically_strained_case(test_case, component, h));
  const double ep2 =
      energy(symmetrically_strained_case(test_case, component, 2.0 * h));
  return (-em2 + 8.0 * em1 - 8.0 * ep1 + ep2) / (12.0 * h);
}

struct DenseVirialFdReference {
  std::vector<double> virial6;
  bool passed = false;
};

template <typename EnergyEvaluator>
DenseVirialFdReference run_dense_virial_fd_scan(
    const std::string& backend,
    const CaseData& test_case,
    const LammpsPrediction& analytic,
    EnergyEvaluator&& energy,
    bool enforce_fp64_gate) {
  static constexpr const char* component_names[6] = {
      "xx", "yy", "zz", "xy", "xz", "yz"};
  static constexpr double steps[] = {3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3};
  std::vector<double> selected(6, 0.0);
  double selected_max_abs = 0.0;
  for (int component = 0; component < 6; ++component) {
    for (double h : steps) {
      const double finite_difference = five_point_negative_strain_derivative(
          energy, test_case, component, h);
      const double abs_error =
          std::abs(analytic.virial6[component] - finite_difference);
      std::cout << std::scientific << std::setprecision(9)
                << "FP64_VIRIAL_FD backend=" << backend
                << " case=" << test_case.name
                << " component=" << component_names[component]
                << " h=" << h
                << " analytic=" << analytic.virial6[component]
                << " finite_difference=" << finite_difference
                << " abs_error=" << abs_error
                << " per_atom_abs="
                << abs_error / static_cast<double>(test_case.atom_count())
                << '\n';
      if (h == steps[3]) {
        selected[component] = finite_difference;
        selected_max_abs = std::max(selected_max_abs, abs_error);
      }
    }
  }

  const std::vector<double> force_moment =
      force_moment_virial6(test_case, analytic.force);
  const ErrorStats moment_stats = error_stats(analytic.virial6, force_moment);
  const double atom_count = static_cast<double>(test_case.atom_count());
  const bool fd_ok = !enforce_fp64_gate || selected_max_abs <= 2.0e-5;
  const bool moment_ok = !enforce_fp64_gate || moment_stats.max_abs <= 2.0e-10;
  std::cout << std::scientific << std::setprecision(9)
            << "FP64_VIRIAL_INDEPENDENT backend=" << backend
            << " case=" << test_case.name
            << " fd_max_abs=" << selected_max_abs
            << " fd_max_per_atom=" << selected_max_abs / atom_count
            << " force_moment_max_abs=" << moment_stats.max_abs
            << " force_moment_max_per_atom=" << moment_stats.max_abs / atom_count
            << " gate=" << (enforce_fp64_gate ? "fp64" : "observe")
            << " status=" << (fd_ok && moment_ok ? "pass" : "fail")
            << '\n';
  return {std::move(selected), fd_ok && moment_ok};
}

bool validate_oracle_lammps(
    const CaseData& test_case,
    const Prediction& batch,
    const LammpsPrediction& lammps) {
  const double energy_diff = std::abs(batch.energy - lammps.energy);
  const double potential_diff = max_abs_diff(batch.potential, lammps.potential);
  const double force_diff = max_abs_diff(batch.force, lammps.force);
  const double virial_diff = max_abs_diff(batch_virial6(batch), lammps.virial6);
  const std::vector<double> lammps_atom_virial =
      lammps_atom_virial_to_batch_order(lammps.atom_virial9);
  double atom_virial_diff = 0.0;
  if (test_case.is_spin()) {
    std::array<double, 9> batch_sum{};
    std::array<double, 9> lammps_sum{};
    for (int atom = 0; atom < test_case.atom_count(); ++atom) {
      for (int component = 0; component < 9; ++component) {
        const std::size_t index = static_cast<std::size_t>(atom) * 9 + component;
        batch_sum[component] += batch.atom_virial[index];
        lammps_sum[component] += lammps_atom_virial[index];
      }
    }
    atom_virial_diff = max_abs_diff(
        std::vector<double>(batch_sum.begin(), batch_sum.end()),
        std::vector<double>(lammps_sum.begin(), lammps_sum.end()));
  } else {
    atom_virial_diff = max_abs_diff(batch.atom_virial, lammps_atom_virial);
  }
  const double mforce_diff = test_case.is_spin()
      ? max_abs_diff(batch.mforce, lammps.mforce)
      : 0.0;
  const bool ok = energy_diff <= 2.0e-10 && potential_diff <= 2.0e-10 &&
                  force_diff <= 2.0e-10 && virial_diff <= 2.0e-10 &&
                  atom_virial_diff <= 2.0e-10 && mforce_diff <= 2.0e-10;
  std::cout << std::scientific << std::setprecision(9)
            << "FP64_ORACLE_LAMMPS_SELF_CHECK case=" << test_case.name
            << " energy=" << energy_diff
            << " potential=" << potential_diff
            << " force=" << force_diff
            << " virial=" << virial_diff
            << " atom_virial=" << atom_virial_diff
            << " mforce=" << mforce_diff
            << " status=" << (ok ? "pass" : "fail") << '\n';
  return ok;
}

bool compare_lammps_prediction(
    const std::string& backend,
    const CaseData& test_case,
    const LammpsPrediction& candidate,
    const LammpsPrediction& oracle,
    const Budgets& budgets,
    bool compare_atom_virial = true) {
  const double atoms = static_cast<double>(test_case.atom_count());
  bool ok = true;
  ok = report_field(
           backend, test_case.name + "_lammps", "energy_per_atom",
           {candidate.energy / atoms}, {oracle.energy / atoms},
           budgets.energy_per_atom) && ok;
  ok = report_field(
           backend, test_case.name + "_lammps", "potential",
           candidate.potential, oracle.potential, budgets.potential) && ok;
  ok = report_field(
           backend, test_case.name + "_lammps", "force",
           candidate.force, oracle.force, budgets.force) && ok;
  ok = report_field(
           backend, test_case.name + "_lammps", "virial6",
           candidate.virial6, oracle.virial6, budgets.virial) && ok;
  if (compare_atom_virial) {
    ok = report_field(
             backend, test_case.name + "_lammps", "atom_virial9",
             candidate.atom_virial9, oracle.atom_virial9,
             budgets.atom_virial) && ok;
  }
  if (test_case.is_spin()) {
    ok = report_field(
             backend, test_case.name + "_lammps", "mforce",
             candidate.mforce, oracle.mforce, budgets.mforce) && ok;
  }
  return ok;
}

bool run_backend_batch_cases(
    const std::string& backend,
    const Budgets& budgets,
    const std::vector<std::pair<CaseData, Prediction>>& nonmag_cases,
    const std::vector<std::pair<CaseData, Prediction>>& spin_cases) {
  bool ok = true;
  ApiRunner nonmag_backend(backend, NEP_ADAPTERS_FP64_MODEL_PATH);
  for (const auto& item : nonmag_cases) {
    const Prediction candidate = evaluate_batch(nonmag_backend, item.first);
    ok = compare_prediction(
             backend, item.first, candidate, item.second, budgets) && ok;
  }

  ApiRunner spin_backend(backend, NEP_ADAPTERS_FP64_SPIN_MODEL_PATH);
  for (const auto& item : spin_cases) {
    const Prediction candidate = evaluate_batch(spin_backend, item.first);
    ok = compare_prediction(
             backend,
             item.first,
             candidate,
             item.second,
             budgets,
             backend != "cuda") && ok;
  }
  return ok;
}

template <typename BackendRunner>
bool run_lammps_case(
    const std::string& backend_name,
    const Budgets& budgets,
    BackendRunner& backend,
    OracleRunner& oracle,
    const CaseData& test_case) {
  LammpsStorage oracle_storage(test_case, model_cutoff_max(oracle));
  const Prediction oracle_batch = evaluate_batch(oracle, test_case);
  const LammpsPrediction oracle_lammps =
      evaluate_lammps(oracle, test_case, oracle_storage);
  bool ok = validate_oracle_lammps(test_case, oracle_batch, oracle_lammps);

  LammpsStorage backend_storage(test_case, model_cutoff_max(backend));
  const LammpsPrediction candidate =
      evaluate_lammps(backend, test_case, backend_storage);
  ok = compare_lammps_prediction(
           backend_name,
           test_case,
           candidate,
           oracle_lammps,
           budgets,
           backend_name != "cuda" || !test_case.is_spin()) && ok;
  return ok;
}

#if defined(NEP_ADAPTERS_FP64_COMPARE_CUDA)
bool run_device_lammps_case(
    const Budgets& budgets,
    ApiRunner& backend,
    OracleRunner& oracle,
    const CaseData& test_case) {
  LammpsStorage oracle_storage(test_case, model_cutoff_max(oracle));
  const Prediction oracle_batch = evaluate_batch(oracle, test_case);
  const LammpsPrediction oracle_lammps =
      evaluate_lammps(oracle, test_case, oracle_storage);
  bool ok = validate_oracle_lammps(test_case, oracle_batch, oracle_lammps);
  const LammpsPrediction candidate =
      evaluate_lammps_device(backend, test_case);
  ok = compare_lammps_prediction(
           "cuda_device", test_case, candidate, oracle_lammps, budgets) && ok;
  return ok;
}

bool run_device_spin_lammps_case(
    const Budgets& budgets,
    ApiRunner& backend,
    OracleRunner& oracle,
    const CaseData& test_case) {
  LammpsStorage oracle_storage(test_case, model_cutoff_max(oracle));
  const Prediction oracle_batch = evaluate_batch(oracle, test_case);
  const LammpsPrediction oracle_lammps =
      evaluate_lammps(oracle, test_case, oracle_storage);
  bool ok = validate_oracle_lammps(test_case, oracle_batch, oracle_lammps);
  const LammpsPrediction candidate =
      evaluate_lammps_device(backend, test_case);
  const double atoms = static_cast<double>(test_case.atom_count());
  ok = report_field(
           "cuda_device",
           test_case.name + "_lammps",
           "energy_per_atom",
           {candidate.energy / atoms},
           {oracle_lammps.energy / atoms},
           budgets.energy_per_atom) && ok;
  ok = report_field(
           "cuda_device", test_case.name + "_lammps", "potential",
           candidate.potential, oracle_lammps.potential, budgets.potential) && ok;
  ok = report_field(
           "cuda_device", test_case.name + "_lammps", "force",
           candidate.force, oracle_lammps.force, budgets.force) && ok;
  std::vector<double> candidate_virial_per_atom = candidate.virial6;
  std::vector<double> oracle_virial_per_atom = oracle_lammps.virial6;
  for (double& value : candidate_virial_per_atom) {
    value /= atoms;
  }
  for (double& value : oracle_virial_per_atom) {
    value /= atoms;
  }
  ok = report_field(
           "cuda_device",
           test_case.name + "_lammps",
           "virial6_per_atom",
           candidate_virial_per_atom,
           oracle_virial_per_atom,
           budgets.virial) && ok;
  ok = report_field(
           "cuda_device",
           test_case.name + "_lammps",
           "atom_virial9",
           candidate.atom_virial9,
           oracle_lammps.atom_virial9,
           budgets.atom_virial) && ok;
  ok = report_field(
           "cuda_device", test_case.name + "_lammps", "mforce",
           candidate.mforce, oracle_lammps.mforce, budgets.mforce) && ok;
  ok = report_field(
           "cuda_device",
           test_case.name + "_lammps",
           "atom_virial9_sum",
           sum_lammps_atom_virial6(candidate.atom_virial9),
           candidate.virial6,
           budgets.virial) && ok;
  return ok;
}
#endif

}  // namespace

int main() {
  try {
    std::cout << "FP64_ORACLE_CONFIG scalar=double openmp=off cblas=off "
                 "radial_table=off fast_math=off fp_contract=off\n";
    if (!nep_adapters::register_cpu_engine()) {
      std::cerr << "failed to register cpu\n";
      return EXIT_FAILURE;
    }
#if defined(NEP_ADAPTERS_FP64_COMPARE_CUDA)
    if (!nep_adapters::register_cuda_engine()) {
      std::cerr << "failed to register cuda\n";
      return EXIT_FAILURE;
    }
#endif

    bool ok = true;
    OracleRunner nonmag_oracle(NEP_ADAPTERS_FP64_MODEL_PATH);
    const CaseData nonmag_fixture = make_nonmag_fixture();
    const Prediction nonmag_fixture_oracle =
        evaluate_batch(nonmag_oracle, nonmag_fixture);
    ok = validate_nonmag_frozen_reference(
             nonmag_fixture_oracle, nonmag_fixture) && ok;
    ok = validate_oracle_derivatives(nonmag_oracle, nonmag_fixture) && ok;

    const CaseData dense_nonmag = make_dense_nonmag_case();
    const Prediction dense_nonmag_oracle =
        evaluate_batch(nonmag_oracle, dense_nonmag);
    const std::vector<std::pair<CaseData, Prediction>> nonmag_cases = {
        {nonmag_fixture, nonmag_fixture_oracle},
        {dense_nonmag, dense_nonmag_oracle},
    };

    OracleRunner spin_oracle(NEP_ADAPTERS_FP64_SPIN_MODEL_PATH);
    const CaseData spin_reference = make_spin_reference_case();
    const Prediction spin_reference_oracle =
        evaluate_batch(spin_oracle, spin_reference);
    ok = validate_spin_frozen_reference(spin_reference_oracle, spin_reference) && ok;

    const CaseData spin_isolated_periodic = make_spin_finite_difference_case();
    const Prediction spin_isolated_periodic_oracle =
        evaluate_batch(spin_oracle, spin_isolated_periodic);
    ok = validate_oracle_derivatives(spin_oracle, spin_isolated_periodic) && ok;
    const CaseData dense_spin = make_dense_spin_case();
    const Prediction dense_spin_oracle = evaluate_batch(spin_oracle, dense_spin);
    const std::vector<std::pair<CaseData, Prediction>> spin_cases = {
        {spin_isolated_periodic, spin_isolated_periodic_oracle},
        {dense_spin, dense_spin_oracle},
    };

    ok = run_backend_batch_cases(
             "cpu",
             cpu_budgets(),
             nonmag_cases,
             spin_cases) && ok;

    CaseData nonmag_lammps = nonmag_fixture;
    nonmag_lammps.name = "nonmag_isolated_periodic";
    nonmag_lammps.pbc = {1, 1, 1};
    ApiRunner cpu_nonmag("cpu", nonmag_lammps.model_path);
    ok = run_lammps_case(
             "cpu",
             cpu_budgets(),
             cpu_nonmag,
             nonmag_oracle,
             nonmag_lammps) && ok;
    ApiRunner cpu_spin("cpu", spin_isolated_periodic.model_path);
    ok = run_lammps_case(
             "cpu",
             cpu_budgets(),
             cpu_spin,
             spin_oracle,
             spin_isolated_periodic) && ok;

#if defined(NEP_ADAPTERS_FP64_COMPARE_CUDA)
    ok = run_backend_batch_cases(
             "cuda",
             cuda_budgets(),
             nonmag_cases,
             spin_cases) && ok;
    ApiRunner cuda_nonmag("cuda", nonmag_lammps.model_path);
    ok = run_lammps_case(
             "cuda",
             cuda_budgets(),
             cuda_nonmag,
             nonmag_oracle,
             nonmag_lammps) && ok;
    CaseData dense_nonmag_lammps = dense_nonmag;
    dense_nonmag_lammps.name = "nonmag_dense_isolated_periodic_device";
    // Keep the periodic cell large enough that no image enters the cutoff.
    dense_nonmag_lammps.box = {
        64.0, 0.0, 0.0,
        0.0, 64.0, 0.0,
        0.0, 0.0, 64.0};
    dense_nonmag_lammps.pbc = {1, 1, 1};
    ok = run_device_lammps_case(
             cuda_budgets(),
             cuda_nonmag,
             nonmag_oracle,
             dense_nonmag_lammps) && ok;
    ApiRunner cuda_spin("cuda", spin_isolated_periodic.model_path);
    ok = validate_cuda_n2_heat_current(
             cuda_spin, spin_oracle, spin_isolated_periodic) && ok;
    ok = run_lammps_case(
             "cuda",
             cuda_budgets(),
             cuda_spin,
             spin_oracle,
             spin_isolated_periodic) && ok;
    ok = run_device_spin_lammps_case(
             cuda_budgets(),
             cuda_spin,
             spin_oracle,
             spin_isolated_periodic) && ok;
    CaseData dense_spin_lammps = dense_spin;
    dense_spin_lammps.name = "spin_chiral_dense_isolated_periodic_device";
    dense_spin_lammps.box = {
        64.0, 0.0, 0.0,
        0.0, 64.0, 0.0,
        0.0, 0.0, 64.0};
    dense_spin_lammps.pbc = {1, 1, 1};
    ok = run_device_spin_lammps_case(
             cuda_budgets(),
             cuda_spin,
             spin_oracle,
             dense_spin_lammps) && ok;

    LammpsStorage dense_spin_oracle_storage(
        dense_spin_lammps, model_cutoff_max(spin_oracle));
    const LammpsPrediction dense_spin_oracle_lammps = evaluate_lammps(
        spin_oracle, dense_spin_lammps, dense_spin_oracle_storage);
    auto oracle_energy = [&](const CaseData& strained) {
      LammpsStorage storage(strained, model_cutoff_max(spin_oracle));
      return evaluate_lammps(spin_oracle, strained, storage).energy;
    };
    const DenseVirialFdReference dense_spin_fd = run_dense_virial_fd_scan(
        "fp64_cpu",
        dense_spin_lammps,
        dense_spin_oracle_lammps,
        oracle_energy,
        true);
    ok = dense_spin_fd.passed && ok;

    const LammpsPrediction dense_spin_cuda_lammps =
        evaluate_lammps_device(cuda_spin, dense_spin_lammps);
    auto cuda_energy = [&](const CaseData& strained) {
      return evaluate_lammps_device(cuda_spin, strained).energy;
    };
    run_dense_virial_fd_scan(
        "cuda_device",
        dense_spin_lammps,
        dense_spin_cuda_lammps,
        cuda_energy,
        false);

    std::vector<double> cuda_virial_per_atom =
        dense_spin_cuda_lammps.virial6;
    std::vector<double> fd_virial_per_atom = dense_spin_fd.virial6;
    std::vector<double> cuda_force_moment_per_atom = force_moment_virial6(
        dense_spin_lammps, dense_spin_cuda_lammps.force);
    const double dense_atoms =
        static_cast<double>(dense_spin_lammps.atom_count());
    for (int component = 0; component < 6; ++component) {
      cuda_virial_per_atom[component] /= dense_atoms;
      fd_virial_per_atom[component] /= dense_atoms;
      cuda_force_moment_per_atom[component] /= dense_atoms;
    }
    ok = report_field(
             "cuda_device",
             dense_spin_lammps.name,
             "virial6_per_atom_vs_fp64_fd",
             cuda_virial_per_atom,
             fd_virial_per_atom,
             {1.0e-3, 0.0}) && ok;
    ok = report_field(
             "cuda_device",
             dense_spin_lammps.name,
             "force_moment6_per_atom_vs_fp64_fd",
             cuda_force_moment_per_atom,
             fd_virial_per_atom,
             {1.0e-3, 0.0}) && ok;
#endif

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << "FP64 oracle test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
