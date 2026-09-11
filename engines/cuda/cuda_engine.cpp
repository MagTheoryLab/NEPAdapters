#include "nep_adapters/engine.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include "device_operations.hpp"
#include "device_model.hpp"
#include "device_workspace.hpp"
#include "force_pipeline.hpp"

#include "host_staging.hpp"
#include "model_protocol.hpp"
#include "nep_adapters/virial_order.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool valid_batch(const NepaStructureBatch& batch) {
  return batch.num_structures > 0 && batch.total_atoms > 0 &&
         batch.atom_counts != nullptr && batch.atom_offsets != nullptr &&
         batch.types != nullptr && batch.positions_aos3 != nullptr &&
         batch.boxes_row_major9 != nullptr;
}

std::size_t max_structure_atom_count(const NepaStructureBatch& batch) {
  std::size_t maximum = 0;
  for (std::int32_t structure = 0; structure < batch.num_structures;
       ++structure) {
    maximum = std::max(
        maximum,
        static_cast<std::size_t>(batch.atom_counts[structure]));
  }
  return maximum;
}

struct StructureBatchView {
  StructureBatchView(const NepaStructureBatch& source, int structure)
      : atom_count(source.atom_counts[structure]),
        atom_offset(source.atom_offsets[structure]) {
    local_atom_counts[0] = atom_count;
    local_atom_offsets[0] = 0;
    batch.num_structures = 1;
    batch.total_atoms = atom_count;
    batch.atom_counts = local_atom_counts;
    batch.atom_offsets = local_atom_offsets;
    batch.types = source.types + atom_offset;
    batch.positions_aos3 =
        source.positions_aos3 + 3 * static_cast<std::size_t>(atom_offset);
    batch.spins_aos3 =
        source.spins_aos3 != nullptr
            ? source.spins_aos3 + 3 * static_cast<std::size_t>(atom_offset)
            : nullptr;
    batch.boxes_row_major9 =
        source.boxes_row_major9 + 9 * static_cast<std::size_t>(structure);
    batch.pbc_flags3 =
        source.pbc_flags3 != nullptr
            ? source.pbc_flags3 + 3 * static_cast<std::size_t>(structure)
            : nullptr;
  }

  std::int32_t local_atom_counts[1]{};
  std::int32_t local_atom_offsets[1]{};
  std::int32_t atom_count = 0;
  std::int32_t atom_offset = 0;
  NepaStructureBatch batch{};
};

bool supports_cuda_force_protocol(
    const nep_adapters::cuda_backend::ModelProtocol& protocol) {
  const auto& body = protocol.body_channels;
  return (protocol.version == 4 || protocol.version == 5) &&
         protocol.charge_mode == 0 &&
         nep_adapters::cuda_backend::supports_cuda_spin_shape(protocol) &&
         (!protocol.has_zbl ||
          (protocol.zbl_outer > 0.0 &&
           (protocol.flexible_zbl ||
            (protocol.zbl_inner >= 0.0 &&
             protocol.zbl_outer > protocol.zbl_inner)))) &&
         body.l_max_3body <= 8;
}

bool supports_cuda_descriptor_protocol(
    const nep_adapters::cuda_backend::ModelProtocol& protocol) {
  if (supports_cuda_force_protocol(protocol)) {
    return true;
  }
  return protocol.version == 4 && protocol.charge_mode > 0 &&
         protocol.spin_mode == 0 && protocol.body_channels.l_max_3body <= 8;
}

bool needs_angular_terms(
    const nep_adapters::cuda_backend::ModelProtocol& protocol) {
  return protocol.body_channels.channel_count() > 0;
}

nep_adapters::cuda_backend::VirialOutputMode select_virial_output_mode(
    bool total_requested,
    bool per_atom_requested) {
  using nep_adapters::cuda_backend::VirialOutputMode;
  if (per_atom_requested) {
    return VirialOutputMode::per_atom_n2;
  }
  return total_requested ? VirialOutputMode::total_only
                         : VirialOutputMode::none;
}

nep_adapters::cuda_backend::ForceEvaluationRequest make_force_evaluation_request(
    nep_adapters::cuda_backend::ForceNeighborTopology topology,
    bool store_potential,
    bool total_virial_requested,
    bool per_atom_virial_requested,
    bool spin_transfer_requested = false) {
  nep_adapters::cuda_backend::ForceEvaluationRequest request;
  request.topology = topology;
  request.virial = select_virial_output_mode(
      total_virial_requested, per_atom_virial_requested);
  request.store_potential = store_potential;
  request.spin_transfer_per_atom = spin_transfer_requested;
  return request;
}

nep_adapters::cuda_backend::VirialTarget select_fp64_virial_target(
    bool virial_requested,
    bool per_atom_requested) {
  using nep_adapters::cuda_backend::VirialTarget;
  if (per_atom_requested) {
    return VirialTarget::neighbor_atom;
  }
  return virial_requested ? VirialTarget::center_atom : VirialTarget::none;
}

int env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  return end == raw ? fallback : static_cast<int>(value);
}

int mpi_rank_from_env() {
  const char* names[] = {
      "OMPI_COMM_WORLD_RANK",
      "PMI_RANK",
      "PMIX_RANK",
      "SLURM_PROCID"};
  for (const char* name : names) {
    const int value = env_int(name, -1);
    if (value >= 0) {
      return value;
    }
  }
  return 0;
}

class LammpsDevicePairProfiler {
 public:
  LammpsDevicePairProfiler(int nlocal, bool rebuild_workspace)
      : nlocal_(nlocal), rebuild_workspace_(rebuild_workspace) {
    const char* raw = std::getenv("NEP_ADAPTERS_PROFILE_PAIR");
    if (raw == nullptr || raw[0] == '\0') {
      return;
    }
    const int rank = mpi_rank_from_env();
    if (rank != env_int("NEP_ADAPTERS_PROFILE_PAIR_RANK", 0)) {
      return;
    }
    static int samples = 0;
    const int max_samples = env_int("NEP_ADAPTERS_PROFILE_PAIR", 5);
    if (max_samples > 0 && samples >= max_samples) {
      return;
    }
    sample_ = ++samples;
    enabled_ = true;
    cudaEventCreate(&mark_);
    cudaEventCreate(&now_);
    cudaEventRecord(mark_);
  }

  ~LammpsDevicePairProfiler() {
    if (mark_ != nullptr) {
      cudaEventDestroy(mark_);
    }
    if (now_ != nullptr) {
      cudaEventDestroy(now_);
    }
  }

  void split(float& target_ms) {
    if (!enabled_) {
      return;
    }
    cudaEventRecord(now_);
    cudaEventSynchronize(now_);
    cudaEventElapsedTime(&target_ms, mark_, now_);
    cudaEventRecord(mark_);
  }

  bool enabled() const { return enabled_; }

  void reset() {
    if (enabled_) {
      cudaEventRecord(mark_);
    }
  }

  void print(
      float stage_ms,
      float clear_ms,
      float descriptor_ann_ms,
      float radial_force_ms,
      float angular_force_ms,
      float zbl_force_ms,
      float spin_onsite_ms,
      float spin_density_ms,
      float spin_density_pull_ms,
      float spin_density_edge_ms,
      float spin_chiral_ms,
      float output_ms) const {
    if (!enabled_) {
      return;
    }
    const float total_ms =
        stage_ms + clear_ms + descriptor_ann_ms + radial_force_ms +
        angular_force_ms + zbl_force_ms + spin_onsite_ms + spin_density_ms +
        spin_chiral_ms + output_ms;
    std::fprintf(
        stderr,
        "NEPA_PAIR_PROFILE sample=%d nlocal=%d rebuild=%d "
        "stage_ms=%.3f clear_ms=%.3f descriptor_ann_ms=%.3f "
        "radial_force_ms=%.3f angular_force_ms=%.3f zbl_force_ms=%.3f "
        "spin_onsite_ms=%.3f spin_density_ms=%.3f "
        "spin_density_pull_ms=%.3f spin_density_edge_ms=%.3f "
        "spin_chiral_ms=%.3f "
        "output_ms=%.3f total_ms=%.3f\n",
        sample_,
        nlocal_,
        rebuild_workspace_ ? 1 : 0,
        stage_ms,
        clear_ms,
        descriptor_ann_ms,
        radial_force_ms,
        angular_force_ms,
        zbl_force_ms,
        spin_onsite_ms,
        spin_density_ms,
        spin_density_pull_ms,
        spin_density_edge_ms,
        spin_chiral_ms,
        output_ms,
        total_ms);
  }

 private:
  bool enabled_ = false;
  int sample_ = 0;
  int nlocal_ = 0;
  bool rebuild_workspace_ = false;
  cudaEvent_t mark_ = nullptr;
  cudaEvent_t now_ = nullptr;
};

class LammpsHostPairProfiler {
 public:
  explicit LammpsHostPairProfiler(int nlocal) : nlocal_(nlocal) {
    const char* raw = std::getenv("NEP_ADAPTERS_PROFILE_PAIR");
    if (raw == nullptr || raw[0] == '\0') {
      return;
    }
    if (mpi_rank_from_env() != env_int("NEP_ADAPTERS_PROFILE_PAIR_RANK", 0)) {
      return;
    }
    static int samples = 0;
    const int max_samples = env_int("NEP_ADAPTERS_PROFILE_PAIR", 5);
    if (max_samples > 0 && samples >= max_samples) {
      return;
    }
    sample_ = ++samples;
    enabled_ = true;
    mark_ = std::chrono::steady_clock::now();
  }

  bool enabled() const { return enabled_; }

  void split(float& target_ms) {
    if (!enabled_) {
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    target_ms = static_cast<float>(
        std::chrono::duration<double, std::milli>(now - mark_).count());
    mark_ = now;
  }

  void print(
      float workspace_wall_ms,
      float neighbor_stage_wall_ms,
      float clear_wall_ms,
      float pipeline_wall_ms,
      const nep_adapters::cuda_backend::ForcePipelineTimings& pipeline,
      float output_wall_ms) const {
    if (!enabled_) {
      return;
    }
    const float nested_pipeline_ms =
        pipeline.descriptor_ann_ms + pipeline.radial_force_ms +
        pipeline.angular_force_ms + pipeline.zbl_force_ms +
        pipeline.spin_onsite_ms + pipeline.spin_density_ms +
        pipeline.spin_chiral_ms;
    const float total_wall_ms =
        workspace_wall_ms + neighbor_stage_wall_ms + clear_wall_ms +
        pipeline_wall_ms + output_wall_ms;
    std::fprintf(
        stderr,
        "NEPA_HOST_PAIR_PROFILE sample=%d nlocal=%d "
        "workspace_wall_ms=%.3f neighbor_stage_wall_ms=%.3f "
        "clear_wall_ms=%.3f pipeline_wall_ms=%.3f "
        "descriptor_ann_ms=%.3f radial_force_ms=%.3f angular_force_ms=%.3f "
        "zbl_force_ms=%.3f spin_onsite_ms=%.3f spin_density_ms=%.3f "
        "spin_density_pull_ms=%.3f spin_density_edge_ms=%.3f "
        "spin_chiral_ms=%.3f pipeline_unaccounted_ms=%.3f "
        "output_wall_ms=%.3f total_wall_ms=%.3f\n",
        sample_,
        nlocal_,
        workspace_wall_ms,
        neighbor_stage_wall_ms,
        clear_wall_ms,
        pipeline_wall_ms,
        pipeline.descriptor_ann_ms,
        pipeline.radial_force_ms,
        pipeline.angular_force_ms,
        pipeline.zbl_force_ms,
        pipeline.spin_onsite_ms,
        pipeline.spin_density_ms,
        pipeline.spin_density_pull_ms,
        pipeline.spin_density_edge_ms,
        pipeline.spin_chiral_ms,
        pipeline_wall_ms - nested_pipeline_ms,
        output_wall_ms,
        total_wall_ms);
  }

 private:
  bool enabled_ = false;
  int sample_ = 0;
  int nlocal_ = 0;
  std::chrono::steady_clock::time_point mark_{};
};

int lammps_atom_capacity(const NepaLammpsNeighborInput& input) {
  int atom_capacity = input.nlocal;
  for (int active = 0; active < input.inum; ++active) {
    const int atom = input.ilist[active];
    if (atom >= 0) {
      atom_capacity = std::max(atom_capacity, atom + 1);
    }
    if (atom < 0 || input.numneigh == nullptr || input.firstneigh == nullptr ||
        input.firstneigh[atom] == nullptr) {
      continue;
    }
    for (int slot = 0; slot < input.numneigh[atom]; ++slot) {
      constexpr int kLammpsNeighborMask = 0x3fffffff;
      const int neighbor = input.firstneigh[atom][slot] & kLammpsNeighborMask;
      atom_capacity = std::max(atom_capacity, neighbor + 1);
    }
  }
  return atom_capacity;
}

nep_adapters::cuda_backend::SimulationBox make_nonperiodic_lammps_box() {
  nep_adapters::cuda_backend::SimulationBox box{};
  box.frac_to_cart[0] = 1.0;
  box.frac_to_cart[4] = 1.0;
  box.frac_to_cart[8] = 1.0;
  box.cart_to_frac[0] = 1.0;
  box.cart_to_frac[4] = 1.0;
  box.cart_to_frac[8] = 1.0;
  return box;
}

bool invert_row_major3(const double* matrix, double* inverse) {
  const double det =
      matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]) -
      matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]) +
      matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
  if (std::abs(det) <= 1.0e-12) {
    return false;
  }
  const double inv_det = 1.0 / det;
  inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inv_det;
  inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inv_det;
  inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inv_det;
  inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inv_det;
  inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inv_det;
  inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inv_det;
  inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inv_det;
  inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inv_det;
  inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inv_det;
  return true;
}

bool make_simulation_box(
    const NepaStructureBatch& batch,
    nep_adapters::cuda_backend::SimulationBox& box) {
  const double* cell = batch.boxes_row_major9;
  for (int component = 0; component < 9; ++component) {
    box.frac_to_cart[component] = cell[component];
  }
  if (!invert_row_major3(box.frac_to_cart, box.cart_to_frac)) {
    return false;
  }
  if (batch.pbc_flags3 == nullptr || batch.pbc_flags3[0] != 1 ||
      batch.pbc_flags3[1] != 1 || batch.pbc_flags3[2] != 1) {
    return false;
  }
  box.pbc[0] = 1;
  box.pbc[1] = 1;
  box.pbc[2] = 1;
  return std::isfinite(box.cart_to_frac[0]) &&
         std::isfinite(box.cart_to_frac[4]) &&
         std::isfinite(box.cart_to_frac[8]);
}

struct ExpandedPeriodicBatch {
  NepaStructureBatch batch{};
  std::vector<std::int32_t> atom_counts;
  std::vector<std::int32_t> atom_offsets;
  std::vector<std::int32_t> types;
  std::vector<double> positions;
  std::vector<double> spins;
  std::vector<double> boxes;
  std::vector<std::int32_t> pbc;
  std::vector<std::int32_t> original_atom;
  std::vector<double> atom_weight;
  std::vector<std::int32_t> replicas_per_structure;
  bool expanded = false;
};

std::array<int, 3> periodic_replication_counts(
    const double* box,
    double cutoff) {
  double inverse[9] = {};
  if (!invert_row_major3(box, inverse)) {
    throw std::invalid_argument("cannot expand a singular periodic box");
  }
  std::array<int, 3> counts = {1, 1, 1};
  for (int axis = 0; axis < 3; ++axis) {
    const double x = inverse[3 * axis + 0];
    const double y = inverse[3 * axis + 1];
    const double z = inverse[3 * axis + 2];
    const double reciprocal_norm = std::sqrt(x * x + y * y + z * z);
    const double required = 2.0 * cutoff * reciprocal_norm;
    if (!std::isfinite(required) ||
        required > static_cast<double>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("periodic box expansion is too large");
    }
    counts[axis] = std::max(
        1,
        static_cast<int>(std::ceil(required - 1.0e-12)));
  }
  return counts;
}

ExpandedPeriodicBatch expand_small_periodic_batch(
    const NepaStructureBatch& source,
    double cutoff) {
  ExpandedPeriodicBatch expanded;
  std::vector<std::array<int, 3>> replication(
      static_cast<std::size_t>(source.num_structures));
  std::size_t expanded_atom_count = 0;

  for (std::int32_t structure = 0;
       structure < source.num_structures;
       ++structure) {
    if (source.pbc_flags3 == nullptr ||
        source.pbc_flags3[3 * structure + 0] != 1 ||
        source.pbc_flags3[3 * structure + 1] != 1 ||
        source.pbc_flags3[3 * structure + 2] != 1) {
      throw std::invalid_argument(
          "CUDA batch evaluation requires three-dimensional periodic boxes");
    }
    const auto cells = periodic_replication_counts(
        source.boxes_row_major9 + 9 * structure,
        cutoff);
    replication[static_cast<std::size_t>(structure)] = cells;
    const std::int64_t replicas =
        static_cast<std::int64_t>(cells[0]) * cells[1] * cells[2];
    const std::int64_t structure_atoms =
        replicas * source.atom_counts[structure];
    if (replicas > std::numeric_limits<std::int32_t>::max() ||
        structure_atoms > std::numeric_limits<std::int32_t>::max() ||
        expanded_atom_count + static_cast<std::size_t>(structure_atoms) >
            static_cast<std::size_t>(
                std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("expanded periodic batch is too large");
    }
    expanded.expanded = expanded.expanded || replicas > 1;
    expanded_atom_count += static_cast<std::size_t>(structure_atoms);
  }
  if (!expanded.expanded) {
    return expanded;
  }

  expanded.atom_counts.reserve(source.num_structures);
  expanded.atom_offsets.reserve(source.num_structures);
  expanded.boxes.reserve(9 * static_cast<std::size_t>(source.num_structures));
  expanded.pbc.reserve(3 * static_cast<std::size_t>(source.num_structures));
  expanded.replicas_per_structure.reserve(source.num_structures);
  expanded.types.reserve(expanded_atom_count);
  expanded.positions.reserve(3 * expanded_atom_count);
  if (source.spins_aos3 != nullptr) {
    expanded.spins.reserve(3 * expanded_atom_count);
  }
  expanded.original_atom.reserve(expanded_atom_count);
  expanded.atom_weight.reserve(expanded_atom_count);

  std::size_t atom_offset = 0;
  for (std::int32_t structure = 0;
       structure < source.num_structures;
       ++structure) {
    const auto cells = replication[static_cast<std::size_t>(structure)];
    const int replicas = cells[0] * cells[1] * cells[2];
    const double weight = 1.0 / replicas;
    expanded.replicas_per_structure.push_back(replicas);
    expanded.atom_offsets.push_back(static_cast<std::int32_t>(atom_offset));
    expanded.atom_counts.push_back(
        source.atom_counts[structure] * replicas);

    const double* box = source.boxes_row_major9 + 9 * structure;
    const double ax = box[0];
    const double ay = box[3];
    const double az = box[6];
    const double bx = box[1];
    const double by = box[4];
    const double bz = box[7];
    const double cx = box[2];
    const double cy = box[5];
    const double cz = box[8];
    expanded.boxes.insert(
        expanded.boxes.end(),
        {ax * cells[0], bx * cells[1], cx * cells[2],
         ay * cells[0], by * cells[1], cy * cells[2],
         az * cells[0], bz * cells[1], cz * cells[2]});
    expanded.pbc.insert(expanded.pbc.end(), {1, 1, 1});

    const int source_offset = source.atom_offsets[structure];
    const int source_count = source.atom_counts[structure];
    for (int ia = 0; ia < cells[0]; ++ia) {
      for (int ib = 0; ib < cells[1]; ++ib) {
        for (int ic = 0; ic < cells[2]; ++ic) {
          const double tx = ia * ax + ib * bx + ic * cx;
          const double ty = ia * ay + ib * by + ic * cy;
          const double tz = ia * az + ib * bz + ic * cz;
          for (int local = 0; local < source_count; ++local) {
            const int atom = source_offset + local;
            expanded.types.push_back(source.types[atom]);
            expanded.positions.insert(
                expanded.positions.end(),
                {source.positions_aos3[3 * atom + 0] + tx,
                 source.positions_aos3[3 * atom + 1] + ty,
                 source.positions_aos3[3 * atom + 2] + tz});
            if (source.spins_aos3 != nullptr) {
              expanded.spins.insert(
                  expanded.spins.end(),
                  {source.spins_aos3[3 * atom + 0],
                   source.spins_aos3[3 * atom + 1],
                   source.spins_aos3[3 * atom + 2]});
            }
            expanded.original_atom.push_back(atom);
            expanded.atom_weight.push_back(weight);
          }
        }
      }
    }
    atom_offset +=
        static_cast<std::size_t>(source_count) *
        static_cast<std::size_t>(replicas);
  }

  expanded.batch.num_structures = source.num_structures;
  expanded.batch.total_atoms = static_cast<std::int32_t>(expanded_atom_count);
  expanded.batch.atom_counts = expanded.atom_counts.data();
  expanded.batch.atom_offsets = expanded.atom_offsets.data();
  expanded.batch.types = expanded.types.data();
  expanded.batch.positions_aos3 = expanded.positions.data();
  expanded.batch.spins_aos3 =
      source.spins_aos3 == nullptr ? nullptr : expanded.spins.data();
  expanded.batch.boxes_row_major9 = expanded.boxes.data();
  expanded.batch.pbc_flags3 = expanded.pbc.data();
  return expanded;
}

struct ExpandedForceResult {
  ExpandedForceResult(
      const ExpandedPeriodicBatch& expanded,
      const NepaFindForceResult& requested)
      : energy(expanded.batch.num_structures),
        forces(3 * static_cast<std::size_t>(expanded.batch.total_atoms)) {
    result.energy_per_structure = energy.data();
    result.forces_aos3 = forces.data();
    const std::size_t atoms =
        static_cast<std::size_t>(expanded.batch.total_atoms);
    const std::size_t structures =
        static_cast<std::size_t>(expanded.batch.num_structures);
    if (requested.potential_per_atom != nullptr) {
      potential.resize(atoms);
      result.potential_per_atom = potential.data();
    }
    if (requested.virials_row_major9 != nullptr) {
      virials.resize(9 * structures);
      result.virials_row_major9 = virials.data();
    }
    if (requested.virials_per_atom_row_major9 != nullptr) {
      atom_virials.resize(9 * atoms);
      result.virials_per_atom_row_major9 = atom_virials.data();
    }
    if (requested.charge_per_atom != nullptr) {
      charge.resize(atoms);
      result.charge_per_atom = charge.data();
    }
    if (requested.bec_per_atom_row_major9 != nullptr) {
      bec.resize(9 * atoms);
      result.bec_per_atom_row_major9 = bec.data();
    }
    if (requested.mforces_aos3 != nullptr) {
      mforces.resize(3 * atoms);
      result.mforces_aos3 = mforces.data();
    }
    if (requested.spin_transfer_per_atom_row_major9 != nullptr) {
      spin_transfer.resize(9 * atoms);
      result.spin_transfer_per_atom_row_major9 = spin_transfer.data();
    }
  }

  NepaFindForceResult result{};
  std::vector<double> energy;
  std::vector<double> potential;
  std::vector<double> forces;
  std::vector<double> virials;
  std::vector<double> atom_virials;
  std::vector<double> charge;
  std::vector<double> bec;
  std::vector<double> mforces;
  std::vector<double> spin_transfer;
};

void reduce_expanded_force_result(
    const ExpandedPeriodicBatch& expanded,
    const ExpandedForceResult& source,
    NepaFindForceResult& target,
    std::size_t original_atom_count) {
  for (std::size_t structure = 0;
       structure < expanded.replicas_per_structure.size();
       ++structure) {
    const double weight =
        1.0 / expanded.replicas_per_structure[structure];
    target.energy_per_structure[structure] =
        source.energy[structure] * weight;
    if (target.virials_row_major9 != nullptr) {
      for (int component = 0; component < 9; ++component) {
        target.virials_row_major9[9 * structure + component] =
            source.virials[9 * structure + component] * weight;
      }
    }
  }

  std::fill(
      target.forces_aos3,
      target.forces_aos3 + 3 * original_atom_count,
      0.0);
  auto clear_if_requested = [original_atom_count](
                                double* values,
                                std::size_t components) {
    if (values != nullptr) {
      std::fill(
          values,
          values + components * original_atom_count,
          0.0);
    }
  };
  clear_if_requested(target.potential_per_atom, 1);
  clear_if_requested(target.virials_per_atom_row_major9, 9);
  clear_if_requested(target.charge_per_atom, 1);
  clear_if_requested(target.bec_per_atom_row_major9, 9);
  clear_if_requested(target.mforces_aos3, 3);
  clear_if_requested(target.spin_transfer_per_atom_row_major9, 9);

  for (std::size_t atom = 0;
       atom < expanded.original_atom.size();
       ++atom) {
    const std::size_t original =
        static_cast<std::size_t>(expanded.original_atom[atom]);
    const double weight = expanded.atom_weight[atom];
    for (int component = 0; component < 3; ++component) {
      target.forces_aos3[3 * original + component] +=
          source.forces[3 * atom + component] * weight;
      if (target.mforces_aos3 != nullptr) {
        target.mforces_aos3[3 * original + component] +=
            source.mforces[3 * atom + component] * weight;
      }
    }
    if (target.potential_per_atom != nullptr) {
      target.potential_per_atom[original] +=
          source.potential[atom] * weight;
    }
    if (target.charge_per_atom != nullptr) {
      target.charge_per_atom[original] += source.charge[atom] * weight;
    }
    for (int component = 0; component < 9; ++component) {
      if (target.virials_per_atom_row_major9 != nullptr) {
        target.virials_per_atom_row_major9[9 * original + component] +=
            source.atom_virials[9 * atom + component] * weight;
      }
      if (target.bec_per_atom_row_major9 != nullptr) {
        target.bec_per_atom_row_major9[9 * original + component] +=
            source.bec[9 * atom + component] * weight;
      }
      if (target.spin_transfer_per_atom_row_major9 != nullptr) {
        target.spin_transfer_per_atom_row_major9[
            9 * original + component] +=
            source.spin_transfer[9 * atom + component] * weight;
      }
    }
  }
}

std::vector<double> copy_device_doubles(const double* device, std::size_t count) {
  std::vector<double> host(count, 0.0);
  const cudaError_t status = cudaMemcpy(
      host.data(),
      device,
      count * sizeof(double),
      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    throw std::runtime_error("failed to copy CUDA double output");
  }
  return host;
}

std::vector<float> copy_device_floats(const float* device, std::size_t count) {
  std::vector<float> host(count, 0.0f);
  const cudaError_t status = cudaMemcpy(
      host.data(),
      device,
      count * sizeof(float),
      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    throw std::runtime_error("failed to copy CUDA float output");
  }
  return host;
}

void copy_device_doubles_to_host(
    const double* device,
    std::size_t count,
    double* host,
    const char* action) {
  const cudaError_t status = cudaMemcpy(
      host,
      device,
      count * sizeof(double),
      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    throw std::runtime_error(action);
  }
}

void clear_device_doubles(double* device, std::size_t count) {
  const cudaError_t status = cudaMemset(device, 0, count * sizeof(double));
  if (status != cudaSuccess) {
    throw std::runtime_error("failed to clear CUDA double buffer");
  }
}

void copy_prepared_batch_results_to_host(
    const nep_adapters::cuda_backend::DeviceWorkspaceView& view,
    int structure_count,
    int atom_count,
    int structure_offset,
    int atom_offset,
    bool spin_model,
    bool charge_model,
    NepaFindForceResult& result) {
  copy_device_doubles_to_host(
      view.structure_energy,
      static_cast<std::size_t>(structure_count),
      result.energy_per_structure + structure_offset,
      "failed to copy CUDA structure energies");
  copy_device_doubles_to_host(
      view.output_forces_aos3,
      static_cast<std::size_t>(atom_count) * 3,
      result.forces_aos3 + 3 * static_cast<std::size_t>(atom_offset),
      "failed to copy CUDA forces");
  if (result.potential_per_atom != nullptr) {
    copy_device_doubles_to_host(
        view.potential,
        static_cast<std::size_t>(atom_count),
        result.potential_per_atom + atom_offset,
        "failed to copy CUDA per-atom potentials");
  }
  if (result.virials_per_atom_row_major9 != nullptr) {
    copy_device_doubles_to_host(
        view.output_virials_per_atom_row_major9,
        static_cast<std::size_t>(atom_count) * 9,
        result.virials_per_atom_row_major9 +
            9 * static_cast<std::size_t>(atom_offset),
        "failed to copy CUDA per-atom virials");
  }
  if (result.virials_row_major9 != nullptr) {
    copy_device_doubles_to_host(
        view.structure_virial_row_major9,
        static_cast<std::size_t>(structure_count) * 9,
        result.virials_row_major9 +
            9 * static_cast<std::size_t>(structure_offset),
        "failed to copy CUDA structure virials");
  }
  if (spin_model && result.spin_transfer_per_atom_row_major9 != nullptr) {
    const std::vector<float> spin_transfer_soa = copy_device_floats(
        view.spin_transfer_soa9,
        view.atom_capacity * 9);
    for (int atom = 0; atom < atom_count; ++atom) {
      for (int component = 0; component < 9; ++component) {
        result.spin_transfer_per_atom_row_major9[
            9 * static_cast<std::size_t>(atom_offset + atom) + component] =
            spin_transfer_soa[
                static_cast<std::size_t>(component) * view.atom_capacity +
                atom];
      }
    }
  }
  if (charge_model && result.charge_per_atom != nullptr) {
    copy_device_doubles_to_host(
        view.charge,
        static_cast<std::size_t>(atom_count),
        result.charge_per_atom + atom_offset,
        "failed to copy CUDA per-atom charges");
  }

  if (!spin_model || result.mforces_aos3 == nullptr) {
    return;
  }
  copy_device_doubles_to_host(
      view.output_mforces_aos3,
      static_cast<std::size_t>(atom_count) * 3,
      result.mforces_aos3 + 3 * static_cast<std::size_t>(atom_offset),
      "failed to copy CUDA mforces");
}

void copy_qnep_bec_to_host(
    const nep_adapters::cuda_backend::DeviceWorkspaceView& view,
    int atom_count,
    int atom_offset,
    double* bec_per_atom_row_major9) {
  if (bec_per_atom_row_major9 == nullptr) {
    return;
  }
  const std::vector<double> bec_soa = copy_device_doubles(
      view.bec_soa9, view.atom_capacity * 9);
  for (int atom = 0; atom < atom_count; ++atom) {
    for (int component = 0; component < 9; ++component) {
      bec_per_atom_row_major9[
          9 * static_cast<std::size_t>(atom_offset + atom) + component] =
          bec_soa[static_cast<std::size_t>(component) * view.atom_capacity + atom];
    }
  }
}


class CudaModel : public nep_adapters::Model {
 public:
  explicit CudaModel(const std::string& model_path)
      :
        host_(nep_adapters::cuda_backend::load_host_model_parameters(model_path)),
        protocol_(host_.protocol),
        device_(host_) {}

  NepaModelKind model_kind() const override {
    if (protocol_.spin_mode > 0) {
      return NEPA_MODEL_KIND_SPIN;
    }
    if (protocol_.charge_mode > 0) {
      return NEPA_MODEL_KIND_CHARGE;
    }
    return NEPA_MODEL_KIND_ORDINARY;
  }

  NepaStatus model_info(NepaModelInfo& out) const override {
    out = {};
    out.cutoff_radial = protocol_.cutoff_radial;
    out.cutoff_angular = protocol_.cutoff_angular;
    out.cutoff_max = protocol_.cutoff_max;
    out.capabilities = nep_adapters::to_mask(nep_adapters::Capability::device_input);
    if (protocol_.spin_mode > 0) {
      out.capabilities |= nep_adapters::to_mask(nep_adapters::Capability::spin);
      out.capabilities |= nep_adapters::to_mask(
          nep_adapters::Capability::spin_energy_transfer);
    }
    if (supports_cuda_force_protocol(protocol_)) {
      out.capabilities |=
          nep_adapters::to_mask(nep_adapters::Capability::batch_find_force);
      out.capabilities |=
          nep_adapters::to_mask(nep_adapters::Capability::external_neighbors);
    } else if (protocol_.charge_mode > 0) {
      out.capabilities |=
          nep_adapters::to_mask(nep_adapters::Capability::batch_find_force);
      out.capabilities |=
          nep_adapters::to_mask(nep_adapters::Capability::charge);
    }
    if (supports_cuda_descriptor_protocol(protocol_)) {
      out.capabilities |=
          nep_adapters::to_mask(nep_adapters::Capability::descriptors);
    }
    out.num_types = protocol_.num_types;
    out.descriptor_dim = protocol_.descriptor_dim;
    return NEPA_STATUS_OK;
  }

  NepaStatus estimate_workspace(
      std::int32_t atom_capacity,
      std::int32_t structure_capacity,
      NepaWorkspaceEstimate& out) const override {
    if (atom_capacity <= 0 || structure_capacity <= 0) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    const auto plan =
        nep_adapters::cuda_backend::make_internal_neighbor_workspace_plan(
            protocol_,
            static_cast<std::size_t>(atom_capacity),
            static_cast<std::size_t>(structure_capacity));
    const std::size_t model_bytes = device_.upload_summary().total_bytes;
    const std::size_t workspace_bytes = plan.total_bytes();
    out = {};
    out.model_bytes = static_cast<std::uint64_t>(model_bytes);
    out.workspace_bytes = static_cast<std::uint64_t>(workspace_bytes);
    out.total_bytes = static_cast<std::uint64_t>(model_bytes + workspace_bytes);
    out.atom_capacity = atom_capacity;
    out.structure_capacity = structure_capacity;
    return NEPA_STATUS_OK;
  }

  NepaStatus find_force_batch(
      const NepaStructureBatch& batch,
      NepaFindForceResult& result) override {
    if (model_kind() == NEPA_MODEL_KIND_CHARGE) {
      return NEPA_STATUS_UNSUPPORTED;
    }
    return find_force_batch_impl(batch, result);
  }

  NepaStatus find_charge_batch(
      const NepaStructureBatch& batch,
      NepaFindForceResult& result) override {
    if (model_kind() != NEPA_MODEL_KIND_CHARGE) {
      return NEPA_STATUS_UNSUPPORTED;
    }
    if (result.charge_per_atom == nullptr ||
        result.bec_per_atom_row_major9 == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    return find_force_batch_impl(batch, result);
  }

  NepaStatus find_force_batch_impl(
      const NepaStructureBatch& batch,
      NepaFindForceResult& result,
      bool allow_periodic_expansion = true) {
    if (!valid_batch(batch) || result.energy_per_structure == nullptr ||
        result.forces_aos3 == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    if (is_cancelled()) {
      return NEPA_STATUS_CANCELLED;
    }

    for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
      const std::int32_t atom_count = batch.atom_counts[structure];
      const std::int32_t atom_offset = batch.atom_offsets[structure];
      if (atom_count <= 0 || atom_offset < 0 ||
          atom_offset + atom_count > batch.total_atoms) {
        return NEPA_STATUS_INVALID_ARGUMENT;
      }
    }

    try {
      if (!supports_cuda_force_protocol(protocol_) && protocol_.charge_mode == 0) {
        return NEPA_STATUS_UNSUPPORTED;
      }
      if (protocol_.spin_mode != 0 && batch.spins_aos3 == nullptr) {
        return NEPA_STATUS_INVALID_ARGUMENT;
      }
      if (result.spin_transfer_per_atom_row_major9 != nullptr &&
          protocol_.spin_mode == 0) {
        return NEPA_STATUS_UNSUPPORTED;
      }
      if (allow_periodic_expansion) {
        ExpandedPeriodicBatch expanded =
            expand_small_periodic_batch(batch, protocol_.cutoff_max);
        if (expanded.expanded) {
          ExpandedForceResult expanded_result(expanded, result);
          const NepaStatus status = find_force_batch_impl(
              expanded.batch,
              expanded_result.result,
              false);
          if (status != NEPA_STATUS_OK) {
            return status;
          }
          reduce_expanded_force_result(
              expanded,
              expanded_result,
              result,
              static_cast<std::size_t>(batch.total_atoms));
          return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
        }
      }
      const bool multi_box_execution =
          protocol_.charge_mode == 0 && protocol_.spin_mode == 0;
      const std::size_t workspace_atom_capacity =
          multi_box_execution
              ? static_cast<std::size_t>(batch.total_atoms)
              : max_structure_atom_count(batch);
      const std::size_t workspace_structure_capacity =
          multi_box_execution
              ? static_cast<std::size_t>(batch.num_structures)
              : 1;
      nep_adapters::cuda_backend::DeviceWorkspace& workspace =
          batch_workspace(
              workspace_atom_capacity,
              workspace_structure_capacity,
              result.spin_transfer_per_atom_row_major9 != nullptr);

      if (multi_box_execution) {
        nep_adapters::cuda_backend::stage_batch_on_device(
            batch, protocol_.num_types, workspace);
        const auto view = workspace.view();
        clear_device_doubles(view.potential, view.atom_capacity);
        clear_device_doubles(view.force_soa3, view.atom_capacity * 3);
        clear_device_doubles(view.virial_soa9, view.atom_capacity * 9);
        nep_adapters::cuda_backend::build_internal_neighbors_batched(
            protocol_,
            batch.num_structures,
            batch.total_atoms,
            workspace,
            nep_adapters::cuda_backend::batch_boxes_are_orthorhombic(batch));
        const auto force_request = make_force_evaluation_request(
            nep_adapters::cuda_backend::
                ForceNeighborTopology::batched_multi_box,
            true,
            true,
            result.virials_per_atom_row_major9 != nullptr);
        nep_adapters::cuda_backend::run_force_pipeline(
            protocol_,
            force_request,
            batch.total_atoms,
            nep_adapters::cuda_backend::SimulationBox{},
            device_,
            workspace);

        nep_adapters::cuda_backend::prepare_batched_outputs(
            batch.num_structures,
            batch.total_atoms,
            workspace);
        copy_prepared_batch_results_to_host(
            view,
            batch.num_structures,
            batch.total_atoms,
            0,
            0,
            false,
            false,
            result);
        return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
      }

      for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
        if (is_cancelled()) {
          return NEPA_STATUS_CANCELLED;
        }
        const StructureBatchView single(batch, structure);

        nep_adapters::cuda_backend::SimulationBox box;
        if (!make_simulation_box(single.batch, box)) {
          return NEPA_STATUS_UNSUPPORTED;
        }
        nep_adapters::cuda_backend::stage_batch_on_device(
            single.batch, protocol_.num_types, workspace);
        const auto view = workspace.view();
        clear_device_doubles(view.potential, view.atom_capacity);
        clear_device_doubles(view.force_soa3, view.atom_capacity * 3);
        if (view.mforce_soa3 != nullptr) {
          clear_device_doubles(view.mforce_soa3, view.atom_capacity * 3);
        }
        clear_device_doubles(view.virial_soa9, view.atom_capacity * 9);
        nep_adapters::cuda_backend::build_internal_neighbors_on_device(
            protocol_,
            single.atom_count,
            box,
            workspace);
        const bool has_angular = needs_angular_terms(protocol_);
        if (protocol_.spin_mode != 0 || protocol_.charge_mode == 0) {
          const auto force_request = make_force_evaluation_request(
              nep_adapters::cuda_backend::
                  ForceNeighborTopology::single_box_symmetric,
              true,
              true,
              result.virials_per_atom_row_major9 != nullptr,
              result.spin_transfer_per_atom_row_major9 != nullptr);
          nep_adapters::cuda_backend::run_force_pipeline(
              protocol_,
              force_request,
              single.atom_count,
              box,
              device_,
              workspace);
        } else {
          nep_adapters::cuda_backend::build_descriptor_core_from_positions_on_device(
              protocol_,
              single.atom_count,
              box,
              device_,
              workspace,
              nep_adapters::cuda_backend::DescriptorCoreTopology::single_box);
          nep_adapters::cuda_backend::evaluate_qnep_ann_on_device(
              protocol_, single.atom_count, device_, workspace);
          if (has_angular) {
            nep_adapters::cuda_backend::build_atom_type_schedule_on_device(
                protocol_, single.atom_count, workspace);
          }
          nep_adapters::cuda_backend::zero_total_charge_on_device(
              single.atom_count,
              workspace);
          nep_adapters::cuda_backend::apply_qnep_charge_terms_on_device(
              protocol_,
              single.atom_count,
              box,
              workspace,
              result.potential_per_atom != nullptr,
              result.virials_per_atom_row_major9 != nullptr);
          nep_adapters::cuda_backend::add_charge_chain_to_fp_on_device(
              protocol_,
              single.atom_count,
              workspace);
          const auto charge_virial_target = select_fp64_virial_target(
              true,
              result.virials_per_atom_row_major9 != nullptr);
          nep_adapters::cuda_backend::accumulate_radial_forces_on_device(
              protocol_,
              single.atom_count,
              box,
              device_,
              workspace,
              charge_virial_target,
              false);
          if (has_angular) {
            nep_adapters::cuda_backend::accumulate_l2_angular_forces_on_device(
                protocol_,
                single.atom_count,
                device_,
                workspace,
                charge_virial_target);
          }
          if (protocol_.has_zbl) {
            nep_adapters::cuda_backend::accumulate_zbl_forces_on_device(
                protocol_,
                single.atom_count,
                box,
                device_,
                workspace,
                true,
                charge_virial_target);
          }
        }

        nep_adapters::cuda_backend::prepare_batched_outputs(
            1, single.atom_count, workspace);
        copy_prepared_batch_results_to_host(
            view,
            1,
            single.atom_count,
            structure,
            single.atom_offset,
            protocol_.spin_mode != 0,
            protocol_.charge_mode > 0,
            result);
        if (protocol_.charge_mode > 0 &&
            result.bec_per_atom_row_major9 != nullptr) {
          if (host_.sqrt_epsilon_inf_offset >= host_.ann_type_major.size()) {
            throw std::runtime_error("qNEP model is missing sqrt_epsilon_inf");
          }
          nep_adapters::cuda_backend::compute_qnep_bec_on_device(
              protocol_,
              single.atom_count,
              box,
              host_.ann_type_major[host_.sqrt_epsilon_inf_offset],
              device_,
              workspace);
          copy_qnep_bec_to_host(
              view,
              single.atom_count,
              single.atom_offset,
              result.bec_per_atom_row_major9);
        }
        if (is_cancelled()) {
          return NEPA_STATUS_CANCELLED;
        }
      }
      return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
    } catch (const std::invalid_argument& error) {
      nep_adapters::set_last_error(error.what());
      return NEPA_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
      nep_adapters::set_last_error(error.what());
      return NEPA_STATUS_RUNTIME_ERROR;
    }
    return NEPA_STATUS_UNSUPPORTED;
  }

  NepaStatus find_descriptors(
      const NepaStructureBatch& batch,
      NepaFindDescriptorResult& result) override {
    if (!valid_batch(batch) || result.descriptors == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    if (is_cancelled()) {
      return NEPA_STATUS_CANCELLED;
    }
    if (protocol_.spin_mode != 0 && batch.spins_aos3 == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    try {
      if (!supports_cuda_descriptor_protocol(protocol_)) {
        return NEPA_STATUS_UNSUPPORTED;
      }
      ExpandedPeriodicBatch expanded =
          expand_small_periodic_batch(batch, protocol_.cutoff_max);
      if (expanded.expanded) {
        std::vector<double> expanded_descriptors(
            static_cast<std::size_t>(expanded.batch.total_atoms) *
            static_cast<std::size_t>(protocol_.descriptor_dim));
        NepaFindDescriptorResult expanded_result{};
        expanded_result.descriptors = expanded_descriptors.data();
        const NepaStatus status =
            find_descriptors(expanded.batch, expanded_result);
        if (status != NEPA_STATUS_OK) {
          return status;
        }
        std::fill(
            result.descriptors,
            result.descriptors +
                static_cast<std::size_t>(batch.total_atoms) *
                    static_cast<std::size_t>(protocol_.descriptor_dim),
            0.0);
        for (std::size_t atom = 0;
             atom < expanded.original_atom.size();
             ++atom) {
          const std::size_t original =
              static_cast<std::size_t>(expanded.original_atom[atom]);
          const double weight = expanded.atom_weight[atom];
          for (int dim = 0; dim < protocol_.descriptor_dim; ++dim) {
            result.descriptors[
                original * static_cast<std::size_t>(protocol_.descriptor_dim) +
                static_cast<std::size_t>(dim)] +=
                expanded_descriptors[
                    atom * static_cast<std::size_t>(protocol_.descriptor_dim) +
                    static_cast<std::size_t>(dim)] *
                weight;
          }
        }
        return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
      }
      nep_adapters::cuda_backend::DeviceWorkspace& workspace =
          batch_workspace(max_structure_atom_count(batch), 1, false);
      for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
        if (is_cancelled()) {
          return NEPA_STATUS_CANCELLED;
        }
        const StructureBatchView single(batch, structure);
        nep_adapters::cuda_backend::SimulationBox box;
        if (!make_simulation_box(single.batch, box)) {
          return NEPA_STATUS_UNSUPPORTED;
        }
        nep_adapters::cuda_backend::stage_batch_on_device(
            single.batch, protocol_.num_types, workspace);
        nep_adapters::cuda_backend::build_internal_neighbors_on_device(
            protocol_,
            single.atom_count,
            box,
            workspace);
        nep_adapters::cuda_backend::build_descriptor_core_from_positions_on_device(
            protocol_,
            single.atom_count,
            box,
            device_,
            workspace,
            nep_adapters::cuda_backend::DescriptorCoreTopology::single_box);
        if (protocol_.spin_mode != 0) {
          nep_adapters::cuda_backend::build_spin_descriptors_on_device(
              protocol_,
              single.atom_count,
              box,
              device_,
              workspace);
        }
        const nep_adapters::cuda_backend::DeviceWorkspaceView view = workspace.view();
        const std::vector<float> descriptor_soa = copy_device_floats(
            view.descriptors,
            view.atom_capacity * static_cast<std::size_t>(protocol_.descriptor_dim));
        for (int atom = 0; atom < single.atom_count; ++atom) {
          const std::size_t global_atom =
              static_cast<std::size_t>(single.atom_offset + atom);
          for (int dim = 0; dim < protocol_.descriptor_dim; ++dim) {
            double value = descriptor_soa[
                static_cast<std::size_t>(dim) * view.atom_capacity +
                static_cast<std::size_t>(atom)];
            if (static_cast<std::size_t>(dim) < host_.q_scaler.size()) {
              value *= host_.q_scaler[static_cast<std::size_t>(dim)];
            }
            result.descriptors[
                global_atom * static_cast<std::size_t>(protocol_.descriptor_dim) +
                static_cast<std::size_t>(dim)] = value;
          }
        }
      }
      return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
    } catch (const std::invalid_argument& error) {
      nep_adapters::set_last_error(error.what());
      return NEPA_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
      nep_adapters::set_last_error(error.what());
      return NEPA_STATUS_RUNTIME_ERROR;
    }
    return NEPA_STATUS_UNSUPPORTED;
  }

  NepaStatus compute_dftd3_batch(
      const NepaStructureBatch&,
      const NepaDftd3Parameters&,
      NepaDftd3Result&,
      bool) override {
    nep_adapters::set_last_error(
        "DFT-D3 is unsupported by the CUDA backend; CPU fallback is disabled");
    return NEPA_STATUS_UNSUPPORTED;
  }

  NepaStatus find_force_lammps_neighbors(
      const NepaLammpsNeighborInput& input,
      NepaLammpsNeighborResult& result) override {
    if (input.nlocal < 0 || input.inum < 0 || input.ilist == nullptr ||
        input.numneigh == nullptr || input.firstneigh == nullptr ||
        input.types == nullptr || input.type_map == nullptr ||
        input.positions == nullptr || result.total_potential == nullptr ||
        result.total_virial6 == nullptr || result.forces == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    if (is_cancelled()) {
      return NEPA_STATUS_CANCELLED;
    }
    try {
      if (!supports_cuda_force_protocol(protocol_)) {
        return NEPA_STATUS_UNSUPPORTED;
      }
      if (result.spin_transfer_per_atom_row_major9 != nullptr &&
          protocol_.spin_mode == 0) {
        return NEPA_STATUS_UNSUPPORTED;
      }
      LammpsHostPairProfiler profiler(input.nlocal);
      float workspace_wall_ms = 0.0f;
      float neighbor_stage_wall_ms = 0.0f;
      float clear_wall_ms = 0.0f;
      float pipeline_wall_ms = 0.0f;
      float output_wall_ms = 0.0f;
      const int atom_capacity = lammps_atom_capacity(input);
      if (atom_capacity <= 0) {
        return NEPA_STATUS_INVALID_ARGUMENT;
      }
      nep_adapters::cuda_backend::DeviceWorkspace workspace(
          nep_adapters::cuda_backend::make_external_neighbor_workspace_plan(
              protocol_,
              static_cast<std::size_t>(atom_capacity),
              static_cast<std::size_t>(input.inum),
              false,
              result.spin_transfer_per_atom_row_major9 != nullptr));
      profiler.split(workspace_wall_ms);
      nep_adapters::cuda_backend::stage_lammps_external_neighbors_on_device(
          input,
          protocol_,
          workspace);
      profiler.split(neighbor_stage_wall_ms);

      const nep_adapters::cuda_backend::SimulationBox box =
          make_nonperiodic_lammps_box();
      const nep_adapters::cuda_backend::DeviceWorkspaceView view = workspace.view();
      clear_device_doubles(view.potential, view.atom_capacity);
      if (view.force_soa3 != nullptr) {
        clear_device_doubles(view.force_soa3, view.atom_capacity * 3);
      }
      if (view.mforce_soa3 != nullptr) {
        clear_device_doubles(view.mforce_soa3, view.atom_capacity * 3);
      }
      if (view.virial_soa9 != nullptr) {
        clear_device_doubles(view.virial_soa9, view.atom_capacity * 9);
      }
      if (profiler.enabled()) {
        const cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
          throw std::runtime_error(
              std::string("synchronize host LAMMPS clear: ") +
              cudaGetErrorString(status));
        }
      }
      profiler.split(clear_wall_ms);
      const auto force_request = make_force_evaluation_request(
          nep_adapters::cuda_backend::
              ForceNeighborTopology::single_box_symmetric,
          true,
          result.total_virial6 != nullptr,
          result.virials_per_atom9 != nullptr,
          result.spin_transfer_per_atom_row_major9 != nullptr);
      nep_adapters::cuda_backend::ForcePipelineTimings pipeline_timings;
      nep_adapters::cuda_backend::run_force_pipeline(
          protocol_,
          force_request,
          atom_capacity,
          box,
          device_,
          workspace,
          profiler.enabled() ? &pipeline_timings : nullptr);
      profiler.split(pipeline_wall_ms);

      const std::vector<double> potential =
          copy_device_doubles(view.potential, view.atom_capacity);
      const std::vector<double> force_soa =
          copy_device_doubles(view.force_soa3, view.atom_capacity * 3);
      const std::vector<double> mforce_soa =
          result.mforces != nullptr && protocol_.spin_mode != 0
              ? copy_device_doubles(view.mforce_soa3, view.atom_capacity * 3)
              : std::vector<double>{};
      const std::vector<double> virial_soa =
          copy_device_doubles(view.virial_soa9, view.atom_capacity * 9);
      const std::vector<float> spin_transfer_soa =
          result.spin_transfer_per_atom_row_major9 != nullptr
              ? copy_device_floats(
                    view.spin_transfer_soa9, view.atom_capacity * 9)
              : std::vector<float>{};

      double total_potential = 0.0;
      double total_raw9[9] = {};
      for (int atom = 0; atom < input.nlocal; ++atom) {
        total_potential += potential[static_cast<std::size_t>(atom)];
        if (result.potential_per_atom != nullptr) {
          result.potential_per_atom[atom] = potential[static_cast<std::size_t>(atom)];
        }
        if (result.forces[atom] != nullptr) {
          result.forces[atom][0] = force_soa[static_cast<std::size_t>(atom)];
          result.forces[atom][1] =
              force_soa[view.atom_capacity + static_cast<std::size_t>(atom)];
          result.forces[atom][2] =
              force_soa[2 * view.atom_capacity + static_cast<std::size_t>(atom)];
        }
        if (result.mforces != nullptr && result.mforces[atom] != nullptr &&
            protocol_.spin_mode != 0) {
          result.mforces[atom][0] = mforce_soa[static_cast<std::size_t>(atom)];
          result.mforces[atom][1] =
              mforce_soa[view.atom_capacity + static_cast<std::size_t>(atom)];
          result.mforces[atom][2] =
              mforce_soa[2 * view.atom_capacity + static_cast<std::size_t>(atom)];
        }
      }
      for (int atom = 0; atom < atom_capacity; ++atom) {
        if (result.virials_per_atom9 != nullptr &&
            result.virials_per_atom9[atom] != nullptr) {
          for (int component = 0; component < 9; ++component) {
            const double value =
                virial_soa[static_cast<std::size_t>(component) *
                               view.atom_capacity +
                           static_cast<std::size_t>(atom)];
            result.virials_per_atom9[atom][component] = value;
          }
        }
        if (result.spin_transfer_per_atom_row_major9 != nullptr &&
            result.spin_transfer_per_atom_row_major9[atom] != nullptr) {
          for (int component = 0; component < 9; ++component) {
            result.spin_transfer_per_atom_row_major9[atom][component] =
                spin_transfer_soa[
                    static_cast<std::size_t>(component) *
                        view.atom_capacity +
                    atom];
          }
        }
      }
      const int total_virial_atoms =
          nep_adapters::cuda_backend::requests_per_atom_virial(force_request)
              ? atom_capacity
              : input.nlocal;
      for (int atom = 0; atom < total_virial_atoms; ++atom) {
        for (int component = 0; component < 9; ++component) {
          total_raw9[component] +=
              virial_soa[static_cast<std::size_t>(component) *
                             view.atom_capacity +
                         static_cast<std::size_t>(atom)];
        }
      }
      *result.total_potential = total_potential;
      for (int component = 0; component < 6; ++component) {
        result.total_virial6[component] =
            nep_adapters::lammps_voigt6_from_lammps_raw9(
                total_raw9,
                component);
      }
      profiler.split(output_wall_ms);
      profiler.print(
          workspace_wall_ms,
          neighbor_stage_wall_ms,
          clear_wall_ms,
          pipeline_wall_ms,
          pipeline_timings,
          output_wall_ms);
      return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
    } catch (const std::invalid_argument& error) {
      nep_adapters::set_last_error(error.what());
      return NEPA_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
      nep_adapters::set_last_error(error.what());
      return NEPA_STATUS_RUNTIME_ERROR;
    }
    return NEPA_STATUS_UNSUPPORTED;
  }

  NepaStatus find_force_lammps_device_neighbors(
      const NepaLammpsDeviceNeighborInput& input,
      NepaLammpsDeviceNeighborResult& result) override {
    const auto invalid_argument = [](const char* message) {
      nep_adapters::set_last_error(message);
      return NEPA_STATUS_INVALID_ARGUMENT;
    };
    if (input.nlocal < 0 || input.nall < input.nlocal || input.inum <= 0 ||
        input.max_neighbors < 0 || input.neighbor_rows <= 0 ||
        input.numneigh_length <= 0 || input.ilist == nullptr ||
        input.numneigh == nullptr || input.neighbors == nullptr ||
        input.neighbor_atom_stride <= 0 || input.neighbor_slot_stride <= 0 ||
        input.types == nullptr || input.positions == nullptr ||
        input.position_atom_stride <= 0 || input.position_component_stride <= 0 ||
        result.forces == nullptr || result.force_atom_stride <= 0 ||
        result.force_component_stride <= 0) {
      return invalid_argument("invalid device LAMMPS neighbor input or force output");
    }
    if ((result.total_potential == nullptr) !=
        (result.total_virial6 == nullptr)) {
      return invalid_argument("device LAMMPS totals require both energy and virial pointers");
    }
    if (result.virials_per_atom9 != nullptr &&
        (result.virial_atom_stride <= 0 ||
         result.virial_component_stride <= 0)) {
      return invalid_argument("invalid device LAMMPS per-atom virial layout");
    }
    if ((input.type_map == nullptr && input.type_map_length != 0) ||
        (input.type_map != nullptr && input.type_map_length <= 0)) {
      return invalid_argument("invalid device LAMMPS type map layout");
    }
    if (protocol_.spin_mode != 0 &&
        (input.spins == nullptr || input.spin_atom_stride <= 0 ||
         input.spin_component_stride <= 0)) {
      return invalid_argument("spin model requires valid device LAMMPS spins");
    }
    if (result.mforces != nullptr &&
        (result.mforce_atom_stride <= 0 || result.mforce_component_stride <= 0)) {
      return invalid_argument("invalid device LAMMPS mforce layout");
    }
    if (result.spin_transfer_per_atom_row_major9 != nullptr &&
        (result.spin_transfer_atom_stride <= 0 ||
         result.spin_transfer_component_stride <= 0)) {
      return invalid_argument("invalid device spin-transfer layout");
    }
    if (result.spin_transfer_per_atom_row_major9 != nullptr &&
        protocol_.spin_mode == 0) {
      return NEPA_STATUS_UNSUPPORTED;
    }
    if (is_cancelled()) {
      return NEPA_STATUS_CANCELLED;
    }

    try {
      if (!supports_cuda_force_protocol(protocol_)) {
        return NEPA_STATUS_UNSUPPORTED;
      }
      auto padded_capacity = [](std::size_t required) {
        return required + required / 64 + 1024;
      };
      const bool positions_are_soa =
          input.position_atom_stride == 1 &&
          input.position_component_stride >= input.nall;
      std::size_t atom_capacity = static_cast<std::size_t>(input.nall);
      if (positions_are_soa) {
        atom_capacity = static_cast<std::size_t>(input.position_component_stride);
      } else if (
          lammps_device_workspace_ != nullptr &&
          atom_capacity <= lammps_device_atom_capacity_) {
        atom_capacity = lammps_device_atom_capacity_;
      } else {
        atom_capacity = padded_capacity(atom_capacity);
      }
      std::size_t active_atom_capacity = static_cast<std::size_t>(input.inum);
      if (lammps_device_workspace_ != nullptr &&
          active_atom_capacity <= lammps_device_active_atom_capacity_) {
        active_atom_capacity = lammps_device_active_atom_capacity_;
      } else {
        active_atom_capacity = padded_capacity(active_atom_capacity);
      }
      active_atom_capacity = std::min(active_atom_capacity, atom_capacity);
      nep_adapters::cuda_backend::ModelProtocol external_protocol = protocol_;
      if (input.max_neighbors >= 0) {
        external_protocol.neighbor_capacity_radial = std::min(
            external_protocol.neighbor_capacity_radial,
            input.max_neighbors);
        external_protocol.neighbor_capacity_angular = std::min(
            external_protocol.neighbor_capacity_angular,
            input.max_neighbors);
      }
      const bool needs_per_atom_virial_sink =
          external_protocol.charge_mode == 0 &&
          result.virials_per_atom9 != nullptr;
      const bool needs_spin_transfer =
          result.spin_transfer_per_atom_row_major9 != nullptr;
      const bool rebuild_workspace =
          lammps_device_workspace_ == nullptr ||
          atom_capacity > lammps_device_atom_capacity_ ||
          active_atom_capacity > lammps_device_active_atom_capacity_ ||
          external_protocol.neighbor_capacity_radial >
              lammps_device_radial_capacity_ ||
          external_protocol.neighbor_capacity_angular >
              lammps_device_angular_capacity_ ||
          (needs_per_atom_virial_sink &&
           !lammps_device_has_per_atom_virial_sink_) ||
          (needs_spin_transfer &&
           !lammps_device_has_spin_transfer_);
      const bool neighbor_capacity_needs_runtime_check =
          input.max_neighbors > external_protocol.neighbor_capacity_radial ||
          input.max_neighbors > external_protocol.neighbor_capacity_angular;
      LammpsDevicePairProfiler profiler(input.nlocal, rebuild_workspace);
      float stage_ms = 0.0f;
      float clear_ms = 0.0f;
      float descriptor_ann_ms = 0.0f;
      float radial_force_ms = 0.0f;
      float angular_force_ms = 0.0f;
      float zbl_force_ms = 0.0f;
      float spin_onsite_ms = 0.0f;
      float spin_density_ms = 0.0f;
      float spin_density_pull_ms = 0.0f;
      float spin_density_edge_ms = 0.0f;
      float spin_chiral_ms = 0.0f;
      float output_ms = 0.0f;
      nep_adapters::cuda_backend::DeviceWorkspace& workspace =
          lammps_device_workspace(
              external_protocol,
              atom_capacity,
              active_atom_capacity,
              needs_per_atom_virial_sink,
              needs_spin_transfer);
      nep_adapters::cuda_backend::stage_lammps_device_neighbors_on_device(
          input,
          external_protocol,
          workspace,
          rebuild_workspace || neighbor_capacity_needs_runtime_check);
      profiler.split(stage_ms);

      const nep_adapters::cuda_backend::SimulationBox box =
          make_nonperiodic_lammps_box();
      const nep_adapters::cuda_backend::DeviceWorkspaceView view = workspace.view();
      const bool store_potential =
          result.total_potential != nullptr || result.potential_per_atom != nullptr;
      const bool accumulate_virial =
          result.total_virial6 != nullptr || result.virials_per_atom9 != nullptr;
      if (store_potential) {
        clear_device_doubles(view.potential, static_cast<std::size_t>(input.nlocal));
      }
      clear_device_doubles(view.force_soa3, view.atom_capacity * 3);
      if (view.mforce_soa3 != nullptr) {
        clear_device_doubles(view.mforce_soa3, view.atom_capacity * 3);
      }
      if (accumulate_virial) {
        clear_device_doubles(view.virial_soa9, view.atom_capacity * 9);
      }
      profiler.split(clear_ms);
      const auto force_request = make_force_evaluation_request(
          nep_adapters::cuda_backend::ForceNeighborTopology::external_full,
          store_potential,
          result.total_virial6 != nullptr,
          result.virials_per_atom9 != nullptr,
          needs_spin_transfer);
      nep_adapters::cuda_backend::ForcePipelineTimings pipeline_timings;
      nep_adapters::cuda_backend::run_force_pipeline(
          external_protocol,
          force_request,
          input.nlocal,
          box,
          device_,
          workspace,
          profiler.enabled() ? &pipeline_timings : nullptr);
      descriptor_ann_ms = pipeline_timings.descriptor_ann_ms;
      radial_force_ms = pipeline_timings.radial_force_ms;
      angular_force_ms = pipeline_timings.angular_force_ms;
      zbl_force_ms = pipeline_timings.zbl_force_ms;
      spin_onsite_ms = pipeline_timings.spin_onsite_ms;
      spin_density_ms = pipeline_timings.spin_density_ms;
      spin_density_pull_ms = pipeline_timings.spin_density_pull_ms;
      spin_density_edge_ms = pipeline_timings.spin_density_edge_ms;
      spin_chiral_ms = pipeline_timings.spin_chiral_ms;
      profiler.reset();
      nep_adapters::cuda_backend::write_lammps_device_outputs(
          input,
          result,
          workspace);
      profiler.split(output_ms);
      profiler.print(
          stage_ms,
          clear_ms,
          descriptor_ann_ms,
          radial_force_ms,
          angular_force_ms,
          zbl_force_ms,
          spin_onsite_ms,
          spin_density_ms,
          spin_density_pull_ms,
          spin_density_edge_ms,
          spin_chiral_ms,
          output_ms);
      return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
    } catch (const std::invalid_argument& error) {
      nep_adapters::set_last_error(error.what());
      return NEPA_STATUS_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
      nep_adapters::set_last_error(error.what());
      return NEPA_STATUS_RUNTIME_ERROR;
    }
    return NEPA_STATUS_UNSUPPORTED;
  }

 private:
  nep_adapters::cuda_backend::DeviceWorkspace& batch_workspace(
      std::size_t atom_capacity,
      std::size_t structure_capacity,
      bool needs_spin_transfer) {
    const bool same_execution_shape =
        (structure_capacity == 1) == (batch_structure_capacity_ == 1);
    if (batch_workspace_ != nullptr && same_execution_shape &&
        atom_capacity <= batch_atom_capacity_ &&
        structure_capacity <= batch_structure_capacity_ &&
        (!needs_spin_transfer || batch_has_spin_transfer_)) {
      return *batch_workspace_;
    }

    batch_workspace_ =
        std::make_unique<nep_adapters::cuda_backend::DeviceWorkspace>(
            nep_adapters::cuda_backend::make_internal_neighbor_workspace_plan(
                protocol_,
                atom_capacity,
                structure_capacity,
                needs_spin_transfer));
    batch_atom_capacity_ = atom_capacity;
    batch_structure_capacity_ = structure_capacity;
    batch_has_spin_transfer_ = needs_spin_transfer;
    return *batch_workspace_;
  }

  nep_adapters::cuda_backend::DeviceWorkspace& lammps_device_workspace(
      const nep_adapters::cuda_backend::ModelProtocol& protocol,
      std::size_t atom_capacity,
      std::size_t active_atom_capacity,
      bool needs_per_atom_virial_sink,
      bool needs_spin_transfer) {
    if (lammps_device_workspace_ != nullptr &&
        atom_capacity <= lammps_device_atom_capacity_ &&
        active_atom_capacity <= lammps_device_active_atom_capacity_ &&
        protocol.neighbor_capacity_radial <= lammps_device_radial_capacity_ &&
        protocol.neighbor_capacity_angular <= lammps_device_angular_capacity_ &&
        (!needs_per_atom_virial_sink ||
         lammps_device_has_per_atom_virial_sink_) &&
        (!needs_spin_transfer || lammps_device_has_spin_transfer_)) {
      return *lammps_device_workspace_;
    }

    auto workspace = std::make_unique<nep_adapters::cuda_backend::DeviceWorkspace>(
        nep_adapters::cuda_backend::make_external_neighbor_workspace_plan(
            protocol,
            atom_capacity,
            active_atom_capacity,
            needs_per_atom_virial_sink,
            needs_spin_transfer));
    lammps_device_workspace_ = std::move(workspace);
    lammps_device_atom_capacity_ = atom_capacity;
    lammps_device_active_atom_capacity_ = active_atom_capacity;
    lammps_device_radial_capacity_ = protocol.neighbor_capacity_radial;
    lammps_device_angular_capacity_ = protocol.neighbor_capacity_angular;
    lammps_device_has_per_atom_virial_sink_ = needs_per_atom_virial_sink;
    lammps_device_has_spin_transfer_ = needs_spin_transfer;
    return *lammps_device_workspace_;
  }

  nep_adapters::cuda_backend::HostModelParameters host_;
  nep_adapters::cuda_backend::ModelProtocol protocol_;
  nep_adapters::cuda_backend::DeviceModel device_;
  std::unique_ptr<nep_adapters::cuda_backend::DeviceWorkspace> batch_workspace_;
  std::size_t batch_atom_capacity_ = 0;
  std::size_t batch_structure_capacity_ = 0;
  bool batch_has_spin_transfer_ = false;
  std::unique_ptr<nep_adapters::cuda_backend::DeviceWorkspace> lammps_device_workspace_;
  std::size_t lammps_device_atom_capacity_ = 0;
  std::size_t lammps_device_active_atom_capacity_ = 0;
  int lammps_device_radial_capacity_ = 0;
  int lammps_device_angular_capacity_ = 0;
  bool lammps_device_has_per_atom_virial_sink_ = false;
  bool lammps_device_has_spin_transfer_ = false;
};

class CudaEngine : public nep_adapters::Engine {
 public:
  nep_adapters::EngineInfo info() const override {
    return {
        "cuda",
        NEP_ADAPTERS_VERSION_STRING,
        nep_adapters::to_mask(nep_adapters::Capability::device_input)};
  }

  NepaStatus load_model(
      const std::string& model_path,
      std::unique_ptr<nep_adapters::Model>& out) override {
    if (model_path.empty()) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }

    try {
      out = std::make_unique<CudaModel>(model_path);
      return NEPA_STATUS_OK;
    } catch (const nep_adapters::cuda_backend::UnsupportedModelProtocol& error) {
      nep_adapters::set_last_error(error.what());
      out.reset();
      return NEPA_STATUS_UNSUPPORTED;
    } catch (const std::exception& error) {
      nep_adapters::set_last_error(error.what());
      out.reset();
      return NEPA_STATUS_RUNTIME_ERROR;
    }
  }
};

}  // namespace

namespace nep_adapters {

bool register_cuda_engine() {
  static CudaEngine engine;
  static const bool registered = register_engine(&engine);
  return registered;
}

}  // namespace nep_adapters

extern "C" int nepa_register_cuda_engine(void) {
  return nep_adapters::register_cuda_engine() ? 1 : 0;
}
