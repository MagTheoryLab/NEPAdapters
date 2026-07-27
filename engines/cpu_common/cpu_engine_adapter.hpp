#pragma once

#include "nep_adapters/engine.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace nep_adapters {

template <typename T, typename = void>
struct HasSpin : std::false_type {};

template <typename T>
struct HasSpin<T, std::void_t<decltype(std::declval<T>().paramb.spin_mode)>>
    : std::true_type {};

// Each structure worker owns a NativeNep copy. These measured work floors keep
// model-copy cost from dominating small ordinary batches.
constexpr std::int64_t kForceBatchAtomsPerTypeWorker = 4;
constexpr std::int64_t kDescriptorBatchAtomsPerTypeWorker = 10;

template <typename NativeNep>
class CpuModel final : public Model {
 public:
  explicit CpuModel(const std::string& model_path) {
    nep_.init_from_file(model_path, false);
  }

  NepaModelKind model_kind() const override {
    if (nep_.paramb.model_type == 1) {
      return NEPA_MODEL_KIND_DIPOLE;
    }
    if (nep_.paramb.model_type == 2) {
      return NEPA_MODEL_KIND_POLARIZABILITY;
    }
    if constexpr (HasSpin<NativeNep>::value) {
      if (nep_.paramb.spin_mode > 0) {
        return NEPA_MODEL_KIND_SPIN;
      }
    }
    if (nep_.paramb.charge_mode > 0) {
      return NEPA_MODEL_KIND_CHARGE;
    }
    return NEPA_MODEL_KIND_ORDINARY;
  }

  NepaStatus model_info(NepaModelInfo& out) const override {
    out = {};
    out.cutoff_radial = nep_.paramb.rc_radial_max;
    out.cutoff_angular = nep_.paramb.rc_angular_max;
    out.cutoff_max = std::max(out.cutoff_radial, out.cutoff_angular);
    if (nep_.zbl.enabled) {
      out.cutoff_max = std::max(out.cutoff_max, nep_.zbl.rc_outer);
    }
    const NepaModelKind kind = model_kind();
    if (kind == NEPA_MODEL_KIND_DIPOLE) {
      out.capabilities = to_mask(Capability::dipole) |
                         to_mask(Capability::descriptors);
    } else if (kind == NEPA_MODEL_KIND_POLARIZABILITY) {
      out.capabilities = to_mask(Capability::polarizability) |
                         to_mask(Capability::descriptors);
    } else {
      out.capabilities = to_mask(Capability::batch_find_force) |
                         to_mask(Capability::external_neighbors) |
                         to_mask(Capability::virial) |
                         to_mask(Capability::descriptors);
      if (kind == NEPA_MODEL_KIND_CHARGE) {
        out.capabilities |= to_mask(Capability::charge);
      } else if (kind == NEPA_MODEL_KIND_SPIN) {
        out.capabilities |= to_mask(Capability::spin);
        out.capabilities |= to_mask(Capability::spin_energy_transfer);
      } else {
        out.capabilities |= to_mask(Capability::dftd3);
        out.capabilities |= NEPA_CAPABILITY_EVALUATE_WITH_DESCRIPTORS;
      }
    }
    out.num_types = static_cast<std::int32_t>(nep_.paramb.num_types);
    out.descriptor_dim = static_cast<std::int32_t>(nep_.annmb.dim);
    return NEPA_STATUS_OK;
  }

  NepaStatus find_force_batch(
      const NepaStructureBatch& batch,
      NepaFindForceResult& result) override {
    if (model_kind() != NEPA_MODEL_KIND_ORDINARY &&
        model_kind() != NEPA_MODEL_KIND_SPIN) {
      return NEPA_STATUS_UNSUPPORTED;
    }
    return find_force_batch_impl(batch, result);
  }

  NepaStatus evaluate_batch(
      const NepaStructureBatch& batch,
      NepaEvaluateResult& result) override {
    if (model_kind() != NEPA_MODEL_KIND_ORDINARY ||
        result.descriptor.descriptors == nullptr) {
      return NEPA_STATUS_UNSUPPORTED;
    }
    return find_force_batch_impl(batch, result.prediction, &result.descriptor);
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
      NepaFindDescriptorResult* descriptor_result = nullptr) {
    if (!valid_batch(batch) || result.energy_per_structure == nullptr ||
        result.forces_aos3 == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    if (is_cancelled()) {
      return NEPA_STATUS_CANCELLED;
    }

    if constexpr (HasSpin<NativeNep>::value) {
      if (nep_.paramb.spin_mode > 0 && batch.spins_aos3 == nullptr) {
        return NEPA_STATUS_INVALID_ARGUMENT;
      }
      if (nep_.paramb.spin_mode == 0 &&
          result.spin_transfer_per_atom_row_major9 != nullptr) {
        return NEPA_STATUS_UNSUPPORTED;
      }
    } else if (result.spin_transfer_per_atom_row_major9 != nullptr) {
      return NEPA_STATUS_UNSUPPORTED;
    }
    auto process_structure = [&](
        NativeNep& native,
        const std::int32_t structure) -> NepaStatus {
        if (is_cancelled()) {
          return NEPA_STATUS_CANCELLED;
        }
        const std::int32_t atom_count = batch.atom_counts[structure];
        const std::int32_t atom_offset = batch.atom_offsets[structure];

        std::vector<int> types(static_cast<std::size_t>(atom_count));
        std::vector<double> positions_soa(static_cast<std::size_t>(atom_count) * 3);
        std::vector<double> box(9);
        std::vector<double> potential(static_cast<std::size_t>(atom_count), 0.0);
        std::vector<double> force_soa(static_cast<std::size_t>(atom_count) * 3, 0.0);
        std::vector<double> virial_soa(static_cast<std::size_t>(atom_count) * 9, 0.0);

        for (std::int32_t atom = 0; atom < atom_count; ++atom) {
          const std::int32_t global_atom = atom_offset + atom;
          types[atom] = batch.types[global_atom];
          positions_soa[atom] = batch.positions_aos3[3 * global_atom + 0];
          positions_soa[static_cast<std::size_t>(atom_count) + atom] =
              batch.positions_aos3[3 * global_atom + 1];
          positions_soa[static_cast<std::size_t>(2) * atom_count + atom] =
              batch.positions_aos3[3 * global_atom + 2];
        }

        std::copy_n(
            batch.boxes_row_major9 + static_cast<std::size_t>(structure) * 9,
            9,
            box.data());

        std::vector<double> charge;
        std::vector<double> bec_soa;
        std::vector<double> descriptor_soa;
        if (descriptor_result != nullptr) {
          descriptor_soa.assign(
              static_cast<std::size_t>(atom_count) * native.annmb.dim, 0.0);
        }
        if constexpr (HasSpin<NativeNep>::value) {
          if (native.paramb.spin_mode > 0) {
            std::vector<double> spins_soa(static_cast<std::size_t>(atom_count) * 3);
            for (std::int32_t atom = 0; atom < atom_count; ++atom) {
              const std::int32_t global_atom = atom_offset + atom;
              spins_soa[atom] = batch.spins_aos3[3 * global_atom + 0];
              spins_soa[static_cast<std::size_t>(atom_count) + atom] =
                  batch.spins_aos3[3 * global_atom + 1];
              spins_soa[static_cast<std::size_t>(2) * atom_count + atom] =
                  batch.spins_aos3[3 * global_atom + 2];
            }
            descriptor_soa.assign(
                static_cast<std::size_t>(atom_count) * native.annmb.dim, 0.0);
            std::vector<double> mforce_soa(static_cast<std::size_t>(atom_count) * 3, 0.0);
            std::vector<double> spin_transfer_soa;
            if (result.spin_transfer_per_atom_row_major9 != nullptr) {
              spin_transfer_soa.assign(
                  static_cast<std::size_t>(atom_count) * 9, 0.0);
            }
            native.compute(
                types,
                box,
                positions_soa,
                spins_soa,
                potential,
                force_soa,
                virial_soa,
                descriptor_soa,
                mforce_soa,
                spin_transfer_soa.empty() ? nullptr : &spin_transfer_soa);
            for (std::int32_t atom = 0; atom < atom_count; ++atom) {
              const std::int32_t global_atom = atom_offset + atom;
              if (result.mforces_aos3 != nullptr) {
                result.mforces_aos3[3 * global_atom + 0] = mforce_soa[atom];
                result.mforces_aos3[3 * global_atom + 1] =
                    mforce_soa[static_cast<std::size_t>(atom_count) + atom];
                result.mforces_aos3[3 * global_atom + 2] =
                    mforce_soa[static_cast<std::size_t>(2) * atom_count + atom];
              }
              if (result.spin_transfer_per_atom_row_major9 != nullptr) {
                for (std::int32_t component = 0; component < 9; ++component) {
                  result.spin_transfer_per_atom_row_major9[
                      9 * static_cast<std::size_t>(global_atom) + component] =
                      spin_transfer_soa[
                          static_cast<std::size_t>(component) * atom_count +
                          atom];
                }
              }
            }
          } else if (native.paramb.charge_mode > 0) {
            charge.assign(static_cast<std::size_t>(atom_count), 0.0);
            bec_soa.assign(static_cast<std::size_t>(atom_count) * 9, 0.0);
            native.compute(
                types,
                box,
                positions_soa,
                potential,
                force_soa,
                virial_soa,
                charge,
                bec_soa);
          } else {
            native.compute(
                types,
                box,
                positions_soa,
                potential,
                force_soa,
                virial_soa,
                descriptor_soa.empty() ? nullptr : &descriptor_soa);
          }
        } else if (native.paramb.charge_mode > 0) {
          charge.assign(static_cast<std::size_t>(atom_count), 0.0);
          bec_soa.assign(static_cast<std::size_t>(atom_count) * 9, 0.0);
          native.compute(
              types,
              box,
              positions_soa,
              potential,
              force_soa,
              virial_soa,
              charge,
              bec_soa);
        } else {
          native.compute(
              types,
              box,
              positions_soa,
              potential,
              force_soa,
              virial_soa,
              descriptor_soa.empty() ? nullptr : &descriptor_soa);
        }

        result.energy_per_structure[structure] =
            std::accumulate(potential.begin(), potential.end(), 0.0);

        for (std::int32_t atom = 0; atom < atom_count; ++atom) {
          const std::int32_t global_atom = atom_offset + atom;
          if (result.potential_per_atom != nullptr) {
            result.potential_per_atom[global_atom] = potential[atom];
          }
          result.forces_aos3[3 * global_atom + 0] = force_soa[atom];
          result.forces_aos3[3 * global_atom + 1] =
              force_soa[static_cast<std::size_t>(atom_count) + atom];
          result.forces_aos3[3 * global_atom + 2] =
              force_soa[static_cast<std::size_t>(2) * atom_count + atom];

          if (result.virials_per_atom_row_major9 != nullptr) {
            for (std::int32_t component = 0; component < 9; ++component) {
              result.virials_per_atom_row_major9[9 * global_atom + component] =
                  virial_soa[static_cast<std::size_t>(component) * atom_count + atom];
            }
          }
          if (result.charge_per_atom != nullptr && !charge.empty()) {
            result.charge_per_atom[global_atom] = charge[atom];
          }
          if (result.bec_per_atom_row_major9 != nullptr && !bec_soa.empty()) {
            for (std::int32_t component = 0; component < 9; ++component) {
              result.bec_per_atom_row_major9[9 * global_atom + component] =
                  bec_soa[static_cast<std::size_t>(component) * atom_count + atom];
            }
          }
          if (descriptor_result != nullptr) {
            for (std::int32_t component = 0;
                 component < native.annmb.dim;
                 ++component) {
              descriptor_result->descriptors[
                  static_cast<std::size_t>(global_atom) * native.annmb.dim +
                  component] = descriptor_soa[
                      static_cast<std::size_t>(component) * atom_count + atom];
            }
          }
        }

        if (result.virials_row_major9 != nullptr) {
          double* virial_out =
              result.virials_row_major9 + static_cast<std::size_t>(structure) * 9;
          std::fill(virial_out, virial_out + 9, 0.0);
          for (std::int32_t component = 0; component < 9; ++component) {
            const std::size_t component_offset =
                static_cast<std::size_t>(component) * atom_count;
            for (std::int32_t atom = 0; atom < atom_count; ++atom) {
              virial_out[component] += virial_soa[component_offset + atom];
            }
          }
        }
        return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
    };

#if defined(_OPENMP)
    const NepaModelKind kind = model_kind();
    const int structure_threads =
        std::min<int>(batch.num_structures, omp_get_max_threads());
    const std::int64_t ordinary_parallel_min_atoms =
        kForceBatchAtomsPerTypeWorker * nep_.paramb.num_types * structure_threads;
    const bool use_structure_parallel = structure_threads > 1 &&
        (kind == NEPA_MODEL_KIND_SPIN ||
         (kind == NEPA_MODEL_KIND_ORDINARY &&
          batch.total_atoms >= ordinary_parallel_min_atoms));
#else
    const bool use_structure_parallel = false;
#endif

    try {
      if (use_structure_parallel) {
#if defined(_OPENMP)
        std::vector<NativeNep> workers(
            static_cast<std::size_t>(structure_threads),
            nep_);
        std::vector<NepaStatus> statuses(
            static_cast<std::size_t>(batch.num_structures),
            NEPA_STATUS_OK);
        std::vector<std::string> errors(
            static_cast<std::size_t>(batch.num_structures));
#pragma omp parallel num_threads(structure_threads)
        {
          omp_set_num_threads(1);
#pragma omp for schedule(dynamic, 1)
          for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
            try {
              statuses[static_cast<std::size_t>(structure)] =
                  process_structure(
                      workers[static_cast<std::size_t>(omp_get_thread_num())],
                      structure);
            } catch (const std::exception& error) {
              errors[static_cast<std::size_t>(structure)] = error.what();
              statuses[static_cast<std::size_t>(structure)] = NEPA_STATUS_RUNTIME_ERROR;
            }
          }
        }
        for (std::size_t structure = 0; structure < statuses.size(); ++structure) {
          if (statuses[structure] != NEPA_STATUS_OK) {
            if (!errors[structure].empty()) {
              set_last_error(errors[structure]);
            }
            return statuses[structure];
          }
        }
#endif
      } else {
        for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
          if (is_cancelled()) {
            return NEPA_STATUS_CANCELLED;
          }
          const NepaStatus status = process_structure(nep_, structure);
          if (status != NEPA_STATUS_OK) {
            return status;
          }
        }
      }
      return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
    } catch (const std::exception& error) {
      set_last_error(error.what());
      return NEPA_STATUS_RUNTIME_ERROR;
    }
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

    const std::int32_t descriptor_dim =
        static_cast<std::int32_t>(nep_.annmb.dim);
    if (descriptor_dim <= 0) {
      return NEPA_STATUS_UNSUPPORTED;
    }

    auto process_structure = [&](
        NativeNep& native,
        const std::int32_t structure) -> NepaStatus {
        if (is_cancelled()) {
          return NEPA_STATUS_CANCELLED;
        }
        const std::int32_t atom_count = batch.atom_counts[structure];
        const std::int32_t atom_offset = batch.atom_offsets[structure];
        if (atom_count <= 0 || atom_offset < 0 ||
            atom_offset + atom_count > batch.total_atoms) {
          return NEPA_STATUS_INVALID_ARGUMENT;
        }

        std::vector<int> types(static_cast<std::size_t>(atom_count));
        std::vector<double> positions_soa(static_cast<std::size_t>(atom_count) * 3);
        std::vector<double> box(9);
        std::vector<double> descriptor_soa(
            static_cast<std::size_t>(atom_count) * descriptor_dim,
            0.0);

        for (std::int32_t atom = 0; atom < atom_count; ++atom) {
          const std::int32_t global_atom = atom_offset + atom;
          types[atom] = batch.types[global_atom];
          positions_soa[atom] = batch.positions_aos3[3 * global_atom + 0];
          positions_soa[static_cast<std::size_t>(atom_count) + atom] =
              batch.positions_aos3[3 * global_atom + 1];
          positions_soa[static_cast<std::size_t>(2) * atom_count + atom] =
              batch.positions_aos3[3 * global_atom + 2];
        }

        std::copy_n(
            batch.boxes_row_major9 + static_cast<std::size_t>(structure) * 9,
            9,
            box.data());

        if constexpr (HasSpin<NativeNep>::value) {
          if (native.paramb.spin_mode > 0) {
            if (batch.spins_aos3 == nullptr) {
              return NEPA_STATUS_INVALID_ARGUMENT;
            }
            std::vector<double> spins_soa(static_cast<std::size_t>(atom_count) * 3);
            for (std::int32_t atom = 0; atom < atom_count; ++atom) {
              const std::int32_t global_atom = atom_offset + atom;
              spins_soa[atom] = batch.spins_aos3[3 * global_atom + 0];
              spins_soa[static_cast<std::size_t>(atom_count) + atom] =
                  batch.spins_aos3[3 * global_atom + 1];
              spins_soa[static_cast<std::size_t>(2) * atom_count + atom] =
                  batch.spins_aos3[3 * global_atom + 2];
            }
            native.find_descriptor(
                types, box, positions_soa, spins_soa, descriptor_soa);
          } else {
            native.find_descriptor(types, box, positions_soa, descriptor_soa);
          }
        } else {
          native.find_descriptor(types, box, positions_soa, descriptor_soa);
        }

        for (std::int32_t atom = 0; atom < atom_count; ++atom) {
          const std::int32_t global_atom = atom_offset + atom;
          for (std::int32_t component = 0; component < descriptor_dim; ++component) {
            result.descriptors[
                static_cast<std::size_t>(global_atom) * descriptor_dim + component] =
                descriptor_soa[
                    static_cast<std::size_t>(component) * atom_count + atom];
          }
        }
        return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
    };

#if defined(_OPENMP)
    const int structure_threads =
        std::min<int>(batch.num_structures, omp_get_max_threads());
    const std::int64_t ordinary_parallel_min_atoms =
        kDescriptorBatchAtomsPerTypeWorker * nep_.paramb.num_types * structure_threads;
    const NepaModelKind kind = model_kind();
    const bool use_structure_parallel = structure_threads > 1 &&
        (kind == NEPA_MODEL_KIND_SPIN ||
         (kind == NEPA_MODEL_KIND_ORDINARY &&
          batch.total_atoms >= ordinary_parallel_min_atoms));
#else
    const bool use_structure_parallel = false;
#endif

    try {
      if (use_structure_parallel) {
#if defined(_OPENMP)
        std::vector<NativeNep> workers(
            static_cast<std::size_t>(structure_threads),
            nep_);
        std::vector<NepaStatus> statuses(
            static_cast<std::size_t>(batch.num_structures),
            NEPA_STATUS_OK);
        std::vector<std::string> errors(
            static_cast<std::size_t>(batch.num_structures));
#pragma omp parallel num_threads(structure_threads)
        {
          omp_set_num_threads(1);
#pragma omp for schedule(dynamic, 1)
          for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
            try {
              statuses[static_cast<std::size_t>(structure)] =
                  process_structure(
                      workers[static_cast<std::size_t>(omp_get_thread_num())],
                      structure);
            } catch (const std::exception& error) {
              errors[static_cast<std::size_t>(structure)] = error.what();
              statuses[static_cast<std::size_t>(structure)] = NEPA_STATUS_RUNTIME_ERROR;
            }
          }
        }
        for (std::size_t structure = 0; structure < statuses.size(); ++structure) {
          if (statuses[structure] != NEPA_STATUS_OK) {
            if (!errors[structure].empty()) {
              set_last_error(errors[structure]);
            }
            return statuses[structure];
          }
        }
#endif
      } else {
        for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
          if (is_cancelled()) {
            return NEPA_STATUS_CANCELLED;
          }
          const NepaStatus status = process_structure(nep_, structure);
          if (status != NEPA_STATUS_OK) {
            return status;
          }
        }
      }

      return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
    } catch (const std::exception& error) {
      set_last_error(error.what());
      return NEPA_STATUS_RUNTIME_ERROR;
    }
  }

  NepaStatus find_dipoles(
      const NepaStructureBatch& batch,
      NepaDipoleResult& result) override {
    if (model_kind() != NEPA_MODEL_KIND_DIPOLE) {
      return NEPA_STATUS_UNSUPPORTED;
    }
    if (!valid_batch(batch) || result.dipoles_row_major3 == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    try {
      for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
        if (is_cancelled()) {
          return NEPA_STATUS_CANCELLED;
        }
        std::vector<int> types;
        std::vector<double> positions_soa;
        std::vector<double> box;
        const NepaStatus status = prepare_native_structure(
            batch, structure, types, positions_soa, box);
        if (status != NEPA_STATUS_OK) {
          return status;
        }
        std::vector<double> dipole(3, 0.0);
        nep_.find_dipole(types, box, positions_soa, dipole);
        if (is_cancelled()) {
          return NEPA_STATUS_CANCELLED;
        }
        std::copy_n(
            dipole.data(),
            3,
            result.dipoles_row_major3 + static_cast<std::size_t>(structure) * 3);
      }
      return NEPA_STATUS_OK;
    } catch (const std::exception& error) {
      set_last_error(error.what());
      return NEPA_STATUS_RUNTIME_ERROR;
    }
  }

  NepaStatus find_polarizabilities(
      const NepaStructureBatch& batch,
      NepaPolarizabilityResult& result) override {
    if (model_kind() != NEPA_MODEL_KIND_POLARIZABILITY) {
      return NEPA_STATUS_UNSUPPORTED;
    }
    if (!valid_batch(batch) || result.polarizabilities_row_major6 == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    try {
      for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
        if (is_cancelled()) {
          return NEPA_STATUS_CANCELLED;
        }
        std::vector<int> types;
        std::vector<double> positions_soa;
        std::vector<double> box;
        const NepaStatus status = prepare_native_structure(
            batch, structure, types, positions_soa, box);
        if (status != NEPA_STATUS_OK) {
          return status;
        }
        std::vector<double> polarizability(6, 0.0);
        nep_.find_polarizability(types, box, positions_soa, polarizability);
        if (is_cancelled()) {
          return NEPA_STATUS_CANCELLED;
        }
        std::copy_n(
            polarizability.data(),
            6,
            result.polarizabilities_row_major6 +
                static_cast<std::size_t>(structure) * 6);
      }
      return NEPA_STATUS_OK;
    } catch (const std::exception& error) {
      set_last_error(error.what());
      return NEPA_STATUS_RUNTIME_ERROR;
    }
  }

  NepaStatus compute_dftd3_batch(
      const NepaStructureBatch& batch,
      const NepaDftd3Parameters& parameters,
      NepaDftd3Result& result,
      bool include_nep) override {
    if (model_kind() != NEPA_MODEL_KIND_ORDINARY) {
      return NEPA_STATUS_UNSUPPORTED;
    }
    if (!valid_batch(batch) || parameters.functional == nullptr ||
        parameters.functional[0] == '\0' || parameters.cutoff <= 0.0 ||
        parameters.cutoff_cn <= 0.0 || result.energy_per_structure == nullptr ||
        result.forces_aos3 == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    try {
      for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
        if (is_cancelled()) {
          return NEPA_STATUS_CANCELLED;
        }
        std::vector<int> types;
        std::vector<double> positions_soa;
        std::vector<double> box;
        const NepaStatus status = prepare_native_structure(
            batch, structure, types, positions_soa, box);
        if (status != NEPA_STATUS_OK) {
          return status;
        }
        const std::int32_t atom_count = batch.atom_counts[structure];
        const std::int32_t atom_offset = batch.atom_offsets[structure];
        std::vector<double> potential(static_cast<std::size_t>(atom_count), 0.0);
        std::vector<double> force_soa(static_cast<std::size_t>(atom_count) * 3, 0.0);
        std::vector<double> virial_soa(static_cast<std::size_t>(atom_count) * 9, 0.0);
        if (include_nep) {
          nep_.compute_with_dftd3(
              parameters.functional,
              parameters.cutoff,
              parameters.cutoff_cn,
              types,
              box,
              positions_soa,
              potential,
              force_soa,
              virial_soa);
        } else {
          nep_.compute_dftd3(
              parameters.functional,
              parameters.cutoff,
              parameters.cutoff_cn,
              types,
              box,
              positions_soa,
              potential,
              force_soa,
              virial_soa);
        }
        if (is_cancelled()) {
          return NEPA_STATUS_CANCELLED;
        }
        result.energy_per_structure[structure] =
            std::accumulate(potential.begin(), potential.end(), 0.0);
        double* structure_virial =
            result.virials_row_major9 == nullptr
                ? nullptr
                : result.virials_row_major9 +
                      static_cast<std::size_t>(structure) * 9;
        if (structure_virial != nullptr) {
          std::fill_n(structure_virial, 9, 0.0);
        }
        for (std::int32_t atom = 0; atom < atom_count; ++atom) {
          const std::int32_t global_atom = atom_offset + atom;
          if (result.potential_per_atom != nullptr) {
            result.potential_per_atom[global_atom] = potential[atom];
          }
          for (std::int32_t component = 0; component < 3; ++component) {
            result.forces_aos3[3 * static_cast<std::size_t>(global_atom) + component] =
                force_soa[static_cast<std::size_t>(component) * atom_count + atom];
          }
          for (std::int32_t component = 0; component < 9; ++component) {
            const double value =
                virial_soa[static_cast<std::size_t>(component) * atom_count + atom];
            if (result.virials_per_atom_row_major9 != nullptr) {
              result.virials_per_atom_row_major9[
                  9 * static_cast<std::size_t>(global_atom) + component] = value;
            }
            if (structure_virial != nullptr) {
              structure_virial[component] += value;
            }
          }
        }
      }
      return NEPA_STATUS_OK;
    } catch (const std::exception& error) {
      set_last_error(error.what());
      return NEPA_STATUS_RUNTIME_ERROR;
    }
  }

  NepaStatus find_force_lammps_neighbors(
      const NepaLammpsNeighborInput& input,
      NepaLammpsNeighborResult& result) override {
    if (model_kind() == NEPA_MODEL_KIND_DIPOLE ||
        model_kind() == NEPA_MODEL_KIND_POLARIZABILITY) {
      return NEPA_STATUS_UNSUPPORTED;
    }
    if (is_cancelled()) {
      return NEPA_STATUS_CANCELLED;
    }
    if (input.nlocal < 0 || input.inum < 0 || input.ilist == nullptr ||
        input.numneigh == nullptr || input.firstneigh == nullptr ||
        input.types == nullptr || input.type_map == nullptr ||
        input.positions == nullptr || result.total_potential == nullptr ||
        result.total_virial6 == nullptr || result.forces == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }

    try {
      double total_potential = 0.0;
      double total_virial[6] = {};
      if constexpr (HasSpin<NativeNep>::value) {
        if (nep_.paramb.spin_mode == 0 &&
            result.spin_transfer_per_atom_row_major9 != nullptr) {
          return NEPA_STATUS_UNSUPPORTED;
        }
        if (nep_.paramb.spin_mode > 0) {
          if (input.spins == nullptr) {
            return NEPA_STATUS_INVALID_ARGUMENT;
          }
          nep_.compute_for_lammps(
              input.nlocal,
              input.inum,
              input.ilist,
              input.numneigh,
              input.firstneigh,
              input.types,
              input.type_map,
              input.positions,
              input.spins,
              total_potential,
              total_virial,
              result.potential_per_atom,
              result.forces,
              result.mforces,
              result.virials_per_atom9,
              result.spin_transfer_per_atom_row_major9);
        } else {
          nep_.compute_for_lammps(
              input.nlocal,
              input.inum,
              input.ilist,
              input.numneigh,
              input.firstneigh,
              input.types,
              input.type_map,
              input.positions,
              total_potential,
              total_virial,
              result.potential_per_atom,
              result.forces,
              result.virials_per_atom9);
        }
      } else {
        if (result.spin_transfer_per_atom_row_major9 != nullptr) {
          return NEPA_STATUS_UNSUPPORTED;
        }
        nep_.compute_for_lammps(
          input.nlocal,
          input.inum,
          input.ilist,
          input.numneigh,
          input.firstneigh,
          input.types,
          input.type_map,
          input.positions,
          total_potential,
          total_virial,
          result.potential_per_atom,
          result.forces,
          result.virials_per_atom9);
      }

      *result.total_potential = total_potential;
      std::copy(total_virial, total_virial + 6, result.total_virial6);
      return is_cancelled() ? NEPA_STATUS_CANCELLED : NEPA_STATUS_OK;
    } catch (const std::exception& error) {
      set_last_error(error.what());
      return NEPA_STATUS_RUNTIME_ERROR;
    }
  }

 private:
  static NepaStatus prepare_native_structure(
      const NepaStructureBatch& batch,
      std::int32_t structure,
      std::vector<int>& types,
      std::vector<double>& positions_soa,
      std::vector<double>& box) {
    if (structure < 0 || structure >= batch.num_structures) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    const std::int32_t atom_count = batch.atom_counts[structure];
    const std::int32_t atom_offset = batch.atom_offsets[structure];
    if (atom_count <= 0 || atom_offset < 0 ||
        atom_offset + atom_count > batch.total_atoms) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }
    types.resize(static_cast<std::size_t>(atom_count));
    positions_soa.resize(static_cast<std::size_t>(atom_count) * 3);
    box.resize(9);
    for (std::int32_t atom = 0; atom < atom_count; ++atom) {
      const std::int32_t global_atom = atom_offset + atom;
      types[atom] = batch.types[global_atom];
      for (std::int32_t component = 0; component < 3; ++component) {
        positions_soa[static_cast<std::size_t>(component) * atom_count + atom] =
            batch.positions_aos3[
                3 * static_cast<std::size_t>(global_atom) + component];
      }
    }
    std::copy_n(
        batch.boxes_row_major9 + static_cast<std::size_t>(structure) * 9,
        9,
        box.data());
    return NEPA_STATUS_OK;
  }

  bool valid_batch(const NepaStructureBatch& batch) const {
    if (batch.num_structures <= 0 || batch.total_atoms <= 0 ||
        batch.atom_counts == nullptr || batch.atom_offsets == nullptr ||
        batch.types == nullptr || batch.positions_aos3 == nullptr ||
        batch.boxes_row_major9 == nullptr || batch.pbc_flags3 == nullptr ||
        nep_.paramb.num_types <= 0) {
      return false;
    }
    std::vector<unsigned char> covered(
        static_cast<std::size_t>(batch.total_atoms), 0);
    for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
      const std::int32_t atom_count = batch.atom_counts[structure];
      const std::int32_t atom_offset = batch.atom_offsets[structure];
      if (atom_count <= 0 || atom_offset < 0 ||
          atom_offset > batch.total_atoms ||
          atom_count > batch.total_atoms - atom_offset) {
        return false;
      }
      for (std::int32_t axis = 0; axis < 3; ++axis) {
        if (batch.pbc_flags3[3 * structure + axis] != 1) {
          return false;
        }
      }
      for (std::int32_t local_atom = 0; local_atom < atom_count; ++local_atom) {
        const std::int32_t atom = atom_offset + local_atom;
        if (covered[static_cast<std::size_t>(atom)] != 0) {
          return false;
        }
        covered[static_cast<std::size_t>(atom)] = 1;
      }
    }
    for (std::int32_t atom = 0; atom < batch.total_atoms; ++atom) {
      if (covered[static_cast<std::size_t>(atom)] == 0 ||
          batch.types[atom] < 0 ||
          static_cast<std::size_t>(batch.types[atom]) >=
              nep_.paramb.num_types) {
        return false;
      }
    }
    return true;
  }

  NativeNep nep_;
};

template <typename NativeNep>
class CpuEngine final : public Engine {
 public:
  explicit CpuEngine(const char* name) : name_(name) {}

  EngineInfo info() const override {
    CapabilityMask capabilities =
        to_mask(Capability::batch_find_force) |
        to_mask(Capability::external_neighbors) |
        to_mask(Capability::virial) |
        to_mask(Capability::descriptors) |
        to_mask(Capability::charge) |
        to_mask(Capability::dipole) |
        to_mask(Capability::polarizability) |
        to_mask(Capability::dftd3);
    if constexpr (HasSpin<NativeNep>::value) {
      capabilities |= to_mask(Capability::spin) |
                      to_mask(Capability::spin_energy_transfer);
    }
    return {name_, NEP_ADAPTERS_VERSION_STRING, capabilities};
  }

  NepaStatus load_model(
      const std::string& model_path,
      std::unique_ptr<Model>& out) override {
    if (model_path.empty()) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }

    out = std::make_unique<CpuModel<NativeNep>>(model_path);
    return NEPA_STATUS_OK;
  }

 private:
  const char* name_;
};

}  // namespace nep_adapters
