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

namespace nep_adapters {

template <typename T, typename = void>
struct HasSpinLite : std::false_type {};

template <typename T>
struct HasSpinLite<T, std::void_t<decltype(std::declval<T>().paramb.spin_mode)>>
    : std::true_type {};

template <typename NativeNep>
class CpuModel final : public Model {
 public:
  explicit CpuModel(const std::string& model_path) : nep_(model_path) {}

  NepaStatus model_info(NepaModelInfo& out) const override {
    out = {};
    out.cutoff_radial = nep_.paramb.rc_radial_max;
    out.cutoff_angular = nep_.paramb.rc_angular_max;
    out.cutoff_max = std::max(out.cutoff_radial, out.cutoff_angular);
    if (nep_.zbl.enabled) {
      out.cutoff_max = std::max(out.cutoff_max, nep_.zbl.rc_outer);
    }
    out.capabilities = to_mask(Capability::batch_find_force) |
                       to_mask(Capability::external_neighbors) |
                       to_mask(Capability::virial) |
                       to_mask(Capability::descriptors);
    if (nep_.paramb.charge_mode > 0) {
      out.capabilities |= to_mask(Capability::charge);
    }
    if constexpr (HasSpinLite<NativeNep>::value) {
      if (nep_.paramb.spin_mode > 0) {
        out.capabilities |= to_mask(Capability::spin);
      }
    }
    out.num_types = static_cast<std::int32_t>(nep_.paramb.num_types);
    out.descriptor_dim = static_cast<std::int32_t>(nep_.annmb.dim);
    return NEPA_STATUS_OK;
  }

  NepaStatus find_force_batch(
      const NepaStructureBatch& batch,
      NepaFindForceResult& result) override {
    if (!valid_batch(batch) || result.energy_per_structure == nullptr ||
        result.forces_aos3 == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }

    try {
      for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
        const std::int32_t atom_count = batch.atom_counts[structure];
        const std::int32_t atom_offset = batch.atom_offsets[structure];
        if (atom_count <= 0 || atom_offset < 0 ||
            atom_offset + atom_count > batch.total_atoms) {
          return NEPA_STATUS_INVALID_ARGUMENT;
        }

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
        if constexpr (HasSpinLite<NativeNep>::value) {
          if (nep_.paramb.spin_mode > 0) {
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
            std::vector<double> descriptor_soa(
                static_cast<std::size_t>(atom_count) * nep_.annmb.dim, 0.0);
            std::vector<double> mforce_soa(static_cast<std::size_t>(atom_count) * 3, 0.0);
            nep_.compute(
                types,
                box,
                positions_soa,
                spins_soa,
                potential,
                force_soa,
                virial_soa,
                descriptor_soa,
                mforce_soa);
            for (std::int32_t atom = 0; atom < atom_count; ++atom) {
              const std::int32_t global_atom = atom_offset + atom;
              if (result.mforces_aos3 != nullptr) {
                result.mforces_aos3[3 * global_atom + 0] = mforce_soa[atom];
                result.mforces_aos3[3 * global_atom + 1] =
                    mforce_soa[static_cast<std::size_t>(atom_count) + atom];
                result.mforces_aos3[3 * global_atom + 2] =
                    mforce_soa[static_cast<std::size_t>(2) * atom_count + atom];
              }
              if (result.tau_aos3 != nullptr) {
                const double sx = batch.spins_aos3[3 * global_atom + 0];
                const double sy = batch.spins_aos3[3 * global_atom + 1];
                const double sz = batch.spins_aos3[3 * global_atom + 2];
                const double mx = mforce_soa[atom];
                const double my = mforce_soa[static_cast<std::size_t>(atom_count) + atom];
                const double mz = mforce_soa[static_cast<std::size_t>(2) * atom_count + atom];
                result.tau_aos3[3 * global_atom + 0] = sy * mz - sz * my;
                result.tau_aos3[3 * global_atom + 1] = sz * mx - sx * mz;
                result.tau_aos3[3 * global_atom + 2] = sx * my - sy * mx;
              }
            }
          } else if (nep_.paramb.charge_mode > 0) {
            charge.assign(static_cast<std::size_t>(atom_count), 0.0);
            bec_soa.assign(static_cast<std::size_t>(atom_count) * 9, 0.0);
            nep_.compute(
                types,
                box,
                positions_soa,
                potential,
                force_soa,
                virial_soa,
                charge,
                bec_soa);
          } else {
            nep_.compute(types, box, positions_soa, potential, force_soa, virial_soa);
          }
        } else if (nep_.paramb.charge_mode > 0) {
          charge.assign(static_cast<std::size_t>(atom_count), 0.0);
          bec_soa.assign(static_cast<std::size_t>(atom_count) * 9, 0.0);
          nep_.compute(
              types,
              box,
              positions_soa,
              potential,
              force_soa,
              virial_soa,
              charge,
              bec_soa);
        } else {
          nep_.compute(types, box, positions_soa, potential, force_soa, virial_soa);
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
      }

      return NEPA_STATUS_OK;
    } catch (const std::exception&) {
      return NEPA_STATUS_RUNTIME_ERROR;
    }
  }

  NepaStatus find_descriptors(
      const NepaStructureBatch& batch,
      NepaFindDescriptorResult& result) override {
    if (!valid_batch(batch) || result.descriptors == nullptr) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }

    const std::int32_t descriptor_dim =
        static_cast<std::int32_t>(nep_.annmb.dim);
    if (descriptor_dim <= 0) {
      return NEPA_STATUS_UNSUPPORTED;
    }

    try {
      for (std::int32_t structure = 0; structure < batch.num_structures; ++structure) {
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

        if constexpr (HasSpinLite<NativeNep>::value) {
          if (nep_.paramb.spin_mode > 0) {
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
            nep_.find_descriptor(types, box, positions_soa, spins_soa, descriptor_soa);
          } else {
            nep_.find_descriptor(types, box, positions_soa, descriptor_soa);
          }
        } else {
          nep_.find_descriptor(types, box, positions_soa, descriptor_soa);
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
      }

      return NEPA_STATUS_OK;
    } catch (const std::exception&) {
      return NEPA_STATUS_RUNTIME_ERROR;
    }
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

    try {
      double total_potential = 0.0;
      double total_virial[6] = {};
      if constexpr (HasSpinLite<NativeNep>::value) {
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
              result.virials_per_atom9);
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
      return NEPA_STATUS_OK;
    } catch (const std::exception&) {
      return NEPA_STATUS_RUNTIME_ERROR;
    }
  }

 private:
  static bool valid_batch(const NepaStructureBatch& batch) {
    return batch.num_structures > 0 && batch.total_atoms > 0 &&
           batch.atom_counts != nullptr && batch.atom_offsets != nullptr &&
           batch.types != nullptr && batch.positions_aos3 != nullptr &&
           batch.boxes_row_major9 != nullptr;
  }

  NativeNep nep_;
};

template <typename NativeNep>
class CpuEngine final : public Engine {
 public:
  explicit CpuEngine(const char* name) : name_(name) {}

  EngineInfo info() const override {
    return {
        name_,
        "external",
        to_mask(Capability::batch_find_force) |
            to_mask(Capability::external_neighbors) |
            to_mask(Capability::virial) |
            to_mask(Capability::descriptors)};
  }

  NepaStatus load_model(
      const std::string& model_path,
      std::unique_ptr<Model>& out) override {
    if (model_path.empty()) {
      return NEPA_STATUS_INVALID_ARGUMENT;
    }

    try {
      out = std::make_unique<CpuModel<NativeNep>>(model_path);
      return NEPA_STATUS_OK;
    } catch (const std::exception&) {
      out.reset();
      return NEPA_STATUS_RUNTIME_ERROR;
    }
  }

 private:
  const char* name_;
};

}  // namespace nep_adapters
