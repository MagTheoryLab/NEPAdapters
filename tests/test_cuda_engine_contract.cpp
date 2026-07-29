#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include "host_staging.hpp"
#include "device_model.hpp"
#include "device_workspace.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef NEP_ADAPTERS_CUDA_TEST_MODEL_PATH
#  error "NEP_ADAPTERS_CUDA_TEST_MODEL_PATH must be defined"
#endif

namespace {

nep_adapters::cuda_backend::ModelProtocol parse_protocol_case(
    const std::string& name,
    const std::string& cutoff,
    const std::string& n_max,
    const std::string& basis_size,
    const std::string& l_max,
    const std::string& ann,
    const std::string& version = "nep4 1 C",
    const std::string& zbl = "") {
  const std::string path =
      (std::filesystem::temp_directory_path() /
       ("nep_adapters_" + name + ".nep"))
          .string();
  std::ofstream out(path);
  out << version << '\n';
  if (!zbl.empty()) {
    out << zbl << '\n';
  }
  out << cutoff << '\n'
      << n_max << '\n'
      << basis_size << '\n'
      << l_max << '\n'
      << ann << '\n';
  out.close();
  return nep_adapters::cuda_backend::parse_model_protocol(path);
}

bool protocol_case_rejected(
    const std::string& name,
    const std::string& cutoff,
    const std::string& n_max,
    const std::string& basis_size,
    const std::string& l_max,
    const std::string& ann,
    const std::string& version = "nep4 1 C",
    const std::string& zbl = "") {
  try {
    (void)parse_protocol_case(
        name, cutoff, n_max, basis_size, l_max, ann, version, zbl);
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

bool protocol_case_unsupported(
    const std::string& name,
    const std::string& cutoff,
    const std::string& n_max,
    const std::string& basis_size,
    const std::string& l_max,
    const std::string& ann,
    const std::string& version = "nep4 1 C",
    const std::string& zbl = "") {
  try {
    (void)parse_protocol_case(
        name, cutoff, n_max, basis_size, l_max, ann, version, zbl);
  } catch (const nep_adapters::cuda_backend::UnsupportedModelProtocol&) {
    return true;
  } catch (const std::exception&) {
    return false;
  }
  return false;
}

}  // namespace

int main() {
  if (!nep_adapters::register_cuda_engine()) {
    return EXIT_FAILURE;
  }

  NepaBackendInfo backend_info{};
  bool found_cuda = false;
  for (int index = 0; index < nepa_backend_count(); ++index) {
    if (nepa_backend_info(index, &backend_info) != NEPA_STATUS_OK) {
      return EXIT_FAILURE;
    }
    if (std::strcmp(backend_info.name, "cuda") == 0) {
      found_cuda = true;
      if (!nep_adapters::has_capability(
              backend_info.capabilities,
              nep_adapters::Capability::device_input)) {
        return EXIT_FAILURE;
      }
    }
  }
  if (!found_cuda) {
    return EXIT_FAILURE;
  }

  NepaModel* model = nullptr;
  if (nepa_load_model("cuda", NEP_ADAPTERS_CUDA_TEST_MODEL_PATH, &model) !=
          NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }

  NepaModelInfo model_info{};
  if (nepa_model_info(model, &model_info) != NEPA_STATUS_OK) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }
  if (model_info.num_types != 2 || model_info.descriptor_dim != 30 ||
      model_info.cutoff_radial != 8.0 || model_info.cutoff_angular != 4.0) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }
  if (!nep_adapters::has_capability(
          model_info.capabilities,
          nep_adapters::Capability::device_input)) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  int atom_counts[] = {1};
  int atom_offsets[] = {0};
  int types[] = {0};
  double positions[] = {0.0, 0.0, 0.0};
  double boxes[] = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  int pbc[] = {1, 1, 1};
  double energy[] = {0.0};
  double forces[] = {0.0, 0.0, 0.0};
  double charges[] = {0.0};
  double becs[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = 1;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions;
  batch.boxes_row_major9 = boxes;
  batch.pbc_flags3 = pbc;

  NepaFindForceResult result{};
  result.energy_per_structure = energy;
  result.forces_aos3 = forces;
  result.charge_per_atom = charges;
  result.bec_per_atom_row_major9 = becs;
  if (nepa_find_force_batch(model, &batch, &result) != NEPA_STATUS_OK ||
      !std::isfinite(energy[0]) ||
      !std::isfinite(forces[0]) ||
      !std::isfinite(forces[1]) ||
      !std::isfinite(forces[2])) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }

  nepa_free_model(model);

  const std::string protocol_model_path =
      (std::filesystem::temp_directory_path() / "cuda_protocol_smoke.nep").string();
  {
    std::ofstream out(protocol_model_path);
    out << "nep4 1 C\n"
        << "cutoff 5 4 8 6\n"
        << "n_max 1 1\n"
        << "basis_size 2 2\n"
        << "l_max 3 2 1 1\n"
        << "ANN 7 0\n";
    for (int value = 1; value <= 139; ++value) {
      out << value << "\n";
    }
  }
  if (nepa_load_model("cuda", protocol_model_path.c_str(), &model) !=
          NEPA_STATUS_OK ||
      model == nullptr) {
    return EXIT_FAILURE;
  }
  if (nepa_model_info(model, &model_info) != NEPA_STATUS_OK) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }
  if (model_info.descriptor_dim != 14) {
    nepa_free_model(model);
    return EXIT_FAILURE;
  }
  nepa_free_model(model);

  const nep_adapters::cuda_backend::ModelProtocol protocol =
      nep_adapters::cuda_backend::parse_model_protocol(protocol_model_path);
  if (protocol.version != 4 || protocol.hidden_neurons != 7 ||
      protocol.body_channels.channel_count() != 6 ||
      protocol.body_channels.abc_count() != 15 ||
      protocol.ann_parameter_count != 113 ||
      protocol.descriptor_parameter_count != 12 ||
      protocol.model_parameter_count != 125 ||
      protocol.q_scaler_count != 14 ||
      protocol.neighbor_capacity_radial != 10 ||
      protocol.neighbor_capacity_angular != 8) {
    return EXIT_FAILURE;
  }

  nep_adapters::cuda_backend::ModelProtocol generic_spin = protocol;
  generic_spin.spin_mode = 1;
  generic_spin.spin_chiral = 0;
  generic_spin.spin_compress = 1;
  generic_spin.spin_basis_size = 0;
  generic_spin.spin_l_max = 4;
  if (!nep_adapters::cuda_backend::supports_cuda_spin_shape(generic_spin)) {
    return EXIT_FAILURE;
  }
  generic_spin.spin_chiral = 1;
  if (!nep_adapters::cuda_backend::supports_cuda_spin_shape(generic_spin)) {
    return EXIT_FAILURE;
  }
  generic_spin.spin_compress = 4;
  generic_spin.spin_basis_size = 3;
  const nep_adapters::cuda_backend::SpinCoreLayout spin_layout =
      nep_adapters::cuda_backend::make_spin_core_layout(generic_spin);
  if (!nep_adapters::cuda_backend::supports_cuda_spin_shape(generic_spin) ||
      spin_layout.channels != 4 || spin_layout.basis_count != 4 ||
      spin_layout.l_max != 4 || spin_layout.chi_channels != 2 ||
      spin_layout.chiral_offset != 58 || spin_layout.descriptor_dim != 68) {
    return EXIT_FAILURE;
  }
  for (int channels = 1; channels <= 4; ++channels) {
    generic_spin.spin_compress = channels;
    generic_spin.spin_basis_size = channels - 1;
    for (int l_max = 0; l_max <= 4; ++l_max) {
      generic_spin.spin_l_max = l_max;
      if (!nep_adapters::cuda_backend::supports_cuda_spin_shape(generic_spin) ||
          nep_adapters::cuda_backend::make_spin_core_layout(generic_spin)
                  .descriptor_dim <= 0) {
        return EXIT_FAILURE;
      }
    }
  }
  generic_spin.spin_compress = 4;
  generic_spin.spin_basis_size = 2;
  if (nep_adapters::cuda_backend::supports_cuda_spin_shape(generic_spin)) {
    return EXIT_FAILURE;
  }
  generic_spin.spin_basis_size = 3;
  generic_spin.spin_l_max = 4;
  const nep_adapters::cuda_backend::WorkspacePlan unified_spin_workspace =
      nep_adapters::cuda_backend::make_internal_neighbor_workspace_plan(
          generic_spin, 4, 1);
  if (unified_spin_workspace.find_array("spin_edge_weights") != nullptr ||
      unified_spin_workspace.find_array("spin_edge_weight_derivatives") != nullptr ||
      unified_spin_workspace.find_array("spin_chiral_pseudodevs") != nullptr ||
      unified_spin_workspace.find_array("spin_density_l1_stf") != nullptr) {
    return EXIT_FAILURE;
  }

  const nep_adapters::cuda_backend::WorkspacePlan workspace =
      nep_adapters::cuda_backend::make_internal_neighbor_workspace_plan(
          protocol, 4, 2);
  const nep_adapters::cuda_backend::WorkspacePlan external_workspace =
      nep_adapters::cuda_backend::make_external_neighbor_workspace_plan(
          protocol, 6, 4);
  const nep_adapters::cuda_backend::DeviceArrayPlan* atom_to_structure =
      workspace.find_array("atom_to_structure");
  const nep_adapters::cuda_backend::DeviceArrayPlan* workspace_boxes =
      workspace.find_array("boxes_row_major9");
  const nep_adapters::cuda_backend::DeviceArrayPlan* fp =
      workspace.find_array("fp");
  const nep_adapters::cuda_backend::DeviceArrayPlan* descriptors =
      workspace.find_array("descriptors");
  const nep_adapters::cuda_backend::DeviceArrayPlan* sum_fxyz =
      workspace.find_array("sum_fxyz");
  const nep_adapters::cuda_backend::DeviceArrayPlan* r12_angular =
      workspace.find_array("r12_angular");
  const nep_adapters::cuda_backend::DeviceArrayPlan* f12x =
      workspace.find_array("f12x");
  const nep_adapters::cuda_backend::DeviceArrayPlan* nl_angular =
      workspace.find_array("nl_angular_slot_major");
  if (workspace.neighbor_source !=
          nep_adapters::cuda_backend::NeighborSource::internal ||
      workspace.structure_capacity != 2 ||
      workspace.active_atom_capacity != 4 ||
      atom_to_structure == nullptr || atom_to_structure->element_count != 4 ||
      workspace_boxes == nullptr || workspace_boxes->element_count != 18 ||
      fp == nullptr || fp->element_count != 56 ||
      descriptors == nullptr || descriptors->element_count != 56 ||
      sum_fxyz == nullptr || sum_fxyz->element_count != 120 ||
      r12_angular == nullptr || r12_angular->element_count != 32 ||
      f12x == nullptr || f12x->element_count != 32 ||
      nl_angular == nullptr || nl_angular->element_count != 32) {
    return EXIT_FAILURE;
  }
  if (workspace.find_array("parameters_and_q_scaler") != nullptr ||
      workspace.find_array("r12_radial") != nullptr ||
      workspace.find_array("fc_radial") != nullptr ||
      workspace.find_array("fn_radial") != nullptr ||
      workspace.find_array("fc_angular") != nullptr ||
      workspace.find_array("fn_angular") != nullptr) {
    return EXIT_FAILURE;
  }
  if (external_workspace.neighbor_source !=
          nep_adapters::cuda_backend::NeighborSource::external ||
      external_workspace.structure_capacity != 0 ||
      external_workspace.active_atom_capacity != 4 ||
      external_workspace.find_array("boxes_row_major9") != nullptr ||
      external_workspace.find_array("active_atom_indices")->element_count != 4 ||
      external_workspace.find_array("nl_radial_slot_major")->element_count !=
          6 * static_cast<std::size_t>(protocol.neighbor_capacity_radial)) {
    return EXIT_FAILURE;
  }
  int staged_atom_counts[] = {2, 2};
  int staged_atom_offsets[] = {0, 2};
  int staged_types[] = {0, 1, 1, 0};
  double staged_positions[] = {
      0.0, 1.0, 2.0,
      3.0, 4.0, 5.0,
      6.0, 7.0, 8.0,
      9.0, 10.0, 11.0,
  };
  double staged_boxes[] = {
      8.0, 0.0, 0.0,
      0.0, 8.0, 0.0,
      0.0, 0.0, 8.0,
      9.0, 0.0, 0.0,
      0.0, 9.0, 0.0,
      0.0, 0.0, 9.0,
  };
  int staged_pbc[] = {1, 1, 1, 1, 1, 1};
  NepaStructureBatch staged_batch{};
  staged_batch.num_structures = 2;
  staged_batch.total_atoms = 4;
  staged_batch.atom_counts = staged_atom_counts;
  staged_batch.atom_offsets = staged_atom_offsets;
  staged_batch.types = staged_types;
  staged_batch.positions_aos3 = staged_positions;
  staged_batch.boxes_row_major9 = staged_boxes;
  staged_batch.pbc_flags3 = staged_pbc;
  const nep_adapters::cuda_backend::HostBatchStaging staged_internal =
      nep_adapters::cuda_backend::stage_batch_for_internal_neighbors(staged_batch);
  if (staged_internal.types[2] != 1 ||
      staged_internal.positions_soa3[1] != 3.0 ||
      staged_internal.positions_soa3[5] != 4.0 ||
      staged_internal.positions_soa3[9] != 5.0 ||
      staged_internal.atom_to_structure[3] != 1 ||
      staged_internal.pbc_flags3[3] != 1 ||
      staged_internal.boxes_row_major9[9] != 9.0) {
    return EXIT_FAILURE;
  }

  int lmp_ilist[] = {0, 2};
  int lmp_numneigh[] = {2, 0, 1, 0};
  int lmp_neigh0[] = {1, 2};
  int lmp_neigh2[] = {3};
  int* lmp_firstneigh[] = {lmp_neigh0, nullptr, lmp_neigh2, nullptr};
  int lmp_types[] = {1, 2, 1, 2};
  int lmp_type_map[] = {-1, 0, 1};
  double lmp_x0[] = {0.0, 1.0, 2.0};
  double lmp_x1[] = {3.0, 4.0, 5.0};
  double lmp_x2[] = {6.0, 7.0, 8.0};
  double lmp_x3[] = {9.0, 10.0, 11.0};
  double* lmp_positions[] = {lmp_x0, lmp_x1, lmp_x2, lmp_x3};
  NepaLammpsNeighborInput staged_lmp_input{};
  staged_lmp_input.nlocal = 3;
  staged_lmp_input.inum = 2;
  staged_lmp_input.ilist = lmp_ilist;
  staged_lmp_input.numneigh = lmp_numneigh;
  staged_lmp_input.firstneigh = lmp_firstneigh;
  staged_lmp_input.types = lmp_types;
  staged_lmp_input.type_map = lmp_type_map;
  staged_lmp_input.positions = lmp_positions;
  const nep_adapters::cuda_backend::HostExternalNeighborStaging staged_external =
      nep_adapters::cuda_backend::stage_lammps_external_neighbors(
          staged_lmp_input,
          protocol);
  if (staged_external.atom_capacity != 4 ||
      staged_external.active_atom_count != 2 ||
      staged_external.types[1] != 1 ||
      staged_external.positions_soa3[2] != 6.0 ||
      staged_external.positions_soa3[6] != 7.0 ||
      staged_external.positions_soa3[10] != 8.0 ||
      staged_external.active_atom_indices[1] != 2 ||
      staged_external.nn_radial[0] != 2 ||
      staged_external.nn_angular[2] != 1 ||
      staged_external.nl_radial_slot_major[4] != 2 ||
      staged_external.nl_angular_slot_major[2] != 3) {
    return EXIT_FAILURE;
  }

  const std::string parameter_model_path =
      (std::filesystem::temp_directory_path() / "cuda_parameter_smoke.nep").string();
  {
    std::ofstream out(parameter_model_path);
    out << "nep4 1 C\n"
        << "cutoff 5 4 8 6\n"
        << "n_max 0 0\n"
        << "basis_size 0 0\n"
        << "l_max 1 0 0\n"
        << "ANN 2 0\n";
    for (int value = 1; value <= 13; ++value) {
      out << value << "\n";
    }
  }
  const nep_adapters::cuda_backend::HostModelParameters packed =
      nep_adapters::cuda_backend::load_host_model_parameters(parameter_model_path);
  if (packed.protocol.descriptor_dim != 2 ||
      packed.ann_type_major.size() != 9 ||
      packed.descriptor_coefficients.size() != 2 ||
      packed.angular_coefficients_center_type_major.size() != 1 ||
      packed.q_scaler.size() != 2 ||
      packed.ann_blocks.size() != 1 ||
      packed.ann_blocks[0].w0_offset != 0 ||
      packed.ann_blocks[0].b0_offset != 4 ||
      packed.ann_blocks[0].w1_offset != 6 ||
      packed.b1_offset != 8 ||
      packed.descriptor_layout.radial_offset != 0 ||
      packed.descriptor_layout.angular_offset != 1 ||
      packed.descriptor_layout.radial_count != 1 ||
      packed.descriptor_layout.angular_count != 1) {
    return EXIT_FAILURE;
  }
  if (packed.ann_type_major[0] != 1.0f || packed.ann_type_major[8] != 9.0f ||
      packed.descriptor_coefficients[0] != 10.0f ||
      packed.descriptor_coefficients[1] != 11.0f ||
      packed.angular_coefficients_center_type_major[0] != 11.0f ||
      packed.q_scaler[0] != 12.0f ||
      packed.q_scaler[1] != 13.0f) {
    return EXIT_FAILURE;
  }

  const std::string multi_type_model_path =
      (std::filesystem::temp_directory_path() /
       "cuda_center_type_major_parameters.nep").string();
  {
    std::ofstream out(multi_type_model_path);
    out << "nep4 2 C H\n"
        << "cutoff 5 4 8 6\n"
        << "n_max 0 0\n"
        << "basis_size 0 1\n"
        << "l_max 1 0 0\n"
        << "ANN 1 0\n";
    for (int value = 1; value <= 23; ++value) {
      out << value << "\n";
    }
  }
  const nep_adapters::cuda_backend::HostModelParameters multi_type_packed =
      nep_adapters::cuda_backend::load_host_model_parameters(
          multi_type_model_path);
  const float expected_center_type_major[] = {
      14.0f, 15.0f, 18.0f, 19.0f,
      16.0f, 17.0f, 20.0f, 21.0f,
  };
  if (multi_type_packed.angular_coefficients_center_type_major.size() != 8) {
    return EXIT_FAILURE;
  }
  for (std::size_t index = 0; index < 8; ++index) {
    if (multi_type_packed.angular_coefficients_center_type_major[index] !=
        expected_center_type_major[index]) {
      return EXIT_FAILURE;
    }
  }

  const std::string qnep_model_path =
      (std::filesystem::temp_directory_path() / "cuda_qnep_protocol_smoke.nep").string();
  {
    std::ofstream out(qnep_model_path);
    out << "nep4_charge1 1 C\n"
        << "cutoff 5 4 8 6\n"
        << "n_max 0 0\n"
        << "basis_size 0 0\n"
        << "l_max 1 0 0\n"
        << "ANN 2 0\n";
    for (int value = 1; value <= 16; ++value) {
      out << value << "\n";
    }
  }
  const nep_adapters::cuda_backend::ModelProtocol qnep_protocol =
      nep_adapters::cuda_backend::parse_model_protocol(qnep_model_path);
  if (qnep_protocol.version != 4 || qnep_protocol.charge_mode != 1 ||
      qnep_protocol.descriptor_dim != 2 ||
      qnep_protocol.ann_parameter_count != 12 ||
      qnep_protocol.descriptor_parameter_count != 2 ||
      qnep_protocol.model_parameter_count != 14 ||
      qnep_protocol.q_scaler_count != 2) {
    std::cerr << "qNEP protocol count mismatch\n";
    return EXIT_FAILURE;
  }
  const nep_adapters::cuda_backend::HostModelParameters qnep_packed =
      nep_adapters::cuda_backend::load_host_model_parameters(qnep_model_path);
  if (!qnep_packed.has_charge || qnep_packed.ann_type_major.size() != 12 ||
      qnep_packed.ann_blocks[0].charge_w1_offset != 8 ||
      qnep_packed.sqrt_epsilon_inf_offset != 10 ||
      qnep_packed.b1_offset != 11 ||
      qnep_packed.descriptor_coefficients[0] != 13.0f ||
      qnep_packed.descriptor_coefficients[1] != 14.0f ||
      qnep_packed.q_scaler[0] != 15.0f ||
      qnep_packed.q_scaler[1] != 16.0f) {
    std::cerr << "qNEP packed parameter layout mismatch "
              << "ann_size=" << qnep_packed.ann_type_major.size()
              << " charge_w1=" << qnep_packed.ann_blocks[0].charge_w1_offset
              << " sqrt=" << qnep_packed.sqrt_epsilon_inf_offset
              << " b1=" << qnep_packed.b1_offset
              << " desc0=" << qnep_packed.descriptor_coefficients[0]
              << " desc1=" << qnep_packed.descriptor_coefficients[1]
              << " q0=" << qnep_packed.q_scaler[0]
              << " q1=" << qnep_packed.q_scaler[1] << '\n';
    return EXIT_FAILURE;
  }
  if (nepa_load_model("cuda", qnep_model_path.c_str(), &model) != NEPA_STATUS_OK ||
      model == nullptr) {
    std::cerr << "qNEP CUDA load failed: " << nepa_last_error_message() << '\n';
    return EXIT_FAILURE;
  }
  if (nepa_model_info(model, &model_info) != NEPA_STATUS_OK ||
      !nep_adapters::has_capability(
          model_info.capabilities,
          nep_adapters::Capability::batch_find_force) ||
      !nep_adapters::has_capability(
          model_info.capabilities,
          nep_adapters::Capability::charge)) {
    std::cerr << "qNEP CUDA capabilities mismatch\n";
    nepa_free_model(model);
    return EXIT_FAILURE;
  }
  const NepaStatus qnep_force_status = nepa_find_charge_batch(model, &batch, &result);
  if (qnep_force_status != NEPA_STATUS_OK) {
    std::cerr << "qNEP CUDA force failed status=" << qnep_force_status
              << " error=" << nepa_last_error_message() << '\n';
    nepa_free_model(model);
    return EXIT_FAILURE;
  }
  nepa_free_model(model);

  const std::string spin_model_path =
      (std::filesystem::temp_directory_path() / "cuda_spin_reject.nep").string();
  {
    std::ofstream out(spin_model_path);
    out << "nep4_spin1 1 C\n";
  }
  if (nepa_load_model("cuda", spin_model_path.c_str(), &model) !=
          NEPA_STATUS_RUNTIME_ERROR) {
    if (model != nullptr) {
      nepa_free_model(model);
    }
    return EXIT_FAILURE;
  }

  const auto compact_l_max = parse_protocol_case(
      "compact_l_max",
      "cutoff 5 4 8 6",
      "n_max 0 0",
      "basis_size 0 0",
      "l_max 4 1",
      "ANN 1 0");
  if (compact_l_max.body_channels.channel_count() != 5 ||
      !compact_l_max.body_channels.has_q_222 ||
      compact_l_max.body_channels.has_q_1111) {
    return EXIT_FAILURE;
  }
  const auto legacy_l_max = parse_protocol_case(
      "legacy_l_max",
      "cutoff 5 4 8 6",
      "n_max 0 0",
      "basis_size 0 0",
      "l_max 4 2 1",
      "ANN 1 0");
  if (legacy_l_max.body_channels.channel_count() != 6 ||
      !legacy_l_max.body_channels.has_q_222 ||
      !legacy_l_max.body_channels.has_q_1111) {
    return EXIT_FAILURE;
  }
  const auto hybrid_l_max = parse_protocol_case(
      "hybrid_l_max",
      "cutoff 5 4 8 6",
      "n_max 0 0",
      "basis_size 0 0",
      "l_max 4 2 0 1 1 1 1",
      "ANN 1 0");
  if (hybrid_l_max.body_channels.channel_count() != 9 ||
      !hybrid_l_max.body_channels.has_q_222 ||
      hybrid_l_max.body_channels.has_q_1111 ||
      !hybrid_l_max.body_channels.has_q_112 ||
      !hybrid_l_max.body_channels.has_q_123 ||
      !hybrid_l_max.body_channels.has_q_233 ||
      !hybrid_l_max.body_channels.has_q_134) {
    return EXIT_FAILURE;
  }
  const auto boundary_protocol = parse_protocol_case(
      "range_boundaries",
      "cutoff 100 100 8 6",
      "n_max 12 8",
      "basis_size 16 12",
      "l_max 8 0 0",
      "ANN 120 0");
  if (boundary_protocol.n_max_radial != 12 ||
      boundary_protocol.n_max_angular != 8 ||
      boundary_protocol.basis_size_radial != 16 ||
      boundary_protocol.basis_size_angular != 12 ||
      boundary_protocol.body_channels.l_max_3body != 8 ||
      boundary_protocol.hidden_neurons != 120) {
    return EXIT_FAILURE;
  }
  const auto type_dependent_cutoff = parse_protocol_case(
      "type_dependent_cutoff",
      "cutoff 5 4 6 3 8 6",
      "n_max 0 0",
      "basis_size 0 0",
      "l_max 4 0 0",
      "ANN 1 0",
      "nep4 2 C H");
  if (type_dependent_cutoff.cutoff_radial != 6.0 ||
      type_dependent_cutoff.cutoff_angular != 4.0 ||
      type_dependent_cutoff.cutoff_radial_by_type.size() != 2 ||
      type_dependent_cutoff.cutoff_radial_by_type[0] != 5.0 ||
      type_dependent_cutoff.cutoff_radial_by_type[1] != 6.0) {
    return EXIT_FAILURE;
  }
  const auto typewise_zbl = parse_protocol_case(
      "typewise_zbl",
      "cutoff 5 4 8 6",
      "n_max 0 0",
      "basis_size 0 0",
      "l_max 4 0 0",
      "ANN 1 0",
      "nep4_zbl 1 C",
      "zbl 1 2 0.7");
  if (!typewise_zbl.use_typewise_cutoff_zbl ||
      typewise_zbl.typewise_cutoff_zbl_factor != 0.7) {
    return EXIT_FAILURE;
  }
  const auto two_hidden_layers = parse_protocol_case(
      "two_hidden_layers",
      "cutoff 5 4 8 6",
      "n_max 0 0",
      "basis_size 0 0",
      "l_max 4 0 0",
      "ANN 32 16");
  if (two_hidden_layers.hidden_neurons2 != 16 ||
      two_hidden_layers.ann_parameter_count != 737) {
    return EXIT_FAILURE;
  }
  std::string too_many_types = "nep4 119";
  for (int type = 0; type < 119; ++type) {
    too_many_types += " C";
  }
  if (!protocol_case_rejected(
          "bad_q_flag",
          "cutoff 5 4 8 6",
          "n_max 0 0",
          "basis_size 0 0",
          "l_max 4 1 0 2",
          "ANN 1 0") ||
      !protocol_case_rejected(
          "too_many_l_max_fields",
          "cutoff 5 4 8 6",
          "n_max 0 0",
          "basis_size 0 0",
          "l_max 4 1 0 0 0 0 0 0",
          "ANN 1 0") ||
      !protocol_case_rejected(
          "n_max_radial_range",
          "cutoff 5 4 8 6",
          "n_max 13 0",
          "basis_size 0 0",
          "l_max 4 0 0",
          "ANN 1 0") ||
      !protocol_case_rejected(
          "n_max_angular_range",
          "cutoff 5 4 8 6",
          "n_max 0 9",
          "basis_size 0 0",
          "l_max 4 0 0",
          "ANN 1 0") ||
      !protocol_case_rejected(
          "basis_radial_range",
          "cutoff 5 4 8 6",
          "n_max 0 0",
          "basis_size 17 0",
          "l_max 4 0 0",
          "ANN 1 0") ||
      !protocol_case_rejected(
          "basis_angular_range",
          "cutoff 5 4 8 6",
          "n_max 0 0",
          "basis_size 0 13",
          "l_max 4 0 0",
          "ANN 1 0") ||
      !protocol_case_rejected(
          "l_max_range",
          "cutoff 5 4 8 6",
          "n_max 0 0",
          "basis_size 0 0",
          "l_max 9 0 0",
          "ANN 1 0") ||
      !protocol_case_rejected(
          "ann_range",
          "cutoff 5 4 8 6",
          "n_max 0 0",
          "basis_size 0 0",
          "l_max 4 0 0",
          "ANN 121 0") ||
      !protocol_case_rejected(
          "type_count_range",
          "cutoff 5 4 8 6",
          "n_max 0 0",
          "basis_size 0 0",
          "l_max 4 0 0",
          "ANN 1 0",
          too_many_types) ||
      !protocol_case_rejected(
          "angular_descriptor_limit",
          "cutoff 5 4 8 6",
          "n_max 0 8",
          "basis_size 0 0",
          "l_max 8 1 1 1 1 1 1",
          "ANN 1 0")) {
    return EXIT_FAILURE;
  }

  const std::string truncated_spin_path =
      (std::filesystem::temp_directory_path() /
       "cuda_truncated_spin_reject.nep").string();
  {
    std::ofstream out(truncated_spin_path);
    out << "nep4_spin 1 Fe\n"
        << "spin_mode 1 1\n"
        << "spin_chiral\n";
  }
  try {
    (void)nep_adapters::cuda_backend::parse_model_protocol(
        truncated_spin_path);
    std::cerr << "CUDA parser accepted truncated spin header\n";
    return EXIT_FAILURE;
  } catch (const std::runtime_error&) {
  }

  return EXIT_SUCCESS;
}
