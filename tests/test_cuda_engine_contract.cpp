#include "nep_adapters/api.h"
#include "nep_adapters/capability.hpp"
#include "nep_adapters/engines/cuda.hpp"

#include "host_staging.hpp"
#include "model_parameters.hpp"
#include "nonspin_pipeline.hpp"
#include "workspace_plan.hpp"

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
  double boxes[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  double energy[] = {0.0};
  double forces[] = {0.0, 0.0, 0.0};

  NepaStructureBatch batch{};
  batch.num_structures = 1;
  batch.total_atoms = 1;
  batch.atom_counts = atom_counts;
  batch.atom_offsets = atom_offsets;
  batch.types = types;
  batch.positions_aos3 = positions;
  batch.boxes_row_major9 = boxes;

  NepaFindForceResult result{};
  result.energy_per_structure = energy;
  result.forces_aos3 = forces;
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
        << "l_max 2 2 1 1 0 1\n"
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
      protocol.body_channels.abc_count() != 8 ||
      protocol.ann_parameter_count != 113 ||
      protocol.descriptor_parameter_count != 12 ||
      protocol.model_parameter_count != 125 ||
      protocol.q_scaler_count != 14 ||
      protocol.neighbor_capacity_radial != 10 ||
      protocol.neighbor_capacity_angular != 8) {
    return EXIT_FAILURE;
  }

  using nep_adapters::cuda_backend::NonSpinDescriptorMode;
  using nep_adapters::cuda_backend::NonSpinNeighborTopology;
  using nep_adapters::cuda_backend::NonSpinPipelineRequest;
  using nep_adapters::cuda_backend::NonSpinRadialForceMode;
  using nep_adapters::cuda_backend::make_nonspin_execution_plan;

  NonSpinPipelineRequest request{};
  request.topology = NonSpinNeighborTopology::external_full;
  request.has_angular = true;
  request.store_potential = false;
  const auto lammps_plan = make_nonspin_execution_plan(protocol, request);
  if (lammps_plan.descriptor_mode != NonSpinDescriptorMode::fused_positions ||
      lammps_plan.radial_force_mode != NonSpinRadialForceMode::external_full) {
    std::cerr << "external-neighbor execution plan mismatch\n";
    return EXIT_FAILURE;
  }

  request = {};
  request.topology = NonSpinNeighborTopology::batched_multi_box;
  request.has_angular = true;
  request.orthorhombic_batched = true;
  const auto batched_plan = make_nonspin_execution_plan(protocol, request);
  if (batched_plan.descriptor_mode != NonSpinDescriptorMode::batched_cached ||
      batched_plan.radial_force_mode != NonSpinRadialForceMode::batched) {
    std::cerr << "batched execution plan mismatch\n";
    return EXIT_FAILURE;
  }

  request = {};
  request.topology = NonSpinNeighborTopology::single_box_symmetric;
  request.has_angular = true;
  const auto single_plan = make_nonspin_execution_plan(protocol, request);
  if (single_plan.descriptor_mode != NonSpinDescriptorMode::fused_positions ||
      single_plan.radial_force_mode != NonSpinRadialForceMode::symmetric) {
    std::cerr << "single-box execution plan mismatch\n";
    return EXIT_FAILURE;
  }

  auto charge_protocol = protocol;
  charge_protocol.charge_mode = 1;
  bool rejected_charge_model = false;
  try {
    (void)make_nonspin_execution_plan(charge_protocol, request);
  } catch (const std::invalid_argument&) {
    rejected_charge_model = true;
  }
  if (!rejected_charge_model) {
    std::cerr << "ordinary non-spin pipeline accepted a charge model\n";
    return EXIT_FAILURE;
  }

  const nep_adapters::cuda_backend::WorkspacePlan workspace =
      nep_adapters::cuda_backend::make_internal_neighbor_workspace_plan(
          protocol, 4, 2);
  const nep_adapters::cuda_backend::WorkspacePlan external_workspace =
      nep_adapters::cuda_backend::make_external_neighbor_workspace_plan(
          protocol, 6, 4);
  const nep_adapters::cuda_backend::WorkspacePlan model_workspace =
      nep_adapters::cuda_backend::make_model_workspace_plan(protocol);
  const nep_adapters::cuda_backend::DeviceArrayPlan* atom_to_structure =
      workspace.find_array("atom_to_structure");
  const nep_adapters::cuda_backend::DeviceArrayPlan* workspace_boxes =
      workspace.find_array("boxes_row_major9");
  const nep_adapters::cuda_backend::DeviceArrayPlan* parameters =
      workspace.find_array("parameters_and_q_scaler");
  const nep_adapters::cuda_backend::DeviceArrayPlan* fp =
      workspace.find_array("fp");
  const nep_adapters::cuda_backend::DeviceArrayPlan* descriptors =
      workspace.find_array("descriptors");
  const nep_adapters::cuda_backend::DeviceArrayPlan* sum_fxyz =
      workspace.find_array("sum_fxyz");
  const nep_adapters::cuda_backend::DeviceArrayPlan* fc_radial =
      workspace.find_array("fc_radial");
  const nep_adapters::cuda_backend::DeviceArrayPlan* fn_radial =
      workspace.find_array("fn_radial");
  const nep_adapters::cuda_backend::DeviceArrayPlan* r12_angular =
      workspace.find_array("r12_angular");
  const nep_adapters::cuda_backend::DeviceArrayPlan* fc_angular =
      workspace.find_array("fc_angular");
  const nep_adapters::cuda_backend::DeviceArrayPlan* fn_angular =
      workspace.find_array("fn_angular");
  const nep_adapters::cuda_backend::DeviceArrayPlan* nl_angular =
      workspace.find_array("nl_angular_slot_major");
  if (workspace.neighbor_source !=
          nep_adapters::cuda_backend::NeighborSource::internal ||
      workspace.structure_capacity != 2 ||
      workspace.active_atom_capacity != 4 ||
      atom_to_structure == nullptr || atom_to_structure->element_count != 4 ||
      workspace_boxes == nullptr || workspace_boxes->element_count != 18 ||
      parameters == nullptr || parameters->element_count != 139 ||
      fp == nullptr || fp->element_count != 56 ||
      descriptors == nullptr || descriptors->element_count != 56 ||
      sum_fxyz == nullptr || sum_fxyz->element_count != 64 ||
      fc_radial == nullptr || fc_radial->element_count != 40 ||
      fn_radial == nullptr || fn_radial->element_count != 120 ||
      r12_angular == nullptr || r12_angular->element_count != 32 ||
      fc_angular == nullptr || fc_angular->element_count != 32 ||
      fn_angular == nullptr || fn_angular->element_count != 96 ||
      nl_angular == nullptr || nl_angular->element_count != 32) {
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
  if (model_workspace.find_array("ann_type_major")->element_count != 113 ||
      model_workspace.find_array("descriptor_coefficients")->element_count != 12 ||
      model_workspace.find_array("q_scaler")->element_count != 14) {
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
  int staged_pbc[] = {1, 1, 1, 0, 0, 0};
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
      staged_internal.pbc_flags3[3] != 0 ||
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
      packed.q_scaler[0] != 12.0f ||
      packed.q_scaler[1] != 13.0f) {
    return EXIT_FAILURE;
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
  const NepaStatus qnep_force_status = nepa_find_force_batch(model, &batch, &result);
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

  return EXIT_SUCCESS;
}
