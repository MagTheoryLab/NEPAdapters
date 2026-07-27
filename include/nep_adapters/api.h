#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(NEP_ADAPTERS_SHARED)
#  if defined(NEP_ADAPTERS_BUILDING)
#    define NEP_ADAPTERS_API __declspec(dllexport)
#  else
#    define NEP_ADAPTERS_API __declspec(dllimport)
#  endif
#else
#  define NEP_ADAPTERS_API
#endif

#define NEP_ADAPTERS_API_VERSION_MAJOR 1
#define NEP_ADAPTERS_API_VERSION_MINOR 0
#define NEP_ADAPTERS_API_VERSION_PATCH 0
#define NEP_ADAPTERS_VERSION_STRING "1.0.0"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NepaStatus {
  NEPA_STATUS_OK = 0,
  NEPA_STATUS_INVALID_ARGUMENT = 1,
  NEPA_STATUS_UNSUPPORTED = 2,
  NEPA_STATUS_BACKEND_UNAVAILABLE = 3,
  NEPA_STATUS_RUNTIME_ERROR = 4,
  NEPA_STATUS_CANCELLED = 5
} NepaStatus;

typedef enum NepaCapabilityFlags {
  NEPA_CAPABILITY_BATCH_FIND_FORCE = 1u << 0u,
  NEPA_CAPABILITY_EXTERNAL_NEIGHBORS = 1u << 1u,
  NEPA_CAPABILITY_DEVICE_INPUT = 1u << 2u,
  NEPA_CAPABILITY_SPIN = 1u << 3u,
  NEPA_CAPABILITY_CHARGE = 1u << 4u,
  NEPA_CAPABILITY_VIRIAL = 1u << 5u,
  NEPA_CAPABILITY_DESCRIPTORS = 1u << 6u,
  NEPA_CAPABILITY_SPIN_ENERGY_TRANSFER = 1u << 7u,
  NEPA_CAPABILITY_DIPOLE = 1u << 8u,
  NEPA_CAPABILITY_POLARIZABILITY = 1u << 9u,
  NEPA_CAPABILITY_DFTD3 = 1u << 10u,
  NEPA_CAPABILITY_EVALUATE_WITH_DESCRIPTORS = 1u << 11u
} NepaCapabilityFlags;

typedef enum NepaModelKind {
  NEPA_MODEL_KIND_ORDINARY = 0,
  NEPA_MODEL_KIND_SPIN = 1,
  NEPA_MODEL_KIND_CHARGE = 2,
  NEPA_MODEL_KIND_DIPOLE = 3,
  NEPA_MODEL_KIND_POLARIZABILITY = 4
} NepaModelKind;

typedef struct NepaModel NepaModel;

typedef struct NepaBackendInfo {
  const char* name;
  const char* version;
  uint64_t capabilities;
} NepaBackendInfo;

typedef struct NepaModelInfo {
  double cutoff_radial;
  double cutoff_angular;
  double cutoff_max;
  uint64_t capabilities;
  int32_t num_types;
  int32_t descriptor_dim;
} NepaModelInfo;

typedef struct NepaWorkspaceEstimate {
  uint64_t model_bytes;
  uint64_t workspace_bytes;
  uint64_t total_bytes;
  int32_t atom_capacity;
  int32_t structure_capacity;
} NepaWorkspaceEstimate;

typedef struct NepaStructureBatch {
  int32_t num_structures;
  int32_t total_atoms;
  const int32_t* atom_counts;
  const int32_t* atom_offsets;
  /* Zero-based model type indices in [0, num_types). */
  const int32_t* types;
  const double* positions_aos3;
  const double* spins_aos3;
  /* 3x3 cell in NEP order: ax, bx, cx, ay, by, cy, az, bz, cz. */
  const double* boxes_row_major9;
  /* Required; every structure must use {1, 1, 1}. Non-periodic inputs are unsupported. */
  const int32_t* pbc_flags3;
} NepaStructureBatch;

typedef struct NepaFindForceResult {
  double* energy_per_structure;
  double* potential_per_atom;
  double* forces_aos3;
  /* Total structure virial in the GPUMD pressure-positive convention.
     For a static structure, ASE stress is -virial / volume.
     Order: xx, xy, xz, yx, yy, yz, zx, zy, zz. */
  double* virials_row_major9;
  /* Per-atom contributions in the same GPUMD convention and raw9 order. */
  double* virials_per_atom_row_major9;
  double* charge_per_atom;
  /* Per-atom 9 components in NEP compute() order: xx, xy, xz, yx, yy, yz, zx, zy, zz. */
  double* bec_per_atom_row_major9;
  double* mforces_aos3;
  /* Optional per-atom spin-energy-transfer tensor A_j with
     A[j,a,alpha] = -sum_i r_ij[a] * dU_i/ds_j[alpha].
     Row-major order is xSx, xSy, xSz, ySx, ySy, ySz, zSx, zSy, zSz. */
  double* spin_transfer_per_atom_row_major9;
} NepaFindForceResult;

typedef struct NepaFindDescriptorResult {
  /* Row-major per-atom descriptors with shape (total_atoms, descriptor_dim). */
  double* descriptors;
} NepaFindDescriptorResult;

typedef struct NepaEvaluateResult {
  NepaFindForceResult prediction;
  NepaFindDescriptorResult descriptor;
} NepaEvaluateResult;

typedef struct NepaDipoleResult {
  /* Row-major values with shape (num_structures, 3). */
  double* dipoles_row_major3;
} NepaDipoleResult;

typedef struct NepaPolarizabilityResult {
  /* Row-major values with shape (num_structures, 6): xx, yy, zz, xy, yz, zx. */
  double* polarizabilities_row_major6;
} NepaPolarizabilityResult;

typedef struct NepaDftd3Parameters {
  const char* functional;
  double cutoff;
  double cutoff_cn;
} NepaDftd3Parameters;

typedef struct NepaDftd3Result {
  double* energy_per_structure;
  double* potential_per_atom;
  double* forces_aos3;
  /* Structure and per-atom raw9 order: xx, xy, xz, yx, yy, yz, zx, zy, zz. */
  double* virials_row_major9;
  double* virials_per_atom_row_major9;
} NepaDftd3Result;

typedef struct NepaLammpsNeighborInput {
  int nlocal;
  int inum;
  int* ilist;
  int* numneigh;
  int** firstneigh;
  int* types;
  int* type_map;
  double** positions;
  /* LAMMPS atom_style spin layout: sp[0..2] is the unit spin direction,
     sp[3] is the magnetic moment magnitude. */
  double** spins;
} NepaLammpsNeighborInput;

typedef struct NepaLammpsNeighborResult {
  double* total_potential;
  /* 6 components in LAMMPS order: xx, yy, zz, xy, xz, yz. */
  double* total_virial6;
  double* potential_per_atom;
  double** forces;
  double** mforces;
  /* Per-atom 9 components in NEP compute_for_lammps() order:
     xx, yy, zz, xy, xz, yz, yx, zx, zy. */
  double** virials_per_atom9;
  /* Same row-major A_j order as NepaFindForceResult; unlike virial this is
     not reordered into the LAMMPS stress-tensor convention. */
  double** spin_transfer_per_atom_row_major9;
} NepaLammpsNeighborResult;

typedef struct NepaLammpsDeviceNeighborInput {
  int nlocal;
  int nall;
  int inum;
  int max_neighbors;
  int neighbor_rows;
  int numneigh_length;
  const int* ilist;
  const int* numneigh;
  /* Strided device neighbor view. LAMMPS/Kokkos legacy and no-legacy layout
     conversion stays in the pair wrapper. */
  const int* neighbors;
  /* Optional length-nall owner ids used only to de-duplicate periodic ghost
     images while preserving the actual neighbor index for geometry. */
  const int* neighbor_owner;
  int neighbor_atom_stride;
  int neighbor_slot_stride;
  /* Device model types, 0-based after any LAMMPS type-map conversion. */
  const int* types;
  /* Optional device LAMMPS type map. When provided, types are interpreted as
     LAMMPS atom types and mapped through type_map[type] on device. */
  const int* type_map;
  int type_map_length;
  const double* positions;
  int position_atom_stride;
  int position_component_stride;
  const double* spins;
  int spin_atom_stride;
  int spin_component_stride;
} NepaLammpsDeviceNeighborInput;

typedef struct NepaLammpsDeviceNeighborResult {
  /* Optional device totals. Provide both pointers to request global energy and
     virial; leave both null for force-only MD steps. */
  double* total_potential;
  /* 6 components in LAMMPS order: xx, yy, zz, xy, xz, yz. */
  double* total_virial6;
  double* potential_per_atom;
  double* forces;
  int force_atom_stride;
  int force_component_stride;
  double* mforces;
  int mforce_atom_stride;
  int mforce_component_stride;
  /* Per-atom 9 components in NEP compute_for_lammps() order:
     xx, yy, zz, xy, xz, yz, yx, zx, zy. */
  double* virials_per_atom9;
  int virial_atom_stride;
  int virial_component_stride;
  /* Device per-atom A_j in fixed row-major spatial-by-spin order. */
  double* spin_transfer_per_atom_row_major9;
  int spin_transfer_atom_stride;
  int spin_transfer_component_stride;
} NepaLammpsDeviceNeighborResult;

NEP_ADAPTERS_API int nepa_api_version(void);
NEP_ADAPTERS_API int nepa_backend_count(void);
NEP_ADAPTERS_API NepaStatus nepa_backend_info(int index, NepaBackendInfo* out);
NEP_ADAPTERS_API NepaStatus nepa_load_model(
    const char* backend_name,
    const char* model_path,
    NepaModel** out);
NEP_ADAPTERS_API NepaStatus nepa_model_info(
    NepaModel* model,
    NepaModelInfo* out);
NEP_ADAPTERS_API NepaStatus nepa_model_kind(
    NepaModel* model,
    NepaModelKind* out);
NEP_ADAPTERS_API NepaStatus nepa_estimate_workspace(
    NepaModel* model,
    int32_t atom_capacity,
    int32_t structure_capacity,
    NepaWorkspaceEstimate* out);
NEP_ADAPTERS_API NepaStatus nepa_find_force_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaFindForceResult* result);
NEP_ADAPTERS_API NepaStatus nepa_evaluate_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaEvaluateResult* result);
NEP_ADAPTERS_API NepaStatus nepa_find_charge_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaFindForceResult* result);
NEP_ADAPTERS_API NepaStatus nepa_find_descriptors(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaFindDescriptorResult* result);
NEP_ADAPTERS_API NepaStatus nepa_find_dipoles(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaDipoleResult* result);
NEP_ADAPTERS_API NepaStatus nepa_find_polarizabilities(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaPolarizabilityResult* result);
NEP_ADAPTERS_API NepaStatus nepa_compute_dftd3_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    const NepaDftd3Parameters* parameters,
    NepaDftd3Result* result);
NEP_ADAPTERS_API NepaStatus nepa_compute_with_dftd3_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    const NepaDftd3Parameters* parameters,
    NepaDftd3Result* result);
NEP_ADAPTERS_API NepaStatus nepa_find_force_lammps_neighbors(
    NepaModel* model,
    const NepaLammpsNeighborInput* input,
    NepaLammpsNeighborResult* result);
NEP_ADAPTERS_API NepaStatus nepa_find_force_lammps_device_neighbors(
    NepaModel* model,
    const NepaLammpsDeviceNeighborInput* input,
    NepaLammpsDeviceNeighborResult* result);
NEP_ADAPTERS_API NepaStatus nepa_cancel_model(NepaModel* model);
NEP_ADAPTERS_API NepaStatus nepa_reset_cancel(NepaModel* model);
NEP_ADAPTERS_API void nepa_free_model(NepaModel* model);
NEP_ADAPTERS_API const char* nepa_status_message(NepaStatus status);
NEP_ADAPTERS_API const char* nepa_last_error_message(void);

#ifdef __cplusplus
}
#endif
