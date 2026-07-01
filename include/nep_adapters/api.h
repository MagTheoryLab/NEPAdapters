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

#define NEP_ADAPTERS_API_VERSION_MAJOR 0
#define NEP_ADAPTERS_API_VERSION_MINOR 1
#define NEP_ADAPTERS_API_VERSION_PATCH 0

#ifdef __cplusplus
extern "C" {
#endif

typedef enum NepaStatus {
  NEPA_STATUS_OK = 0,
  NEPA_STATUS_INVALID_ARGUMENT = 1,
  NEPA_STATUS_UNSUPPORTED = 2,
  NEPA_STATUS_BACKEND_UNAVAILABLE = 3,
  NEPA_STATUS_RUNTIME_ERROR = 4
} NepaStatus;

typedef enum NepaCapabilityFlags {
  NEPA_CAPABILITY_BATCH_FIND_FORCE = 1u << 0u,
  NEPA_CAPABILITY_EXTERNAL_NEIGHBORS = 1u << 1u,
  NEPA_CAPABILITY_DEVICE_INPUT = 1u << 2u,
  NEPA_CAPABILITY_SPIN = 1u << 3u,
  NEPA_CAPABILITY_CHARGE = 1u << 4u,
  NEPA_CAPABILITY_VIRIAL = 1u << 5u,
  NEPA_CAPABILITY_DESCRIPTORS = 1u << 6u
} NepaCapabilityFlags;

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
} NepaModelInfo;

typedef struct NepaStructureBatch {
  int32_t num_structures;
  int32_t total_atoms;
  const int32_t* atom_counts;
  const int32_t* atom_offsets;
  const int32_t* types;
  const double* positions_aos3;
  const double* boxes_row_major9;
  const int32_t* pbc_flags3;
} NepaStructureBatch;

typedef struct NepaFindForceResult {
  double* energy_per_structure;
  double* potential_per_atom;
  double* forces_aos3;
  /* 9 components in NEP compute() order: xx, xy, xz, yx, yy, yz, zx, zy, zz. */
  double* virials_row_major9;
  /* Per-atom 9 components in NEP compute() order: xx, xy, xz, yx, yy, yz, zx, zy, zz. */
  double* virials_per_atom_row_major9;
} NepaFindForceResult;

typedef struct NepaLammpsNeighborInput {
  int nlocal;
  int inum;
  int* ilist;
  int* numneigh;
  int** firstneigh;
  int* types;
  int* type_map;
  double** positions;
} NepaLammpsNeighborInput;

typedef struct NepaLammpsNeighborResult {
  double* total_potential;
  /* 6 components in LAMMPS order: xx, yy, zz, xy, xz, yz. */
  double* total_virial6;
  double* potential_per_atom;
  double** forces;
  /* Per-atom 9 components in NEP compute_for_lammps() order:
     xx, yy, zz, xy, xz, yz, yx, zx, zy. */
  double** virials_per_atom9;
} NepaLammpsNeighborResult;

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
NEP_ADAPTERS_API NepaStatus nepa_find_force_batch(
    NepaModel* model,
    const NepaStructureBatch* batch,
    NepaFindForceResult* result);
NEP_ADAPTERS_API NepaStatus nepa_find_force_lammps_neighbors(
    NepaModel* model,
    const NepaLammpsNeighborInput* input,
    NepaLammpsNeighborResult* result);
NEP_ADAPTERS_API void nepa_free_model(NepaModel* model);
NEP_ADAPTERS_API const char* nepa_status_message(NepaStatus status);

#ifdef __cplusplus
}
#endif
