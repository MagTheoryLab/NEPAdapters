# cuda Engine

This directory contains the CUDA engine scaffold.

CUDA dependencies must stay scoped to this engine and CUDA-enabled packages.
The core runtime and CPU-only Python package must remain buildable without CUDA.

The current implementation deliberately starts with a non-spin model boundary:

- `cuda_engine.cpp` registers the `cuda` engine and parses enough `nep.txt`
  metadata to report cutoffs, type count, descriptor dimension, and the initial
  device-input capability.
- The ordinary NEP4/NEP5 `nep.txt` protocol follows
  `/Users/superbing/Desktop/Workspace/torchnep/src/force/nep.cu`; in particular,
  descriptor dimension is computed from `n_max` plus the newer `l_max ...
  has_q_*` body-channel flags rather than inferred from the `ANN` line.
- `model_protocol.*` owns protocol parsing and parameter counts.
  `model_parameters.*` owns host-side parameter packing. `workspace_plan.*`
  owns high-level device buffer sizing and names. Kernel code should depend on
  these internal contracts instead of re-parsing text or duplicating size math.
- Parameters are split for future cache locality: `ann_type_major` keeps each
  element type's W0/B0/W1 block contiguous, descriptor coefficients keep radial
  and angular coefficient regions contiguous, and `q_scaler` is uploaded as its
  own linear array.
- `device_model.cu` is compiled only when
  `NEP_ADAPTERS_CUDA_ENABLE_DEVICE_RUNTIME=ON`; it uploads these packed arrays to
  CUDA device memory and exposes stable device pointers for later kernels.
- `device_workspace.cu` follows the same opt-in rule for per-call memory. It
  allocates typed device arrays from `WorkspacePlan` and exposes a plain
  `DeviceWorkspaceView` so kernels can receive model parameters and execution
  buffers without knowing about allocation ownership.
- The CUDA engine has two caller shapes. `find_force_batch` accepts coordinates,
  boxes, and structure metadata, then uses an internal-neighbor workspace that
  owns neighbor construction. `find_force_lammps_neighbors` accepts a LAMMPS-style
  external neighbor list, stages active atoms, and reuses the caller's neighbor
  topology.
- Batch boxes use the NEP/GPUMD 3x3 order `ax,bx,cx, ay,by,cy, az,bz,cz`; the
  CUDA triclinic path uses this matrix directly for fractional-to-Cartesian
  conversion.
- Host staging lives in `host_staging.*`: direct batch input is converted from
  AoS positions to SoA positions plus per-structure metadata; LAMMPS host
  neighbors are converted from `ilist`/`numneigh`/`firstneigh` into active atom
  indices and slot-major neighbor arrays.
- Device staging lives in `device_staging.cu`: direct batch input can copy raw
  host arrays and run CUDA kernels for AoS-to-SoA positions and
  `atom_to_structure`; LAMMPS host-neighbor input still has to gather
  `double**`/`int**` rows on the host, then CUDA kernels map types and pack dense
  neighbor rows into slot-major execution arrays.
- Internal neighbor construction starts in `internal_neighbor_builder.cu`. It is
  a large-system cell-list path, not an all-pairs prototype: atoms are binned by
  cell counts, a prefix scan builds CSR cell offsets, atoms are scattered into
  cell-contiguous storage, then each center atom traverses neighboring cells to
  fill radial and angular slot-major neighbor lists.
- Both caller shapes stage the execution neighbor lists as slot-major
  (`center + atom_capacity * slot`) to match the current NEP CUDA kernels and
  keep per-slot center threads contiguous.
- The internal-neighbor builder supports orthorhombic and triclinic boxes with
  PBC. It bins atoms in fractional coordinates, keeps the production path
  cell-list based, and computes Cartesian pair vectors from the full 3x3 box
  before descriptor and force kernels consume them.
- Neighbor-list correctness is checked against a self-contained brute-force
  PBC oracle in the CUDA device test, including a tilted triclinic box. The
  production builder stays cell-list based; the all-pairs path is only a
  small-test oracle so descriptor and force work does not inherit an unvalidated
  neighbor contract.
- Pair geometry caching starts in `pair_geometry_cache.cu`. It consumes the
  validated slot-major neighbor lists and writes radial pair distances plus
  angular minimum-image deltas. Angular deltas are defined as
  `neighbor_position - center_position`; the CUDA device test compares these
  cached values against the same brute-force PBC oracle used for neighbor sets.
- Radial basis caching starts in `radial_basis_cache.cu`. It consumes
  `r12_radial` and writes `fc_radial` plus `fn_radial` using the current
  torchnep cutoff and Chebyshev radial basis formulas. `fn_radial` is laid out as
  `atom + atom_capacity * (slot + radial_capacity * basis_index)` so a future
  descriptor kernel can keep center atoms contiguous for a fixed slot and basis
  channel.
- Radial descriptor accumulation starts in `radial_descriptor.cu`. It consumes
  the radial basis cache plus packed descriptor coefficients and writes
  `descriptors` as `atom + atom_capacity * descriptor_index`; the first device
  test covers the same coefficient indexing used by torchnep:
  `(n * (basis_size_radial + 1) + k) * num_type_pairs + type_pair`.
- Angular basis caching starts in `angular_basis_cache.cu` and mirrors the
  radial cache layout with `r12_angular`, `fc_angular`, and `fn_angular`.
- Angular descriptor accumulation lives in `angular_descriptor.cu`. It covers
  ordinary 3-body channels through `L=4` plus the current high-body flags
  `q222`, `q1111`, `q112`, `q123`, `q233`, and `q134`; `sum_fxyz` is staged as
  `atom + atom_capacity * (n * abc_count + abc)` for the later force path.
- ANN energy evaluation starts in `ann_energy.cu`. The kernel handles packed
  NEP4/NEP5 one-hidden-layer layouts, writes per-atom `potential`, and stages
  descriptor derivatives in `fp` so force backpropagation can be added without
  changing the descriptor layout.
- Radial force accumulation starts in `radial_force.cu`. The public
  `find_force_batch` path now runs a real single-structure NEP4
  pipeline for orthorhombic and triclinic boxes: device staging, internal
  cell-list neighbors,
  descriptors, ANN energy, force accumulation, and host output copyback.
- Universal non-flexible ZBL accumulation lives in `zbl_force.cu`. It reuses the
  radial neighbor list when `zbl_outer <= cutoff_radial`, adds per-atom ZBL
  potential/force/virial after ANN force backpropagation, and is covered by a
  finite-difference force gate. Flexible ZBL and typewise ZBL cutoffs still
  return `NEPA_STATUS_UNSUPPORTED`.
- Angular force accumulation starts in `angular_force.cu`. The public force
  gates support regular angular channels through `L=4` plus
  `q222/q1111/q112/q123/q233/q134`; these paths are checked by
  finite-difference energy/force tests. The direct batch path runs these
  angular and high-body force channels through the same batched workspace used
  by radial/ZBL execution.
- Batched result preparation lives in `batch_output.cu`. It keeps CUDA backend
  reductions and layout conversion on the GPU: per-atom forces are packed from
  SoA to the public AoS layout, per-atom virials are converted to public
  row-major order, and per-structure energy/virial are reduced before the final
  public host-result copy.
- The local LAMMPS Kokkos source uses `X_FLOAT*[3]` coordinate views accessed as
  `x(i,0..2)` and Kokkos neighbor views accessed as `neighbors(i,j)`, with layout
  controlled by Kokkos/LAMMPS build macros. A future Kokkos frontend can pass
  device views more directly, but the host-neighbor API should continue to stage
  into this engine-owned execution layout.
- Ordinary potential models are accepted; spin, charge, dipole,
  polarizability, and temperature model tags are rejected until their contracts
  are explicit.
- `find_force_batch` validates the host batch contract and supports the
  device path described above. Other ordinary NEP shapes still return
  `NEPA_STATUS_UNSUPPORTED` until their force paths are verified.
- Tests are split by caller shape: ordinary batch, LAMMPS host-neighbor
  simulation, and LAMMPS Kokkos-style device staging. The Kokkos simulation is
  intentionally a workspace/layout contract until the public device-input ABI is
  defined.

The maintained NEP_GPU code path should be mined for kernels and measured
optimizations, but this engine should own its staging and batch/neighbor
contracts inside NEPAdapters rather than exposing the old LAMMPS bridge shape.
