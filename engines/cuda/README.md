# CUDA Engine

This directory contains the CUDA engine implementation.

CUDA dependencies must stay scoped to this engine and CUDA-enabled packages.
The core runtime and CPU-only Python package must remain buildable without CUDA.

The directory is organized around a small orchestration interface and separate
CUDA compilation units:

- `cuda_engine.cpp` owns the engine interface and caller-specific staging.
- `force_pipeline.*` owns the complete force dataflow behind one model-driven
  interface. Callers provide topology and requested outputs; ordinary/spin
  selection, ZBL composition, and virial placement stay inside the module.
- `device_operations.hpp` is the single private interface for device staging,
  neighbor construction, descriptors, ANN evaluation, forces, and output.
  Individual `.cu` files remain separate CUDA compilation units; they are not
  separate public modules.
- `spin_onsite.cu` owns spin orchestration, with descriptor and force device
  implementation kept in two private `.cuh` fragments.
- The ordinary NEP4/NEP5 `nep.txt` protocol follows
  `/Users/superbing/Desktop/Workspace/torchnep/src/force/nep.cu`; in particular,
  descriptor dimension is computed from `n_max` plus the newer `l_max ...
  has_q_*` body-channel flags rather than inferred from the `ANN` line.
- `model_protocol.*` owns protocol parsing and parameter counts.
  `device_model.hpp` groups the model interface; `model_parameters.cpp` and
  `device_model.cu` implement its host-packing and device-upload halves.
  `device_workspace.hpp` groups planning, allocation, and view interfaces;
  `workspace_plan.cpp` and `device_workspace.cu` implement those two halves.
  Kernel code should depend on these internal contracts instead of re-parsing
  text or duplicating size math.
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
  cell-list based, and leaves pair geometry to the descriptor core so neighbor
  construction does not materialize data that the next stage would overwrite.
- Neighbor-list correctness is checked against a self-contained brute-force
  PBC oracle in the CUDA device test, including a tilted triclinic box. The
  production builder stays cell-list based; the all-pairs path is only a
  small-test oracle so descriptor and force work does not inherit an unvalidated
  neighbor contract.
- `angular_descriptor.cu` owns the shared structural descriptor core. It reads
  positions and slot-major neighbor lists directly, forms radial basis sums in
  per-block scratch, and accumulates angular tiles without global radial or
  angular basis-cache arrays. The same core serves ordinary, spin, and charge
  orchestration; model-specific descriptor additions and ANN evaluation remain
  explicit following stages.
- The descriptor core writes `descriptors` as
  `atom + atom_capacity * descriptor_index`. For angular models it also keeps
  only the state needed by force backpropagation: minimum-image pair vectors,
  distances, and `sum_fxyz`. Supported channels are ordinary 3-body terms
  through `L=4` plus `q222`, `q1111`, `q112`, `q123`, `q233`, and `q134`.
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
- Device result preparation lives in `device_output.cu`. It keeps both batched
  and LAMMPS reductions/layout conversion on the GPU: per-atom forces are
  packed from SoA to the public layout, virial order is converted at the output
  seam, and requested totals are reduced before copyback.
- The local LAMMPS Kokkos source uses `X_FLOAT*[3]` coordinate views accessed as
  `x(i,0..2)` and Kokkos neighbor views accessed as `neighbors(i,j)`, with layout
  controlled by Kokkos/LAMMPS build macros. A future Kokkos frontend can pass
  device views more directly, but the host-neighbor API should continue to stage
  into this engine-owned execution layout.
- Ordinary and spin execution share the one `force_pipeline` interface while
  keeping their implementation stages private. Charge-specific device
  operations remain isolated in `qnep_charge.cu`; a future charge integration
  can compose the same structural and short-range stages without changing the
  caller interface.
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
