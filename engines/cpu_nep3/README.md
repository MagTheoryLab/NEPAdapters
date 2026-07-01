# cpu_nep3 Engine

This directory is the planned shim around the official or existing NEP CPU class.
The `nep3` name is historical: in this project it means the current NEP CPU
code path exposed by that class, not an intentionally old NEP3-only model.

Its purpose is to provide the first reference engine and parity oracle. It should
stay thin:

- translate core runtime views into the NEP CPU class input convention;
- expose capabilities honestly;
- preserve upstream license and attribution when source is vendored or linked;
- avoid performance-oriented rewrites.

Optimizations belong in `engines/cpu_opt/`, not here.

Current M0 scope:

- supported: model load, batch energy, force, summed virial, per-atom virial,
  descriptors, and the LAMMPS neighbor-list path for ordinary potential models;
- tested: bundled small NEP fixture and bundled `nep89.txt` large model smoke;
- not yet exposed: spin and charge.
