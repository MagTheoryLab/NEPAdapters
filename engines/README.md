# Engines

Engines implement the core engine SPI. They do the numerical work and should not
depend on Python or LAMMPS frontends.

Supported engines:

- `cpu`: CPU engine for ordinary, spin, and charge NEP models.
- `cuda`: CUDA engine; the LAMMPS frontend requires CUDA-enabled Kokkos.
Only tests should compare engines with each other. Production code should select
engines through core capabilities and registry behavior.
