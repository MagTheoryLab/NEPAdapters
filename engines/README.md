# Engines

Engines implement the core engine SPI. They do the numerical work and should not
depend on Python or LAMMPS frontends.

Planned engines:

- `cpu_nep3`: reference/oracle engine that adapts the official or existing NEP
  CPU class.
- `cpu_opt`: optimized CPU engine for OpenMP/SIMD/layout work.
- `cuda`: CUDA engine corresponding to the maintained NEP_GPU direction.

Only tests should compare engines with each other. Production code should select
engines through core capabilities and registry behavior.
