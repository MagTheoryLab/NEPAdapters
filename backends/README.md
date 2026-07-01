# Legacy Backend Notes

This directory is retained only as a transitional note from the earlier
"adapter/backend" layout.

New compute implementations should go under `engines/`:

- `engines/cpu_nep3/`
- `engines/cpu_opt/`
- `engines/cuda/`

LAMMPS and Python code should go under `frontends/`, not `backends/`.
