# LAMMPS Benchmarks

LAMMPS benchmarks should be real LAMMPS input decks, not a separate simulator.

Recommended layout:

```text
benchmarks/lammps/
  cases/
    case-name/
      in.nep
      data.system
      README.md
```

Each case should document:

- atom count and chemistry;
- model file expected by the case;
- LAMMPS build assumptions;
- pair style name being tested;
- MD steps and thermo output cadence;
- CPU/GPU/MPI command lines used for timing.

CTest labels for future LAMMPS runs should include
`bench;performance;frontend;lammps`.

The current CPU milestone only compiles the `nep/cpu` pair/plugin. A
real MD benchmark should be added here once a local LAMMPS executable with
`PLUGIN` is part of the test environment.
