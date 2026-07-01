# cuda Engine

This directory is reserved for the CUDA engine.

CUDA dependencies must stay scoped to this engine and CUDA-enabled packages.
The core runtime and CPU-only Python package must remain buildable without CUDA.

The maintained NEP_GPU code path should be adapted here rather than copied into
frontends.
