# Python Benchmarks

Python frontend benchmarks belong here once the Python package exists.

Use `pytest-benchmark` for Python-facing API timings so reports stay comparable
with normal Python tooling. Keep low-level C++ engine timing in
`benchmarks/cpp/`.

CTest labels for future Python runs should include
`bench;performance;frontend;python`.
