# 性能测试

本目录只放性能探针。正确性测试属于 `tests/`；吞吐、扩展性和 profiler 采样属于这里。

优先使用项目现有 runner：

- C/C++ engine microbenchmark 由 CMake 构建，通过 CTest 的 `bench` 标签运行；
- Python 前端性能测试使用 `pytest-benchmark`；
- LAMMPS 性能测试使用 `benchmarks/lammps/` 下的真实 LAMMPS 输入，不另写 MD 模拟器。

## 本地运行

```sh
cmake -S . -B .build/bench \
  -DNEP_ADAPTERS_BUILD_TESTS=ON \
  -DNEP_ADAPTERS_BUILD_BENCHMARKS=ON
cmake --build .build/bench -j2
ctest --test-dir .build/bench -L bench --output-on-failure
```

benchmark 可执行文件结束前至少输出一行 JSON，包含：

- benchmark 名称；
- engine 或 frontend 名称；
- model 或 case 名称；
- 原子数和迭代数；
- 是否编译 OpenMP；
- OpenMP 线程数；
- 总耗时；
- 每秒计算次数；
- 每秒 atom-step 数。

不要提交生成的 benchmark 结果。临时结果统一放在 `benchmarks/results/`。

## 生成综合报告

```sh
source /Users/superbing/miniconda3/etc/profile.d/conda.sh
conda activate mysci
python3 tools/generate_test_report.py --openmp-threads 4
```

报告写入本地产物目录 `reports/`。同一次运行还会生成 `reports/performance_conditions.json`，记录自动饱和扫描选出的固定测试规模。默认报告是单进程 OpenMP 测试。

有本地 LAMMPS 和 plugin 时，可以加入 MPI 正确性检查：

```sh
python3 tools/generate_test_report.py \
  --lammps-source-dir /path/to/lammps \
  --lmp-executable /path/to/lmp \
  --lammps-plugin /path/to/nepadaptersplugin.so
```

传入 `--lmp-executable` 后，脚本会先使用 `tests/fixtures/cpu_baseline/` 运行真实 LAMMPS baseline smoke，再运行 MPI smoke。

更重的本地扩展性扫描：

```sh
python3 tools/generate_test_report.py \
  --auto-scales 1x1x1,2x2x1,2x2x2,3x3x3,4x4x4 \
  --iterations 3 \
  --openmp-threads 4 \
  --min-saturation-atoms 2000
```
