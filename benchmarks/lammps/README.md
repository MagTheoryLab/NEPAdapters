# LAMMPS 性能测试

LAMMPS benchmark 必须使用真实 LAMMPS 输入，不要维护另一套模拟器。

推荐目录结构：

```text
benchmarks/lammps/
  cases/
    case-name/
      in.nep
      data.system
      README.md
```

每个 case 应说明：

- 原子数和元素组成；
- 使用的模型文件；
- LAMMPS 构建条件；
- 测试的 pair style；
- MD 步数和 thermo 输出频率；
- CPU、GPU 和 MPI 计时命令。

LAMMPS 性能测试使用 `bench;performance;frontend;lammps` 标签。CPU 使用 `nep/cpu`，CUDA Kokkos 使用 `nep/gpu`。性能结果必须来自真实 MD 运行，并同时记录硬件、线程或 rank、模型和输入规模。
