# 前端

前端负责把上层软件的数据约定转换为公共核心 API，它们不是计算引擎。

当前前端：

- `python`：Python 包、pybind11 绑定、NumPy 高层接口和可选 ASE 适配；
- `lammps`：LAMMPS pair style 与 runtime plugin。

前端只能依赖公共 API，不应包含 CPU 或 CUDA 引擎内部头文件，也不能复制后端算法。
