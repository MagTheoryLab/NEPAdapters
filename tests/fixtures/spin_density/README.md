# 独立密度描述符参考数据

上游来源：GPUMD-MagTheoryLab `c516@cd6ed6b9f46646176486dafc418a36453fb68c70`。

- `soc/`：原样复制上游 `tests_pytest/fixtures/nep_spin/c1l0s1`、`c3l2s2`、
  `c2l4s4` 的 `.nep` 与 `.json`；每套包含 large/triclinic/zero/isolated，合计 12 状态。
  标签由上游 `tests_pytest/generate_spin_oracles.py` 使用独立 e3nn FP64/autograd
  参考模型生成；JSON 保留模型 SHA256 和全部参考 Python 源文件的 SHA256。
- `nosoc/`：用上游 `tests_pytest/validate_spin_nosoc.py` 的独立 Legendre oracle
  生成，适配后的生成器为 `tests/generate_spin_density_nosoc.py`。
  六套形状、45 状态；包含独立空间/自旋旋转、时间反演、单 spin 旋转响应。
  模型固定种子，JSON 保留模型 SHA256 和不同 FD 步长的一致性误差。
  非 SOC 模型 ANN 的结构输入权重为零，独立 descriptor 标签只含 spin 列，
  不能把结构 descriptor 假定为零。
- `.dat`：由 `tests/convert_spin_density_oracles.py` 无计算地序列化 JSON，供 C++
  测试读取；盒矩阵从 ASE 行向量转置为底层 API 约定。
- `lammps/`：SOC C3L2S2 large 案例的几何和标签，直接使用 `soc/c3l2s2.nep`。
  包含原始非磁 Bi/O 自旋，用于确认输入保留和内部有效自旋屏蔽。

再生成非 SOC 数据并序列化：

```sh
python tests/generate_spin_density_nosoc.py --generate-fixtures tests/fixtures/spin_density/nosoc
python tests/convert_spin_density_oracles.py
```

使用仓库 `mysci` 环境；重新生成 SOC 需要上游 generator 指定的独立参考目录，
该目录未复制到本仓库。冻结模型不是材料拟合势，只用于推理正确性。
