# CUDA 性能测试

CUDA engine benchmark 使用标签 `bench;performance;engine;cuda`。与 CPU 对比时应尽量复用相同 case 名称和模型。

Profiler 分工：

- Nsight Systems：检查时间线、launch 开销和 CPU/GPU 重叠；
- Nsight Compute：检查 kernel occupancy、访存和指令组成。

不要把 profiler 输出提交到 Git。临时采样放在 `benchmarks/results/`。

`nep_adapters_bench_cuda_batch` 接受 `--model radial|angular|q134`，可用同一 batch driver 比较纯 radial、常规 `L=4` angular 和更重的 high-body force 路径。Profiler 采样建议加 `--batch-only`，避免单结构循环主导时间线。

输出中的 `batched_vs_single_loop` 比较一次 batch 调用与使用同一 CUDA engine 逐结构重复调用。相对 baseline 的优化比例应从固定 baseline commit 的保存结果计算，不能把当前单结构循环临时当成历史 baseline。
