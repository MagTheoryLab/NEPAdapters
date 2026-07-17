# Python 性能测试

Python 前端性能测试使用 `pytest-benchmark`，便于与常规 Python 工具链比较。底层 C++ engine 计时仍放在 `benchmarks/cpp/`。

CTest 标签使用 `bench;performance;frontend;python`。测试必须区分 Python 调用开销和 native 计算时间，不能用 mock 模型代替真实后端吞吐。
