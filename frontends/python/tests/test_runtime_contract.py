import hashlib
import os

import nep_adapters


def main():
    model_path = os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"]
    info = nep_adapters.inspect_model(model_path)
    expected_hash = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
    if info.sha256 != expected_hash:
        raise AssertionError("inspect_model SHA256 mismatch")
    if info.model_type != "ordinary" or not info.supports("batch_find_force"):
        raise AssertionError("inspect_model semantic capabilities mismatch")
    if info.num_types != len(info.elements) or info.descriptor_dim <= 0:
        raise AssertionError("inspect_model metadata is incomplete")

    cpu = nep_adapters.backend_status("cpu")
    if not cpu.installed or not cpu.available or cpu.reason != "available":
        raise AssertionError("CPU backend status mismatch")

    try:
        nep_adapters.backend_status("gpu")
    except nep_adapters.InvalidInputError as error:
        if error.code != "invalid_input" or error.operation != "backend_status":
            raise AssertionError("invalid backend error metadata mismatch")
    else:
        raise AssertionError("legacy gpu backend name must be rejected")

    with nep_adapters.NEPCalculator(model_path, backend="cpu") as calculator:
        if calculator.model_info.sha256 != info.sha256:
            raise AssertionError("calculator and inspect_model metadata differ")
        try:
            calculator.estimate_workspace(100)
        except nep_adapters.UnsupportedModelError as error:
            if error.backend != "cpu" or error.operation != "workspace_estimate":
                raise AssertionError("workspace error metadata mismatch")
        else:
            raise AssertionError("CPU backend must not report CUDA workspace")

    print(
        "python runtime contract:",
        f"model_type={info.model_type}",
        f"elements={','.join(info.elements)}",
        f"sha256={info.sha256[:12]}",
    )


if __name__ == "__main__":
    main()
