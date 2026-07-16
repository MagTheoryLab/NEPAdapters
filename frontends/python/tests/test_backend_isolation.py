import importlib.util
import os
import sys

import nep_adapters


def main():
    gpu_module_name = "nep_adapters.nep_gpu"
    if gpu_module_name in sys.modules:
        raise AssertionError("import nep_adapters must not eagerly load nep_gpu")

    gpu_extension_present = importlib.util.find_spec(gpu_module_name) is not None
    if gpu_module_name in sys.modules:
        raise AssertionError("probing nep_gpu availability must not import it")

    with nep_adapters.load_model(
        "cpu",
        os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"],
    ) as model:
        descriptor_dim = int(model.model_info()["descriptor_dim"])
    if descriptor_dim <= 0:
        raise AssertionError("cpu model metadata is invalid")
    if gpu_module_name in sys.modules:
        raise AssertionError("cpu model loading must not import nep_gpu")

    try:
        nep_adapters.load_model(
            "cpu_opt",
            os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"],
        )
    except ValueError as error:
        if "unknown backend" not in str(error):
            raise
    else:
        raise AssertionError("the removed cpu_opt backend name must not be accepted")

    if not gpu_extension_present:
        try:
            nep_adapters.register_cuda()
        except RuntimeError as error:
            if "nep_gpu could not be loaded" not in str(error):
                raise
        else:
            raise AssertionError("CPU-only package must reject CUDA registration")

        with nep_adapters.load_model(
            "cpu",
            os.environ["NEP_ADAPTERS_PYTHON_TEST_MODEL"],
        ) as model:
            if int(model.model_info()["descriptor_dim"]) != descriptor_dim:
                raise AssertionError("failed CUDA import changed the CPU backend")

    print(
        "python backend isolation:",
        f"gpu_extension_present={int(gpu_extension_present)}",
        f"gpu_eagerly_loaded={int(gpu_module_name in sys.modules)}",
        f"cpu_descriptor_dim={descriptor_dim}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
