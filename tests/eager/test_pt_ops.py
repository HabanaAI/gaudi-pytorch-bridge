import functools
import torch
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.dynamo.compile_backend

def test_argmax(mode):
    def test(func, cpu_tensor):
        hpu_tensor = cpu_tensor.to("hpu")

        if mode == "graph":
            result_nocompile = func(hpu_tensor).to("cpu")
            compiled_function_training = torch.compile(func, backend="aot_hpu_training_backend")
            result_compile = compiled_function_training(hpu_tensor).to("cpu")
            result_compile = result_compile.to(torch.int64)    # TODO: SW-140031
            assert torch.allclose(result_nocompile, result_compile, rtol=0, atol=0)
        else:
            result_cpu = func(cpu_tensor)
            result_hpu = func(hpu_tensor).to("cpu")
            assert torch.allclose(result_cpu, result_hpu, rtol=0, atol=0)

    B0 = 4
    test(lambda x: torch.argmax(x), torch.randn(B0))
    test(lambda x: torch.argmax(x), torch.randn(B0, 2, 3))
    test(lambda x: torch.argmax(x, dim=0), torch.randn(B0, 2, 3))
    test(lambda x: torch.argmax(x, dim=-1), torch.randn(B0, 2, 3))
    test(lambda x: torch.argmax(x, dim=2, keepdim=True), torch.randn(B0, 2, 3))

def test_alias(mode):
    def raw_function(x):
        y = x[...]
        y = y + 2
        return y

    x = torch.randn(3, 4)
    hx = x.to("hpu")

    result_cpu = raw_function(x)

    if mode == "graph":
        compiled_function = torch.compile(raw_function, backend="aot_hpu_training_backend")
        result_compile = compiled_function(hx).to("cpu")
        assert torch.allclose(result_cpu, result_compile, rtol=1e-3, atol=1e-3)
    else:
        result_hpu = raw_function(hx).to("cpu")
        assert torch.allclose(result_cpu, result_hpu, rtol=1e-3, atol=1e-3)
