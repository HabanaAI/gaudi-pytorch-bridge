import functools
import numpy as np
import pytest
import torch
import pytest
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

def test_div(mode):
    cpu_tensor = torch.randn(9, 9, dtype=torch.float32)
    hpu_tensor = cpu_tensor.to("hpu")

    def test_div_(x):
        return torch.div(x, 2)

    result_cpu = test_div_(cpu_tensor)

    if mode == "graph":
        compiled_function = torch.compile(test_div_, backend="aot_hpu_training_backend")
        result_hpu = compiled_function(hpu_tensor)
        assert torch.allclose(result_cpu, result_hpu.cpu(), rtol=1e-3, atol=1e-3)
    else:
        result_hpu = test_div_(hpu_tensor)
        assert torch.allclose(result_cpu, result_hpu.cpu(), rtol=1e-3, atol=1e-3)

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

@pytest.mark.parametrize("memory_format", [None, torch.contiguous_format])
def test_clone(mode, memory_format):
    def raw_function(x):
        return torch.clone(x, memory_format=memory_format)

    cpu_tensor = torch.randn(4, 4)
    hpu_tensor = cpu_tensor.to("hpu")

    if mode == "graph":
        result_nocompile = raw_function(hpu_tensor).to("cpu")
        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend")
        result_compile = compiled_function_training(hpu_tensor).to("cpu")
        assert torch.equal(result_nocompile, result_compile)
    else:
        result_cpu = raw_function(cpu_tensor)
        result_hpu = raw_function(hpu_tensor).to("cpu")
        assert torch.equal(result_cpu, result_hpu)

@pytest.mark.parametrize("size_stride", [
                        ((20, 20), (20, 1)),
                        ((20, 20), (30, 1))])
def test_empty_strided(mode, size_stride):
    def test(size, stride, device):
        x = torch.empty_strided(size, stride, device=device)
        return x

    size, stride = size_stride
    hpu_device = torch.device("hpu")
    cpu_device = torch.device("cpu")

    if mode == "graph":
        result_nocompile = test(size, stride, hpu_device)
        compiled_function_training = torch.compile(test, backend="aot_hpu_training_backend")
        result_compile = compiled_function_training(size, stride, hpu_device)
        assert (result_nocompile.size() == result_compile.size() \
            and result_nocompile.dtype == result_compile.dtype \
            and result_nocompile.stride() == result_compile.stride())
    else:
        result_cpu = test(size, stride, cpu_device)
        result_hpu = test(size, stride, hpu_device)
        assert (result_hpu.size() == result_cpu.size() \
            and result_hpu.dtype == result_cpu.dtype \
            and result_hpu.stride() == result_cpu.stride())

@pytest.mark.parametrize("memory_format", [
                        None,
                        torch.contiguous_format])
@pytest.mark.parametrize("size", [(2, 3, 4, 5)])
def test_empty_memory_format(mode, size, memory_format):
    def test(size, device, memory_format):
        x = torch.empty(size, device=device, memory_format=memory_format)
        return x

    hpu_device = torch.device("hpu")
    cpu_device = torch.device("cpu")

    if mode == "graph":
        result_nocompile = test(size, hpu_device, memory_format)
        compiled_function = torch.compile(test, backend="aot_hpu_training_backend")
        result_compile = compiled_function(size, hpu_device, memory_format)
        assert (result_nocompile.size() == result_compile.size() \
            and result_nocompile.dtype == result_compile.dtype)
    else:
        result_cpu = test(size, cpu_device, memory_format)
        result_hpu = test(size, hpu_device, memory_format)
        assert (result_hpu.size() == result_cpu.size() \
            and result_hpu.dtype == result_cpu.dtype)

def test_to_copy_dtype(mode):
    def raw_function(x, dtype):
        return torch.ops.aten._to_copy(x, dtype=dtype)

    input_tensor = torch.Tensor(np.random.randint(-1, 1, (20, 20)))
    dtype = input_tensor.dtype
    cpu_tensor = input_tensor.ge(0)
    hpu_tensor = cpu_tensor.to("hpu")

    if mode == "graph":
        result_nocompile = raw_function(hpu_tensor, dtype).to("cpu")
        compiled_function_training = torch.compile(raw_function, backend="aot_hpu_training_backend")
        result_compile = compiled_function_training(hpu_tensor, dtype).to("cpu")
        assert torch.equal(result_nocompile, result_compile)
    else:
        result_cpu = raw_function(cpu_tensor, dtype)
        result_hpu = raw_function(hpu_tensor, dtype).to("cpu")
        assert torch.equal(result_cpu, result_hpu)
