###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import torch
import torch.nn as nn
import pytest
from torch.testing._internal.common_methods_invocations import op_db
import habana_frameworks.torch.dynamo.compile_backend  # noqa: F401
import habana_frameworks.torch.utils.experimental as htexp
from functools import reduce
from test_utils import generic_setup_teardown_env


all_dtypes = [
    torch.bfloat16,
    torch.float,
    torch.int,
    torch.int16,
    torch.int8,
    torch.bool,
]
@pytest.fixture(autouse=True, scope="module")
def setup_teardown_env():
    def callback():
        pass

    generic_setup_teardown_env(temp_test_env={"PT_HPU_LAZY_MODE": 0}, callback=callback)


if htexp._get_device_type() != htexp.synDeviceType.synDeviceGaudi:
    all_dtypes += [torch.half]


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize(
    "memory_format", [torch.channels_last, torch.contiguous_format]
)
@pytest.mark.parametrize("torch_func", [torch.empty_like, torch.zeros_like])
def test_empty_and_zeros_like(dtype, memory_format, torch_func):
    requires_grad = False
    layout = torch.strided

    def fn(tensor, dtype, layout, requires_grad, memory_format, torch_func):
        return torch_func(
            tensor,
            dtype=dtype,
            layout=layout,
            requires_grad=requires_grad,
            memory_format=memory_format,
        )

    tensor = torch.randn(4, 3, 2, 5)

    compiled_cpu = torch.compile(fn)
    cpu_res = compiled_cpu(
        tensor, dtype, layout, requires_grad, memory_format, torch_func
    )

    compiled_hpu = torch.compile(fn, backend="aot_hpu_training_backend")
    hpu_res = compiled_hpu(
        tensor.to("hpu"), dtype, layout, requires_grad, memory_format, torch_func
    )

    assert cpu_res.size() == hpu_res.size()
    assert cpu_res.dtype == hpu_res.dtype


@pytest.mark.skip(reason="https://jira.habana-labs.com/browse/SW-150162")
@pytest.mark.parametrize(
    "dtype, layout, device",
    [(torch.int, torch.strided, torch.device("hpu")), (None, None, None)],
)
def test_new_empty_strided(dtype, layout, device):
    def fn(tensor, size, stride, dtype, layout, device):
        return tensor.new_empty_strided(
            size=size, stride=stride, dtype=dtype, layout=layout, device=device
        )

    tensor = torch.randn(4, 3, 2, 5)
    size = (5, 4, 3)
    stride = (2, 3, 5)

    compiled_cpu = torch.compile(fn)
    cpu_result = compiled_cpu(tensor, size, stride, dtype, layout, device)

    compiled_hpu = torch.compile(fn, backend="aot_hpu_training_backend")
    hpu_result = compiled_hpu(tensor.to("hpu"), size, stride, dtype, layout, device)

    assert hpu_result.size() == cpu_result.size()
    assert hpu_result.dtype == cpu_result.dtype
    assert hpu_result.layout == cpu_result.layout


def run_test(aten_name, dtype):
    def get_op_info(aten_name):
        return next((x for x in op_db if x.aten_name == aten_name), None)

    opinfo = get_op_info(aten_name)
    for sample_input in opinfo.reference_inputs("cpu", dtype):
        t_inp, t_args, t_kwargs = (
            sample_input.input,
            sample_input.args,
            sample_input.kwargs,
        )

        def fn(op, t_inp, t_args, t_kwargs):
            return op(t_inp, *t_args, **t_kwargs)

        compiled_cpu = torch.compile(fn)
        result_cpu = compiled_cpu(opinfo.op, t_inp, t_args, t_kwargs)

        compiled_hpu = torch.compile(fn, backend="aot_hpu_training_backend")
        result_hpu = compiled_hpu(
            opinfo.op,
            t_inp.to("hpu"),
            (
                *(
                    arg.to("hpu") if isinstance(arg, torch.Tensor) else arg
                    for arg in t_args
                ),
            ),
            t_kwargs,
        )

        results = list(zip(result_cpu, result_hpu))
        return results
    return []


@pytest.mark.parametrize("dtype", all_dtypes)
def test_as_strided(dtype):
    results = run_test("as_strided", dtype)
    for result_cpu, result_hpu in results:
        assert result_hpu.size() == result_cpu.size()
        assert result_hpu.dtype == result_cpu.dtype
        assert result_hpu.layout == result_cpu.layout
        assert result_hpu.cpu().equal(result_cpu)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_as_strided_scatter(dtype):
    results = run_test("as_strided_scatter", dtype)
    for result_cpu, result_hpu in results:
        assert result_hpu.size() == result_cpu.size()
        assert result_hpu.dtype == result_cpu.dtype
        assert result_hpu.layout == result_cpu.layout
        assert result_hpu.cpu().equal(result_cpu)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_slice_scatter(dtype):
    results = run_test("slice_scatter", dtype)
    for result_cpu, result_hpu in results:
        assert result_hpu.size() == result_cpu.size()
        assert result_hpu.dtype == result_cpu.dtype
        assert result_hpu.layout == result_cpu.layout
        assert result_hpu.cpu().equal(result_cpu)


@pytest.mark.parametrize("dtype", all_dtypes)
def test_expand(dtype):
    if dtype == torch.half:
        pytest.skip("Half is not supported for expand.")
    """
    expand is a view op.
    For instance, if we perform inplace update on expand o/p,
    the expand input should also reflect the change.
    In our design, view output are eagerized.
    To test graph flow, we need to keep expand as a graph intermediate.
    """

    def fn(tensor, sizes):
        exp_t = tensor.expand(sizes)
        return exp_t.clone()

    tensor = torch.randn(3, 1).to(dtype)

    compiled_cpu = torch.compile(fn)
    cpu_res = compiled_cpu(tensor, (3, 4))

    compiled_hpu = torch.compile(fn, backend="aot_hpu_training_backend")
    hpu_res = compiled_hpu(tensor.to("hpu"), (3, 4))

    assert cpu_res.size() == hpu_res.size()
    assert cpu_res.dtype == hpu_res.dtype


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("dim", [-1, 0])
def test_unsqueeze(dtype, dim):
    def raw_function(x):
        x = x * 2
        b = x.unsqueeze(dim)
        c = b.clone()
        return c

    cpu_tensor = torch.randn(96).to(dtype)
    hpu_tensor = cpu_tensor.to("hpu")

    compiled_cpu = torch.compile(raw_function)
    cpu_res = compiled_cpu(cpu_tensor)

    compiled_hpu = torch.compile(raw_function, backend="aot_hpu_training_backend")
    hpu_res = compiled_hpu(hpu_tensor)

    assert torch.equal(cpu_res, hpu_res.to("cpu"))



def test_constant_pad_nd():
    def raw_function(x, device):
        m = nn.ConstantPad2d(2, 3.5).to(device)
        return m(x)

    cpu_tensor = torch.randn(1, 2, 2)
    hpu_tensor = cpu_tensor.to("hpu")

    compiled_cpu = torch.compile(raw_function)
    cpu_res = compiled_cpu(cpu_tensor, "cpu")

    compiled_hpu = torch.compile(raw_function, backend="aot_hpu_training_backend")
    hpu_res = compiled_hpu(hpu_tensor, "hpu")

    assert torch.allclose(cpu_res, hpu_res.to('cpu'), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", all_dtypes)
@pytest.mark.parametrize("torch_func", [torch.logical_and, torch.logical_xor, torch.logical_or])
def test_logical_bin_ops(dtype, torch_func):
    def raw_function(a, b):
        return torch_func(a, b)

    cpu_tensor_a = torch.randn(16).to(dtype)
    hpu_tensor_a = cpu_tensor_a.to("hpu")

    cpu_tensor_b = torch.randn(16).to(dtype)
    hpu_tensor_b = cpu_tensor_b.to("hpu")

    compiled_cpu = torch.compile(raw_function)
    cpu_res = compiled_cpu(cpu_tensor_a, cpu_tensor_b)

    compiled_hpu = torch.compile(raw_function, backend="aot_hpu_training_backend")
    hpu_res = compiled_hpu(hpu_tensor_a, hpu_tensor_b)

    assert torch.equal(cpu_res, hpu_res.to("cpu"))


@pytest.mark.parametrize("dtype", all_dtypes)
def test_logical_not(dtype):
    def raw_function(a):
        return torch.logical_not(a)

    cpu_tensor = torch.randn(16).to(dtype)
    hpu_tensor = cpu_tensor.to("hpu")

    compiled_cpu = torch.compile(raw_function)
    cpu_res = compiled_cpu(cpu_tensor)

    compiled_hpu = torch.compile(raw_function, backend="aot_hpu_training_backend")
    hpu_res = compiled_hpu(hpu_tensor)

    assert torch.equal(cpu_res, hpu_res.to("cpu"))


@pytest.mark.xfail(reason="KeyError: 'torch_dynamo_backends'")
def test_cat():
    def raw_function(t1, t2):
        return torch.cat((t1, t2))

    compiled_fnc = torch.compile(raw_function, backend="aot_hpu_training_backend")

    t1 = torch.rand(8, 8)
    t2 = torch.rand(8, 8)

    t1_cpu = t1.to(device="cpu")
    t2_cpu = t2.to(device="cpu")
    cpu_reference = raw_function(t1_cpu, t2_cpu)

    hpu_output = compiled_fnc(t1, t2)

    torch.allclose(hpu_output.to(device="cpu"), cpu_reference)

@pytest.mark.parametrize("dtype", all_dtypes)
def test_unbind_opdbtest(dtype):
    results = run_test("unbind", dtype)
    for (a, b) in results:
        assert torch.allclose(a, b.cpu(), atol = 0.001, rtol = 0.001)

@pytest.mark.parametrize("shape_in", [(4, 4), (2, 3, 4, 4, 4)])
def test_nonzero(shape_in):
    def fn(tensor):
        return torch.nonzero(tensor)
    cpu_tensor = torch.randint(10, shape_in) > 5
    hpu_tensor = cpu_tensor.to("hpu")

    compiled_cpu = torch.compile(fn)
    cpu_res = compiled_cpu(cpu_tensor)

    compiled_hpu = torch.compile(fn, backend="aot_hpu_training_backend")
    hpu_res = compiled_hpu(hpu_tensor)

    assert torch.equal(cpu_res, hpu_res.to("cpu"))
