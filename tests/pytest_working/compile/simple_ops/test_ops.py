###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
import habana_frameworks.torch.dynamo.compile_backend  # noqa: F401
import pytest
import torch
from test_utils import (
    compile_function_if_compile_mode,
    format_tc,
    generic_setup_teardown_env,
    is_gaudi1,
)
from torch.testing._internal.common_methods_invocations import op_db

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


if not is_gaudi1():
    all_dtypes.append(torch.float16)


@pytest.mark.parametrize("dtype", all_dtypes, ids=format_tc)
@pytest.mark.parametrize("memory_format", [torch.channels_last, torch.contiguous_format], ids=format_tc)
@pytest.mark.parametrize("torch_func", [torch.empty_like, torch.zeros_like], ids=format_tc)
def test_empty_and_zeros_like(dtype, memory_format, torch_func):
    if pytest.mode == "compile" and torch_func == torch.zeros_like and dtype == torch.bool:
        pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")
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

    cpu_res = fn(tensor, dtype, layout, requires_grad, memory_format, torch_func)

    compiled_hpu = compile_function_if_compile_mode(fn)
    hpu_res = compiled_hpu(tensor.to("hpu"), dtype, layout, requires_grad, memory_format, torch_func)

    assert cpu_res.size() == hpu_res.size()
    assert cpu_res.dtype == hpu_res.dtype


@pytest.mark.parametrize(
    "dtype, layout, device_none", [(torch.float32, torch.strided, False), (None, None, True)], ids=format_tc
)
def test_new_empty_strided(dtype, layout, device_none):
    def fn(tensor, size, stride, dtype, layout, device):
        return tensor.new_empty_strided(size=size, stride=stride, dtype=dtype, layout=layout, device=device)

    tensor = torch.randn(4, 3, 2, 5)
    size = (5, 4, 3)
    stride = (2, 3, 5)

    cpu_result = fn(tensor, size, stride, dtype, layout, None if device_none else "cpu")

    compiled_hpu = compile_function_if_compile_mode(fn)
    hpu_result = compiled_hpu(tensor.to("hpu"), size, stride, dtype, layout, None if device_none else "hpu")

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

        torch._dynamo.reset()
        result_cpu = fn(opinfo.op, t_inp, t_args, t_kwargs)

        compiled_hpu = compile_function_if_compile_mode(fn)
        result_hpu = compiled_hpu(
            opinfo.op,
            t_inp.to("hpu"),
            (*(arg.to("hpu") if isinstance(arg, torch.Tensor) else arg for arg in t_args),),
            t_kwargs,
        )

        results = list(zip(result_cpu, result_hpu, strict=False))
        return results
    return []


@pytest.mark.parametrize("dtype", all_dtypes, ids=format_tc)
def test_as_strided(dtype):
    results = run_test("as_strided", dtype)
    for result_cpu, result_hpu in results:
        assert result_hpu.size() == result_cpu.size()
        assert result_hpu.dtype == result_cpu.dtype
        assert result_hpu.layout == result_cpu.layout
        assert result_hpu.cpu().equal(result_cpu)


@pytest.mark.parametrize("dtype", all_dtypes, ids=format_tc)
def test_as_strided_scatter(dtype):
    results = run_test("as_strided_scatter", dtype)
    for result_cpu, result_hpu in results:
        assert result_hpu.size() == result_cpu.size()
        assert result_hpu.dtype == result_cpu.dtype
        assert result_hpu.layout == result_cpu.layout
        assert result_hpu.cpu().equal(result_cpu)


@pytest.mark.parametrize("dtype", all_dtypes, ids=format_tc)
def test_slice_scatter(dtype):
    results = run_test("slice_scatter", dtype)
    for result_cpu, result_hpu in results:
        assert result_hpu.size() == result_cpu.size()
        assert result_hpu.dtype == result_cpu.dtype
        assert result_hpu.layout == result_cpu.layout
        assert result_hpu.cpu().equal(result_cpu)


@pytest.mark.parametrize("dtype", all_dtypes, ids=format_tc)
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

    cpu_res = fn(tensor, (3, 4))

    compiled_hpu = compile_function_if_compile_mode(fn)
    hpu_res = compiled_hpu(tensor.to("hpu"), (3, 4))

    assert cpu_res.size() == hpu_res.size()
    assert cpu_res.dtype == hpu_res.dtype


@pytest.mark.parametrize("dtype", all_dtypes, ids=format_tc)
@pytest.mark.parametrize("dim", [-1, 0])
def test_unsqueeze(dtype, dim):
    def raw_function(x):
        x = x * 2
        b = x.unsqueeze(dim)
        c = b.clone()
        return c

    cpu_tensor = torch.randn(96).to(dtype)
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_res = raw_function(cpu_tensor)

    compiled_hpu = compile_function_if_compile_mode(raw_function)
    hpu_res = compiled_hpu(hpu_tensor)

    assert torch.equal(cpu_res, hpu_res.to("cpu"))


def test_constant_pad_nd():
    def raw_function(x, device):
        return torch.constant_pad_nd(x, (1, 1), -1.0)

    cpu_tensor = torch.randn(1, 2, 2)
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_res = raw_function(cpu_tensor, "cpu")

    compiled_hpu = compile_function_if_compile_mode(raw_function)
    hpu_res = compiled_hpu(hpu_tensor, "hpu")

    assert torch.allclose(cpu_res, hpu_res.to("cpu"), rtol=1e-3, atol=1e-3)


logical_dtypes = [
    torch.bfloat16,
    torch.float,
    torch.int,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.long,
]

if not is_gaudi1():
    logical_dtypes.append(torch.float16)

logical_ops_not_supported_dtypes = {
    torch.logical_and: [torch.int16],
    torch.logical_or: [torch.long, torch.int, torch.int16],
    torch.logical_xor: [torch.long, torch.int, torch.int16],
    torch.logical_not: [torch.bfloat16, torch.float, torch.long, torch.int, torch.int16],
}


@pytest.mark.parametrize("dtype", logical_dtypes, ids=format_tc)
@pytest.mark.parametrize("torch_func", [torch.logical_and, torch.logical_xor, torch.logical_or], ids=format_tc)
def test_logical_bin_ops(dtype, torch_func):
    if dtype in logical_ops_not_supported_dtypes[torch_func]:
        pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")

    cpu_tensor_a = torch.randn(16).to(dtype)
    hpu_tensor_a = cpu_tensor_a.to("hpu")

    cpu_tensor_b = torch.randn(16).to(dtype)
    hpu_tensor_b = cpu_tensor_b.to("hpu")

    cpu_res = torch_func(cpu_tensor_a, cpu_tensor_b)

    compiled_hpu = compile_function_if_compile_mode(torch_func)
    hpu_res = compiled_hpu(hpu_tensor_a, hpu_tensor_b)

    assert torch.equal(cpu_res, hpu_res.to("cpu"))


@pytest.mark.parametrize("dtype", logical_dtypes, ids=format_tc)
def test_logical_not(dtype):
    if dtype in logical_ops_not_supported_dtypes[torch.logical_not]:
        pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")

    cpu_tensor = torch.randn(16).to(dtype)
    hpu_tensor = cpu_tensor.to("hpu")

    cpu_res = torch.logical_not(cpu_tensor)

    compiled_hpu = compile_function_if_compile_mode(torch.logical_not)
    hpu_res = compiled_hpu(hpu_tensor)

    assert torch.equal(cpu_res, hpu_res.to("cpu"))
