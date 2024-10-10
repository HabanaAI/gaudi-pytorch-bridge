###############################################################################
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import habana_frameworks.torch.internal.bridge_config as bc
import pytest
import torch
from test_utils import compare_tensors, format_tc, is_gaudi1

basic_dtypes = extended_dtypes = [torch.float32, torch.bfloat16, torch.int]
if not is_gaudi1():
    extended_dtypes = basic_dtypes + [torch.float8_e5m2, torch.float8_e4m3fn, torch.float16]


@pytest.fixture(autouse=True)
def skip_unsupported_compile(request):
    dtype = request.node.callspec.params["dtype"]
    if pytest.mode == "compile" and dtype in (
        torch.bfloat16,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    ):
        pytest.skip(reason="https://jira.habana-labs.com/browse/SW-167770")


def get_hpu_fn(fn):
    if pytest.mode == "compile":
        return torch.compile(fn, backend="hpu_backend", dynamic=False)
    else:
        return fn


def create_rand_tensors(shape, dtype):
    if dtype == torch.int:
        cpu_tensor = torch.randint(low=-127, high=127, size=shape, dtype=dtype)
    else:
        cpu_tensor = torch.randn(shape).to(dtype)
    if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
        cpu_tensor = cpu_tensor.float()
    return cpu_tensor, cpu_tensor.to("hpu")


def create_empty_tensors(shape, dtype):
    cpu_tensor = torch.empty(shape, dtype=dtype)
    return cpu_tensor, cpu_tensor.to("hpu")


@pytest.mark.parametrize("shape", [(20, 10), (2, 4, 6, 8)], ids=format_tc)
@pytest.mark.parametrize("dtype", extended_dtypes, ids=format_tc)
def test_median(dtype, shape):
    def fn(input):
        return torch.median(input=input)

    cpu_input, hpu_input = create_rand_tensors(shape, dtype)
    hpu_fn = get_hpu_fn(fn)

    cpu_output = fn(cpu_input)
    hpu_output = hpu_fn(hpu_input)

    compare_tensors(hpu_output, cpu_output, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("shape", [(20, 10), (2, 4, 6, 8)], ids=format_tc)
@pytest.mark.parametrize("dtype", basic_dtypes, ids=format_tc)
@pytest.mark.parametrize("dim", [0, -1], ids=format_tc)
@pytest.mark.parametrize("keepdim", [True, False], ids=format_tc)
def test_median_dim(dtype, shape, dim, keepdim):
    def fn(input, dim, keepdim):
        return torch.median(input=input, dim=dim, keepdim=keepdim)

    cpu_input, hpu_input = create_rand_tensors(shape, dtype)
    hpu_fn = get_hpu_fn(fn)

    cpu_output = fn(cpu_input, dim, keepdim)
    hpu_output = hpu_fn(hpu_input, dim, keepdim)

    compare_tensors(hpu_output.values, cpu_output.values, atol=0.0, rtol=0.0)
    compare_tensors(hpu_output.indices, cpu_output.indices, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("shape", [(20, 10), (2, 4, 6, 8)], ids=format_tc)
@pytest.mark.parametrize("dtype", basic_dtypes, ids=format_tc)
@pytest.mark.parametrize("dim", [0, -1], ids=format_tc)
@pytest.mark.parametrize("keepdim", [True, False], ids=format_tc)
def test_median_dim_out(dtype, shape, dim, keepdim):
    if pytest.mode == "lazy" and not bc.get_pt_enable_int64_support():
        pytest.skip(reason="index exceed int32 range which is unsupported")

    def fn(input, dim, keepdim, out):
        torch.median(input, dim=dim, keepdim=keepdim, out=out)

    expected_shape = list(shape)
    if keepdim:
        expected_shape[dim] = 1
    else:
        expected_shape.pop(dim)

    cpu_input, hpu_input = create_rand_tensors(shape, dtype)
    cpu_value, hpu_value = create_empty_tensors(expected_shape, dtype)
    cpu_index, hpu_index = create_empty_tensors(expected_shape, torch.int64)
    cpu_out = cpu_value, cpu_index
    hpu_out = hpu_value, hpu_index
    hpu_fn = get_hpu_fn(fn)

    fn(cpu_input, dim, keepdim, out=cpu_out)
    hpu_fn(hpu_input, dim, keepdim, out=hpu_out)

    compare_tensors(hpu_out[0], cpu_out[0], atol=0.0, rtol=0.0)
    compare_tensors(hpu_out[1], cpu_out[1], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("shape", [[1], [20, 10]], ids=format_tc)
@pytest.mark.parametrize("dtype", extended_dtypes, ids=format_tc)
def test_2_iterations(shape, dtype):
    def fn(*args):
        return torch.median(*args)

    hpu_fn = get_hpu_fn(fn)

    for iter in range(2):
        actual_shape = [d * (iter + 1) for d in shape]
        cpu_input, hpu_input = create_rand_tensors(actual_shape, dtype)

    res_hpu = hpu_fn(hpu_input)
    res_cpu = fn(cpu_input)

    compare_tensors(res_hpu, res_cpu, atol=0.0, rtol=0.0)
