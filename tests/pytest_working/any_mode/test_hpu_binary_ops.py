###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################
import pytest
import torch
from habana_frameworks.torch.dynamo.compile_backend import config as hpu_backend_config
from test_utils import (
    check_ops_executed_in_jit_ir,
    clear_t_compile_logs,
    compare_tensors,
    is_gaudi1,
    is_pytest_mode_compile,
)

dtypes = [torch.float32, torch.bfloat16, torch.int]
fp8_dtypes = [torch.float8_e5m2, torch.float8_e4m3fn]
if not is_gaudi1():
    dtypes += fp8_dtypes


def generate_tensor(shape, dtype):
    if dtype in fp8_dtypes:
        return torch.rand(shape).to(dtype)
    return (torch.randn(shape) * 3.0).to(dtype)


@pytest.mark.parametrize("func", [torch.add, torch.sub, torch.rsub])
@pytest.mark.parametrize("shape_a, shape_b", [[(), ()], [(1,), (2,)], [(4, 4), (1, 1)], [(16, 12), (16, 12)]])
@pytest.mark.parametrize("alpha", [1, 3])
@pytest.mark.parametrize("dtype", dtypes)
def test_binary(func, shape_a, shape_b, alpha, dtype):
    def fn(input, other, alpha):
        return func(input, other, alpha=alpha)

    if is_pytest_mode_compile():
        orig_flag = hpu_backend_config.reinplace_add
        hpu_backend_config.reinplace_add = False
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    input = generate_tensor(shape_a, dtype)
    other = generate_tensor(shape_b, dtype)

    input_hpu = input.to("hpu")
    other_hpu = other.to("hpu")
    if dtype in fp8_dtypes:
        input = input.float()
        other = other.float()

    expected = func(input, other, alpha=alpha)
    result = fn(input_hpu, other_hpu, alpha)

    if dtype in fp8_dtypes:
        expected = expected.to(dtype)

    if dtype == torch.bfloat16:
        tol = 0.05
    elif dtype in fp8_dtypes:
        tol = 0.25
    else:
        tol = 1e-06
    compare_tensors(result, expected, atol=tol, rtol=tol)
    if is_pytest_mode_compile():
        name = "add" if func == torch.add else "sub"
        check_ops_executed_in_jit_ir(name)
        hpu_backend_config.reinplace_add = orig_flag


@pytest.mark.skipif(is_gaudi1(), reason="G1 unsupported trunc mode")
@pytest.mark.parametrize("shape_a, shape_b", [[(), ()], [(1,), (2,)], [(4, 4), (1, 1)], [(16, 12), (16, 12)]])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8])
def test_mul_trunc(shape_a, shape_b, dtype):
    def fn(input, other):
        return torch.mul(input, other)

    if is_pytest_mode_compile():
        clear_t_compile_logs()
        torch._dynamo.reset()
        fn = torch.compile(fn, backend="hpu_backend")

    input = torch.randint(low=torch.iinfo(dtype).min, high=torch.iinfo(dtype).max, size=shape_a, dtype=dtype)
    other = torch.randint(low=torch.iinfo(dtype).min, high=torch.iinfo(dtype).max, size=shape_b, dtype=dtype)

    input_hpu = input.to("hpu")
    other_hpu = other.to("hpu")

    expected = torch.mul(input, other)
    result = fn(input_hpu, other_hpu)

    compare_tensors(result, expected, atol=0, rtol=0)
    if is_pytest_mode_compile():
        check_ops_executed_in_jit_ir("mul")
