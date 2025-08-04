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

import pytest
import torch
from compile.test_dynamo_utils import use_eager_fallback
from test_utils import compile_function_if_compile_mode, format_tc

igamma_dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
igammac_dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
lgamma_dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
digamma_dtypes = [
    torch.bfloat16,
    torch.float16,
    torch.bool,
    torch.float32,
    torch.float64,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.int8,
    torch.uint8,
]
shape_list = [[0], [1], [3, 4]]


@pytest.mark.parametrize("dtype", lgamma_dtypes, ids=format_tc)
@pytest.mark.parametrize("input_shape", shape_list, ids=format_tc)
def test_hpu_lgamma(dtype, input_shape):
    def fn(input_x):
        return torch.lgamma(input_x)

    with use_eager_fallback():
        cpu_input_x = torch.randn(input_shape).to(dtype)
        hpu_input_x = cpu_input_x.to("hpu")
        fn = compile_function_if_compile_mode(fn)
        cpu_output = torch.lgamma(cpu_input_x)
        hpu_output = fn(hpu_input_x).cpu()
    assert torch.isclose(cpu_output, hpu_output, atol=0.001, rtol=0.001, equal_nan=True).all()


@pytest.mark.parametrize("dtype", igamma_dtypes, ids=format_tc)
@pytest.mark.parametrize("input_shape", shape_list, ids=format_tc)
def test_hpu_igamma(dtype, input_shape):
    def fn(input_x, input_y):
        return torch.igamma(input_x, input_y)

    with use_eager_fallback():
        cpu_input_x = torch.randn(input_shape).to(dtype)
        cpu_input_y = torch.randn(input_shape).to(dtype)
        hpu_input_x = cpu_input_x.to("hpu")
        hpu_input_y = cpu_input_y.to("hpu")
        fn = compile_function_if_compile_mode(fn)
        cpu_output = torch.igamma(cpu_input_x, cpu_input_y)
        hpu_output = fn(hpu_input_x, hpu_input_y).cpu()
    assert torch.isclose(cpu_output, hpu_output, atol=0.001, rtol=0.001, equal_nan=True).all()


@pytest.mark.parametrize("dtype", igammac_dtypes, ids=format_tc)
@pytest.mark.parametrize("input_shape", shape_list, ids=format_tc)
def test_hpu_igammac(dtype, input_shape):
    def fn(input_x, input_y):
        return torch.igammac(input_x, input_y)

    with use_eager_fallback():
        cpu_input_x = torch.randn(input_shape).to(dtype)
        cpu_input_y = torch.randn(input_shape).to(dtype)
        hpu_input_x = cpu_input_x.to("hpu")
        hpu_input_y = cpu_input_y.to("hpu")
        fn = compile_function_if_compile_mode(fn)
        cpu_output = torch.igammac(cpu_input_x, cpu_input_y)
        hpu_output = fn(hpu_input_x, hpu_input_y).cpu()
    assert torch.isclose(cpu_output, hpu_output, atol=0.001, rtol=0.001, equal_nan=True).all()


@pytest.mark.parametrize("dtype", digamma_dtypes, ids=format_tc)
@pytest.mark.parametrize("input_shape", shape_list, ids=format_tc)
def test_hpu_digamma(dtype, input_shape):
    def fn(input):
        return torch.digamma(input)

    with use_eager_fallback():
        cpu_input = torch.randn(input_shape).to(dtype)
        hpu_input = cpu_input.to("hpu")
        fn = compile_function_if_compile_mode(fn)
        cpu_output = torch.digamma(cpu_input)
        hpu_output = fn(hpu_input).cpu()
    assert torch.isclose(cpu_output, hpu_output, atol=0.001, rtol=0.001, equal_nan=True).all()
